"""CPU evidence checks for original VDRM; no inference or GT-driven decisions."""

import hashlib
import io
import json
import math
from collections import Counter
from pathlib import Path, PurePosixPath
import struct
import tarfile

import numpy as np

ARMS = ('Tclean', 'Rfreeze', 'Rdisc')
PLAN = {
    'GOT-10k_Train_003240': (2, 100),
    'GOT-10k_Train_008333': (2, 74),
    'GOT-10k_Train_001664': (2, 100),
    'GOT-10k_Train_007878': (2, 130),
    'GOT-10k_Train_008311': (2, 100),
}
RECORDS = sum(stop - start + 1 for start, stop in PLAN.values())
STEPS = sum(stop - 1 for _, stop in PLAN.values()) * len(ARMS)
CONFIGURATIONS = {
    'Tclean': 'vitb_256_mae_ce_vdrm_v8_tclean_s42_32x4_ep300',
    'Rfreeze': 'vitb_256_mae_ce_vdrm_rfreeze_s42_32x4_ep300',
    'Rdisc': 'vitb_256_mae_ce_vdrm_rdisc_s42_32x4_ep300',
}


def visual_geometry(anchor, shape, gt):
    """Independent geometry with sample_target's final-row/column padding rule."""
    side = math.ceil(math.sqrt(anchor[2] * anchor[3]) * 4.)
    x, y = [round(anchor[i] + .5 * anchor[i+2] - .5 * side) for i in (0, 1)]
    height, width = shape
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + side, width - 1), min(y + side, height - 1)
    region = [x0, y0, max(0, x1-x0), max(0, y1-y0)]
    def fraction(box):
        if min(gt[2:]) <= 0:
            return None
        size = np.maximum(np.minimum(np.asarray(gt[:2]) + gt[2:], np.asarray(box[:2]) + box[2:])
                          - np.maximum(gt[:2], box[:2]), 0.)
        return float(size.prod() / (gt[2] * gt[3]))
    return dict(square_xywh=[x, y, side, side], valid_source_xywh=region,
                gt_fraction_in_square=fraction([x, y, side, side]),
                gt_fraction_in_visual_support=fraction(region))


def ious(boxes, gt):
    boxes, gt = np.asarray(boxes, float), np.asarray(gt, float)
    intersect = np.maximum(0, np.minimum(boxes[:, :2] + boxes[:, 2:], gt[:2] + gt[2:])
                           - np.maximum(boxes[:, :2], gt[:2])).prod(1)
    union = boxes[:, 2:].prod(1) + gt[2:].prod() - intersect
    return np.divide(intersect, union, out=np.zeros_like(intersect), where=union > 0)


def decode_candidates(arrays, row, shape):
    """Float32 CENTER arithmetic then native Python mapping/clip, all cells."""
    side = arrays['score_map'].shape[-1]
    yy, xx = np.indices((side, side))
    offset, size = arrays['offset_map'][0], arrays['size_map'][0]
    normalized = np.stack(((xx.astype(np.float32) + offset[0]) / side,
                           (yy.astype(np.float32) + offset[1]) / side,
                           size[0], size[1]), -1).reshape(-1, 4)
    decoded = (normalized * np.float32(256) / np.float32(row['resize_factor'])).tolist()
    a = row['anchor_xywh']
    cx0, cy0 = a[0] + .5 * a[2], a[1] + .5 * a[3]
    half = .5 * 256 / row['resize_factor']
    height, width = shape
    boxes = []
    for cx, cy, w, h in decoded:
        x, y = cx + (cx0 - half) - .5 * w, cy + (cy0 - half) - .5 * h
        x1, y1 = min(max(0, x), width - 10), min(max(0, y), height - 10)
        x2, y2 = min(max(10, x + w), width), min(max(10, y + h), height)
        boxes.append([x1, y1, max(10, x2 - x1), max(10, y2 - y1)])
    return np.asarray(boxes)


def validate_arrays(a):
    needed = {'template_tokens', 'search_tokens', 'part_masks', 'prototypes', 'similarity',
              'q', 'part_valid', 'route', 'route_logits', 'indices', 'delta', 'alpha',
              'score_map', 'size_map', 'offset_map', 'hann', 'final_kept', 'padding',
              'template_bbox', 'search_input', 'template_input', 'match_scale', 'match_bias'}
    base_keys = {'base_score_map', 'base_size_map', 'base_offset_map'}
    if set(a) not in (needed, needed | base_keys) or any(not np.isfinite(v).all() for v in a.values()):
        raise ValueError('missing/unexpected/nonfinite evidence arrays')
    if a['search_tokens'].ndim != 3 or a['template_tokens'].shape != (1, 64, 768):
        raise ValueError('wrong V8 template/search shape')
    _, length, channels = a['search_tokens'].shape
    if (channels != 768 or not 1 <= length <= 256 or a['prototypes'].shape != (1, 4, 768)
            or a['part_masks'].shape != (1, 4, 64) or a['q'].shape != (1, 4)
            or a['part_valid'].shape != (1, 4) or a['indices'].shape != (1, length)
            or a['alpha'].size != 1 or a['template_bbox'].shape != (1, 4)
            or a['template_input'].shape != (1, 3, 128, 128)
            or a['search_input'].shape != (1, 3, 256, 256)):
        raise ValueError('wrong internal evidence shape')
    for key in ('similarity', 'route', 'route_logits'):
        if a[key].shape != (1, 4, length):
            raise ValueError('wrong part evidence shape: ' + key)
    if a['delta'].shape != (1, length, 768):
        raise ValueError('wrong delta shape')
    for key, shape in [('score_map', (1, 1, 16, 16)), ('size_map', (1, 2, 16, 16)),
                       ('offset_map', (1, 2, 16, 16)), ('hann', (1, 1, 16, 16)),
                       ('final_kept', (1, 256)), ('padding', (256, 256))]:
        if a[key].shape != shape:
            raise ValueError('wrong output evidence shape: ' + key)
    if base_keys <= set(a):
        for key in ('score_map', 'size_map', 'offset_map'):
            if a['base_' + key].shape != a[key].shape:
                raise ValueError('wrong same-crop baseline shape')
    indices = a['indices'][0]
    if (not np.array_equal(indices, indices.astype(int)) or indices.min() < 0
            or indices.max() >= 256 or len(np.unique(indices)) != length):
        raise ValueError('invalid retained CE indices')
    if not np.isin(a['final_kept'], [0, 1]).all() or not np.isin(a['padding'], [0, 1]).all():
        raise ValueError('invalid support masks')
    if not set(np.flatnonzero(a['final_kept'])) <= set(indices.astype(int)):
        raise ValueError('final CE indices not a subset of insertion indices')
    if not np.isin(a['part_masks'], [0, 1]).all() or not np.isin(a['part_valid'], [0, 1]).all():
        raise ValueError('invalid part masks')
    if not np.array_equal(a['part_valid'], a['part_masks'].any(-1)):
        raise ValueError('part validity mismatch')
    for key in ('q', 'route', 'score_map', 'hann'):
        if (a[key] < 0).any() or (a[key] > 1).any():
            raise ValueError('invalid gate/response range: ' + key)
    count = np.maximum(a['part_masks'].sum(-1, keepdims=True), 1)
    expected = np.einsum('bkl,blc->bkc', a['part_masks'], a['template_tokens']) / count
    if not np.allclose(expected, a['prototypes'], rtol=1e-5, atol=2e-5):
        raise ValueError('prototype does not match original masked mean')
    p = a['prototypes'] / np.maximum(np.linalg.norm(a['prototypes'], axis=-1, keepdims=True), 1e-6)
    s = a['search_tokens'] / np.maximum(np.linalg.norm(a['search_tokens'], axis=-1, keepdims=True), 1e-6)
    if not np.allclose(np.einsum('bkc,blc->bkl', p, s), a['similarity'], atol=2e-5):
        raise ValueError('similarity does not match original features')
    evidence = np.sort(a['similarity'], axis=-1)[..., -min(4, length):].mean(-1)
    expected_q = 1 / (1 + np.exp(-np.clip(a['match_scale'].item() * evidence + a['match_bias'].item(), -80, 80)))
    expected_q *= a['part_valid']
    if not np.allclose(expected_q, a['q'], atol=2e-6):
        raise ValueError('q does not match original topk calibration')
    expected_route = 1 / (1 + np.exp(-np.clip(a['route_logits'], -80, 80)))
    expected_route *= a['part_valid'][..., None]
    if not np.allclose(expected_route, a['route'], atol=2e-6):
        raise ValueError('route does not match logits')
    residual = np.einsum('bkl,bkc->blc', a['route'] * a['q'][..., None], a['prototypes'])
    residual /= max(int(a['part_valid'].sum()), 1)
    if not np.allclose(residual * a['alpha'].item(), a['delta'], rtol=3e-4, atol=3e-5):
        raise ValueError('delta does not match V8 residual equation')


def frame_metrics(a, record, row, arm):
    validate_arrays(a)
    if (arm == 'Rdisc') != ('base_score_map' in a):
        raise ValueError('same-crop baseline maps attached to wrong arm')
    geometry = visual_geometry(row['anchor_xywh'], record['image_shape_hw'], record['gt_xywh'])
    if geometry != record['arms'][arm]['geometry']:
        raise ValueError('saved visual geometry mismatch')
    if abs(row['resize_factor'] - 256 / geometry['square_xywh'][2]) > 1e-12:
        raise ValueError('crop resize mismatch')
    boxes = decode_candidates(a, row, record['image_shape_hw'])
    peak = int((a['score_map'] * a['hann']).argmax())
    error = float(np.abs(boxes[peak] - row['bbox_xywh']).max())
    if error > .002:
        raise ValueError('native CENTER decode mismatch')
    gt = np.asarray(record['gt_xywh'])
    valid = bool((gt[2:] > 0).all())
    overlap = ious(boxes, gt) if valid else None
    if valid and abs(float(ious([row['bbox_xywh']], gt)[0]) - row['iou']) > 1e-10:
        raise ValueError('saved trajectory IoU mismatch')
    visible = valid and record['cover'] > 0 and record['absence'] == 0
    coverage = record['arms'][arm]['geometry']['gt_fraction_in_visual_support']
    if not visible:
        category = 'not_visible_or_invalid'
    elif coverage == 0:
        category = 'outside_visual_crop'
    elif overlap.max() < .5:
        category = 'no_good_candidate'
    elif overlap[peak] < .5:
        category = 'good_candidate_not_selected'
    else:
        category = 'selected_good_candidate'
    p = a['prototypes'][0]
    normalized = p / np.maximum(np.linalg.norm(p, axis=-1, keepdims=True), 1e-6)
    cosine = normalized @ normalized.T
    delta_norm = np.linalg.norm(a['delta'], axis=-1)
    reference = np.maximum(np.linalg.norm(a['search_tokens'], axis=-1), 1e-6)
    result = dict(sequence=record['sequence'], frame_1based=record['frame_1based'], arm=arm,
        category=category, visible=bool(visible), decode_error_pixels=error, native_iou=row['iou'],
        candidate_best_iou=float(overlap.max()) if valid else None,
        good_candidates=int((overlap >= .5).sum()) if valid else None,
        visual_support_fraction=coverage, part_token_counts=a['part_masks'].sum(-1)[0].tolist(),
        part_valid=a['part_valid'][0].tolist(), q=a['q'][0].tolist(),
        q_spread=float(np.ptp(a['q'][0][a['part_valid'][0].astype(bool)])) if a['part_valid'].any() else None,
        prototype_cosine=cosine.tolist(), route_mean_by_part=a['route'].mean(-1)[0].tolist(),
        delta_relative_norm_mean=float((delta_norm / reference).mean()),
        retained_at_insertion=int(a['search_tokens'].shape[1]), retained_final=int(a['final_kept'].sum()),
        automatic_identity_verdict=False)
    if arm == 'Rdisc':
        base = {key: a['base_' + key] for key in ('score_map', 'size_map', 'offset_map')}
        base_boxes = decode_candidates(base, row, record['image_shape_hw'])
        base_peak = int((base['score_map'] * a['hann']).argmax())
        base_iou = float(ious([base_boxes[base_peak]], gt)[0]) if valid else None
        result.update(same_crop_Tclean_iou=base_iou,
                      same_crop_iou_delta=row['iou'] - base_iou if valid else None,
                      same_crop_peak_changed=base_peak != peak,
                      same_crop_Tclean_bbox=base_boxes[base_peak].tolist())
    return result


def check_bundle(path):
    """One streaming archive pass; no extraction, pickle, GPU, or dataset."""
    frames, contents, image_shapes = [], {}, {}
    with tarfile.open(path, 'r|gz') as archive:
        iterator = iter(archive)
        first = next(iterator)
        if first.name != 'manifest.json' or not first.isfile():
            raise ValueError('manifest must be first')
        manifest = json.load(archive.extractfile(first))
        expected = manifest.get('members_sha256', {})
        if (manifest.get('kind') != 'vdrm_module1_step01' or manifest.get('version') != 1
                or manifest.get('plan') != {k: list(v) for k, v in PLAN.items()}
                or manifest.get('records') != RECORDS or manifest.get('native_steps') != STEPS
                or manifest.get('optimizer_steps') != 0 or manifest.get('arms') != list(ARMS)
                or manifest.get('dataset') != 'got10k_vdrm_dev'
                or manifest.get('configurations') != CONFIGURATIONS):
            raise ValueError('wrong step01 protocol')
        if set(manifest.get('provenance', {})) != set(ARMS):
            raise ValueError('missing model provenance')
        for provenance in manifest['provenance'].values():
            if provenance['state_sha256_before'] != provenance['state_sha256_after']:
                raise ValueError('model state changed')
        seen = {'manifest.json'}
        for member in iterator:
            name = member.name
            if (name in seen or name not in expected or not member.isfile()
                    or PurePosixPath(name).is_absolute() or '..' in PurePosixPath(name).parts or '\\' in name):
                raise ValueError('unsafe/duplicate/unexpected member')
            seen.add(name)
            data = archive.extractfile(member).read()
            if hashlib.sha256(data).hexdigest() != expected[name]:
                raise ValueError('evidence checksum mismatch: ' + name)
            if name.endswith('.json'):
                contents[name] = json.loads(data)
            elif name.endswith('.npz'):
                if 'records.json' not in contents or 'trajectories.json' not in contents:
                    raise ValueError('metadata must precede arrays')
                matching = [(r, arm) for r in contents['records.json'] for arm in ARMS
                            if r['arms'][arm]['arrays_member'] == name]
                if len(matching) != 1:
                    raise ValueError('array member has wrong record ownership')
                record, arm = matching[0]
                trace = next(t for t in contents['trajectories.json'] if t['sequence'] == record['sequence'])
                row = trace['native'][arm][record['frame_1based'] - 1]
                with np.load(io.BytesIO(data), allow_pickle=False) as arrays:
                    frames.append(frame_metrics({k: arrays[k] for k in arrays.files}, record, row, arm))
            elif name.endswith('.png'):
                if data[:8] != b'\x89PNG\r\n\x1a\n' or data[12:16] != b'IHDR':
                    raise ValueError('invalid PNG evidence')
                width, height = struct.unpack('>II', data[16:24])
                image_shapes[name] = [height, width]
            else:
                raise ValueError('unexpected member type')
        if seen != set(expected) | {'manifest.json'}:
            raise ValueError('missing archive members')
    records, traces = contents['records.json'], contents['trajectories.json']
    expected_pairs = {(name, frame) for name, (start, stop) in PLAN.items() for frame in range(start, stop + 1)}
    pairs = [(r['sequence'], r['frame_1based']) for r in records]
    if len(pairs) != RECORDS or set(pairs) != expected_pairs or len(frames) != RECORDS * len(ARMS):
        raise ValueError('incomplete/duplicate evidence frames')
    if len(traces) != len(PLAN) or {t['sequence'] for t in traces} != set(PLAN):
        raise ValueError('incomplete trajectory set')
    for trace in traces:
        stop = PLAN[trace['sequence']][1]
        if any(len(trace[k]) != stop for k in ('gt_xywh', 'cover', 'absence')):
            raise ValueError('wrong trajectory labels')
        if set(trace['native']) != set(ARMS):
            raise ValueError('wrong native arms')
        for arm, rows in trace['native'].items():
            if len(rows) != stop or rows[0]['bbox_xywh'] != trace['gt_xywh'][0]:
                raise ValueError('wrong trajectory initialization/length')
            for index, row in enumerate(rows):
                if row['frame_1based'] != index + 1 or not np.isfinite(row['bbox_xywh']).all() or min(row['bbox_xywh'][2:]) <= 0:
                    raise ValueError('wrong float trajectory row')
                if index and (row['anchor_xywh'] != rows[index-1]['bbox_xywh'] or not row['exact_replay']):
                    raise ValueError('broken native float state chain/replay')
                if index:
                    gt = np.asarray(trace['gt_xywh'][index], float)
                    expected_iou = float(ious([row['bbox_xywh']], gt)[0]) if (gt[2:] > 0).all() else None
                    if ((expected_iou is None) != (row['iou'] is None)
                            or (expected_iou is not None and abs(expected_iou-row['iou']) > 1e-10)):
                        raise ValueError('prefix trajectory IoU mismatch')
    for record in records:
        trace = next(t for t in traces if t['sequence'] == record['sequence'])
        index = record['frame_1based'] - 1
        if any(record[k] != trace[k][index] for k in ('gt_xywh', 'cover', 'absence')):
            raise ValueError('record labels differ from trajectory')
        if (set(record['arms']) != set(ARMS) or record['alpha0_exact_Tclean'] is not True
                or image_shapes.get(record['image_member']) != record['image_shape_hw']):
            raise ValueError('missing same-crop bypass check')
    required_images = {r['image_member'] for r in records} | {f'images/{name}/000001.png' for name in PLAN}
    if set(image_shapes) != required_images:
        raise ValueError('incomplete/unexpected original images')
    summary = dict(records=RECORDS, native_steps=STEPS, optimizer_steps=0,
        category_counts={arm: dict(Counter(f['category'] for f in frames if f['arm'] == arm)) for arm in ARMS},
        max_decode_error_pixels=max(f['decode_error_pixels'] for f in frames),
        source_sha256=_file_hash(path), full_dev_gate_passed=False, automatic_identity_verdict=False)
    summary['native_first_box_difference_vs_Tclean'] = {
        trace['sequence']: {arm: next((row['frame_1based'] for row, baseline in zip(trace['native'][arm], trace['native']['Tclean'])
                                      if row['bbox_xywh'] != baseline['bbox_xywh']), None)
                            for arm in ('Rfreeze', 'Rdisc')} for trace in traces}
    samecrop = [f for f in frames if f['arm'] == 'Rdisc' and f['visible']]
    summary['Rdisc_same_crop_visible'] = dict(frames=len(samecrop),
        improve_gt_iou_gt_5pp=sum(f['same_crop_iou_delta'] > .05 for f in samecrop),
        harm_gt_iou_gt_5pp=sum(f['same_crop_iou_delta'] < -.05 for f in samecrop),
        changed_peak=sum(f['same_crop_peak_changed'] for f in samecrop))
    report = ('# Module 1 step01 evidence\n\n'
              f'{RECORDS} saved frames; {STEPS} native steps; no optimization.\n\n'
              '| Arm | Category | Saved frames |\n|---|---|---:|\n')
    for arm, counts in summary['category_counts'].items():
        report += ''.join(f'| {arm} | {category} | {count} |\n' for category, count in sorted(counts.items()))
    report += '\nPosthoc five-case diagnosis only. Counts are not identity-switch counts or full-development performance.\n'
    return summary, frames, report


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(data)
    return digest.hexdigest()
