"""CPU verification and paired summaries for step03; no tracking imports."""

from collections import Counter
import hashlib
import io
import json
from pathlib import PurePosixPath
import tarfile

import numpy as np

from .module1_evidence import PLAN, CONFIGURATIONS, _file_hash, check_bundle as check_source, decode_candidates, ious, visual_geometry

MODES = ('control', 'qflat', 'routeflat', 'bothflat', 'alpha0')
SHARED_MODES = MODES + ('qflat_norm', 'routeflat_norm', 'bothflat_norm')
ARMS = ('Tclean', 'Rdisc')


def protocol_plan(smoke=False):
    return {name: (2, 3 if smoke else stop) for name, (_, stop) in PLAN.items()}


def read_source(path):
    verified, _, _ = check_source(path)
    metadata = {}
    with tarfile.open(path, 'r|gz') as archive:
        for member in archive:
            if member.name in ('manifest.json', 'records.json', 'trajectories.json'):
                metadata[member.name] = json.load(archive.extractfile(member))
            if len(metadata) == 3:
                break
    return dict(sha256=verified['source_sha256'], manifest=metadata['manifest.json'],
                records=metadata['records.json'], trajectories=metadata['trajectories.json'])


def validate_gate_arrays(a, mode):
    base = mode[:-5] if mode.endswith('_norm') else mode
    valid = a['part_valid'].astype(bool)
    q, route = a['raw_q'], a['raw_route']
    length = route.shape[-1]
    if (q.shape != (1, 4) or route.shape != (1, 4, length) or valid.shape != (1, 4)
            or a['effective_q'].shape != q.shape or a['effective_route'].shape != route.shape
            or a['prototype_gram'].shape != (1, 4, 4) or a['indices'].shape != (1, length)
            or a['norm_gain'].shape != (1, 1, 1) or a['alpha'].size != 1
            or a['residual_norm'].shape != (1, length)
            or a['injected_energy'].shape != (1, length)
            or a['injected_relative_norm'].shape != (1, length)):
        raise ValueError('gate evidence shape mismatch')
    for key in ('raw_q', 'raw_route', 'effective_q', 'effective_route'):
        if (a[key] < 0).any() or (a[key] > 1).any():
            raise ValueError('gate outside [0,1]')
    if (q[~valid] != 0).any() or (route[~valid] != 0).any():
        raise ValueError('invalid part has nonzero gate')
    count = max(int(valid.sum()), 1)
    expected_q = q.sum(-1, keepdims=True) / count * valid if base in ('qflat', 'bothflat') else q
    expected_route = np.broadcast_to(route.mean(-1, keepdims=True), route.shape) if base in ('routeflat', 'bothflat') else route
    if (not np.allclose(a['effective_q'], expected_q, atol=2e-7, rtol=1e-6)
            or not np.allclose(a['effective_route'], expected_route, atol=2e-7, rtol=1e-6)):
        raise ValueError('effective gates do not implement requested intervention')
    def energy(q_value, route_value):
        weights = (q_value[..., None] * route_value / count).astype(np.float64)
        return np.maximum(0, np.einsum('bkl,bkm,bml->bl', weights, a['prototype_gram'].astype(np.float64), weights))
    original_energy = energy(q, route)
    changed_energy = energy(a['effective_q'], a['effective_route'])
    expected_gain = 1.
    if mode.endswith('_norm'):
        if changed_energy.sum() == 0 and original_energy.sum() > 0:
            raise ValueError('undefined norm compensation')
        if changed_energy.sum() > 0:
            expected_gain = np.sqrt(original_energy.sum() / changed_energy.sum())
    gain = float(a['norm_gain'].item())
    if not np.isclose(gain, expected_gain, rtol=2e-5, atol=2e-6):
        raise ValueError('incorrect residual norm gain')
    if not np.allclose(a['residual_norm'] ** 2, changed_energy * gain**2, rtol=1e-4, atol=1e-5):
        raise ValueError('residual formula does not match Gram-matrix norm')
    alpha = float(a['alpha'].item())
    expected_norm = np.sqrt(changed_energy) * gain * abs(alpha)
    actual_norm = np.sqrt(a['injected_energy'])
    if (a['injected_energy'] < 0).any() or (a['injected_relative_norm'] < 0).any():
        raise ValueError('negative injected norm/energy')
    # output-search subtraction rounds in float32; store actual energy separately.
    if not np.allclose(actual_norm, expected_norm, rtol=3e-3, atol=3e-4):
        raise ValueError('actual injection norm differs from effective gate formula')
    if base == 'alpha0' and (alpha != 0 or np.any(a['injected_energy'] != 0)):
        raise ValueError('alpha0 did not disable injection')


def frame_metrics(a, row, gate_validator=validate_gate_arrays):
    if any(not np.isfinite(value).all() for value in a.values()):
        raise ValueError('nonfinite gate evidence')
    gate_validator(a, row['mode'])
    shapes = dict(score_map=(1, 1, 16, 16), size_map=(1, 2, 16, 16), offset_map=(1, 2, 16, 16),
                  hann=(1, 1, 16, 16), final_kept=(1, 256), padding=(256, 256))
    if any(a[key].shape != shape for key, shape in shapes.items()):
        raise ValueError('wrong tracking map shape')
    if not np.isin(a['part_valid'], [0, 1]).all() or any(not np.isin(a[k], [0, 1]).all() for k in ('final_kept', 'padding')):
        raise ValueError('invalid mask')
    indices = a['indices'][0]
    if (len(indices) == 0 or len(indices) > 256 or not np.array_equal(indices, indices.astype(int))
            or len(np.unique(indices)) != len(indices) or indices.min() < 0 or indices.max() >= 256
            or not set(np.flatnonzero(a['final_kept'])) <= set(indices.astype(int))):
        raise ValueError('invalid retained indices')
    for key in ('score_map', 'hann'):
        if (a[key] < 0).any() or (a[key] > 1).any():
            raise ValueError('invalid response/window')
    geometry = visual_geometry(row['anchor_xywh'], row['image_shape_hw'], row['gt_xywh'])
    if abs(row['resize_factor'] - 256 / geometry['square_xywh'][2]) > 1e-12:
        raise ValueError('incorrect native crop resize')
    boxes = decode_candidates(a, row, row['image_shape_hw'])
    scores = (a['score_map'] * a['hann']).reshape(-1)
    peak = int(scores.argmax())
    error = float(np.abs(boxes[peak] - row['bbox_xywh']).max())
    if error > .002:
        raise ValueError('native float decode mismatch')
    gt = np.asarray(row['gt_xywh'])
    valid = bool((gt[2:] > 0).all())
    overlap = ious(boxes, gt) if valid else None
    overlap_selected = float(ious([row['bbox_xywh']], gt)[0]) if valid else None
    if (overlap_selected is None) != (row['iou'] is None) or (valid and abs(overlap_selected-row['iou']) > 1e-10):
        raise ValueError('saved IoU mismatch')
    visible = valid and row['cover'] > 0 and row['absence'] == 0
    coverage = geometry['gt_fraction_in_visual_support']
    category = ('not_visible_or_invalid' if not visible else 'outside_visual_crop' if coverage == 0
                else 'no_good_candidate' if overlap.max() < .5 else 'good_candidate_not_selected' if overlap[peak] < .5
                else 'selected_good_candidate')
    margin = None
    if visible and (overlap >= .5).any() and (overlap < .5).any():
        margin = float(scores[overlap >= .5].max() - scores[overlap < .5].max())
    return dict(sequence=row['sequence'], frame_1based=row['frame_1based'], scope=row['scope'], mode=row['mode'],
        visible=bool(visible), iou=overlap_selected, category=category, peak=peak, candidate_best_iou=float(overlap.max()) if valid else None,
        candidate_score_margin=margin, visual_support_fraction=coverage, retained_final=int(a['final_kept'].sum()),
        q=a['raw_q'][0].tolist(), effective_q=a['effective_q'][0].tolist(), norm_gain=float(a['norm_gain'].item()),
        injected_frobenius_norm=float(np.sqrt(a['injected_energy'].sum())),
        injected_relative_norm_mean=float(a['injected_relative_norm'].mean()), decode_error_pixels=error)


def summarize(frames, smoke):
    result = dict(smoke_only=smoke, performance_passed=False, automatic_identity_verdict=False, optimizer_steps=0, scopes={})
    for scope, modes in [('closed', MODES), ('shared', SHARED_MODES)]:
        selected = [r for r in frames if r['scope'] == scope]
        by_key = {(r['sequence'], r['frame_1based'], r['mode']): r for r in selected}
        sequences = sorted({r['sequence'] for r in selected})
        details = {}
        for name in sequences:
            rows = [r for r in selected if r['sequence'] == name and r['mode'] == 'control' and r['visible']]
            comparisons = {}
            for mode in modes:
                pairs = [(r, by_key[name, r['frame_1based'], mode]) for r in rows]
                deltas = [b['iou'] - a['iou'] for a, b in pairs]
                comparisons[mode] = dict(visible_frames=len(pairs), mean_iou=float(np.mean([b['iou'] for _, b in pairs])) if pairs else None,
                    mean_delta_iou=float(np.mean(deltas)) if deltas else None,
                    improve_gt_5pp=sum(d > .05 for d in deltas), harm_gt_5pp=sum(d < -.05 for d in deltas),
                    repair_at_50=sum(a['iou'] < .5 <= b['iou'] for a, b in pairs),
                    lose_at_50=sum(b['iou'] < .5 <= a['iou'] for a, b in pairs),
                    peak_changes=sum(a['peak'] != b['peak'] for a, b in pairs) if scope == 'shared' else None,
                    category_counts=dict(Counter(b['category'] for _, b in pairs)),
                    first_harm_gt_5pp=next((a['frame_1based'] for a, b in pairs if b['iou'] - a['iou'] < -.05), None))
            interactions = [by_key[name, r['frame_1based'], 'bothflat']['iou'] - by_key[name, r['frame_1based'], 'qflat']['iou']
                            - by_key[name, r['frame_1based'], 'routeflat']['iou'] + r['iou'] for r in rows]
            details[name] = dict(modes=comparisons, interaction_mean_iou=float(np.mean(interactions)) if interactions else None)
        macro = {}
        for mode in modes:
            values = [v['modes'][mode]['mean_delta_iou'] for v in details.values()
                      if v['modes'][mode]['mean_delta_iou'] is not None]
            macro[mode] = float(np.mean(values)) if values else None
        result['scopes'][scope] = dict(per_sequence=details, sequence_macro_delta_iou=macro)
    return result


def check_bundle(path):
    contents, seen = {}, set()
    with tarfile.open(path, 'r|gz') as archive:
        for member in archive:
            name = member.name
            if (not member.isfile() or name in seen or PurePosixPath(name).is_absolute()
                    or '..' in PurePosixPath(name).parts or '\\' in name):
                raise ValueError('unsafe/duplicate bundle member')
            seen.add(name)
            data = archive.extractfile(member).read()
            if len(seen) == 1:
                if name != 'manifest.json':
                    raise ValueError('manifest must be first')
                manifest = json.loads(data)
            else:
                if hashlib.sha256(data).hexdigest() != manifest['members_sha256'].get(name):
                    raise ValueError('missing checksum or changed member: ' + name)
                contents[name] = data
    if seen != {'manifest.json'} | set(manifest['members_sha256']):
        raise ValueError('incomplete bundle')
    smoke = manifest.get('smoke_only')
    plan = protocol_plan(smoke)
    if (type(smoke) is not bool or manifest.get('kind') != 'vdrm_module1_step03_gates' or manifest.get('version') != 1
            or manifest.get('dataset') != 'got10k_vdrm_dev' or manifest.get('plan') != {k: list(v) for k, v in plan.items()}
            or manifest.get('modes') != list(MODES) or manifest.get('shared_modes') != list(SHARED_MODES)
            or manifest.get('optimizer_steps') != 0 or manifest.get('source_verified') is not True
            or manifest.get('configurations') != {arm: CONFIGURATIONS[arm] for arm in ARMS}):
        raise ValueError('wrong step03 protocol')
    source = json.loads(contents['source_reference.json'])
    if source['sha256'] != manifest['source_sha256'] or len(source['sha256']) != 64:
        raise ValueError('source reference mismatch')
    for arm in ARMS:
        p, ref = manifest['provenance'][arm], source['provenance'][arm]
        if (p['state_sha256_before'] != p['state_sha256_after'] or p['state_sha256_before'] != ref['state_sha256_before']
                or p['checkpoint_sha256'] != ref['checkpoint_sha256']):
            raise ValueError('changed/wrong checkpoint or model state')
    rows = json.loads(contents['rows.json'])
    expected = {(name, frame, scope, mode) for name, (start, stop) in plan.items() for frame in range(start, stop+1)
                for scope, modes in [('closed', MODES), ('shared', SHARED_MODES)] for mode in modes}
    actual = [(r['sequence'], r['frame_1based'], r['scope'], r['mode']) for r in rows]
    count = sum(stop-start+1 for start, stop in plan.values())
    if (len(rows) != len(expected) or set(actual) != expected or manifest.get('native_steps') != count*5
            or manifest.get('forward_calls') != count*18):
        raise ValueError('incomplete/duplicate frames or wrong forward counts')
    references = {(r['sequence'], r['frame_1based']): r for r in source['frames']}
    if len(references) != count or set(references) != {(n, f) for n, (start, stop) in plan.items() for f in range(start, stop+1)}:
        raise ValueError('incomplete source reference')
    by_key, arrays, frames = dict(zip(actual, rows)), {}, []
    members = {r['arrays_member'] for r in rows}
    if set(contents) != members | {'rows.json', 'source_reference.json'}:
        raise ValueError('unexpected/missing data artifacts')
    for member in members:
        with np.load(io.BytesIO(contents[member]), allow_pickle=False) as raw:
            arrays[member] = {key: raw[key] for key in raw.files}
    for row in rows:
        name, frame, scope, mode = row['sequence'], row['frame_1based'], row['scope'], row['mode']
        ref = references[name, frame]
        if (any(row[k] != ref[k] for k in ('gt_xywh', 'cover', 'absence', 'image_shape_hw'))
                or row.get('image_sha256') != ref['image_sha256']
                or row.get('exact_replay' if scope == 'closed' else 'native_input_verified') is not True):
            raise ValueError('source labels/image or replay mismatch')
        closed_control = by_key[name, frame, 'closed', 'control']
        if scope == 'closed':
            previous = source['initial_boxes'][name] if frame == 2 else by_key[name, frame-1, scope, mode]['bbox_xywh']
            if row['anchor_xywh'] != previous:
                raise ValueError('broken closed-loop float state chain')
            if mode in ('control', 'alpha0'):
                old = ref['native']['Rdisc' if mode == 'control' else 'Tclean']
                if any(row[k] != old[k] for k in ('bbox_xywh', 'anchor_xywh', 'input_sha256')):
                    raise ValueError('native baseline differs from step01')
        else:
            if any(row[k] != closed_control[k] for k in ('anchor_xywh', 'resize_factor', 'input_sha256')):
                raise ValueError('shared input differs from current control')
            if mode == 'control' and (row['bbox_xywh'] != closed_control['bbox_xywh'] or row['arrays_member'] != closed_control['arrays_member']):
                raise ValueError('shared control differs from native')
        a = arrays[row['arrays_member']]
        if scope == 'shared':
            control = arrays[closed_control['arrays_member']]
            for field in ('raw_q', 'raw_route', 'prototype_gram', 'indices', 'part_valid', 'padding'):
                if not np.array_equal(a[field], control[field]):
                    raise ValueError('shared intervention changed pre-residual evidence: ' + field)
            if mode == 'alpha0':
                if row.get('alpha0_exact_Tclean') is not True:
                    raise ValueError('missing alpha0/Tclean check')
                for field in ('score_map', 'size_map', 'offset_map', 'final_kept'):
                    if not np.array_equal(a[field], a['tclean_' + field]):
                        raise ValueError('alpha0 visual maps differ from Tclean')
        frames.append(frame_metrics(a, row))
    summary = summarize(frames, smoke)
    summary.update(source_sha256=source['sha256'], evidence_sha256=_file_hash(path), records=len(rows), native_steps=count*5, forward_calls=count*18,
                   max_decode_error_pixels=max(r['decode_error_pixels'] for r in frames))
    return summary, frames


def report_text(summary):
    text = '# Step03 gate intervention diagnosis\n\n'
    text += f"Smoke only: {summary['smoke_only']}. Five posthoc development cases; no performance or identity pass.\n\n"
    text += '| Scope | Sequence | Mode | Visible frames | Mean IoU | Delta IoU (pp) | Improve/harm >5pp | Repair/lose at .5 |\n|---|---|---|---:|---:|---:|---|---|\n'
    interactions = []
    for scope, value in summary['scopes'].items():
        for name, detail in value['per_sequence'].items():
            for mode, r in detail['modes'].items():
                if r['mean_iou'] is None:
                    continue
                text += f"| {scope} | {name} | {mode} | {r['visible_frames']} | {r['mean_iou']:.6f} | {r['mean_delta_iou']*100:.4f} | {r['improve_gt_5pp']}/{r['harm_gt_5pp']} | {r['repair_at_50']}/{r['lose_at_50']} |\n"
            interactions.append(f"{scope}/{name}: interaction = bothflat - qflat - routeflat + control = {detail['interaction_mean_iou']} IoU.")
    text += '\n' + '\n\n'.join(interactions) + '\n'
    text += '\nShared crops measure immediate effects. Closed loops include changing histories. Norm controls only match global pre-injection Frobenius norm; local energy and direction may differ. Peak indices across different crops are not directly comparable. GT labels do not prove instance identity.\n'
    return text
