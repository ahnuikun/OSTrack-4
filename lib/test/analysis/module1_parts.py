"""Static part readouts from verified step01 features; no tracker decisions."""

import hashlib
import io
import json
from pathlib import Path
import tarfile

import numpy as np

from .module1_evidence import ARMS, check_bundle

METHODS = ('whole_mean', 'part_mean', 'part_qmean', 'part_max', 'token_top4')
ROI_SCALES = (1., .8)


def normalized(value):
    return value / np.maximum(np.linalg.norm(value, axis=-1, keepdims=True), 1e-6)


def rank_metrics(scores, positive, negative):
    """Token-pair AUC with exact tie handling; not tracking success AUC."""
    p, n = np.asarray(scores)[positive], np.asarray(scores)[negative]
    if len(p) < 2 or len(n) < 2:
        return None
    difference = p[:, None] - n[None, :]
    return dict(pair_auc=float(((difference > 0) + .5 * (difference == 0)).mean()),
                mean_gap=float(p.mean() - n.mean()), peak_gap=float(p.max() - n.max()),
                positive_tokens=len(p), negative_tokens=len(n))


def inside(points, box):
    box = np.asarray(box, float)
    return ((points >= box[:2]) & (points < box[:2] + box[2:])).all(1)


def scaled_box(box, scale):
    box = np.asarray(box, float)
    size = box[2:] * scale
    return np.r_[box[:2] + (box[2:] - size) / 2, size]


def retained_points(arrays, geometry):
    indices = arrays['indices'][0].astype(int)
    x, y, side, _ = geometry['square_xywh']
    points = np.stack(((indices % 16 + .5) * side / 16 + x,
                       (indices // 16 + .5) * side / 16 + y), axis=-1)
    # Pixel-block support: exclude a token if any pixel in its 16x16 block is padded.
    pad = arrays['padding'].reshape(16, 16, 16, 16).any(axis=(1, 3)).reshape(-1)
    return points, ~pad[indices]


def readouts(arrays):
    valid = arrays['part_valid'][0].astype(bool)
    mask = arrays['part_masks'][0].any(0)
    if not valid.any() or not mask.any():
        return None
    search = normalized(arrays['search_tokens'][0])
    template = arrays['template_tokens'][0, mask]
    whole = normalized(template.mean(0, keepdims=True))[0]
    part = arrays['similarity'][0, valid]
    q = arrays['q'][0, valid]
    token = np.einsum('kc,lc->kl', normalized(template), search)
    k = min(4, token.shape[0])
    scores = dict(whole_mean=np.einsum('c,lc->l', whole, search), part_mean=part.mean(0),
        part_qmean=(part * q[:, None]).sum(0) / max(float(q.sum()), 1e-6),
        part_max=part.max(0), token_top4=np.sort(token, axis=0)[-k:].mean(0))
    return scores


def part_frame(arrays, record, arm, roi=None):
    geometry = record['arms'][arm]['geometry']
    points, support = retained_points(arrays, geometry)
    gt = np.asarray(record['gt_xywh'], float)
    visible = bool((gt[2:] > 0).all() and record['cover'] > 0 and record['absence'] == 0)
    target = inside(points, gt) & support if visible else np.zeros(len(points), bool)
    cell = geometry['square_xywh'][2] / 16
    guard = np.r_[gt[:2] - cell, gt[2:] + 2 * cell]
    negative = ~inside(points, guard) & support
    scores = readouts(arrays)
    valid_parts = arrays['part_valid'][0].astype(bool)
    prototypes = normalized(arrays['prototypes'][0])
    pair_mask = valid_parts[:, None] & valid_parts[None, :] & ~np.eye(4, dtype=bool)
    offdiagonal = (prototypes @ prototypes.T)[pair_mask]
    route_mass = (arrays['q'][..., None] * arrays['route']).sum(axis=1)[0] / max(int(valid_parts.sum()), 1)
    total_mass = float(route_mass[support].sum())
    q_topk = []
    distractor = (inside(points, roi['distractor_xywh']) & support) if roi else None
    for index in range(4):
        if not valid_parts[index]:
            q_topk.append(None)
            continue
        sim = arrays['similarity'][0, index]
        k = min(4, len(sim))
        top = np.argsort(-sim, kind='stable')[:k]
        # Ties at kth boundary make individual ownership non-unique.
        threshold = np.sort(sim)[-k]
        ties = int((sim == threshold).sum()) > int((sim[top] == threshold).sum())
        q_topk.append(dict(gt_fraction=float(target[top].mean()),
            support_fraction=float(support[top].mean()), cutoff_tie=ties,
            distractor_fraction=float(distractor[top].mean()) if roi else None))
    reason = ('not_visible_or_invalid' if not visible else 'no_valid_template_part' if scores is None
              else 'insufficient_gt_tokens' if target.sum() < 2
              else 'insufficient_outside_gt_tokens' if negative.sum() < 2 else None)
    result = dict(sequence=record['sequence'], frame_1based=record['frame_1based'], arm=arm,
        visible=visible, target_tokens=int(target.sum()), supported_tokens=int(support.sum()),
        outside_gt_tokens=int(negative.sum()), exclusion_reason=reason,
        part_token_counts=arrays['part_masks'][0].sum(-1).tolist(),
        prototype_offdiagonal_mean=float(offdiagonal.mean()) if len(offdiagonal) else None,
        q=arrays['q'][0].tolist(), q_top4_ownership=q_topk,
        route_mass_gt_fraction=float(route_mass[target].sum()) / total_mass if total_mass > 0 else None,
        outside_gt_readouts=({key: rank_metrics(value, target, negative) for key, value in scores.items()}
                             if reason is None else None), identity=[])
    if roi:
        for scale in ROI_SCALES:
            pos = inside(points, scaled_box(gt, scale)) & support
            neg = inside(points, scaled_box(roi['distractor_xywh'], scale)) & support
            overlap = pos & neg
            pos &= ~overlap
            neg &= ~overlap
            why = ('not_visible_or_invalid' if not visible else 'no_valid_template_part' if scores is None
                   else 'insufficient_gt_tokens' if pos.sum() < 2
                   else 'insufficient_distractor_tokens' if neg.sum() < 2 else None)
            count = int(pos.sum()), int(neg.sum())
            item = dict(roi_scale=scale, positive_tokens=count[0], negative_tokens=count[1],
                excluded_overlap_tokens=int(overlap.sum()), exclusion_reason=why,
                readouts=({key: rank_metrics(value, pos, neg) for key, value in scores.items()} if why is None else None),
                per_part=([rank_metrics(arrays['similarity'][0, i], pos, neg) if valid_parts[i] else None
                           for i in range(4)] if why is None else None),
                actual_route_mass=rank_metrics(route_mass, pos, neg) if why is None else None)
            result['identity'].append(item)
    return result


def read_annotations(path, source_hash):
    raw = Path(path).read_bytes()
    annotations = json.loads(raw)
    if (annotations.get('kind') != 'vdrm_module1_identity_rois' or annotations.get('version') != 1
            or annotations.get('source_sha256') != source_hash or annotations.get('posthoc') is not True):
        raise ValueError('annotation source/protocol mismatch')
    result = {}
    for item in annotations['entries']:
        key = item['sequence'], item['frame_1based']
        box = np.asarray(item['distractor_xywh'], float)
        if (key in result or box.shape != (4,) or not np.isfinite(box).all() or min(box[2:]) <= 0
                or item['identity_relation'] != 'different_same_class_instance'
                or item['reviewed_by'] != 'assistant_visual_inspection'):
            raise ValueError('invalid/duplicate/unreviewed identity ROI')
        result[key] = item
    if not result:
        raise ValueError('empty identity annotations')
    return result, hashlib.sha256(raw).hexdigest()


def summarize(frames):
    summary = dict(posthoc_only=True, tracking_performance_passed=False,
                   methods=list(METHODS), roi_scales=list(ROI_SCALES), arms={})
    def aggregate(items):
        if not items:
            return dict(eligible_frames=0, sequences=0, sequence_macro=None, per_sequence={})
        sequences = sorted({name for name, _ in items})
        per_sequence = {name: {method: {metric: float(np.mean([value[method][metric] for seq, value in items if seq == name]))
                    for metric in ('pair_auc', 'mean_gap', 'peak_gap')} for method in METHODS} for name in sequences}
        macro = {method: {metric: float(np.mean([per_sequence[seq][method][metric] for seq in sequences]))
                 for metric in ('pair_auc', 'mean_gap', 'peak_gap')} for method in METHODS}
        return dict(eligible_frames=len(items), sequences=len(sequences), sequence_macro=macro, per_sequence=per_sequence)
    for arm in ARMS:
        selected = [r for r in frames if r['arm'] == arm]
        outside = [(r['sequence'], r['outside_gt_readouts']) for r in selected if r['outside_gt_readouts'] is not None]
        identity = {str(scale): aggregate([(r['sequence'], item['readouts']) for r in selected
                    for item in r['identity'] if item['roi_scale'] == scale and item['readouts'] is not None]) for scale in ROI_SCALES}
        exclusions = {}
        for r in selected:
            reason = r['exclusion_reason'] or 'eligible'
            exclusions[reason] = exclusions.get(reason, 0) + 1
        summary['arms'][arm] = dict(outside_gt=aggregate(outside), identity=identity, eligibility=exclusions)
    return summary


def analyze(bundle, annotations_path, progress=None):
    verified, _, _ = check_bundle(bundle)
    annotations, annotation_hash = read_annotations(annotations_path, verified['source_sha256'])
    if progress:
        progress('STEP01 SOURCE VERIFIED; computing fixed static part readouts', flush=True)
    frames = []
    with tarfile.open(bundle, 'r|gz') as archive:
        for member in archive:
            if member.name == 'manifest.json':
                manifest = json.load(archive.extractfile(member))
            elif member.name == 'records.json':
                records = json.load(archive.extractfile(member))
                index = {r['arms'][arm]['arrays_member']: (r, arm) for r in records for arm in ARMS}
                valid_keys = {(r['sequence'], r['frame_1based']) for r in records}
                if not set(annotations) <= valid_keys:
                    raise ValueError('annotation frame absent from source')
                for r in records:
                    item = annotations.get((r['sequence'], r['frame_1based']))
                    if item and (item['image_member'] != r['image_member']
                                 or item['image_sha256'] != manifest['members_sha256'][r['image_member']]):
                        raise ValueError('annotation image mismatch')
                    if item:
                        h, w = r['image_shape_hw']
                        x, y, bw, bh = item['distractor_xywh']
                        if x < 0 or y < 0 or x+bw > w+.002 or y+bh > h+.002:
                            raise ValueError('annotation box outside original image')
            elif member.name.endswith('.npz'):
                data = archive.extractfile(member).read()
                if hashlib.sha256(data).hexdigest() != manifest['members_sha256'][member.name]:
                    raise ValueError('array changed after initial verification')
                record, arm = index[member.name]
                with np.load(io.BytesIO(data), allow_pickle=False) as raw:
                    # Images and full inputs are already checked by step01; read only feature fields here.
                    keys = ('part_valid', 'part_masks', 'search_tokens', 'template_tokens', 'similarity',
                            'prototypes', 'q', 'route', 'indices', 'padding')
                    arrays = {key: raw[key] for key in keys}
                frames.append(part_frame(arrays, record, arm, annotations.get((record['sequence'], record['frame_1based']))))
                if progress and len(frames) % 300 == 0:
                    progress(f'PARTS: {len(frames)}/1497 cached arm records', flush=True)
    if len(frames) != 1497:
        raise ValueError('incomplete part analysis')
    summary = summarize(frames)
    summary.update(source_sha256=verified['source_sha256'], annotations_sha256=annotation_hash,
                   records=len(frames), annotated_frames=len(annotations), optimizer_steps=0)
    report = '# Step02 static part readouts\n\nPosthoc cache diagnosis; no training or new tracking trajectory.\n\n'
    report += '| Arm | Scope | Eligible frames/sequences | Method | Token-pair AUC | Mean gap |\n|---|---|---|---|---:|---:|\n'
    for arm, values in summary['arms'].items():
        for scope, result in [('outside_gt', values['outside_gt']), *[('identity_' + k, v) for k, v in values['identity'].items()]]:
            for method, metrics in (result['sequence_macro'] or {}).items():
                report += f"| {arm} | {scope} | {result['eligible_frames']}/{result['sequences']} | {method} | {metrics['pair_auc']:.6f} | {metrics['mean_gap']:.6f} |\n"
    report += '\nToken AUC is not tracking AUC. ROI scores are conditional on two supported tokens per side. Related frames/arms/scales are not independent samples.\n'
    return summary, frames, report
