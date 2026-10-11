"""Independent CPU verification of signed residual sweeps and head selection."""

from collections import Counter
import hashlib
import io
import json
from pathlib import PurePosixPath
import tarfile

import numpy as np

from .module1_evidence import _file_hash, decode_candidates, ious
from .module1_gates import ARMS, CONFIGURATIONS, frame_metrics, protocol_plan
from .module1_location import SCALES, support_mask, validate_proxy_arrays

GAINS = (-4., -1., 0., 1., 4., 16.)
LOCATION_GAINS = tuple(g for g in GAINS if g != 0)


def schedule(eligibility):
    # Evaluate exact native control first; gain zero exactly matches saved Tclean.
    return [('original', g, None) for g in (1., 0., -1., -4., 4., 16.)] + [
        ('route_target_norm', g, s) for s in SCALES
        if eligibility[str(s)]['reason'] is None for g in LOCATION_GAINS]


def selection_metrics(a, row, baseline):
    """Readouts are diagnostics on the same maps, not new tracker trajectories.

    Fixed good/bad masks come from ORIGINAL predicted boxes, then stay fixed
    across strengths. Per-arm candidate availability is also reported separately.
    Removed cells remain in the actual decoder; retained-only counts expose them.
    """
    boxes = decode_candidates(a, row, row['image_shape_hw'])
    base_boxes = decode_candidates(baseline, row, row['image_shape_hw'])
    gt = row['gt_xywh']
    valid = bool(min(gt[2:]) > 0 and row['cover'] > 0 and row['absence'] == 0)
    overlap, base_overlap = ious(boxes, gt), ious(base_boxes, gt)
    raw = a['score_map'].reshape(-1)
    raw_peak = int(raw.argmax())
    kept = a['final_kept'].reshape(-1).astype(bool)
    fixed_good = base_overlap >= .5
    result = dict(raw_peak=raw_peak, raw_peak_bbox_xywh=boxes[raw_peak].tolist(),
                  raw_peak_iou=float(overlap[raw_peak]) if valid else None,
                  best_retained_iou=float(overlap[kept].max()) if valid and kept.any() else None,
                  good_retained_candidates=int(((overlap >= .5) & kept).sum()) if valid else None,
                  good_removed_candidates=int(((overlap >= .5) & ~kept).sum()) if valid else None,
                  final_kept_changes=int((a['final_kept'] != baseline['final_kept']).sum()),
                  score_max_abs_change=float(np.abs(a['score_map'] - baseline['score_map']).max()))
    for label, scores in (('raw', raw), ('hann', (a['score_map'] * a['hann']).reshape(-1))):
        result['fixed_' + label + '_margin'] = (
            float(scores[fixed_good].max() - scores[~fixed_good].max())
            if valid and fixed_good.any() and (~fixed_good).any() else None)
    return result


def validate_strength_arrays(a, row, control):
    mode = 'control' if row['family'] == 'original' else 'route_target_norm'
    validate_proxy_arrays(a, mode)
    expected_alpha = np.float32(control['alpha'].item()) * np.float32(row['strength'])
    if (not np.array_equal(a['reference_alpha'], control['alpha'])
            or not np.isclose(a['alpha'].item(), expected_alpha, rtol=1e-6, atol=1e-9)):
        raise ValueError('wrong signed effective alpha')
    norm2 = a['residual_norm'].astype(float)**2
    signed_dot = a['alpha'].item() * norm2
    if (a['injected_residual_dot'].shape != norm2.shape
            or not np.allclose(a['injected_residual_dot'], signed_dot, rtol=3e-3, atol=3e-3)):
        raise ValueError('actual signed injection projection mismatch')
    if a['injection_roundoff_max'].shape != (1,) or not 0 <= a['injection_roundoff_max'].item() <= 1e-4:
        raise ValueError('incorrect residual injection roundoff')
    expected_relative = np.sqrt(a['injected_energy']) / np.maximum(a['search_norm'], 1e-6)
    if a['search_norm'].shape != norm2.shape or not np.allclose(a['injected_relative_norm'], expected_relative, rtol=1e-5, atol=1e-7):
        raise ValueError('incorrect relative injection norm')
    if row['strength'] == 0 and (np.any(a['injected_energy']) or np.any(a['injected_residual_dot'])):
        raise ValueError('zero gain injected a nonzero update')


def summarize(frames, cases, smoke):
    lookup = {(f['sequence'], f['frame_1based'], f['family'], f['strength'], f['roi_scale']): f for f in frames}
    groups = {}
    # Every gain has one fixed population per family/scale; no per-frame best gain.
    for family, scales, gains in [('original', (None,), GAINS), ('route_target_norm', SCALES, LOCATION_GAINS)]:
        for scale in scales:
            selected = [c for c in cases if (scale is None or c['eligibility'][str(scale)]['reason'] is None)
                        and c['cover'] > 0 and c['absence'] == 0 and min(c['gt_xywh'][2:]) > 0]
            for gain in gains:
                sequences = {}
                for name in sorted({c['sequence'] for c in selected}):
                    pairs = [(lookup[name, c['frame_1based'], 'original', 1., None],
                              lookup[name, c['frame_1based'], family, gain, scale]) for c in selected if c['sequence'] == name]
                    deltas = [b['iou'] - a['iou'] for a, b in pairs]
                    original_same_gain = [b['iou'] - lookup[name,b['frame_1based'],'original',gain,None]['iou'] for _,b in pairs]
                    family_gain_one = [b['iou'] - lookup[name,b['frame_1based'],family,1.,scale]['iou'] for _,b in pairs]
                    sequences[name] = dict(frames=len(pairs), mean_delta_iou=float(np.mean(deltas)),
                        mean_delta_vs_original_same_gain=float(np.mean(original_same_gain)),
                        mean_delta_vs_family_gain_one=float(np.mean(family_gain_one)),
                        improve_gt_5pp=sum(d > .05 for d in deltas), harm_gt_5pp=sum(d < -.05 for d in deltas),
                        repair_at_50=sum(a['iou'] < .5 <= b['iou'] for a, b in pairs),
                        lose_at_50=sum(b['iou'] < .5 <= a['iou'] for a, b in pairs),
                        raw_repair_vs_native_at_50=sum(a['iou'] < .5 <= b['raw_peak_iou'] for a, b in pairs),
                        raw_lose_vs_native_at_50=sum(b['raw_peak_iou'] < .5 <= a['iou'] for a, b in pairs),
                        peak_changes=sum(a['peak'] != b['peak'] for a, b in pairs))
                groups[f'{family}/{scale}/{gain:g}'] = dict(family=family, roi_scale=scale, strength=gain,
                    frames=len(selected), per_sequence=sequences,
                    sequence_macro_delta_iou=float(np.mean([d['mean_delta_iou'] for d in sequences.values()])) if sequences else None)
    return dict(smoke_only=smoke, optimizer_steps=0, tracking_performance_passed=False,
                fixed_input_only=True, current_gt_used_by_location_family=True,
                gain_exceeds_trained_bound_allowed=True, per_frame_gain_selection=False,
                groups=groups, eligibility={str(s):dict(Counter(c['eligibility'][str(s)]['reason'] or 'eligible' for c in cases)) for s in SCALES})


def check_bundle(path):
    contents, seen = {}, set()
    with tarfile.open(path, 'r|gz') as archive:
        for member in archive:
            name = member.name
            if (not member.isfile() or name in seen or PurePosixPath(name).is_absolute()
                    or '..' in PurePosixPath(name).parts or '\\' in name):
                raise ValueError('unsafe/duplicate strength member')
            seen.add(name)
            data = archive.extractfile(member).read()
            if len(seen) == 1:
                if name != 'manifest.json': raise ValueError('manifest must be first')
                manifest = json.loads(data)
            else:
                if hashlib.sha256(data).hexdigest() != manifest['members_sha256'].get(name):
                    raise ValueError('strength artifact checksum mismatch')
                contents[name] = data
    if seen != set(manifest['members_sha256']) | {'manifest.json'}:
        raise ValueError('incomplete strength bundle')
    smoke = manifest.get('smoke_only')
    plan = protocol_plan(smoke)
    if (type(smoke) is not bool or manifest.get('kind') != 'vdrm_module1_step05_strength' or manifest.get('version') != 1
            or manifest.get('plan') != {k:list(v) for k,v in plan.items()} or manifest.get('gains') != list(GAINS)
            or manifest.get('scales') != list(SCALES) or manifest.get('optimizer_steps') != 0
            or manifest.get('source_verified') is not True or manifest.get('fixed_input_only') is not True
            or manifest.get('configurations') != {a:CONFIGURATIONS[a] for a in ARMS}
            or len(manifest.get('source_sha256','')) != 64):
        raise ValueError('wrong strength protocol')
    for arm in ARMS:
        p, old = manifest['provenance'][arm], manifest['source_provenance'][arm]
        if p['state_sha256_before'] != p['state_sha256_after'] or any(p[k] != old[k] for k in ('checkpoint_sha256','state_sha256_before')):
            raise ValueError('changed/wrong strength model')
    cases, rows = json.loads(contents['cases.json']), json.loads(contents['rows.json'])
    expected_cases = {(n,f) for n,(start,stop) in plan.items() for f in range(start,stop+1)}
    case_index = {(c['sequence'],c['frame_1based']):c for c in cases}
    if len(cases) != len(expected_cases) or set(case_index) != expected_cases:
        raise ValueError('missing/duplicate strength cases')
    expected = {(c['sequence'],c['frame_1based'],family,g,s) for c in cases for family,g,s in schedule(c['eligibility'])}
    keys = [(r['sequence'],r['frame_1based'],r['family'],r['strength'],r['roi_scale']) for r in rows]
    if len(keys) != len(expected) or set(keys) != expected or manifest.get('forward_calls') != len(rows):
        raise ValueError('missing/duplicate strength rows or count')
    members = {r['arrays_member'] for r in rows}
    if len(members) != len(rows) or set(contents) != {'rows.json','cases.json'} | members:
        raise ValueError('unexpected strength evidence')
    arrays = {}
    for member in members:
        with np.load(io.BytesIO(contents[member]), allow_pickle=False) as raw:
            arrays[member] = {k:raw[k] for k in raw.files}
    index = dict(zip(keys, rows))
    for case in cases:
        base = index[case['sequence'],case['frame_1based'],'original',1.,None]
        if base.get('source_maps_exact') is not True or base['bbox_xywh'] != case['source_bbox_xywh']:
            raise ValueError('strength control differs from native source')
        a = arrays[base['arrays_member']]
        for scale in SCALES:
            mask, reason = support_mask(a,case,scale)
            if case['eligibility'][str(scale)] != dict(reason=reason,target_tokens=int(mask.sum())):
                raise ValueError('wrong strength eligibility')
    frames = []
    for row in rows:
        case = case_index[row['sequence'],row['frame_1based']]
        if any(row[k] != case[k] for k in ('anchor_xywh','resize_factor','input_sha256','gt_xywh','cover','absence','image_shape_hw')):
            raise ValueError('strength changed fixed input/labels')
        mode = 'control' if row['family'] == 'original' else 'route_target_norm'
        if row.get('scope') != 'shared_residual_strength' or row['mode'] != mode:
            raise ValueError('wrong strength row mode/scope')
        a = arrays[row['arrays_member']]
        base = arrays[index[row['sequence'],row['frame_1based'],'original',1.,None]['arrays_member']]
        for key in ('raw_q','raw_route','prototype_gram','indices','part_valid','padding','similarity','match_scale','match_bias','hann','search_norm'):
            if not np.array_equal(a[key],base[key]): raise ValueError('changed strength prefix: '+key)
        if row['family'] != 'original':
            mask, reason = support_mask(a,row,row['roi_scale'])
            if reason is not None or not np.array_equal(mask,a['target_mask'][0]): raise ValueError('wrong strength target support')
        if row['strength'] == 0:
            if row.get('saved_Tclean_exact') is not True: raise ValueError('missing zero/Tclean check')
            for k in ('score_map','size_map','offset_map'):
                if not np.array_equal(a[k],a['source_'+k]): raise ValueError('zero differs from saved Tclean')
        validate_strength_arrays(a,row,base)
        f = frame_metrics(a,row,gate_validator=validate_proxy_arrays)
        f.update({k:row[k] for k in ('family','strength','roi_scale')})
        f.update(selection_metrics(a,row,base))
        frames.append(f)
    summary = summarize(frames,cases,smoke)
    summary.update(source_sha256=manifest['source_sha256'],evidence_sha256=_file_hash(path),
                   cases=len(cases),records=len(rows),forward_calls=len(rows),
                   max_decode_error_pixels=max(f['decode_error_pixels'] for f in frames))
    return summary,frames
