"""CPU checks for explicit GT-location proxies, with support exclusions."""

from collections import Counter
import hashlib
import io
import json
from pathlib import PurePosixPath
import tarfile

import numpy as np

from .module1_evidence import _file_hash, visual_geometry
from .module1_gates import ARMS, CONFIGURATIONS, frame_metrics, protocol_plan, validate_gate_arrays
from .module1_parts import retained_points, inside, scaled_box

SCALES = (1., .8)
PROXIES = ('q_target', 'route_target', 'joint_target', 'q_target_norm', 'route_target_norm', 'joint_target_norm')


def support_mask(arrays, row, scale):
    geometry = visual_geometry(row['anchor_xywh'], row['image_shape_hw'], row['gt_xywh'])
    points, support = retained_points(arrays, geometry)
    gt = np.asarray(row['gt_xywh'])
    visible = bool((gt[2:] > 0).all() and row['cover'] > 0 and row['absence'] == 0)
    mask = inside(points, scaled_box(gt, scale)) & support
    reason = ('not_visible_or_invalid' if not visible else 'no_valid_template_part' if not arrays['part_valid'].any()
              else 'insufficient_gt_tokens' if mask.sum() < 4 else None)
    return mask, reason


def validate_proxy_arrays(a, mode):
    if mode in ('control', 'alpha0'):
        validate_gate_arrays(a, mode)
        return
    if mode not in PROXIES:
        raise ValueError('unregistered location proxy')
    base = mode[:-5] if mode.endswith('_norm') else mode
    q, g, valid = a['raw_q'], a['raw_route'], a['part_valid'].astype(bool)
    mask = a['target_mask'][0].astype(bool)
    if (a['target_mask'].shape != (1, g.shape[-1]) or a['similarity'].shape != g.shape or mask.sum() < 4
            or not np.isin(a['target_mask'], [0, 1]).all()):
        raise ValueError('invalid target support/similarity')
    expected_q = q
    if base in ('q_target', 'joint_target'):
        selected = np.sort(a['similarity'][..., mask], axis=-1)[..., -4:].mean(-1)
        expected_q = 1 / (1 + np.exp(-np.clip(a['match_scale'].item() * selected + a['match_bias'].item(), -80, 80))) * valid
    expected_g = np.broadcast_to(mask[None, None, :], g.shape) * valid[..., None] if base in ('route_target', 'joint_target') else g
    if (not np.allclose(a['effective_q'], expected_q, rtol=1e-6, atol=2e-6)
            or not np.allclose(a['effective_route'], expected_g, rtol=1e-6, atol=2e-6)):
        raise ValueError('location intervention does not match declared formula')
    count = max(int(valid.sum()), 1)
    gram = a['prototype_gram'].astype(float)
    def energy(q_value, g_value):
        weights = q_value.astype(float)[..., None] * g_value.astype(float) / count
        return np.maximum(0, np.einsum('bkl,bkm,bml->bl', weights, gram, weights))
    original, changed = energy(q, g), energy(a['effective_q'], a['effective_route'])
    gain = float(a['norm_gain'].item())
    expected_gain = 1.
    if mode.endswith('_norm'):
        if changed.sum() == 0 and original.sum() > 0:
            raise ValueError('undefined location norm gain')
        if changed.sum() > 0:
            expected_gain = np.sqrt(original.sum() / changed.sum())
    if not np.isclose(gain, expected_gain, rtol=2e-5, atol=2e-6):
        raise ValueError('wrong location norm gain')
    if (not np.allclose(a['residual_norm']**2, changed*gain**2, rtol=1e-4, atol=1e-5)
            or not np.allclose(np.sqrt(a['injected_energy']), np.sqrt(changed)*gain*abs(a['alpha'].item()), rtol=3e-3, atol=3e-4)):
        raise ValueError('location residual does not match actual injection')


def summarize(frames, cases, smoke):
    index = {(r['sequence'], r['frame_1based'], r['mode'], r['roi_scale']): r for r in frames}
    result = dict(smoke_only=smoke, gt_location_proxy_only=True, true_part_visibility_verified=False,
                  tracking_performance_passed=False, optimizer_steps=0, scales={})
    for scale in SCALES:
        eligible = [(c['sequence'], c['frame_1based']) for c in cases if c['eligibility'][str(scale)]['reason'] is None]
        reasons = Counter(c['eligibility'][str(scale)]['reason'] or 'eligible' for c in cases)
        sequences = sorted({name for name, frame in eligible})
        per_sequence = {}
        for name in sequences:
            detail = {}
            for mode in ('alpha0', *PROXIES):
                pairs = [(index[n, f, 'control', None], index[n, f, mode, None if mode == 'alpha0' else scale])
                         for n, f in eligible if n == name]
                deltas = [b['iou']-a['iou'] for a, b in pairs]
                detail[mode] = dict(eligible_frames=len(pairs), mean_delta_iou=float(np.mean(deltas)),
                    improve_gt_5pp=sum(d > .05 for d in deltas), harm_gt_5pp=sum(d < -.05 for d in deltas),
                    repair_at_50=sum(a['iou'] < .5 <= b['iou'] for a, b in pairs),
                    lose_at_50=sum(b['iou'] < .5 <= a['iou'] for a, b in pairs),
                    peak_changes=sum(a['peak'] != b['peak'] for a, b in pairs))
            per_sequence[name] = detail
        macro = {mode: float(np.mean([d[mode]['mean_delta_iou'] for d in per_sequence.values()])) if per_sequence else None
                 for mode in ('alpha0', *PROXIES)}
        result['scales'][str(scale)] = dict(eligibility=dict(reasons), eligible_frames=len(eligible),
            sequence_macro_delta_iou=macro, per_sequence=per_sequence)
    return result


def check_bundle(path):
    content, seen = {}, set()
    with tarfile.open(path, 'r|gz') as archive:
        for member in archive:
            name = member.name
            if (name in seen or not member.isfile() or PurePosixPath(name).is_absolute()
                    or '..' in PurePosixPath(name).parts or '\\' in name):
                raise ValueError('unsafe/duplicate location artifact')
            seen.add(name)
            data = archive.extractfile(member).read()
            if len(seen) == 1:
                if name != 'manifest.json': raise ValueError('manifest must be first')
                manifest = json.loads(data)
            else:
                if hashlib.sha256(data).hexdigest() != manifest['members_sha256'].get(name):
                    raise ValueError('location artifact checksum mismatch')
                content[name] = data
    if seen != set(manifest['members_sha256']) | {'manifest.json'}:
        raise ValueError('incomplete location bundle')
    smoke = manifest.get('smoke_only')
    plan = protocol_plan(smoke)
    if (type(smoke) is not bool or manifest.get('kind') != 'vdrm_module1_step04_location' or manifest.get('version') != 1
            or manifest.get('plan') != {k: list(v) for k,v in plan.items()} or manifest.get('optimizer_steps') != 0
            or manifest.get('proxies') != list(PROXIES) or manifest.get('scales') != list(SCALES)
            or manifest.get('gt_location_proxy_only') is not True or manifest.get('source_verified') is not True
            or manifest.get('configurations') != {a: CONFIGURATIONS[a] for a in ARMS}):
        raise ValueError('wrong location protocol')
    for arm in ARMS:
        p, old = manifest['provenance'][arm], manifest['source_provenance'][arm]
        if (p['state_sha256_before'] != p['state_sha256_after'] or p['state_sha256_before'] != old['state_sha256_before']
                or p['checkpoint_sha256'] != old['checkpoint_sha256']):
            raise ValueError('wrong/changed location checkpoint')
    cases, rows = json.loads(content['cases.json']), json.loads(content['rows.json'])
    expected_cases = {(name, f) for name, (start, stop) in plan.items() for f in range(start,stop+1)}
    case_index = {(c['sequence'], c['frame_1based']): c for c in cases}
    if len(cases) != len(expected_cases) or set(case_index) != expected_cases:
        raise ValueError('missing/duplicate location cases')
    expected_rows = {(n,f,m,None) for n,f in expected_cases for m in ('control','alpha0')}
    for case in cases:
        for scale in SCALES:
            if case['eligibility'][str(scale)]['reason'] is None:
                expected_rows |= {(case['sequence'],case['frame_1based'],m,scale) for m in PROXIES}
    keys = [(r['sequence'], r['frame_1based'], r['mode'], r['roi_scale']) for r in rows]
    if len(keys) != len(expected_rows) or set(keys) != expected_rows or manifest['forward_calls'] != len(rows):
        raise ValueError('missing/duplicate proxy rows or wrong forward count')
    if set(content) != {'cases.json','rows.json'} | {r['arrays_member'] for r in rows}:
        raise ValueError('missing/unexpected location artifact')
    arrays, frame_index, metrics = {}, {}, []
    for row in rows:
        member = row['arrays_member']
        with np.load(io.BytesIO(content[member]), allow_pickle=False) as raw:
            arrays[member] = {key: raw[key] for key in raw.files}
        frame_index[row['sequence'],row['frame_1based'],row['mode'],row['roi_scale']] = row
    for case in cases:
        name, frame = case['sequence'],case['frame_1based']
        control_row = frame_index[name,frame,'control',None]
        control = arrays[control_row['arrays_member']]
        if control_row['bbox_xywh'] != case['source_bbox_xywh'] or control_row.get('source_maps_exact') is not True:
            raise ValueError('control differs from source')
        for scale in SCALES:
            mask, why = support_mask(control, case, scale)
            if case['eligibility'][str(scale)] != dict(reason=why, target_tokens=int(mask.sum())):
                raise ValueError('incorrect support eligibility')
    for row in rows:
        name, frame, mode, scale = (row[k] for k in ('sequence','frame_1based','mode','roi_scale'))
        case = case_index[name,frame]
        if any(row[k] != case[k] for k in ('anchor_xywh','resize_factor','input_sha256','gt_xywh','cover','absence','image_shape_hw')):
            raise ValueError('proxy changed its fixed input or labels')
        a = arrays[row['arrays_member']]
        control = arrays[frame_index[name,frame,'control',None]['arrays_member']]
        for key in ('raw_q','raw_route','prototype_gram','indices','part_valid','padding','similarity','match_scale','match_bias'):
            if not np.array_equal(a[key],control[key]): raise ValueError('changed pre-residual location evidence: '+key)
        if mode in PROXIES:
            mask, reason = support_mask(a,row,scale)
            if reason is not None or not np.array_equal(mask,a['target_mask'][0]):
                raise ValueError('location proxy uses wrong target mask')
            if a['alpha'].item() != control['alpha'].item(): raise ValueError('location proxy changed alpha')
        if mode == 'alpha0':
            if row.get('saved_Tclean_exact') is not True: raise ValueError('missing saved Tclean check')
            for key in ('score_map','size_map','offset_map'):
                if not np.array_equal(a[key],a['source_'+key]): raise ValueError('alpha0 differs from saved Tclean')
        metric = frame_metrics(a,row,gate_validator=validate_proxy_arrays)
        metric['roi_scale'] = scale
        metrics.append(metric)
    summary = summarize(metrics,cases,smoke)
    summary.update(source_sha256=manifest['source_sha256'], evidence_sha256=_file_hash(path), records=len(rows),
        cases=len(cases), forward_calls=len(rows), max_decode_error_pixels=max(r['decode_error_pixels'] for r in metrics))
    return summary, metrics


def report_text(summary):
    text = '# Step04 GT-location proxy diagnosis\n\nGT is used by these diagnostic interventions. No native closed-loop result, real visibility oracle or tracking performance claim.\n\n'
    text += '| ROI scale | Sequence | Mode | Eligible frames | Delta IoU (pp) | Improve/harm >5pp | Repair/lose at .5 |\n|---|---|---|---:|---:|---|---|\n'
    for scale, value in summary['scales'].items():
        for name, detail in value['per_sequence'].items():
            for mode, r in detail.items():
                text += f"| {scale} | {name} | {mode} | {r['eligible_frames']} | {r['mean_delta_iou']*100:.4f} | {r['improve_gt_5pp']}/{r['harm_gt_5pp']} | {r['repair_at_50']}/{r['lose_at_50']} |\n"
    text += '\nq_target recalibrates exactly top4 similarities inside the GT region with original learned scalars. At least four supported target tokens are required for every proxy, preserving the original k=4 estimator and a common comparison set. route_target replaces routing by a binary GT-region mask. These proxies neither know visible parts nor guarantee optimal residual direction. Norm controls preserve global residual Frobenius norm, not local energy. Missing token support is excluded.\n'
    return text
