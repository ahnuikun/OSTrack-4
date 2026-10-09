"""CPU evidence for frozen visual search coverage; GT never selects a crop."""
import hashlib
import json
from pathlib import PurePosixPath
import tarfile

import numpy as np

from lib.test.analysis.candidate_cues import read_npz
from lib.test.analysis.got10k_annotations import validate_native_annotations
from lib.test.analysis.identity_guard import partition
from lib.test.analysis.identity_native import interval, overlaps


PROTOCOL = dict(version=1, seed=42, sequence_count=128, sampling_stride=4, source_partition='guard_fit_only',
    crop_names=['center4', 'wide6', 'left4', 'right4', 'up4', 'down4'],
    search_size=256, source_factor=4., wide_factor=6., offset_side_fraction=.5,
    success_iou=.5, normal_cover_minimum=7, normal_tolerance_pp=.5,
    minimum_added_candidate_frames=30, minimum_added_candidate_sequences=10,
    feedback='saved_Tclean_float_anchors_only', optimizer_steps=0, current_gt_in_selection=False,
    purpose='development_candidate_generation_feasibility', deployment_approved=False)
GROUPS = {'wide6': [0, 1], 'tiles4': [0, 2, 3, 4, 5]}


def selected_ids(train_ids):
    return partition(train_ids)['fit']


def crop_specs(anchor):
    """Original state alone determines all centers and scales, never current GT."""
    anchor = np.asarray(anchor, dtype=np.float64)
    if anchor.shape != (4,) or not np.isfinite(anchor).all() or (anchor[2:] <= 0).any():
        raise ValueError('invalid search anchor')
    side = np.ceil(np.sqrt(anchor[2:].prod())*PROTOCOL['source_factor'])
    offset = side*PROTOCOL['offset_side_fraction']
    shifts = [(0., 0.), (0., 0.), (-offset, 0.), (offset, 0.), (0., -offset), (0., offset)]
    anchors = np.stack([anchor+np.array([x, y, 0., 0.]) for x, y in shifts])
    factors = np.array([4., 6., 4., 4., 4., 4.])
    sizes = np.ceil(np.sqrt(anchors[:, 2:].prod(1))*factors)
    origins = np.round(anchors[:, :2]+anchors[:, 2:]/2-sizes[:, None]/2)
    return anchors, factors, np.column_stack((origins, sizes, sizes))


def gt_crop_fraction(squares, image_hw, gt):
    """Post-inference pixel-support geometry matching sample_target's slicing.

    The native crop excludes the final image row/column on padded boundaries
    (x2_pad/y2_pad includes +1). Respect that behavior rather than silently
    treating a theoretical square as fully observed visual evidence.
    """
    low = np.maximum(squares[..., :2], 0)
    limits = image_hw[:, [1, 0]]-1
    high = np.minimum(squares[..., :2]+squares[..., 2:], limits[:, None])
    low_gt, high_gt = gt[:, None, :2], gt[:, None, :2]+gt[:, None, 2:]
    intersection = np.maximum(np.minimum(high, high_gt)-np.maximum(low, low_gt), 0).prod(-1)
    valid = (gt[:, 2:] > 0).all(1)
    area = np.where(valid, gt[:, 2:].prod(1), 0)
    return np.divide(intersection, area[:, None], out=np.zeros_like(intersection),
                     where=valid[:, None] & (area[:, None] > 0))


def choose_group(scores, supported, group):
    """Explicit naive max-Hann control, not a calibrated cross-crop score.

    The native choice is always available, including on unsupported source
    cells. Additional crops may propose only retained, nonpadding cells.
    Exact ties keep the native box.
    """
    rows = np.arange(len(scores)); native = scores[:, 0].argmax(1)
    selected_crop = np.zeros(len(scores), dtype=np.int64)
    selected_cell = native.copy(); best = scores[rows, 0, native].copy()
    for crop in group[1:]:
        eligible = np.where(supported[:, crop], scores[:, crop], -np.inf)
        cell = eligible.argmax(1); value = eligible[rows, cell]
        change = value > best
        selected_crop[change], selected_cell[change], best[change] = crop, cell[change], value[change]
    return selected_crop, selected_cell


def sequence_evidence(name, arrays):
    n = len(arrays['native_boxes']); history = len(arrays['source_native_boxes'])
    crops = len(PROTOCOL['crop_names'])
    shapes = dict(native_boxes=(n, 4), anchors=(n, 4), boxes=(n, crops, 256, 4),
        scores=(n, crops, 256), supported=(n, crops, 256), crop_anchors=(n, crops, 4),
        crop_squares=(n, crops, 4), image_hw=(n, 2), frame_indices=(n,),
        gt=(history+1, 4), cover=(history+1,), absence=(history+1,),
        source_boxes=(history, 256, 4), source_scores=(history, 256), source_native_boxes=(history, 4), source_anchors=(history, 4))
    if n < 1 or set(arrays) != set(shapes): raise ValueError('invalid coverage fields/frame count')
    for key, shape in shapes.items():
        if arrays[key].shape != shape or not np.isfinite(arrays[key]).all():
            raise ValueError('invalid coverage array: '+key)
    masks, _ = validate_native_annotations(arrays['gt'], arrays['cover'], arrays['absence'], name)
    indices = arrays['frame_indices']
    if indices.dtype.kind not in 'iu' or not np.array_equal(indices, np.arange(0, history, PROTOCOL['sampling_stride'])):
        raise ValueError('coverage frame sampling differs from registered stride')
    if ((arrays['boxes'][..., 2:] <= 0).any() or (arrays['source_boxes'][..., 2:] <= 0).any() or
        (arrays['image_hw'] < 2).any() or arrays['image_hw'].dtype.kind not in 'iu'):
        raise ValueError('invalid coverage box/image geometry')
    if (arrays['supported'].dtype != np.bool_ or (arrays['scores'] < 0).any() or (arrays['scores'] > 1.00001).any() or
        (arrays['source_scores'] < 0).any() or (arrays['source_scores'] > 1.00001).any()):
        raise ValueError('invalid coverage support/response')
    for actual, source in (('native_boxes','source_native_boxes'), ('anchors','source_anchors')):
        if not np.array_equal(arrays[actual], arrays[source][indices]):
            raise ValueError('coverage source native trajectory changed: '+actual)
    if not np.array_equal(arrays['boxes'][:, 0], arrays['source_boxes'][indices]) or not np.array_equal(arrays['scores'][:, 0], arrays['source_scores'][indices]):
        raise ValueError('coverage center output differs from source guard')
    expected = np.concatenate((arrays['gt'][:1], arrays['source_native_boxes'][:-1]))
    if not np.array_equal(arrays['source_anchors'], expected): raise ValueError('broken coverage source float history')
    source_rows = np.arange(history); source_native = arrays['source_scores'].argmax(1)
    if not np.array_equal(arrays['source_boxes'][source_rows, source_native], arrays['source_native_boxes']):
        raise ValueError('coverage source dense/native decode differs')
    for i, anchor in enumerate(arrays['anchors']):
        expected_anchors, _, squares = crop_specs(anchor)
        if not np.array_equal(arrays['crop_anchors'][i], expected_anchors) or not np.array_equal(arrays['crop_squares'][i], squares):
            raise ValueError('coverage crop centers/scales changed')
    rows = np.arange(n); native = arrays['scores'][:, 0].argmax(1)
    if not np.array_equal(arrays['boxes'][rows, 0, native], arrays['native_boxes']):
        raise ValueError('coverage dense/native decode differs')
    gt = arrays['gt'][indices+1]
    iou = np.stack([overlaps(arrays['boxes'][:, c], gt) for c in range(crops)], 1)
    baseline_iou = iou[rows, 0, native]
    visible = (gt[:, 2:] > 0).all(1) & (arrays['cover'][indices+1] > 0) & (arrays['absence'][indices+1] == 0)
    stages = dict(visible=visible, normal=visible & (arrays['cover'][indices+1] >= PROTOCOL['normal_cover_minimum']),
        severe_partial=visible & masks['severe_partial'][indices+1], reappearance=visible & masks['reappearance'][indices+1])
    fraction = gt_crop_fraction(arrays['crop_squares'], arrays['image_hw'], gt)
    base_good = (iou[:, 0] >= .5).any(1); no_good = visible & ~base_good
    result = dict(sequence=name, counts=dict(frames=n, original_history_steps=history, visible=int(visible.sum()),
        invisible_or_invalid=int((~visible).sum()), baseline_wrong=int((visible & (baseline_iou < .5)).sum()),
        baseline_no_candidate=int(no_good.sum()),
        no_candidate_outside_source=int((no_good & (fraction[:, 0] == 0)).sum()),
        no_candidate_inside90_source=int((no_good & (fraction[:, 0] >= .9)).sum()),
        no_candidate_partial_source=int((no_good & (fraction[:, 0] > 0) & (fraction[:, 0] < .9)).sum())), groups={})
    for name, group in GROUPS.items():
        # Source all-256 oracle retained; additional boxes require actual cells.
        oracle = iou[:, 0].max(1).copy()
        for crop in group[1:]:
            oracle = np.maximum(oracle, np.where(arrays['supported'][:, crop], iou[:, crop], 0).max(1))
        recovered = no_good & (oracle >= .5)
        selected_crop, selected_cell = choose_group(arrays['scores'], arrays['supported'], group)
        chosen = iou[rows, selected_crop, selected_cell]
        values = {}
        for stage, mask in stages.items():
            values[stage] = dict(frames=int(mask.sum()),
                naive_delta_pp=float((chosen[mask]-baseline_iou[mask]).mean()*100) if mask.any() else None,
                oracle_delta_pp=float((oracle[mask]-iou[mask, 0].max(1)).mean()*100) if mask.any() else None)
        result['groups'][name] = dict(added_candidate_frames=int(recovered.sum()),
            added_outside_source=int((recovered & (fraction[:, 0] == 0)).sum()),
            added_inside90_source=int((recovered & (fraction[:, 0] >= .9)).sum()),
            added_partial_source=int((recovered & (fraction[:, 0] > 0) & (fraction[:, 0] < .9)).sum()),
            added_candidate_frames_1based=(indices[recovered]+2).tolist(),
            selected_crop_indices=selected_crop.tolist(), selected_cell_indices=selected_cell.tolist(),
            naive_repairs=int((visible & (baseline_iou < .5) & (chosen >= .5)).sum()),
            naive_damages=int((visible & (baseline_iou >= .5) & (chosen < .5)).sum()),
            stages=values)
    return result


def summarize(records):
    result = dict(sequences=len(records), counts={k:sum(r['counts'][k] for r in records) for k in records[0]['counts']},
        groups={}, deployment_approved=False, closed_loop_tracking_passed=False,
        residual_effectiveness_passed=False, reliability_effectiveness_passed=False)
    for name in GROUPS:
        groups = [r['groups'][name] for r in records]
        frames = sum(g['added_candidate_frames'] for g in groups)
        sequences = sum(g['added_candidate_frames'] > 0 for g in groups)
        stages = {stage:dict(frames=sum(g['stages'][stage]['frames'] for g in groups),
            naive=interval([g['stages'][stage]['naive_delta_pp'] for g in groups if g['stages'][stage]['frames']]),
            oracle=interval([g['stages'][stage]['oracle_delta_pp'] for g in groups if g['stages'][stage]['frames']]))
            for stage in groups[0]['stages']}
        opportunity = frames >= PROTOCOL['minimum_added_candidate_frames'] and sequences >= PROTOCOL['minimum_added_candidate_sequences']
        visible_ci, normal_ci = stages['visible']['naive']['ci95_pp'], stages['normal']['naive']['ci95_pp']
        result['groups'][name] = dict(added_candidate_frames=frames, added_candidate_sequences=sequences,
            added_outside_source=sum(g['added_outside_source'] for g in groups),
            added_inside90_source=sum(g['added_inside90_source'] for g in groups),
            added_partial_source=sum(g['added_partial_source'] for g in groups),
            naive_repairs=sum(g['naive_repairs'] for g in groups), naive_damages=sum(g['naive_damages'] for g in groups),
            stages=stages, candidate_opportunity_sufficient=opportunity,
            naive_same_anchor_screen_passed=bool(opportunity and visible_ci is not None and visible_ci[0] > 0 and
                normal_ci is not None and normal_ci[0] >= -PROTOCOL['normal_tolerance_pp']))
    result['next_action'] = ('design_GT_free_crop_selection' if any(g['candidate_opportunity_sufficient'] for g in result['groups'].values())
                             else 'stop_multicrop_candidate_generation_path')
    return result


def render_report(result):
    lines = ['# Frozen search coverage probe', '',
        '128 existing guard FIT sequences; frames 2,6,10,...; full source float histories retained.',
        'Sampled same-anchor AO only. Saved Tclean anchors determine inputs; GT oracle is an upper bound.',
        'Naive cross-crop max-Hann scores are not calibrated reliability.', '',
        'Counts: '+str(result['counts']), '']
    for name, group in result['groups'].items():
        lines += [name+': added good candidates '+str(group['added_candidate_frames'])+' frames / '+str(group['added_candidate_sequences'])+' sequences',
            'Added candidates outside/partial/inside90 source support: '+str((group['added_outside_source'],group['added_partial_source'],group['added_inside90_source'])),
            'Visible naive AO delta pp: '+str(group['stages']['visible']['naive']),
            'Normal naive AO delta pp: '+str(group['stages']['normal']['naive']),
            'Naive repairs/damages: '+str((group['naive_repairs'],group['naive_damages'])),
            'Candidate opportunity sufficient: '+str(group['candidate_opportunity_sufficient']),
            'Naive same-anchor screen passed: '+str(group['naive_same_anchor_screen_passed']), '']
    lines += ['Next action: '+result['next_action'],
        'Deployment/closed-loop/residual/reliability approval: False', '']
    return '\n'.join(lines)


def check_bundle(path):
    with tarfile.open(path, 'r:gz') as archive:
        members = archive.getmembers(); names = [m.name for m in members]
        if len(names) != len(set(names)) or any(not m.isfile() or PurePosixPath(m.name).is_absolute() or
            '..' in PurePosixPath(m.name).parts or '\\' in m.name for m in members):
            raise ValueError('unsafe coverage bundle')
        manifest = json.load(archive.extractfile('manifest.json')); specs = manifest['sequences']
        ids = selected_ids(manifest['source_screen_train_ids'])
        if (manifest.get('kind') != 'vdrm_search_coverage' or manifest.get('protocol') != PROTOCOL or
            [s['sequence_id'] for s in specs] != ids or
            any(s['name'] != f'GOT-10k_Train_{s["sequence_id"]+1:06d}' for s in specs) or
            manifest.get('optimizer_steps') != 0 or manifest.get('parameter_gradient_tensors') != 0 or
            manifest['state_before'] != manifest['state_after']):
            raise ValueError('coverage partition/state contract differs')
        for key in ('source_guard_sha256', 'checkpoint_sha256', 'data_list_sha256', 'state_before'):
            value = manifest[key]
            if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
                raise ValueError('invalid coverage source/state hash')
        by_member = {s['member']:s for s in specs}
        required = set(by_member) | {'records.json','summary.json','report.md'}
        if len(by_member) != len(ids) or set(names) != required | {'manifest.json'} or set(manifest['members_sha256']) != required:
            raise ValueError('coverage bundle member set differs')
        records, documents = [], {}
        for member in members:
            if member.name == 'manifest.json': continue
            data = archive.extractfile(member).read()
            if hashlib.sha256(data).hexdigest() != manifest['members_sha256'][member.name]:
                raise ValueError('coverage checksum differs')
            if member.name in by_member:
                spec = by_member[member.name]; arrays = read_npz(data)
                if len(arrays['source_native_boxes']) != spec['frames']-1 or spec['exact_replay_frames'] != [2]:
                    raise ValueError('incomplete coverage native trajectory/replay')
                records.append(sequence_evidence(spec['name'], arrays))
            else: documents[member.name] = data
        records.sort(key=lambda r:r['sequence']); result = summarize(records)
        if records != json.loads(documents['records.json']) or result != json.loads(documents['summary.json']):
            raise ValueError('independent coverage recomputation differs')
        if render_report(result) != documents['report.md'].decode('utf-8'):
            raise ValueError('coverage report differs')
        expected_steps = sum(s['frames']-1 for s in specs)
        replayed = sum((s['frames']-2)//PROTOCOL['sampling_stride']+1 for s in specs)
        if manifest['source_history_steps'] != expected_steps or manifest['replayed_native_steps'] != replayed or manifest['extra_crop_forwards'] != replayed*5:
            raise ValueError('coverage compute/frame budget differs')
    return manifest, result
