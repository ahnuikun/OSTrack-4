"""Posthoc diagnosis and TRAIN-only fit/held-out audit of a conservative identity rule.

This is a rejection/retention screen, never a tracking integration approval.
"""
import numpy as np

from lib.train.identity_offline import PROJECT, ARMS
from lib.test.evaluation.got10kdataset import _vdrm_dev_sequence_ids
from lib.test.analysis.identity_native import (overlaps, candidate_pools, choose_candidates,
    interval, sequence_evidence)

PROTOCOL = dict(version=1, seed=42, fit_sequences=128, audit_sequences=64,
    localization_overlap=0.5, thresholds=[0., .025, .05, .1, .2, .4, 2.01],
    normal_cover_minimum=7, normal_tolerance_pp=.5,
    minimum_switch_frames=30, minimum_switch_sequences=10,
    candidate_pool='original_top5', original_heads_frozen=True,
    feedback='Tclean_only', disabled_threshold=2.01,
    official_dev_used_for_fitting=False, automatic_integration_approved=False)


def partition(screen_train_ids):
    all_train = {int(x) for x in (PROJECT/'lib/train/data_specs/got10k_vot_train_split.txt').read_text().split()}
    forbidden = set(screen_train_ids) | set(_vdrm_dev_sequence_ids())
    available = sorted(all_train-forbidden)
    count = PROTOCOL['fit_sequences']+PROTOCOL['audit_sequences']
    if len(available) < count:
        raise ValueError('insufficient disjoint training-domain sequences')
    chosen = np.random.default_rng(PROTOCOL['seed']).choice(available, count, replace=False).tolist()
    return dict(fit=sorted(chosen[:PROTOCOL['fit_sequences']]), audit=sorted(chosen[PROTOCOL['fit_sequences']:]))


def guard_select(base_scores, identity_scores, boxes, common_cells, threshold):
    """Only predictions enter decisions; no GT, visibility or future state."""
    if threshold not in PROTOCOL['thresholds']:
        raise ValueError('unregistered identity margin threshold')
    row = np.arange(len(base_scores))
    native = base_scores.argmax(1)
    candidate, _, _ = choose_candidates(base_scores, identity_scores,
        candidate_pools(base_scores, common_cells)['top5'])
    mutual = overlaps(boxes, boxes[row, native])[row, candidate]
    advantage = identity_scores[row, candidate]-identity_scores[row, native]
    permitted = ((candidate != native) & (mutual < PROTOCOL['localization_overlap']) &
                 (common_cells[row, native] >= 3) & (advantage > threshold))
    return np.where(permitted, candidate, native)


def masks_for(arrays):
    gt = arrays['gt'][1:]
    visible = (gt[:, 2:] > 0).all(1) & (arrays['cover'][1:] > 0) & (arrays['absence'][1:] == 0)
    return visible, visible & (arrays['cover'][1:] >= PROTOCOL['normal_cover_minimum'])


def selection_stats(arrays, selected):
    visible, normal = masks_for(arrays)
    rows = np.arange(len(selected)); native = arrays['base_scores'].argmax(1)
    iou = overlaps(arrays['boxes'], arrays['gt'][1:])
    baseline, candidate = iou[rows, native], iou[rows, selected]
    difference = candidate-baseline
    return dict(visible_frames=int(visible.sum()), normal_frames=int(normal.sum()),
        delta_pp=float(difference[visible].mean()*100) if visible.any() else None,
        normal_delta_pp=float(difference[normal].mean()*100) if normal.any() else None,
        switch_frames=int((visible & (selected != native)).sum()),
        invisible_switch_frames=int((~visible & (selected != native)).sum()),
        repairs=int((visible & (baseline < .5) & (candidate >= .5)).sum()),
        damages=int((visible & (baseline >= .5) & (candidate < .5)).sum()))


def summarize_policy(rows):
    result = interval([r['delta_pp'] for r in rows if r['delta_pp'] is not None])
    result.update(normal=interval([r['normal_delta_pp'] for r in rows if r['normal_delta_pp'] is not None]),
        switch_frames=sum(r['switch_frames'] for r in rows),
        switch_sequences=sum(r['switch_frames'] > 0 for r in rows),
        invisible_switch_frames=sum(r['invisible_switch_frames'] for r in rows),
        repairs=sum(r['repairs'] for r in rows), damages=sum(r['damages'] for r in rows))
    return result


def policy_key(arm, threshold):
    return arm+':'+str(float(threshold))


def grid_record(name, split, arrays, selected=None):
    # Reuse original closure/geometry checks, not any of its success gates.
    sequence_evidence(name, arrays)
    record = dict(sequence=name, split=split, policies={})
    if split == 'fit' and selected is None:
        policies = [(a, t) for a in ARMS for t in PROTOCOL['thresholds']]
    elif split == 'audit' and selected is not None:
        policies = [(selected['arm'], selected['threshold'])]
    else:
        raise ValueError('fit searches the registered grid; audit evaluates only the frozen choice')
    for arm, threshold in policies:
        indices = guard_select(arrays['base_scores'], arrays[arm+'_scores'], arrays['boxes'],
                               arrays['common_cells'], threshold)
        record['policies'][policy_key(arm, threshold)] = selection_stats(arrays, indices)
    return record


def fit_policy(records):
    """Selection cannot inspect audit rows, even if the caller supplies them."""
    if not records or any(r['split'] != 'fit' for r in records):
        raise ValueError('only fit-partition rows may select a threshold/head')
    table = {}
    selected = dict(arm='mean_rank', threshold=PROTOCOL['disabled_threshold'], fit_delta_pp=0.)
    best = 0.
    # Exact ties prefer the simpler mean_rank; within an arm prefer a stricter margin.
    for arm in ARMS:
        for threshold in reversed(PROTOCOL['thresholds']):
            values = [r['policies'][policy_key(arm, threshold)] for r in records]
            summary = summarize_policy(values)
            table[policy_key(arm, threshold)] = summary
            normal = summary['normal']['delta_pp']
            sufficient = (summary['switch_frames'] >= PROTOCOL['minimum_switch_frames'] and
                          summary['switch_sequences'] >= PROTOCOL['minimum_switch_sequences'])
            if sufficient and normal is not None and normal >= -PROTOCOL['normal_tolerance_pp'] and summary['delta_pp'] is not None and summary['delta_pp'] > best:
                best = summary['delta_pp']
                selected = dict(arm=arm, threshold=threshold, fit_delta_pp=best)
    return selected, table


def audit_policy(records, selected):
    if not records or any(r['split'] != 'audit' for r in records):
        raise ValueError('fixed held-out audit rows required')
    key = policy_key(selected['arm'], selected['threshold'])
    if any(set(r['policies']) != {key} for r in records):
        raise ValueError('audit must evaluate exactly the frozen fit choice')
    summary = summarize_policy([r['policies'][key] for r in records])
    reasons = []
    if selected['threshold'] == PROTOCOL['disabled_threshold']:
        reasons.append('fit_selected_disabled_baseline')
    if summary['switch_frames'] < PROTOCOL['minimum_switch_frames'] or summary['switch_sequences'] < PROTOCOL['minimum_switch_sequences']:
        reasons.append('insufficient_independent_switch_coverage')
    if summary['ci95_pp'] is None or summary['ci95_pp'][0] <= 0:
        reasons.append('no_positive_heldout_AO_lower_bound')
    # Conservative hold-out protection uses the interval, not only the point estimate.
    if summary['normal']['ci95_pp'] is None or summary['normal']['ci95_pp'][0] < -PROTOCOL['normal_tolerance_pp']:
        reasons.append('normal_frame_noninferiority_not_demonstrated')
    return dict(selected=selected, heldout=summary, rejection_reasons=reasons,
        heldout_screen_passed=not reasons, automatic_integration_approved=False,
        next_action='review_for_new_native_closed_loop_screen' if not reasons else 'stop_existing_identity_reranking_path')


def decide(records):
    selected, table = fit_policy([r for r in records if r['split'] == 'fit'])
    if selected['threshold'] == PROTOCOL['disabled_threshold']:
        if any(r['split'] == 'audit' for r in records):
            raise ValueError('disabled fit must stop before spending the held-out audit')
        return dict(protocol=PROTOCOL, fit_table=table, selected=selected, heldout=None,
            rejection_reasons=['fit_selected_disabled_baseline'], heldout_screen_passed=False,
            automatic_integration_approved=False, next_action='stop_existing_identity_reranking_path')
    audit = audit_policy([r for r in records if r['split'] == 'audit'], selected)
    return dict(protocol=PROTOCOL, fit_table=table, **audit)


def diagnose_sequence(name, arrays):
    """Posthoc explanatory decomposition; not a deployment threshold search."""
    sequence_evidence(name, arrays)
    visible, _ = masks_for(arrays); n = len(visible); rows = np.arange(n)
    base = arrays['base_scores']; native = base.argmax(1)
    iou = overlaps(arrays['boxes'], arrays['gt'][1:]); bi = iou[rows, native]
    pool = candidate_pools(base, arrays['common_cells'])['top5']
    mutual = overlaps(arrays['boxes'], arrays['native_boxes'])
    anchor, gt = arrays['anchors'], arrays['gt'][1:]
    # Theoretical sample_target square only: original image dimensions/padding
    # are not in this bundle. Do NOT claim exact visual-support visibility.
    side = np.ceil(np.sqrt(anchor[:, 2:].prod(1))*4)
    origin = np.round(anchor[:, :2]+anchor[:, 2:]/2-side[:, None]/2)
    intersection = np.maximum(0, np.minimum(origin+side[:, None], gt[:, :2]+gt[:, 2:])-np.maximum(origin, gt[:, :2])).prod(1)
    area = gt[:, 2:].prod(1)
    fraction = np.divide(intersection, area, out=np.zeros(n), where=area > 0)
    no_good = visible & (iou.max(1) < .5)
    result = dict(sequence=name, visible_frames=int(visible.sum()),
        no_good=int(no_good.sum()), no_good_gt_outside_square=int((no_good & (fraction == 0)).sum()),
        no_good_gt_at_least90pct_inside=int((no_good & (fraction >= .9)).sum()),
        no_good_gt_partial_square=int((no_good & (fraction > 0) & (fraction < .9)).sum()), heads={})
    for arm in ARMS:
        selected, _, _ = choose_candidates(base, arrays[arm+'_scores'], pool)
        ai = iou[rows, selected]; delta = ai-bi
        groups = dict(both_correct=visible & (bi >= .5) & (ai >= .5),
            damage=visible & (bi >= .5) & (ai < .5), repair=visible & (bi < .5) & (ai >= .5),
            both_wrong=visible & (bi < .5) & (ai < .5),
            changed_near=visible & (selected != native) & (mutual[rows, selected] >= .5),
            changed_far=visible & (selected != native) & (mutual[rows, selected] < .5))
        # Localization-only rollback differs from guard: no source-support/advantage
        # gate. It exactly isolates observed nearby-box changes, without GT decisions.
        locked = np.where(mutual[rows, selected] >= .5, native, selected)
        result['heads'][arm] = dict(groups={k: dict(frames=int(mask.sum()),
            contribution_pp=float(delta[mask].sum()/visible.sum()*100) if visible.any() else None)
            for k, mask in groups.items()}, localization_lock=selection_stats(arrays, locked))
    return result


def diagnose_summary(records):
    summary = dict(posthoc=True, deployment_approved=False, sequences=len(records), heads={})
    for key in ('visible_frames', 'no_good', 'no_good_gt_outside_square',
                'no_good_gt_at_least90pct_inside', 'no_good_gt_partial_square'):
        summary[key] = sum(r[key] for r in records)
    for arm in ARMS:
        groups = {}
        for key in records[0]['heads'][arm]['groups']:
            values = [r['heads'][arm]['groups'][key] for r in records]
            groups[key] = dict(frames=sum(v['frames'] for v in values),
                **interval([v['contribution_pp'] for v in values if v['contribution_pp'] is not None]))
        summary['heads'][arm] = dict(groups=groups,
            localization_lock=summarize_policy([r['heads'][arm]['localization_lock'] for r in records]))
    return summary
