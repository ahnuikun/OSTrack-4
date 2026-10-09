"""Detached-cache native-box usability screen; no tracker or crop selection."""
import numpy as np

from lib.test.analysis.identity_native import interval, overlaps
from lib.test.analysis.search_coverage import choose_group


FEATURES = ['peak', 'separated_peak', 'relative_gap', 'entropy', 'local_mass',
            'peak_radius', 'displacement', 'log_area_change', 'log_aspect_change',
            'past_peak_mean', 'peak_change', 'past_displacement', 'motion_innovation']
PROTOCOL = dict(version=1, seed=42, folds=5, history=5, l2=.01,
                maximum_iterations=100, gradient_tolerance=1e-8,
                target_fpr=.10, minimum_recall=.70, minimum_bad_frames=30,
                minimum_bad_sequences=10, features=FEATURES,
                target='invisible_or_invalid_or_native_iou_below_0.5',
                aggregation='sequence_macro', source_partition='guard_fit_only',
                calibration_fold='(test_fold+1)%5', threshold_comparison='strict_greater',
                tracker_optimizer_steps=0, current_gt_in_features=False,
                purpose='development_out_of_fold_usability_screen')


def causal_features(scores, boxes, anchors):
    """All inputs are predictions. Frame t features use only frames <= t."""
    scores, boxes, anchors = (np.asarray(v, dtype=np.float64) for v in (scores, boxes, anchors))
    n = len(scores)
    if (n < 1 or scores.shape != (n, 256) or boxes.shape != (n, 4) or anchors.shape != (n, 4)
            or not all(np.isfinite(v).all() for v in (scores, boxes, anchors))
            or (scores < 0).any() or (scores > 1.00001).any()
            or (boxes[:, 2:] <= 0).any() or (anchors[:, 2:] <= 0).any()):
        raise ValueError('invalid native prediction features')
    grid = np.column_stack(np.unravel_index(np.arange(256), (16, 16)))
    cell = scores.argmax(1)
    distance = np.abs(grid[None] - grid[cell, None]).max(2)
    peak = scores[np.arange(n), cell]
    second = np.where(distance > 2, scores, 0).max(1)
    probability = scores / np.maximum(scores.sum(1, keepdims=True), 1e-12)
    entropy = -(probability * np.log(np.maximum(probability, 1e-12))).sum(1) / np.log(256)
    mass = np.where(distance <= 1, probability, 0).sum(1)
    radius = np.linalg.norm((grid[cell] - 7.5) / 7.5, axis=1)
    side = np.sqrt(anchors[:, 2:].prod(1))
    velocity = (boxes[:, :2] + boxes[:, 2:]/2 - anchors[:, :2] - anchors[:, 2:]/2) / side[:, None]
    logs = np.log(boxes[:, 2:] / anchors[:, 2:])
    past_peak = np.empty(n); past_velocity = np.zeros((n, 2))
    for t in range(n):
        start = max(0, t-PROTOCOL['history'])
        past_peak[t] = peak[start:t].mean() if t else peak[t]
        if t:
            past_velocity[t] = velocity[start:t].mean(0)
    result = np.column_stack((peak, second, (peak-second)/np.maximum(peak, 1e-12),
        entropy, mass, radius, np.linalg.norm(velocity, axis=1), logs.sum(1), logs[:, 0]-logs[:, 1],
        past_peak, peak-past_peak, np.linalg.norm(past_velocity, axis=1),
        np.linalg.norm(velocity-past_velocity, axis=1)))
    if not np.isfinite(result).all():
        raise ValueError('nonfinite causal features')
    return result


def fold_map(sequence_ids):
    ids = sorted(int(i) for i in sequence_ids)
    if len(ids) < PROTOCOL['folds'] or len(ids) != len(set(ids)):
        raise ValueError('need unique sequences for all folds')
    shuffled = np.random.default_rng(PROTOCOL['seed']).permutation(ids)
    return {int(seq): int(i % PROTOCOL['folds']) for i, seq in enumerate(shuffled)}


def sequence_weights(groups):
    _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    if not len(counts):
        raise ValueError('empty sequence weighting population')
    return 1. / (len(counts) * counts[inverse])


def sigmoid(x):
    # This symmetric form avoids overflow without truncating logits.
    z = np.exp(-np.abs(x))
    return np.where(x >= 0, 1/(1+z), z/(1+z))


def fit_logistic(x, y, groups):
    """Fixed small convex fit. Normalization never sees calibration/test rows."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if x.shape != (len(y), len(FEATURES)) or not np.isfinite(x).all() or set(np.unique(y)) != {0., 1.}:
        raise ValueError('logistic training needs finite features and both label classes')
    weight = sequence_weights(groups)
    mean = (x * weight[:, None]).sum(0)
    scale = np.maximum(np.sqrt(((x-mean)**2 * weight[:, None]).sum(0)), 1e-6)
    design = np.column_stack((np.ones(len(x)), (x-mean)/scale))
    penalty = np.full(design.shape[1], PROTOCOL['l2']); penalty[0] = 0
    beta = np.zeros(design.shape[1])
    prevalence = float(weight @ y)
    beta[0] = np.log(prevalence/(1-prevalence))

    def loss(b):
        logits = design @ b
        return float(weight @ (np.logaddexp(0, logits) - y*logits) + .5*np.sum(penalty*b*b))

    for iteration in range(PROTOCOL['maximum_iterations']):
        probability = sigmoid(design @ beta)
        gradient = design.T @ (weight*(probability-y)) + penalty*beta
        norm = float(np.abs(gradient).max())
        if norm <= PROTOCOL['gradient_tolerance']:
            return dict(mean=mean.tolist(), scale=scale.tolist(), beta=beta.tolist(),
                        iterations=iteration, gradient_inf=norm, loss=loss(beta))
        curvature = weight * probability * (1-probability)
        hessian = design.T @ (design*curvature[:, None]) + np.diag(penalty+1e-10)
        step = np.linalg.solve(hessian, gradient)
        before = loss(beta); rate = 1.
        for _ in range(40):
            proposed = beta-rate*step
            if loss(proposed) <= before - 1e-4*rate*float(gradient @ step):
                beta = proposed
                break
            rate *= .5
        else:
            raise ValueError('logistic line search did not converge')
    raise ValueError('logistic solver iteration budget exhausted')


def predict(model, x):
    z = (np.asarray(x)-np.asarray(model['mean'])) / np.asarray(model['scale'])
    return sigmoid(np.column_stack((np.ones(len(z)), z)) @ np.asarray(model['beta']))


def calibrate_threshold(risk, bad, groups):
    """Smallest observed threshold with <=10% macro FPR, using strict >."""
    good = ~np.asarray(bad, dtype=bool)
    if not good.any():
        raise ValueError('calibration partition has no usable native boxes')
    score = np.asarray(risk, dtype=np.float64)[good]
    if not np.isfinite(score).all():
        raise ValueError('invalid calibration risks')
    weight = sequence_weights(np.asarray(groups)[good])
    values, inverse = np.unique(score, return_inverse=True)
    mass = np.bincount(inverse, weights=weight)
    tail = np.maximum(0, 1-np.cumsum(mass))
    eligible = np.flatnonzero(tail <= PROTOCOL['target_fpr']+1e-12)
    threshold = float(values[eligible[0]])
    actual_fpr = float(weight @ (score > threshold))
    return dict(threshold=threshold, calibration_fpr=actual_fpr,
                usable_frames=int(good.sum()), usable_sequences=int(len(np.unique(np.asarray(groups)[good]))))


def proposal_diagnostics(arrays):
    """Post-inference diagnostic only; labels do not choose proposals."""
    ids = arrays['frame_indices']; rows = np.arange(len(ids)); gt = arrays['gt'][ids+1]
    visible = ((arrays['cover'][ids+1] > 0) & (arrays['absence'][ids+1] == 0) & (gt[:, 2:] > 0).all(1))
    iou = np.stack([overlaps(arrays['boxes'][:, c], gt) for c in range(6)], 1)
    baseline = overlaps(arrays['native_boxes'][:, None], gt)[:, 0]
    no_good = visible & (iou[:, 0].max(1) < .5)
    all_good = np.zeros(len(ids), dtype=bool); top_good = all_good.copy()
    for crop in (2, 3, 4, 5):
        eligible = np.where(arrays['supported'][:, crop], arrays['scores'][:, crop], -np.inf)
        cell = eligible.argmax(1)
        top_good |= arrays['supported'][rows, crop, cell] & (iou[rows, crop, cell] >= .5)
        all_good |= np.where(arrays['supported'][:, crop], iou[:, crop], 0).max(1) >= .5
    crop, cell = choose_group(arrays['scores'], arrays['supported'], [0, 2, 3, 4, 5])
    chosen = iou[rows, crop, cell]
    mutual = overlaps(arrays['boxes'][rows, crop, cell][:, None], arrays['native_boxes'])[:, 0]
    counts = {}
    for label, good in (('all_cell_added', all_good), ('per_crop_top1_added', top_good)):
        hit = no_good & good
        counts[label+'_frames'] = int(hit.sum()); counts[label+'_sequences'] = int(hit.any())
    for label, condition in (('near', mutual >= .5), ('far', mutual < .5)):
        changed = visible & (crop != 0) & condition
        counts[label+'_changes'] = int(changed.sum())
        counts[label+'_repairs'] = int((changed & (baseline < .5) & (chosen >= .5)).sum())
        counts[label+'_damages'] = int((changed & (baseline >= .5) & (chosen < .5)).sum())
    return counts


def dataset_rows(sequence_id, arrays):
    x = causal_features(arrays['source_scores'], arrays['source_native_boxes'], arrays['source_anchors'])
    gt = arrays['gt'][1:]
    visible = (gt[:, 2:] > 0).all(1) & (arrays['cover'][1:] > 0) & (arrays['absence'][1:] == 0)
    iou = overlaps(arrays['source_native_boxes'][:, None], gt)[:, 0]
    return dict(features=x, bad=~visible | (iou < .5), sequence_id=np.full(len(x), sequence_id, dtype=np.int64),
                frame=np.arange(2, len(x)+2, dtype=np.int64), visible=visible,
                normal=visible & (arrays['cover'][1:] >= 7), native_iou=iou)


def cross_validate(rows):
    x, y, groups = rows['features'], rows['bad'], rows['sequence_id']
    assignment = fold_map(np.unique(groups))
    folds = np.array([assignment[int(i)] for i in groups])
    predictions = dict(fold=folds, risk=np.zeros(len(y)), peak_risk=-x[:, 0],
                       flag=np.zeros(len(y), dtype=bool), peak_flag=np.zeros(len(y), dtype=bool))
    models = []
    for test in range(PROTOCOL['folds']):
        calibration = (test+1) % PROTOCOL['folds']
        train_mask = (folds != test) & (folds != calibration)
        cal_mask, test_mask = folds == calibration, folds == test
        model = fit_logistic(x[train_mask], y[train_mask], groups[train_mask])
        learned = calibrate_threshold(predict(model, x[cal_mask]), y[cal_mask], groups[cal_mask])
        peak = calibrate_threshold(-x[cal_mask, 0], y[cal_mask], groups[cal_mask])
        predictions['risk'][test_mask] = predict(model, x[test_mask])
        predictions['flag'][test_mask] = predictions['risk'][test_mask] > learned['threshold']
        predictions['peak_flag'][test_mask] = predictions['peak_risk'][test_mask] > peak['threshold']
        roles = {role: np.unique(groups[mask]).tolist() for role, mask in
                 (('train', train_mask), ('calibration', cal_mask), ('test', test_mask))}
        models.append(dict(test_fold=test, sequences=roles, model=model, learned=learned, peak=peak))
    return predictions, models


def _rate(mask, flag):
    return float(np.asarray(flag)[mask].mean()) if mask.any() else None


def evaluate(rows, predictions):
    y, groups = rows['bad'], rows['sequence_id']
    records = []
    for group in np.unique(groups):
        mask = groups == group; bad = y[mask]; normal = rows['normal'][mask] & ~bad
        record = dict(sequence_id=int(group), frames=int(mask.sum()), bad_frames=int(bad.sum()), arms={})
        for name, key in (('logistic', 'flag'), ('low_peak', 'peak_flag')):
            flag = predictions[key][mask]
            record['arms'][name] = dict(fpr=_rate(~bad, flag), recall=_rate(bad, flag),
                                        normal_usable_fpr=_rate(normal, flag))
        records.append(record)
    result = dict(sequences=len(records), frames=len(y), bad_frames=int(y.sum()),
                  bad_sequences=sum(r['bad_frames'] > 0 for r in records), arms={})
    for name, key in (('logistic', 'flag'), ('low_peak', 'peak_flag')):
        macro = {metric: interval([100*r['arms'][name][metric] for r in records
                                  if r['arms'][name][metric] is not None])
                 for metric in ('fpr', 'recall', 'normal_usable_fpr')}
        result['arms'][name] = dict(macro_percent=macro,
            pooled=dict(fpr=_rate(~y, predictions[key]), recall=_rate(y, predictions[key]),
                        false_alarms=int((~y & predictions[key]).sum()), detected_bad=int((y & predictions[key]).sum())))
    weight = sequence_weights(groups); risk = predictions['risk']
    bins = np.minimum((risk*10).astype(int), 9)
    ece = 0.; calibration_bins = []
    for i in range(10):
        mask = bins == i; mass = float(weight[mask].sum())
        predicted = float(weight[mask] @ risk[mask]/mass) if mass else None
        observed = float(weight[mask] @ y[mask]/mass) if mass else None
        if mass:
            ece += mass*abs(predicted-observed)
        calibration_bins.append(dict(bin=i, sequence_weight=mass, predicted=predicted, observed=observed))
    result['calibration'] = dict(brier=float(weight @ (risk-y)**2), ece_10_bins=ece, bins=calibration_bins)
    metrics = result['arms']['logistic']['macro_percent']
    fpr, recall = metrics['fpr']['delta_pp'], metrics['recall']['delta_pp']
    sufficient = result['bad_frames'] >= PROTOCOL['minimum_bad_frames'] and result['bad_sequences'] >= PROTOCOL['minimum_bad_sequences']
    passed = bool(sufficient and fpr is not None and recall is not None and
                  fpr <= PROTOCOL['target_fpr']*100 and recall >= PROTOCOL['minimum_recall']*100)
    result.update(development_continuation_gate=passed,
                  next_action='design_separate_crop_acceptance_and_fresh_audit' if passed else 'stop_response_history_only_gate',
                  independent_audit_passed=False, closed_loop_tracking_passed=False,
                  residual_effectiveness_passed=False, reliability_effectiveness_passed=False, deployment_approved=False)
    return result, records


def render_report(result):
    lines = ['# Cached native-usability gate', '',
        'Previously used 128 guard FIT sequences; grouped development cross-validation, NOT an independent audit.',
        'No GT features, no tracker updates, no crop replacement, no closed-loop gain claim.', '',
        f"Frames: {result['frames']}; unusable: {result['bad_frames']} / {result['bad_sequences']} sequences", '',
        '| Diagnostic | Macro false-positive % (95% CI) | Macro recall % (95% CI) |',
        '| --- | --- | --- |']
    for arm, values in result['arms'].items():
        formatted = []
        for key in ('fpr', 'recall'):
            v = values['macro_percent'][key]
            formatted.append(f"{v['delta_pp']:.3f} {v['ci95_pp']}" if v['delta_pp'] is not None else 'N/A')
        lines.append('| '+arm+' | '+' | '.join(formatted)+' |')
    lines += ['', 'FPR thresholds are fitted on separate calibration sequences; achieved test FPRs need not match.',
        f"Brier: {result['calibration']['brier']:.6f}; ECE: {result['calibration']['ece_10_bins']:.6f}",
        'Tiles proposal/naive-selection diagnosis: '+str(result.get('proposal_diagnostics', {})),
        'Development continuation gate: '+str(result['development_continuation_gate']),
        'Next action: '+result['next_action'],
        'Independent-audit/closed-loop/residual/reliability/deployment approval: False', '']
    return '\n'.join(lines)
