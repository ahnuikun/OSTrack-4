"""GT-defined development stages and paired recovery; no model selection GT."""

import json
from pathlib import Path

import numpy as np
import torch

from .extract_results import calc_seq_err_robust
from lib.test.evaluation.result_paths import resolve_result_bbox_path


def visibility_protocol(cover, absence):
    """Exclude initialization; reappearance window=10, recovery horizon<=30."""
    if cover.ndim != 1 or absence.shape != cover.shape or not np.isfinite(cover).all() or not np.isfinite(absence).all():
        raise ValueError('invalid visibility annotation shape/values')
    if not len(cover) or not np.isin(cover, np.arange(9)).all() or not np.isin(absence, (0, 1)).all():
        raise ValueError('GOT-10k requires integer cover=0..8 and binary absence')
    visible = (cover > 0) & (absence == 0)
    reappearance = np.zeros(len(cover), dtype=bool)
    events = []
    invisible = ~visible
    padded = np.r_[False, invisible, False].astype(int)
    starts, ends = np.flatnonzero(np.diff(padded) == 1), np.flatnonzero(np.diff(padded) == -1)
    for a, end in zip(starts, ends):
        if end >= len(cover):
            continue
        next_invisible = np.flatnonzero(invisible[end:])
        stop = end + int(next_invisible[0]) if len(next_invisible) else len(cover)
        reappearance[end:min(end + 10, stop)] = True
        if end - a >= 3 and min(stop, end + 30) - end >= 5:
            events.append(dict(occlusion_start_1based=int(a + 1), reappearance_1based=int(end + 1),
                               start=int(end), stop=int(min(stop, end + 30))))
    masks = dict(normal=(cover == 8) & visible, partial=(cover >= 1) & (cover <= 7) & visible,
                 severe_partial=(cover >= 1) & (cover <= 3) & visible,
                 reappearance=reappearance & visible)
    for mask in masks.values():
        mask[0] = False
    return masks, events


def recovery_delay(iou, event):
    """First five-frame visible run at IoU>=.5; None means unrecovered."""
    for start in range(event['start'], event['stop'] - 4):
        if np.all(iou[start:start + 5] >= .5):
            return int(start - event['start'])
    return None


def summarize_stages(trackers, dataset, reference, report_dir):
    names = [t.parameter_name for t in trackers]
    ref = names.index(reference)
    rows = []
    for sequence in dataset:
        folder = Path(sequence.frames[0]).parent
        cover = np.loadtxt(folder / 'cover.label', ndmin=1)
        absence = np.loadtxt(folder / 'absence.label', ndmin=1)
        gt = sequence.ground_truth_rect
        if cover.shape != (len(gt),) or len(sequence.frames) != len(gt):
            raise ValueError('stage GT/image/visibility length mismatch: ' + sequence.name)
        masks, events = visibility_protocol(cover, absence)
        record = dict(sequence=sequence.name, stages={}, recovery_events=[])
        overlaps = []
        validity = None
        for tracker in trackers:
            prediction = np.loadtxt(resolve_result_bbox_path(tracker.results_dir, sequence), ndmin=2)
            if prediction.shape != gt.shape or not np.isfinite(prediction).all() or (prediction[:, 2:] <= 0).any():
                raise ValueError('stage prediction/GT mismatch: ' + sequence.name)
            iou, _, _, valid = calc_seq_err_robust(torch.tensor(prediction), torch.tensor(gt), 'got10k')
            overlaps.append(iou.numpy())
            current_validity = valid.numpy()
            if validity is not None and not np.array_equal(validity, current_validity):
                raise RuntimeError('paired GT validity differs across trackers: ' + sequence.name)
            validity = current_validity
        for stage, mask in masks.items():
            selected = mask & validity
            record['stages'][stage] = dict(frames=int(selected.sum()),
                ao_percent=[float(100 * v[selected].mean()) if selected.any() else None for v in overlaps])
        for event in events:
            record['recovery_events'].append(dict(**event,
                delay_frames=[recovery_delay(iou, event) for iou in overlaps]))
        rows.append(record)
    result = dict(dataset='got10k_vdrm_dev', sequences=len(dataset), trackers=names, reference=reference,
                  protocol=dict(normal='cover=8 and absence=0', partial='cover=1..7 and absence=0',
                    severe_partial='cover=1..3 and absence=0', initialization_excluded=True,
                    reappearance_window_frames=10, minimum_invisible_run_frames=3,
                    recovery_horizon_frames=30, recovery_run_frames=5, recovery_iou=.5),
                  per_sequence=rows, stages={}, recovery={})
    rng = np.random.default_rng(42)
    print('\nGT-defined stages (sequence-equal AO; initialization excluded)')
    for stage in ('normal', 'partial', 'severe_partial', 'reappearance'):
        eligible = [r['stages'][stage] for r in rows if r['stages'][stage]['frames']]
        summary = dict(eligible_sequences=len(eligible), frames=sum(r['frames'] for r in eligible), arms={})
        if eligible:
            values = np.asarray([r['ao_percent'] for r in eligible])
            resample = rng.integers(0, len(values), (5000, len(values)))
            for i, name in enumerate(names):
                delta = values[:, i] - values[:, ref]
                summary['arms'][name] = dict(ao_percent=float(values[:, i].mean()), delta_pp=float(delta.mean()),
                    paired_ci95_pp=np.quantile(delta[resample].mean(1), [.025, .975]).tolist())
                print(f'{stage} | {name} | AO={values[:, i].mean():.3f} | delta={delta.mean():+.3f} pp '
                      f'| {len(eligible)} sequences, {summary["frames"]} frames')
        result['stages'][stage] = summary
    event_rows = [r for r in rows if r['recovery_events']]
    total = sum(len(r['recovery_events']) for r in event_rows)
    for i, name in enumerate(names):
        recovered = sum(e['delay_frames'][i] is not None for r in event_rows for e in r['recovery_events'])
        paired = [(e['delay_frames'][i], e['delay_frames'][ref]) for r in event_rows for e in r['recovery_events']
                  if e['delay_frames'][i] is not None and e['delay_frames'][ref] is not None]
        rates = np.array([np.mean([e['delay_frames'][i] is not None for e in r['recovery_events']]) for r in event_rows])
        reference_rates = np.array([np.mean([e['delay_frames'][ref] is not None for e in r['recovery_events']]) for r in event_rows])
        delta = 100 * (rates - reference_rates)
        ci = None
        if len(delta):
            sampled = rng.integers(0, len(delta), (5000, len(delta)))
            ci = np.quantile(delta[sampled].mean(1), [.025, .975]).tolist()
        summary = dict(eligible_sequences=len(event_rows), eligible_events=total, recovered_events=recovered,
            pooled_recovery_rate=recovered / total if total else None,
            sequence_mean_recovery_rate=float(rates.mean()) if len(rates) else None,
            paired_rate_delta_pp=float(delta.mean()) if len(delta) else None, paired_rate_ci95_pp=ci,
            both_recovered_events=len(paired),
            paired_delay_delta_frames=float(np.mean([a - b for a, b in paired])) if paired else None)
        result['recovery'][name] = summary
        print(f'Recovery | {name} | {recovered}/{total} events | paired delay delta '
              f'{summary["paired_delay_delta_frames"]} frames (both recovered only)')
    path = Path(report_dir) / 'visibility_stages.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print('Stage/recovery report:', path.resolve())
    return result
