"""Read-only candidate feasibility and first-failure analysis of Rresp bundles.

GT labels only already-computed maps/boxes. Counterfactual scores never advance
tracker state and are not a new tracker or a closed-loop performance estimate.
"""

import io
import json
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from .response_probe import CONFIGS, check_bundle, model_protocol
from lib.models.layers.head import CenterPredictor
from lib.test.tracker.ostrack import OSTrack
from lib.train.data.processing_utils import transform_image_to_crop
from lib.utils.box_ops import clip_box
from tracking.diagnose_vdrm_frozen_failures import crop_geometry
from tracking.diagnose_vdrm_response import focus_summary
from tracking.vdrm_diagnostics.backend_pair_metrics import bbox_iou_xywh

DECODE_TOLERANCE_PIXELS = .002


def candidate_boxes(maps, row, shape):
    """Enumerate actual CENTER candidates with the production decoder/mapping."""
    side = maps['score_map'].shape[-1]
    count = side * side
    normalized = CenterPredictor.cal_bbox(SimpleNamespace(feat_sz=side),
        torch.eye(count).reshape(count, 1, side, side),
        torch.from_numpy(maps['size_map']).repeat(count, 1, 1, 1),
        torch.from_numpy(maps['offset_map']).repeat(count, 1, 1, 1))
    boxes = (normalized * 256 / row['resize_factor']).tolist()
    tracker = SimpleNamespace(state=row['anchor_xywh'], params=SimpleNamespace(search_size=256))
    return np.asarray([clip_box(OSTrack.map_box_back(tracker, box, row['resize_factor']),
                                *shape, margin=10) for box in boxes])


def _sigmoid(logits):
    return torch.from_numpy(np.asarray(logits, dtype=np.float32)).sigmoid().clamp(1e-4, 1-1e-4).numpy()


def energy_peak(base_score, hann, delta):
    # Log of the native float product preserves its exact ties when delta=0.
    with np.errstate(divide='ignore'):
        energy = np.log((base_score * hann).astype(np.float64)) + delta
    return int(energy.argmax())


def _peak_result(index, ious, boxes):
    return dict(peak_xy=[int(index % 16), int(index // 16)],
                iou=None if ious is None else float(ious[index]), bbox_xywh=boxes[index].tolist())


def analyze_frame(record, trace, maps_by_arm, shape, candidate_arm='Rresp'):
    frame = record['frame_1based']
    gt = np.asarray(trace['gt_xywh'][frame - 1], dtype=float)
    valid_gt = bool((gt[2:] > 0).all())
    visible = bool(valid_gt and trace['cover'][frame - 1] > 0 and trace['absence'][frame - 1] == 0)
    response_maps = maps_by_arm[candidate_arm]
    hann = response_maps['hann_window']
    if hann.shape != (1, 1, 16, 16) or not np.isfinite(hann).all() or (hann < 0).any() or hann.max() <= 0:
        raise ValueError('invalid native Hann window')
    result = dict(sequence=trace['sequence'], frame_1based=frame, visible=visible, valid_gt=valid_gt,
                  gt_xywh=gt.tolist(), cover=trace['cover'][frame-1], absence=trace['absence'][frame-1],
                  arms={}, image_member=record['image_member'])
    response_boxes = response_ious = None
    for arm, maps in maps_by_arm.items():
        row = trace['native'][arm][frame - 1]
        crop = crop_geometry(row['anchor_xywh'], 4., shape, gt)
        if abs(row['resize_factor'] - 256 / crop['square_xywh'][2]) > 1e-12:
            raise ValueError('native crop/resize mismatch')
        boxes = candidate_boxes(maps, row, shape)
        ious = np.asarray([bbox_iou_xywh(box, gt) for box in boxes]) if valid_gt else None
        peak = int((maps['score_map'] * hann).argmax())
        error = float(np.abs(boxes[peak] - row['bbox_xywh']).max())
        expected_peak = [int(row['response_windowed'][k]) for k in ('p1_x', 'p1_y')]
        if error > DECODE_TOLERANCE_PIXELS or expected_peak != [peak % 16, peak // 16]:
            raise ValueError('saved native response/box does not match independent decode: ' + arm)
        result['arms'][arm] = dict(iou=row['iou'], bbox_xywh=row['bbox_xywh'], anchor_xywh=row['anchor_xywh'],
            resize_factor=row['resize_factor'], search_side_pixels=crop['square_xywh'][2], crop=crop,
            peak_xy=[peak % 16, peak // 16], decode_error_pixels=error,
            oracle_max_iou=float(ious.max()) if valid_gt else None,
            candidates_iou_ge_0_5=int((ious >= .5).sum()) if valid_gt else None)
        if arm == candidate_arm:
            response_boxes, response_ious = boxes, ious

    row = trace['native'][candidate_arm][frame - 1]
    maps, boxes, ious = response_maps, response_boxes, response_ious
    quality_head = candidate_arm == 'Rqual'
    delta_key = 'response_delta_energy' if quality_head else 'response_delta_logits'
    base, score, delta = maps['Tclean_score_map'], maps['score_map'], maps[delta_key]
    base_peak, peak = int((base * hann).argmax()), int((score * hann).argmax())
    if np.max(np.abs(boxes[base_peak] - record['base_samecrop_bbox'])) > DECODE_TOLERANCE_PIXELS:
        raise ValueError('same-crop Tclean decode differs from saved box')
    if not np.array_equal(base, maps['response_base_score_map']):
        raise ValueError('response base score is not same-crop Tclean')
    base_logits = maps['score_logits'] if quality_head else maps.get('Tclean_score_logits', maps['score_logits'] - delta)
    expected_score = base * np.exp(delta - delta.max()) if quality_head else _sigmoid(maps['score_logits'])
    if (not np.allclose(_sigmoid(base_logits), base, rtol=0, atol=2e-6)
            or not np.allclose(expected_score, score, rtol=0, atol=2e-6)):
        raise ValueError('saved logits do not reproduce score maps')
    samecrop = dict(Tclean=_peak_result(base_peak, ious, boxes), **{candidate_arm: _peak_result(peak, ious, boxes)},
        same_peak=base_peak == peak, delta_at_Tclean_peak=float(delta.flat[base_peak]),
        base_windowed_at_Tclean_peak=float((base * hann).flat[base_peak]),
        corrected_windowed_at_Tclean_peak=float((score * hann).flat[base_peak]))
    samecrop['delta_at_' + candidate_arm + '_peak'] = float(delta.flat[peak])
    samecrop['base_windowed_at_' + candidate_arm + '_peak'] = float((base * hann).flat[peak])
    samecrop['corrected_windowed_at_' + candidate_arm + '_peak'] = float((score * hann).flat[peak])
    # These controls depend only on saved predictions, never on GT selection.
    controls = {
        'uniform_plus2_all_logits': int((_sigmoid(base_logits + 2.) * hann).argmax()),
        'delta_in_final_log_energy': energy_peak(base, hann, delta),
    }
    if quality_head:
        controls = {'zero_correction': base_peak}
    kept = maps.get('final_retained_mask')
    active = delta.flatten() != 0
    delta_stats = dict(nonzero_cells=int(active.sum()), actual_retained_mask_available=kept is not None,
        mean_delta_nonzero=float(delta.flatten()[active].mean()) if active.any() else None,
        saturation_fraction_nonzero=float((np.abs(delta.flatten()[active]) >= 1.9).mean()) if active.any() else None)
    if kept is not None:
        retained = kept.flatten()
        common = float(delta.flatten()[retained].mean()) if retained.any() else 0.
        centered = delta - np.asarray(common, dtype=np.float32) * kept
        if not quality_head:
            controls['remove_retained_mean_in_logits'] = int((_sigmoid(base_logits + centered) * hann).argmax())
        delta_stats.update(retained_cells=int(retained.sum()), mean_delta_retained=common)
    result['same_crop'] = samecrop
    result['delta'] = delta_stats
    result['one_step_controls'] = {name: _peak_result(index, ious, boxes) for name, index in controls.items()}
    result['one_step_controls_are_closed_loop'] = False

    target = transform_image_to_crop(torch.tensor(gt, dtype=torch.float32),
        torch.tensor(row['anchor_xywh'], dtype=torch.float32), row['resize_factor'],
        torch.tensor([256, 256]), normalize=True)
    points = (torch.arange(16, dtype=torch.float32) + .5) / 16
    yy, xx = torch.meshgrid(points, points, indexing='ij')
    xy = torch.stack((xx, yy), -1).reshape(256, 2)
    inside = (((xy >= target[:2]) & (xy < target[:2] + target[2:])).all(1).numpy()
              if valid_gt else np.zeros(256, dtype=bool))
    label = dict(gt_crop_size_pixels=(target[2:] * 256).tolist(), gt_grid_center_cells=int(inside.sum()),
        gt_grid_positive_low_iou_cells=int(((ious < .1) & inside).sum()) if valid_gt else None,
        nonzero_correction_gt_cells=int((inside & active).sum()), retained_gt_cells=None,
        retained_nonpadding_gt_cells=None,
        note='Offline geometry of the point-label rule; not measured training eligibility/frequency.')
    if kept is not None:
        padding = F.adaptive_max_pool2d(torch.from_numpy(maps['search_padding_mask']).float()[None, None],
                                       (16, 16)).numpy().flatten().astype(bool)
        label['retained_gt_cells'] = int((inside & retained).sum())
        label['retained_nonpadding_gt_cells'] = int((inside & retained & ~padding).sum())
    result['point_label_geometry'] = label
    oracle = dict(max_iou=float(ious.max()) if valid_gt else None,
        best=_peak_result(int(ious.argmax()), ious, boxes) if valid_gt else None,
        good_candidates=int((ious >= .5).sum()) if valid_gt else None,
        retained_good_candidates=int(((ious >= .5) & retained).sum()) if valid_gt and kept is not None else None)
    result['frozen_candidate_oracle'] = oracle
    if quality_head:
        valid = maps['candidate_valid_mask'].flatten()
        predicted = torch.from_numpy(maps['candidate_quality_logits']).sigmoid().numpy().flatten()
        result['candidate_quality'] = dict(valid_candidates=int(valid.sum()),
            predicted_quality_at_selected=float(predicted[peak]),
            predicted_quality_at_Tclean=float(predicted[base_peak]),
            predicted_quality_at_oracle=float(predicted[ious.argmax()]) if valid_gt else None,
            valid_good_candidates=int((valid & (ious >= .5)).sum()) if valid_gt else None,
            note='Native image-clipped IoU for diagnosis; training uses normalized crop IoU. Not a calibrated reliability claim.')
    if not visible:
        category = 'not_visible_or_invalid_gt'
    elif result['arms'][candidate_arm]['crop']['gt_fraction_in_visual_support'] == 0:
        category = 'target_outside_visual_crop'
    elif oracle['good_candidates'] == 0:
        category = 'no_frozen_candidate_iou_ge_0_5'
    elif samecrop[candidate_arm]['iou'] < .5:
        category = 'good_candidate_exists_but_not_selected'
    else:
        category = 'selected_candidate_iou_ge_0_5'
    result['candidate_status'] = category
    return result


def _first(rows, predicate):
    return next((row['frame_1based'] for row in rows if predicate(row)), None)


def summarize(traces, frames, manifest):
    candidate_arm = manifest.get('candidate_arm', 'Rresp')
    configs, _ = model_protocol(candidate_arm)
    cases = []
    for trace in traces:
        name = trace['sequence']
        selected = [r for r in frames if r['sequence'] == name]
        native = []
        for index in range(1, len(trace['gt_xywh'])):
            a, b = (trace['native'][arm][index] for arm in (candidate_arm, 'Tclean'))
            visible = bool(trace['cover'][index] > 0 and trace['absence'][index] == 0
                           and (np.asarray(trace['gt_xywh'][index])[2:] > 0).all())
            native.append(dict(frame_1based=index+1, visible=visible, candidate_iou=a['iou'], Tclean_iou=b['iou'],
                bbox_changed=a['bbox_xywh'] != b['bbox_xywh'],
                crop_changed=a['input_sha256']['search'] != b['input_sha256']['search']))
        first_change = _first(native, lambda r: r['bbox_changed'])
        first_loss = _first(native, lambda r: r['visible'] and r['candidate_iou'] < r['Tclean_iou'] - .05)
        lookup = {r['frame_1based']: r for r in selected}
        events = dict(first_native_bbox_difference=first_change,
            first_native_crop_difference=_first(native, lambda r: r['crop_changed']),
            first_visible_native_loss_gt5pp=first_loss,
            first_visible_native_iou_lt0_1=_first(native, lambda r: r['visible'] and r['candidate_iou'] < .1),
            first_saved_visible_zero_visual_support=_first(selected, lambda r: r['visible'] and
                r['arms'][candidate_arm]['crop']['gt_fraction_in_visual_support'] == 0),
            first_saved_visible_no_good_candidate=_first(selected, lambda r: r['visible'] and
                r['frozen_candidate_oracle']['good_candidates'] == 0),
            first_saved_visible_good_candidate_missed=_first(selected, lambda r: r['candidate_status'] ==
                'good_candidate_exists_but_not_selected'),
            first_saved_visible_samecrop_loss_gt5pp=_first(selected, lambda r: r['visible'] and
                r['same_crop'][candidate_arm]['iou'] < r['same_crop']['Tclean']['iou'] - .05))
        event_evidence = {key: (dict(frame_1based=value, maps_saved=value in lookup,
            candidate_status=lookup[value]['candidate_status'] if value in lookup else None,
            same_crop=lookup[value]['same_crop'] if value in lookup else None,
            oracle=lookup[value]['frozen_candidate_oracle'] if value in lookup else None)
            if value is not None else None) for key, value in events.items()}
        counts = {status: sum(r['candidate_status'] == status for r in selected)
                  for status in sorted({r['candidate_status'] for r in selected})}
        cases.append(dict(sequence=name, saved_window_1based=manifest['plan'][name], frames=len(selected),
            first_events=events, event_evidence=event_evidence, candidate_status_counts=counts))
    return dict(kind='vdrm_response_failure_analysis', version=1,
        profile=manifest.get('profile', 'focus'), records=len(frames), optimizer_steps=0,
        gt_used_for_tracking=False, one_step_controls_are_closed_loop=False,
        automatic_identity_verdict=False, full_dev_gate_passed=False,
        maximum_decode_error_pixels=max(r['arms'][a]['decode_error_pixels'] for r in frames for a in configs),
        original_focus_summary=focus_summary(traces), cases=cases)


def render_report(summary):
    candidate_arm = summary.get('candidate_arm', 'Rresp')
    configs, _ = model_protocol(candidate_arm)
    lines = [f'# {candidate_arm} early failure audit', '',
        'Integrity and numerical analysis completed. This is NOT a performance or identity pass.', '',
        f"Profile: `{summary['profile']}`; saved frames: {summary['records']}; optimizer steps: 0.", '',
        'All frame numbers are 1-based. Candidate oracle uses GT only for offline evaluation.',
        f'Score controls are independent single-step calculations on the saved {candidate_arm} crop; '
        'they do not form a new closed-loop trajectory.', '',
        '| Sequence | Saved frames | First box change | First visible loss >5 pp | First saved crop miss | First saved no good box |',
        '| --- | --- | ---: | ---: | ---: | ---: |']
    def cell(value):
        return '-' if value is None else str(value)
    for case in summary['cases']:
        e = case['first_events']
        a, b = case['saved_window_1based']
        values = [case['sequence'], f'{a}-{b}', e['first_native_bbox_difference'],
            e['first_visible_native_loss_gt5pp'], e['first_saved_visible_zero_visual_support'],
            e['first_saved_visible_no_good_candidate']]
        lines.append('| ' + ' | '.join(cell(v) for v in values) + ' |')
    lines += ['', 'The first saved event need not be the first event in the full sequence. '
        'See event_evidence.maps_saved before attributing the first native loss to a mechanism.', '',
        '## Original five windows (AO %, not AUC)', '',
        f'| Sequence | Stage | Frames | Tclean | Rfreeze | Rdisc | {candidate_arm} |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for case in summary['original_focus_summary']['cases']:
        for stage, values in case['stages'].items():
            if stage == 'visible' or not values['frames']:
                continue
            row = [case['sequence'], stage, str(values['frames'])]
            row += [f"{values['ao_percent'][arm]:.4f}" for arm in configs]
            lines.append('| ' + ' | '.join(row) + ' |')
    lines += ['', '## Interpretation', '',
        '- Good frozen candidates exist but are missed: inspect identity and Hann-weighted ranking.',
        '- No frozen candidate reaches IoU 0.5: response reordering cannot select a good box on this crop.',
        '- Target outside the visual crop: this crop contains no current GT visual evidence.',
        '- These categories describe the saved state, not necessarily its earliest cause.',
        '- Point-label counts are geometry diagnostics, not measurements of training-sample frequency.',
        '- A high response or IoU change alone does not establish correct instance identity.',
        '- Inspect the original images in the source probe before deciding the next model design.', '']
    return '\n'.join(lines)


def analyze_bundle(path):
    manifest, payloads, shapes = check_bundle(path, include_payloads=True)
    candidate_arm = manifest.get('candidate_arm', 'Rresp')
    configs, _ = model_protocol(candidate_arm)
    traces = json.loads(payloads['trajectories.json'])
    by_name = {trace['sequence']: trace for trace in traces}
    records = json.loads(payloads['records.json'])
    # Recompute native IoU independently, including prefixes without saved maps.
    for trace in traces:
        for rows in trace['native'].values():
            for index, row in enumerate(rows[1:], 1):
                gt = np.asarray(trace['gt_xywh'][index])
                expected = bbox_iou_xywh(row['bbox_xywh'], gt) if (gt[2:] > 0).all() else None
                if ((expected is None) != (row['iou'] is None)
                        or expected is not None and abs(expected - row['iou']) > 1e-12):
                    raise ValueError('saved native IoU differs from bbox/GT recomputation')
    original = focus_summary(traces)
    if original != json.loads(payloads['focus_summary.json']):
        raise ValueError('saved focus summary differs from native trajectories')
    if manifest.get('profile') == 'early_failure':
        expanded = focus_summary(traces, {n: tuple(b) for n, b in manifest['plan'].items()})
        if expanded != json.loads(payloads['expanded_summary.json']):
            raise ValueError('saved expanded summary differs from native trajectories')
    frames = []
    for record in records:
        maps_by_arm = {}
        for arm in configs:
            with np.load(io.BytesIO(payloads[record['arms'][arm]['maps_member']]), allow_pickle=False) as maps:
                maps_by_arm[arm] = {key: maps[key] for key in maps.files}
        frames.append(analyze_frame(record, by_name[record['sequence']], maps_by_arm, shapes[record['image_member']], candidate_arm))
    summary = summarize(traces, frames, manifest)
    if candidate_arm != 'Rresp':
        summary['candidate_arm'] = candidate_arm
    return summary, frames, render_report(summary)
