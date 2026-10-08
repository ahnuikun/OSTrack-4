"""Native Rresp or Rqual / TC / RF / RD five-case isolation test and bundle."""

import argparse
from pathlib import Path
import platform
import sys

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from lib.test.analysis.frozen_failure_diagnostics import dump_json, write_bundle
from lib.test.analysis.frozen_vdrm_stages import visibility_protocol, recovery_delay
from lib.test.analysis.response_probe import CONFIG, CONFIGS, FOCUS_PLAN, PROFILES, protocol, check_bundle, model_protocol
from lib.test.evaluation import get_dataset
from lib.train.frozen_response import prepare_response_models, validate_response_checkpoint, baseline_hash
from lib.train.frozen_vdrm import file_hash
from tracking.diagnose_vdrm_frozen_failures import (
    SpatialCapture, config_for, checkpoint_paths, load_models, png_bytes, state_hash)
from tracking.diagnose_vdrm_native_float import (
    NativeTracker, prepare_cases, read_rgb, native_step, decode_native, clone_output,
    assert_exact, input_hashes, save_inputs, npz_bytes)


def focus_summary(trajectories, plan=None):
    plan = FOCUS_PLAN if plan is None else plan
    cases = []
    for trace in trajectories:
        name = trace['sequence']
        start, stop = plan[name]
        cover, absence = np.asarray(trace['cover']), np.asarray(trace['absence'])
        masks, events = visibility_protocol(cover, absence)
        masks = dict(visible=(cover > 0) & (absence == 0), **masks)
        window = np.arange(stop) >= start - 1
        valid = (np.asarray(trace['gt_xywh'])[:, 2:] > 0).all(1)
        overlaps = {arm: np.asarray([r.get('iou') if r.get('iou') is not None else -1.
                                    for r in rows]) for arm, rows in trace['native'].items()}
        case = dict(sequence=name, window_1based=[start, stop], stages={}, recovery_events=[])
        for stage, mask in masks.items():
            selected = mask & window & valid
            case['stages'][stage] = dict(frames=int(selected.sum()), ao_percent={
                a: float(v[selected].mean() * 100) if selected.any() else None for a, v in overlaps.items()})
        for event in events:
            # Do not count recovery when the registered prefix truncates its horizon.
            if event['start'] >= start - 1 and event['stop'] < stop:
                case['recovery_events'].append(dict(**event,
                    delay_frames={a: recovery_delay(v, event) for a, v in overlaps.items()}))
        cases.append(case)
    return dict(posthoc_cases_only=True, automatic_identity_verdict=False,
                full_dev_gate_passed=False, cases=cases)


def collect(cases, models, profile='focus', candidate_arm='Rresp'):
    plan, _, _ = protocol(profile)
    configs, _ = model_protocol(candidate_arm)
    captures = {a: SpatialCapture() for a in configs}
    hooks = [models[a].backbone.vdrm.register_forward_hook(captures[a], with_kwargs=True) for a in configs]
    payloads, records, trajectories = {}, [], []
    try:
        with torch.no_grad():
            for sequence, gt, cover, absence, start, stop in cases:
                name = sequence.name
                trackers = {a: NativeTracker(m, config_for(configs[a])) for a, m in models.items()}
                initial = read_rgb(sequence.frames[0])
                for tracker in trackers.values():
                    tracker.initialize(initial, dict(init_bbox=gt[0].tolist()))
                traces = {a: [dict(frame_1based=1, bbox_xywh=gt[0].tolist())] for a in configs}
                for frame in range(2, stop + 1):
                    image = read_rgb(sequence.frames[frame - 1])
                    snapshots, rows = {}, {}
                    for arm, tracker in trackers.items():
                        snapshots[arm], rows[arm] = native_step(tracker, image, gt[frame - 1], frame, captures[arm])
                        traces[arm].append(rows[arm])
                    if frame >= start:
                        candidate, snapshot = trackers[candidate_arm], snapshots[candidate_arm]
                        anchor = rows[candidate_arm]['anchor_xywh']
                        inputs = snapshot['inputs']
                        reference = clone_output(models['Tclean'](**inputs))
                        try:
                            models[candidate_arm].response_residual_disabled = True
                            bypass = clone_output(models[candidate_arm](**inputs))
                        finally:
                            models[candidate_arm].response_residual_disabled = False
                        assert_exact(reference, bypass)
                        assert_exact(reference, snapshot['output'], keys=('size_map', 'offset_map', 'backbone_feat'))
                        base_box = decode_native(candidate, reference, anchor, snapshot['resize'], image.shape)
                        bypass_box = decode_native(candidate, bypass, anchor, snapshot['resize'], image.shape)
                        same = torch.equal((candidate.output_window * reference['score_map']).flatten(1).argmax(1),
                            (candidate.output_window * snapshot['output']['score_map']).flatten(1).argmax(1))
                        if bypass_box != base_box or (same and base_box != rows[candidate_arm]['bbox_xywh']):
                            raise RuntimeError('response bypass/same-peak bbox is not exactly Tclean')
                        if input_hashes(inputs) != snapshot['input_sha256'] or list(candidate.state) != rows[candidate_arm]['bbox_xywh']:
                            raise RuntimeError('response same-crop replay modified input/state')
                        prefix = f'frames/{name}_{frame:06d}'
                        image_member = prefix + '_image.png'
                        payloads[image_member] = png_bytes(image)
                        artifacts = {}
                        for arm, snap in snapshots.items():
                            crop_member, maps_member = prefix + '_' + arm + '_crop.png', prefix + '_' + arm + '_maps.npz'
                            payloads[crop_member] = png_bytes(snap['patch'])
                            maps = {key: snap['output'][key].cpu().numpy() for key in ('score_map', 'size_map', 'offset_map')}
                            if arm == candidate_arm:
                                extra = ('response_delta_logits',) if arm == 'Rresp' else (
                                    'response_delta_energy', 'candidate_quality_logits', 'candidate_boxes',
                                    'candidate_valid_mask', 'candidate_roi_support')
                                maps.update({key: snap['output'][key].cpu().numpy() for key in
                                             ('score_logits', 'response_base_score_map') + extra})
                                maps.update({'Tclean_' + key: reference[key].cpu().numpy() for key in
                                             ('score_map', 'size_map', 'offset_map')})
                                maps['bypass_score_map'] = bypass['score_map'].cpu().numpy()
                                maps['hann_window'] = candidate.output_window.cpu().numpy()
                                if profile == 'early_failure':
                                    kept = torch.ones_like(reference['score_map'], dtype=torch.bool)
                                    for removed in snap['output']['removed_indexes_s']:
                                        if removed is not None:
                                            kept.flatten(1).scatter_(1, removed.long(), False)
                                    maps['final_retained_mask'] = kept.cpu().numpy()
                                    maps['search_padding_mask'] = np.asarray(snap['padding'], dtype=bool)
                                    maps['Tclean_score_logits'] = reference['score_logits'].cpu().numpy()
                            payloads[maps_member] = npz_bytes(maps)
                            artifacts[arm] = dict(crop_member=crop_member, maps_member=maps_member,
                                                 bbox_xywh=rows[arm]['bbox_xywh'])
                        records.append(dict(sequence=name, frame_1based=frame, arms=artifacts,
                            image_member=image_member, inputs_member=save_inputs(payloads, prefix, snapshot),
                            bypass_exact=True, frozen_regression_exact=True, same_hann_peak=bool(same),
                            base_samecrop_bbox=base_box, bypass_bbox=bypass_box))
                    if frame % 20 == 0 or frame == stop:
                        print(f'{name}: native {candidate_arm}/TC/RF/RD prefix {frame}/{stop}; exact replays', flush=True)
                trajectories.append(dict(sequence=name, source_frames=sequence.frames[:stop],
                    gt_xywh=gt[:stop].tolist(), cover=cover[:stop].tolist(), absence=absence[:stop].tolist(), native=traces))
    finally:
        for hook in hooks:
            hook.remove()
    payloads['records.json'] = dump_json(records)
    payloads['trajectories.json'] = dump_json(trajectories)
    payloads['focus_summary.json'] = dump_json(focus_summary(trajectories))
    if profile != 'focus':
        payloads['expanded_summary.json'] = dump_json(focus_summary(trajectories, plan))
    return payloads, records


def run(save_dir, output, device, profile='focus', candidate_arm='Rresp'):
    plan, selected_frames, native_steps = protocol(profile)
    configs, epoch = model_protocol(candidate_arm)
    config = configs[candidate_arm]
    if Path(output).exists():
        raise FileExistsError('refusing to overwrite response bundle: ' + str(output))
    paths = checkpoint_paths(save_dir)
    candidate_path = Path(save_dir) / 'checkpoints/train/ostrack' / config / f'OSTrack_ep{epoch:04d}.pth.tar'
    if not candidate_path.is_file():
        raise FileNotFoundError(candidate_arm + ' checkpoint missing: ' + str(candidate_path))
    cases = prepare_cases(get_dataset('got10k_vdrm_dev'), plan=plan)
    models, _, provenance = load_models(paths, torch.device(device))
    cfg = config_for(config)
    model, baseline, source = prepare_response_models(cfg, save_dir, config, candidate_path)
    validate_response_checkpoint(model, torch.load(candidate_path, map_location='cpu', weights_only=False),
                                 cfg=cfg, expected_epoch=epoch)
    if baseline_hash(baseline.state_dict()) != baseline_hash(models['Tclean'].state_dict()):
        raise RuntimeError('response probe/source Tclean states differ')
    del baseline
    models[candidate_arm] = model.to(device).requires_grad_(False).eval()
    provenance[candidate_arm] = dict(checkpoint_sha256=file_hash(candidate_path), source_sha256=file_hash(source),
        frozen_response_contract=model.frozen_response_contract, state_sha256_before=state_hash(model))
    try:
        payloads, records = collect(cases, models, profile, candidate_arm)
    finally:
        for arm, model in models.items():
            after = state_hash(model)
            if (after != provenance[arm]['state_sha256_before'] or any(p.grad is not None for p in model.parameters())
                    or any(m.training or any(p.requires_grad for p in m.parameters(recurse=False)) for m in model.modules())):
                raise RuntimeError('response diagnostic changed state/mode or produced gradients: ' + arm)
            provenance[arm]['state_sha256_after'] = after
    if len(records) != selected_frames:
        raise RuntimeError('incomplete response focus records')
    manifest = dict(kind='vdrm_response_probe', version=1, dataset='got10k_vdrm_dev', configurations=configs,
        profile=profile, plan={n: list(b) for n, b in plan.items()}, records=len(records), selected_frames=selected_frames,
        native_tracking_steps=native_steps, optimizer_steps=0, provenance=provenance,
        environment=dict(python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
                         device=str(device)),
        protocol='Posthoc fixed training-dev cases, not held-out benchmark; native float closed loop from frame 1; '
                 'current GT is analysis-only; no optimizer; integrity pass is NOT performance/identity pass.')
    if candidate_arm != 'Rresp':
        manifest['candidate_arm'] = candidate_arm
    write_bundle(output, manifest, payloads)
    verified = check_bundle(output)
    print(f'{candidate_arm.upper()} PROBE CHECK PASSED: {native_steps} native steps, {selected_frames} records; NOT a performance pass')
    print('Bundle:', Path(output).resolve())
    return verified


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--save_dir', default='./output')
    parser.add_argument('--output', help='New .tar.gz path; existing bundles are never overwritten')
    parser.add_argument('--check_bundle', help='Standalone integrity check; no GPU/data/weights required')
    parser.add_argument('--profile', choices=tuple(PROFILES), default='focus',
                        help='early_failure also saves the previously missing early failure frames')
    parser.add_argument('--candidate', choices=('Rresp', 'Rqual'), default='Rresp')
    args = parser.parse_args()
    if args.check_bundle:
        verified = check_bundle(args.check_bundle)
        print(verified.get('candidate_arm', 'Rresp').upper() + ' BUNDLE CHECK PASSED:', verified['records'], 'records')
        return
    if not torch.cuda.is_available():
        parser.error('native tracking requires CUDA')
    relative = ('analysis/frozen_response/rresp_probe.tar.gz' if args.profile == 'focus'
                else 'analysis/frozen_response_early/rresp_early_probe.tar.gz')
    if args.candidate == 'Rqual':
        relative = 'analysis/frozen_quality/rqual_' + args.profile + '_probe.tar.gz'
    output = args.output or str(Path(args.save_dir) / relative)
    if args.candidate != 'Rresp':
        run(args.save_dir, output, 'cuda', profile=args.profile, candidate_arm=args.candidate)
    elif args.profile == 'focus':
        run(args.save_dir, output, 'cuda')
    else:
        run(args.save_dir, output, 'cuda', profile=args.profile)


if __name__ == '__main__':
    main()
