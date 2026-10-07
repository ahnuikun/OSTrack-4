"""Native Rresp/TC/RF/RD five-case test, isolation replay, analysis and bundle."""

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
from lib.test.analysis.response_probe import CONFIG, CONFIGS, FOCUS_PLAN, SELECTED_FRAMES, NATIVE_STEPS, check_bundle
from lib.test.evaluation import get_dataset
from lib.train.frozen_response import prepare_response_models, validate_response_checkpoint, baseline_hash
from lib.train.frozen_vdrm import file_hash
from tracking.diagnose_vdrm_frozen_failures import (
    SpatialCapture, config_for, checkpoint_paths, load_models, png_bytes, state_hash)
from tracking.diagnose_vdrm_native_float import (
    NativeTracker, prepare_cases, read_rgb, native_step, decode_native, clone_output,
    assert_exact, input_hashes, save_inputs, npz_bytes)


def focus_summary(trajectories):
    cases = []
    for trace in trajectories:
        name = trace['sequence']
        start, stop = FOCUS_PLAN[name]
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


def collect(cases, models):
    captures = {a: SpatialCapture() for a in CONFIGS}
    hooks = [models[a].backbone.vdrm.register_forward_hook(captures[a], with_kwargs=True) for a in CONFIGS]
    payloads, records, trajectories = {}, [], []
    try:
        with torch.no_grad():
            for sequence, gt, cover, absence, start, stop in cases:
                name = sequence.name
                trackers = {a: NativeTracker(m, config_for(CONFIGS[a])) for a, m in models.items()}
                initial = read_rgb(sequence.frames[0])
                for tracker in trackers.values():
                    tracker.initialize(initial, dict(init_bbox=gt[0].tolist()))
                traces = {a: [dict(frame_1based=1, bbox_xywh=gt[0].tolist())] for a in CONFIGS}
                for frame in range(2, stop + 1):
                    image = read_rgb(sequence.frames[frame - 1])
                    snapshots, rows = {}, {}
                    for arm, tracker in trackers.items():
                        snapshots[arm], rows[arm] = native_step(tracker, image, gt[frame - 1], frame, captures[arm])
                        traces[arm].append(rows[arm])
                    if frame >= start:
                        candidate, snapshot = trackers['Rresp'], snapshots['Rresp']
                        anchor = rows['Rresp']['anchor_xywh']
                        inputs = snapshot['inputs']
                        reference = clone_output(models['Tclean'](**inputs))
                        try:
                            models['Rresp'].response_residual_disabled = True
                            bypass = clone_output(models['Rresp'](**inputs))
                        finally:
                            models['Rresp'].response_residual_disabled = False
                        assert_exact(reference, bypass)
                        assert_exact(reference, snapshot['output'], keys=('size_map', 'offset_map', 'backbone_feat'))
                        base_box = decode_native(candidate, reference, anchor, snapshot['resize'], image.shape)
                        bypass_box = decode_native(candidate, bypass, anchor, snapshot['resize'], image.shape)
                        same = torch.equal((candidate.output_window * reference['score_map']).flatten(1).argmax(1),
                            (candidate.output_window * snapshot['output']['score_map']).flatten(1).argmax(1))
                        if bypass_box != base_box or (same and base_box != rows['Rresp']['bbox_xywh']):
                            raise RuntimeError('response bypass/same-peak bbox is not exactly Tclean')
                        if input_hashes(inputs) != snapshot['input_sha256'] or list(candidate.state) != rows['Rresp']['bbox_xywh']:
                            raise RuntimeError('response same-crop replay modified input/state')
                        prefix = f'frames/{name}_{frame:06d}'
                        image_member = prefix + '_image.png'
                        payloads[image_member] = png_bytes(image)
                        artifacts = {}
                        for arm, snap in snapshots.items():
                            crop_member, maps_member = prefix + '_' + arm + '_crop.png', prefix + '_' + arm + '_maps.npz'
                            payloads[crop_member] = png_bytes(snap['patch'])
                            maps = {key: snap['output'][key].cpu().numpy() for key in ('score_map', 'size_map', 'offset_map')}
                            if arm == 'Rresp':
                                maps.update({key: snap['output'][key].cpu().numpy() for key in
                                             ('score_logits', 'response_delta_logits', 'response_base_score_map')})
                                maps.update({'Tclean_' + key: reference[key].cpu().numpy() for key in
                                             ('score_map', 'size_map', 'offset_map')})
                                maps['bypass_score_map'] = bypass['score_map'].cpu().numpy()
                                maps['hann_window'] = candidate.output_window.cpu().numpy()
                            payloads[maps_member] = npz_bytes(maps)
                            artifacts[arm] = dict(crop_member=crop_member, maps_member=maps_member,
                                                 bbox_xywh=rows[arm]['bbox_xywh'])
                        records.append(dict(sequence=name, frame_1based=frame, arms=artifacts,
                            image_member=image_member, inputs_member=save_inputs(payloads, prefix, snapshot),
                            bypass_exact=True, frozen_regression_exact=True, same_hann_peak=bool(same),
                            base_samecrop_bbox=base_box, bypass_bbox=bypass_box))
                    if frame % 20 == 0 or frame == stop:
                        print(f'{name}: native Rresp/TC/RF/RD prefix {frame}/{stop}; exact replays', flush=True)
                trajectories.append(dict(sequence=name, source_frames=sequence.frames[:stop],
                    gt_xywh=gt[:stop].tolist(), cover=cover[:stop].tolist(), absence=absence[:stop].tolist(), native=traces))
    finally:
        for hook in hooks:
            hook.remove()
    payloads['records.json'] = dump_json(records)
    payloads['trajectories.json'] = dump_json(trajectories)
    payloads['focus_summary.json'] = dump_json(focus_summary(trajectories))
    return payloads, records


def run(save_dir, output, device):
    if Path(output).exists():
        raise FileExistsError('refusing to overwrite response bundle: ' + str(output))
    paths = checkpoint_paths(save_dir)
    candidate_path = Path(save_dir) / 'checkpoints/train/ostrack' / CONFIG / 'OSTrack_ep0300.pth.tar'
    if not candidate_path.is_file():
        raise FileNotFoundError('Rresp checkpoint missing: ' + str(candidate_path))
    cases = prepare_cases(get_dataset('got10k_vdrm_dev'), plan=FOCUS_PLAN)
    models, _, provenance = load_models(paths, torch.device(device))
    cfg = config_for(CONFIG)
    model, baseline, source = prepare_response_models(cfg, save_dir, CONFIG, candidate_path)
    validate_response_checkpoint(model, torch.load(candidate_path, map_location='cpu', weights_only=False),
                                 cfg=cfg, expected_epoch=300)
    if baseline_hash(baseline.state_dict()) != baseline_hash(models['Tclean'].state_dict()):
        raise RuntimeError('response probe/source Tclean states differ')
    del baseline
    models['Rresp'] = model.to(device).requires_grad_(False).eval()
    provenance['Rresp'] = dict(checkpoint_sha256=file_hash(candidate_path), source_sha256=file_hash(source),
        frozen_response_contract=model.frozen_response_contract, state_sha256_before=state_hash(model))
    try:
        payloads, records = collect(cases, models)
    finally:
        for arm, model in models.items():
            after = state_hash(model)
            if (after != provenance[arm]['state_sha256_before'] or any(p.grad is not None for p in model.parameters())
                    or any(m.training or any(p.requires_grad for p in m.parameters(recurse=False)) for m in model.modules())):
                raise RuntimeError('response diagnostic changed state/mode or produced gradients: ' + arm)
            provenance[arm]['state_sha256_after'] = after
    if len(records) != SELECTED_FRAMES:
        raise RuntimeError('incomplete response focus records')
    manifest = dict(kind='vdrm_response_probe', version=1, dataset='got10k_vdrm_dev', configurations=CONFIGS,
        plan={n: list(b) for n, b in FOCUS_PLAN.items()}, records=len(records), selected_frames=SELECTED_FRAMES,
        native_tracking_steps=NATIVE_STEPS, optimizer_steps=0, provenance=provenance,
        environment=dict(python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
                         device=str(device)),
        protocol='Posthoc fixed training-dev cases, not held-out benchmark; native float closed loop from frame 1; '
                 'current GT is analysis-only; no optimizer; integrity pass is NOT performance/identity pass.')
    write_bundle(output, manifest, payloads)
    verified = check_bundle(output)
    print(f'RRESP PROBE CHECK PASSED: {NATIVE_STEPS} native steps, {SELECTED_FRAMES} records; NOT a performance pass')
    print('Bundle:', Path(output).resolve())
    return verified


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--save_dir', default='./output')
    parser.add_argument('--output', help='New .tar.gz path; existing bundles are never overwritten')
    parser.add_argument('--check_bundle', help='Standalone integrity check; no GPU/data/weights required')
    args = parser.parse_args()
    if args.check_bundle:
        verified = check_bundle(args.check_bundle)
        print('RRESP BUNDLE CHECK PASSED:', verified['records'], 'records')
        return
    if not torch.cuda.is_available():
        parser.error('native tracking requires CUDA')
    output = args.output or str(Path(args.save_dir) / 'analysis/frozen_response/rresp_probe.tar.gz')
    run(args.save_dir, output, 'cuda')


if __name__ == '__main__':
    main()
