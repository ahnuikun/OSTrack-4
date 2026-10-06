"""Read-only, native-float Rscore closed-loop probe on five fixed dev cases."""

import argparse
from pathlib import Path
import platform
import sys

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from lib.test.analysis.frozen_failure_diagnostics import CONFIGS, dump_json, write_bundle
from lib.test.analysis.frozen_vdrm_stages import recovery_delay, visibility_protocol
from lib.test.analysis.rscore_probe import ARMS, FOCUS_PLAN, NATIVE_STEPS, SELECTED_FRAMES, check_bundle
from lib.test.evaluation import get_dataset
from lib.test.tracker.ostrack_rscore import PROBE_CONFIG, ScoreOnlyNetwork
from tracking.diagnose_vdrm_frozen_failures import (
    EXACT_KEYS, SpatialCapture, checkpoint_paths, config_for, intervention, load_models, png_bytes, state_hash)
from tracking.diagnose_vdrm_native_float import (
    NativeTracker, assert_exact, clone_output, decode_native, input_hashes, native_step, npz_bytes,
    prepare_cases, read_rgb, save_inputs)

MAP_KEYS = ('score_map', 'size_map', 'offset_map', 'search_global_index')


def focus_summary(trajectories):
    """Visible-frame posthoc case summaries; never automatic identity labels."""
    summaries = []
    for trace in trajectories:
        name = trace['sequence']
        start, stop = FOCUS_PLAN[name]
        cover, absence = np.asarray(trace['cover']), np.asarray(trace['absence'])
        masks, events = visibility_protocol(cover, absence)
        window = np.arange(stop) >= start - 1
        masks = dict(visible=(cover > 0) & (absence == 0), **masks)
        result = dict(sequence=name, window_1based=[start, stop], stages={}, recovery_events=[])
        overlaps = {a: np.asarray([r.get('iou') if r.get('iou') is not None else -1.
                                  for r in trace['native'][a]]) for a in ARMS}
        valid_gt = (np.asarray(trace['gt_xywh'])[:, 2:] > 0).all(axis=1)
        for stage, mask in masks.items():
            selected = mask & window & valid_gt
            result['stages'][stage] = dict(frames=int(selected.sum()), ao_percent={
                a: float(v[selected].mean() * 100) if selected.any() else None for a, v in overlaps.items()})
        for event in events:
            # A truncated prefix must not count a horizon it never observed.
            if event['start'] >= start - 1 and event['stop'] < stop:
                result['recovery_events'].append(dict(**event,
                    delay_frames={a: recovery_delay(v, event) for a, v in overlaps.items()}))
        summaries.append(result)
    return dict(posthoc_cases_only=True, automatic_identity_verdict=False,
                full_dev_gate_passed=False, cases=summaries)


def collect(cases, models):
    probe = ScoreOnlyNetwork(models['Tclean'], models['Rdisc'])
    captures = {a: SpatialCapture() for a in CONFIGS}
    captures['Rscore'] = captures['Rdisc']
    hooks = [models[a].backbone.vdrm.register_forward_hook(captures[a], with_kwargs=True) for a in CONFIGS]
    payloads, trajectories, records = {}, [], []
    try:
        with torch.no_grad():
            for sequence, gt, cover, absence, start, stop in cases:
                name = sequence.name
                trackers = {a: NativeTracker(m, config_for(CONFIGS[a])) for a, m in models.items()}
                trackers['Rscore'] = NativeTracker(probe, config_for(CONFIGS['Rdisc']))
                initial = read_rgb(sequence.frames[0])
                for t in trackers.values():
                    t.initialize(initial, dict(init_bbox=gt[0].tolist()))
                traces = {a: [dict(frame_1based=1, bbox_xywh=gt[0].tolist())] for a in ARMS}
                for frame in range(2, stop + 1):
                    image = read_rgb(sequence.frames[frame - 1])
                    snapshots, rows = {}, {}
                    for arm, tracker in trackers.items():
                        snapshots[arm], rows[arm] = native_step(tracker, image, gt[frame - 1], frame, captures[arm])
                        traces[arm].append(rows[arm])
                    if frame >= start:
                        t, snapshot = trackers['Rscore'], snapshots['Rscore']
                        base, residual = clone_output(probe.last_components)
                        anchor = rows['Rscore']['anchor_xywh']
                        base_box = decode_native(t, base, anchor, snapshot['resize'], image.shape)
                        residual_box = decode_native(t, residual, anchor, snapshot['resize'], image.shape)
                        same_peak = torch.equal((t.output_window * base['score_map']).flatten(1).argmax(1),
                                                (t.output_window * snapshot['output']['score_map']).flatten(1).argmax(1))
                        if same_peak and base_box != rows['Rscore']['bbox_xywh']:
                            raise RuntimeError('same-peak Rscore changed base regression bbox')
                        state_before = (list(t.state), t.frame_id)
                        with intervention(models['Rdisc'], 'alpha0'):
                            zero = clone_output(probe.forward(**snapshot['inputs']))
                            assert_exact(base, zero, keys=EXACT_KEYS + ('search_global_index',))
                            zero_box = decode_native(t, zero, anchor, snapshot['resize'], image.shape)
                            if zero_box != base_box:
                                raise RuntimeError('Rscore alpha0 bbox is not exactly Tclean')
                        if state_before != (list(t.state), t.frame_id) or input_hashes(snapshot['inputs']) != snapshot['input_sha256']:
                            raise RuntimeError('Rscore audit modified native state/inputs')
                        prefix = f'frames/{name}/{frame:06d}'
                        image_member = prefix + '.png'
                        payloads[image_member] = png_bytes(image)
                        artifacts = {}
                        for arm in ARMS:
                            arrays = {k: snapshots[arm]['output'][k].cpu().numpy() for k in MAP_KEYS}
                            if arm == 'Rscore':
                                arrays.update({a + '_' + k: o[k].cpu().numpy()
                                               for a, o in (('Tclean', base), ('Rdisc', residual)) for k in MAP_KEYS})
                                arrays['alpha0_score_map'] = zero['score_map'].cpu().numpy()
                                arrays['hann_window'] = t.output_window.cpu().numpy()
                            crop_member, maps_member = prefix + '_' + arm + '.png', prefix + '_' + arm + '.npz'
                            payloads[crop_member] = png_bytes(snapshots[arm]['patch'])
                            payloads[maps_member] = npz_bytes(arrays)
                            artifacts[arm] = dict(bbox_xywh=rows[arm]['bbox_xywh'],
                                                 crop_member=crop_member, maps_member=maps_member)
                        records.append(dict(sequence=name, frame_1based=frame, image_member=image_member,
                            arms=artifacts, same_hann_peak=bool(same_peak), base_samecrop_bbox=base_box,
                            residual_samecrop_bbox=residual_box, alpha0_bbox=zero_box, alpha0_exact=True,
                            inputs_member=save_inputs(payloads, prefix + '_Rscore', snapshot)))
                    if frame % 10 == 0 or frame == stop:
                        print(f'{name}: Rscore native prefix {frame}/{stop}; no GT reset; exact replays', flush=True)
                trajectories.append(dict(sequence=name, source_frames=sequence.frames[:stop],
                    gt_xywh=gt[:stop].tolist(), cover=cover[:stop].tolist(), absence=absence[:stop].tolist(), native=traces))
    finally:
        for hook in hooks:
            hook.remove()
    payloads['trajectories.json'], payloads['records.json'] = dump_json(trajectories), dump_json(records)
    payloads['focus_summary.json'] = dump_json(focus_summary(trajectories))
    return payloads, records


def run(save_dir, output, device):
    if Path(output).exists():
        raise FileExistsError('refusing to overwrite Rscore bundle: ' + str(output))
    paths = checkpoint_paths(save_dir)
    cases = prepare_cases(get_dataset('got10k_vdrm_dev'), plan=FOCUS_PLAN)
    print('Preflight passed: 3 checkpoints; 5 complete image prefixes; no prediction TXT required', flush=True)
    models, _, provenance = load_models(paths, torch.device(device))
    try:
        payloads, records = collect(cases, models)
    finally:
        for arm, model in models.items():
            after = state_hash(model)
            if (after != provenance[arm]['state_sha256_before'] or any(p.grad is not None for p in model.parameters())
                    or any(m.training or any(p.requires_grad for p in m.parameters(recurse=False)) for m in model.modules())):
                raise RuntimeError('Rscore audit changed weights/mode or produced gradients: ' + arm)
            provenance[arm]['state_sha256_after'] = after
    manifest = dict(kind='vdrm_rscore_probe', version=1, dataset='got10k_vdrm_dev',
        configurations=CONFIGS, probe_parameter=PROBE_CONFIG, plan={n: list(b) for n, b in FOCUS_PLAN.items()},
        records=len(records), native_tracking_steps=NATIVE_STEPS, optimizer_steps=0, provenance=provenance,
        environment=dict(python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
                         device=str(device), gpu=torch.cuda.get_device_name() if torch.device(device).type == 'cuda' else None),
        protocol='Posthoc five-case inference-only probe, not unbiased benchmark or identity verdict; '
                 'Rdisc score + Tclean size/offset on identical native inputs; own float trajectory from frame 1; '
                 'unchanged Hann/crop/CE/mapping/update; GT only initialization and post-inference labels; '
                 'exact replays, same-peak bbox equality and alpha0 visual/bbox equality required; '
                 'no optimizer/training/reliability/M2 and no overwrite of normal tracking results.')
    if len(records) != SELECTED_FRAMES:
        raise RuntimeError('incomplete Rscore focus collection')
    payloads['README.txt'] = manifest['protocol'].encode('utf-8')
    write_bundle(output, manifest, payloads)
    verified = check_bundle(output)
    print(f'RSCORE PROBE CHECK PASSED: {NATIVE_STEPS} native steps, {SELECTED_FRAMES} records; '
          'this is a numerical check, NOT a performance pass', flush=True)
    print('Bundle:', Path(output).resolve(), flush=True)
    return verified


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--save_dir', default='./output')
    parser.add_argument('--output', help='New .tar.gz path; existing bundles are never overwritten')
    parser.add_argument('--check_bundle', help='Independent bundle check; no weights/data/GPU required')
    args = parser.parse_args()
    if args.check_bundle:
        manifest = check_bundle(args.check_bundle)
        print(f'RSCORE BUNDLE CHECK PASSED: {manifest["records"]} records')
        return
    if not torch.cuda.is_available():
        parser.error('native tracking requires CUDA')
    torch.manual_seed(42)
    output = args.output or str(Path(args.save_dir) / 'analysis/frozen_vdrm/rscore_probe.tar.gz')
    run(args.save_dir, output, 'cuda')


if __name__ == '__main__':
    main()
