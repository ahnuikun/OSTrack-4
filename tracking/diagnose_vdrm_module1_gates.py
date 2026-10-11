"""Step03: Rdisc q/route factorial on shared inputs and native closed loops."""

import argparse
import hashlib
from pathlib import Path
import platform
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

import numpy as np
import torch

from lib.test.analysis.frozen_failure_diagnostics import CONFIGS, dump_json, write_bundle
from lib.test.analysis.module1_gates import ARMS, MODES, SHARED_MODES, check_bundle, protocol_plan, read_source
from lib.test.evaluation import get_dataset
from tracking.diagnose_vdrm_frozen_failures import EXACT_KEYS, SpatialCapture, config_for, load_models, png_bytes, state_hash
from tracking.diagnose_vdrm_native_float import (
    NativeTracker, assert_exact, clone_output, decode_native, input_hashes, native_step, npz_bytes, prepare_cases, read_rgb)
from tracking.vdrm_diagnostics.backend_pair_metrics import bbox_iou_xywh
from tracking.vdrm_diagnostics.module1_gate_interventions import gate_intervention


def final_kept(out):
    kept = torch.ones((1, 256), device=out['score_map'].device, dtype=torch.bool)
    for removed in out['removed_indexes_s']:
        if removed is not None:
            kept.scatter_(1, removed.long(), False)
    return kept


def evidence_arrays(out, tracker, snapshot, values, capture):
    tensors = dict(values)
    tensors.update({key: out[key] for key in ('score_map', 'size_map', 'offset_map')})
    tensors.update(alpha=out['vdrm_alpha'], indices=out['search_global_index'],
                   final_kept=final_kept(out), hann=tracker.output_window)
    arrays = {key: value.detach().cpu().numpy().copy() for key, value in tensors.items()}
    arrays.update({key: value.copy() for key, value in capture.maps.items()})
    arrays['padding'] = np.asarray(snapshot['padding']).copy()
    return arrays


def make_reference(source, plan):
    traces = {r['sequence']: r for r in source['trajectories']}
    reference = dict(sha256=source['sha256'], provenance={a: source['manifest']['provenance'][a] for a in ARMS},
        initial_boxes={name: traces[name]['gt_xywh'][0] for name in plan}, frames=[])
    for record in source['records']:
        name, frame = record['sequence'], record['frame_1based']
        if name not in plan or not plan[name][0] <= frame <= plan[name][1]:
            continue
        row = {key: record[key] for key in ('sequence', 'frame_1based', 'gt_xywh', 'cover', 'absence', 'image_shape_hw')}
        row['image_sha256'] = source['manifest']['members_sha256'][record['image_member']]
        row['native'] = {arm: traces[name]['native'][arm][frame-1] for arm in ARMS}
        reference['frames'].append(row)
    return reference


def preflight_source(cases, source):
    traces = {r['sequence']: r for r in source['trajectories']}
    hashes = source['manifest']['members_sha256']
    for sequence, gt, cover, absence, start, stop in cases:
        trace = traces[sequence.name]
        for key, value in [('gt_xywh', gt), ('cover', cover), ('absence', absence)]:
            if not np.array_equal(np.asarray(trace[key][:stop]), value[:stop]):
                raise ValueError('dataset labels differ from step01: ' + sequence.name + '/' + key)
        for frame, path in enumerate(sequence.frames[:stop], 1):
            digest = hashlib.sha256(png_bytes(read_rgb(path))).hexdigest()
            if digest != hashes[f'images/{sequence.name}/{frame:06d}.png']:
                raise ValueError(f'decoded image differs from step01: {sequence.name}/{frame}')


def assert_source_native(row, ref, mode):
    arm = 'Rdisc' if mode == 'control' else 'Tclean'
    if any(row[key] != ref['native'][arm][key] for key in ('bbox_xywh', 'anchor_xywh', 'input_sha256')):
        raise RuntimeError(f'{ref["sequence"]}/{ref["frame_1based"]}: {mode} differs from step01 {arm}; '
                           'check checkpoint, software/GPU and crop determinism before attributing interventions')


def collect(cases, models, cfg, reference):
    model = models['Rdisc']
    capture = SpatialCapture()
    hook = model.backbone.vdrm.register_forward_hook(capture, with_kwargs=True)
    refs = {(r['sequence'], r['frame_1based']): r for r in reference['frames']}
    payloads, rows = {}, []
    try:
        with torch.no_grad():
            for sequence, gt, cover, absence, start, stop in cases:
                trackers = {mode: NativeTracker(model, cfg) for mode in MODES}
                initial = read_rgb(sequence.frames[0])
                for tracker in trackers.values():
                    tracker.initialize(initial, dict(init_bbox=gt[0].tolist()))
                for frame in range(start, stop+1):
                    image = read_rgb(sequence.frames[frame-1])
                    ref = refs[sequence.name, frame]
                    common = {key: ref[key] for key in ('sequence', 'frame_1based', 'gt_xywh', 'cover', 'absence',
                                                       'image_shape_hw', 'image_sha256')}
                    closed = {}
                    for mode, tracker in trackers.items():
                        with gate_intervention(model, mode) as values:
                            snapshot, native_row = native_step(tracker, image, gt[frame-1], frame, capture)
                            arrays = evidence_arrays(snapshot['output'], tracker, snapshot, values, capture)
                        if mode in ('control', 'alpha0'):
                            assert_source_native(native_row, ref, mode)
                        member = f'arrays/{sequence.name}/{frame:06d}_closed_{mode}.npz'
                        payloads[member] = npz_bytes(arrays)
                        row = dict(common, **{k: v for k, v in native_row.items() if k != 'frame_1based'},
                                   scope='closed', mode=mode, arrays_member=member)
                        rows.append(row)
                        closed[mode] = row
                        if mode == 'control':
                            control_snapshot = snapshot
                            control_evidence = arrays
                    rows.append(dict(closed['control'], scope='shared', native_input_verified=True))
                    before = {mode: (list(t.state), t.frame_id) for mode, t in trackers.items()}
                    tracker = trackers['control']
                    anchor = closed['control']['anchor_xywh']
                    inputs = control_snapshot['inputs']
                    for mode in SHARED_MODES[1:]:
                        with gate_intervention(model, mode) as values:
                            out = clone_output(model.forward(**inputs))
                            arrays = evidence_arrays(out, tracker, control_snapshot, values, capture)
                            box = decode_native(tracker, out, anchor, control_snapshot['resize'], image.shape)
                        for field in ('raw_q', 'raw_route', 'prototype_gram', 'indices', 'part_valid'):
                            if not np.array_equal(arrays[field], control_evidence[field]):
                                raise RuntimeError('shared input changed pre-residual evidence: ' + field)
                        alpha_check = False
                        if mode == 'alpha0':
                            baseline = models['Tclean'].forward(**inputs)
                            assert_exact(baseline, out, keys=EXACT_KEYS + ('search_global_index',))
                            for key in ('score_map', 'size_map', 'offset_map'):
                                arrays['tclean_' + key] = baseline[key].detach().cpu().numpy().copy()
                            arrays['tclean_final_kept'] = final_kept(baseline).cpu().numpy()
                            alpha_check = True
                        if input_hashes(inputs) != control_snapshot['input_sha256']:
                            raise RuntimeError('intervention modified native inputs')
                        member = f'arrays/{sequence.name}/{frame:06d}_shared_{mode}.npz'
                        payloads[member] = npz_bytes(arrays)
                        rows.append(dict(common, scope='shared', mode=mode, arrays_member=member,
                            anchor_xywh=anchor, bbox_xywh=box, resize_factor=control_snapshot['resize'],
                            input_sha256=control_snapshot['input_sha256'], native_input_verified=True,
                            alpha0_exact_Tclean=alpha_check,
                            iou=bbox_iou_xywh(box, gt[frame-1]) if (gt[frame-1, 2:] > 0).all() else None))
                    if before != {mode: (list(t.state), t.frame_id) for mode, t in trackers.items()}:
                        raise RuntimeError('shared interventions modified tracker state')
                    if frame % 10 == 0 or frame == stop:
                        print(f'{sequence.name}: {frame}/{stop}; native replay and source parity passed', flush=True)
    finally:
        hook.remove()
    return {'rows.json': dump_json(rows), 'source_reference.json': dump_json(reference), **payloads}


def run(save_dir, source_path, output, smoke=False):
    if Path(output).exists():
        raise FileExistsError('refusing to overwrite gate evidence: ' + str(output))
    if not torch.cuda.is_available():
        raise RuntimeError('native tracker requires CUDA')
    paths = {arm: Path(save_dir) / 'checkpoints/train/ostrack' / CONFIGS[arm] / 'OSTrack_ep0300.pth.tar' for arm in ARMS}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError('missing registered checkpoint: ' + str(path))
    print('Checking step01 source bundle on CPU...', flush=True)
    source = read_source(source_path)
    plan = protocol_plan(smoke)
    cases = prepare_cases(get_dataset('got10k_vdrm_dev'), plan=plan)
    preflight_source(cases, source)
    # This loader returns ONE shared cfg, not a dictionary indexed by arms.
    models, _, provenance = load_models(paths, 'cuda', arms=ARMS)
    cfg = config_for(CONFIGS['Rdisc'])
    for arm in ARMS:
        old = source['manifest']['provenance'][arm]
        if any(provenance[arm][key] != old[key] for key in ('checkpoint_sha256', 'state_sha256_before')):
            raise ValueError('loaded model differs from step01: ' + arm)
    reference = make_reference(source, plan)
    count = sum(stop-start+1 for start, stop in plan.values())
    print(f'STEP03 PRECHECK PASSED: smoke={smoke}, {len(cases)} cases, {count*5} native steps, {count*18} forwards', flush=True)
    try:
        payloads = collect(cases, models, cfg, reference)
    finally:
        for arm, model in models.items():
            after = state_hash(model)
            if (after != provenance[arm]['state_sha256_before'] or any(p.grad is not None for p in model.parameters())
                    or any(m.training for m in model.modules())):
                raise RuntimeError('experiment changed model state/mode or produced gradients: ' + arm)
            provenance[arm]['state_sha256_after'] = after
    manifest = dict(kind='vdrm_module1_step03_gates', version=1, dataset='got10k_vdrm_dev',
        configurations={arm: CONFIGS[arm] for arm in ARMS}, smoke_only=smoke,
        modes=list(MODES), shared_modes=list(SHARED_MODES), plan={k: list(v) for k, v in plan.items()},
        optimizer_steps=0, native_steps=count*5, forward_calls=count*18, source_verified=True,
        source_sha256=source['sha256'], provenance=provenance,
        environment=dict(python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(), matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
            cudnn_tf32=torch.backends.cudnn.allow_tf32, cudnn_benchmark=torch.backends.cudnn.benchmark))
    write_bundle(output, manifest, payloads)
    checked, _ = check_bundle(output)
    marker = 'STEP03 SMOKE PASSED' if smoke else 'STEP03 EVIDENCE CHECK PASSED'
    print(f'{marker}: {checked["native_steps"]} native steps; diagnosis only, no performance pass', flush=True)
    print('Download:', Path(output).resolve(), flush=True)
    return checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--save_dir', default='./output')
    parser.add_argument('--source', help='Verified step01 module1_evidence.tar.gz')
    parser.add_argument('--output')
    parser.add_argument('--smoke', action='store_true', help='Two native steps per case; source is still fully checked')
    args = parser.parse_args()
    root = Path(args.save_dir) / 'analysis'
    source = args.source or str(root / 'module1_step01/module1_evidence.tar.gz')
    output = args.output or str(root / 'module1_step03' / ('module1_gates_smoke.tar.gz' if args.smoke else 'module1_gates.tar.gz'))
    run(args.save_dir, source, output, args.smoke)


if __name__ == '__main__':
    main()
