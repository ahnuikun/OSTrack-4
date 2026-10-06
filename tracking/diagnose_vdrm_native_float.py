"""Read-only native-float replay and closed-loop forks of two VDRM cases.

Only initialization GT enters tracking. Later GT is used after inference.
Never load integer prediction TXT files or write normal tracking_results.
"""

import argparse
from contextlib import contextmanager, nullcontext
from copy import copy, deepcopy
import hashlib
import io
from pathlib import Path
import platform
import sys
from types import SimpleNamespace

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from lib.test.analysis.frozen_failure_diagnostics import CONFIGS, dump_json, write_bundle
from lib.test.analysis.frozen_vdrm_stages import visibility_protocol
from lib.test.analysis.native_float_audit import AUDIT_PLAN, check_bundle
from lib.test.evaluation import get_dataset
from lib.test.tracker import ostrack as native
from lib.test.tracker.basetracker import BaseTracker
from lib.test.tracker.data_utils import Preprocessor
from lib.test.utils.hann import hann2d
from lib.train.data.processing_utils import transform_image_to_crop
from lib.utils.box_ops import clip_box
from tracking.diagnose_vdrm_frozen_failures import (
    EXACT_KEYS, SpatialCapture, checkpoint_paths, config_for, crop_geometry, intervention,
    load_models, png_bytes, state_hash, summarize_output)
from tracking.vdrm_diagnostics.backend_pair_metrics import bbox_iou_xywh, response_statistics
from tracking.vdrm_diagnostics.diagnose_vdrm_backend_pairs import read_rgb

REPLAY_KEYS = EXACT_KEYS + ('part_route_logits', 'search_global_index')
FORKS = ('control', 'route0', 'alpha0')
INPUT_KEYS = {'template', 'search', 'ce_template_mask', 'template_bbox'}


class NativeTracker(native.OSTrack):
    """Bind already provenance-checked models; use native initialize/track.

    Only constructor loading is replaced: load_models already strictly loaded
    all three checkpoints. No crop, normalization, decode or update override.
    """

    def __init__(self, model, cfg):
        params = SimpleNamespace(cfg=cfg, debug=0, save_all_boxes=False,
            template_factor=cfg.TEST.TEMPLATE_FACTOR, template_size=cfg.TEST.TEMPLATE_SIZE,
            search_factor=cfg.TEST.SEARCH_FACTOR, search_size=cfg.TEST.SEARCH_SIZE)
        BaseTracker.__init__(self, params)
        self.cfg, self.network = cfg, model
        self.preprocessor = Preprocessor()
        self.feat_sz = cfg.TEST.SEARCH_SIZE // cfg.MODEL.BACKBONE.STRIDE
        self.output_window = hann2d(torch.tensor([self.feat_sz] * 2).long(), centered=True).to(
            next(model.parameters()).device)
        self.state, self.frame_id = None, 0
        self.debug = self.use_visdom = self.save_all_boxes = False
        self.z_dict1 = {}


def tensor_hash(value):
    if value is None:
        return None
    array = value.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(tuple(array.shape)).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def input_hashes(inputs):
    return {key: tensor_hash(value) for key, value in inputs.items()}


def clone_output(value):
    # Raw-alpha diagnostics may alias the parameter. Clone while intervention
    # is active; otherwise restoring alpha would silently rewrite the record.
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {k: clone_output(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(clone_output(v) for v in value)
    return deepcopy(value)


@contextmanager
def capture_native(tracker):
    """Intercept actual native crop/forward, including network.forward calls.

    OSTrack calls network.forward directly, so a network __call__ hook alone
    would miss it. Temporary methods/functions are restored even on errors.
    """
    model = tracker.network
    original_forward, original_crop = model.forward, native.sample_target
    had_override = 'forward' in model.__dict__
    snapshot = dict(crop_calls=0, forward_calls=0)

    def crop(*args, **kwargs):
        result = original_crop(*args, **kwargs)
        snapshot['crop_calls'] += 1
        snapshot['patch'], snapshot['resize'], snapshot['padding'] = result
        return result

    def forward(*args, **kwargs):
        if args or set(kwargs) != INPUT_KEYS:
            raise RuntimeError('unexpected native forward inputs; no current GT is allowed')
        snapshot['forward_calls'] += 1
        snapshot['inputs'] = kwargs
        snapshot['input_sha256'] = input_hashes(kwargs)
        out = original_forward(**kwargs)
        snapshot['output'] = clone_output(out)
        return out

    model.forward, native.sample_target = forward, crop
    try:
        yield snapshot
        if snapshot['crop_calls'] != 1 or snapshot['forward_calls'] != 1:
            raise RuntimeError('expected exactly one native crop and forward')
        if snapshot['input_sha256'] != input_hashes(snapshot['inputs']):
            raise RuntimeError('native forward modified input tensors')
    finally:
        native.sample_target = original_crop
        if had_override:
            model.forward = original_forward
        else:
            del model.forward


def decode_native(tracker, out, anchor, resize, shape):
    """Same native Hann/head/mapping/clip; never advance the tracker state."""
    state, frame_id = tracker.state, tracker.frame_id
    try:
        tracker.state = list(anchor)
        response = tracker.output_window * out['score_map']
        boxes = tracker.network.box_head.cal_bbox(response, out['size_map'], out['offset_map']).view(-1, 4)
        box = (boxes.mean(dim=0) * tracker.params.search_size / resize).tolist()
        return clip_box(tracker.map_box_back(box, resize), *shape[:2], margin=10)
    finally:
        tracker.state, tracker.frame_id = state, frame_id


def assert_exact(reference, replay, keys=REPLAY_KEYS):
    for key in keys:
        if (not torch.isfinite(reference[key]).all() or not torch.isfinite(replay[key]).all()
                or not torch.equal(reference[key], replay[key])):
            raise RuntimeError('native replay is not exact: ' + key)
    removed_a, removed_b = reference['removed_indexes_s'], replay['removed_indexes_s']
    if len(removed_a) != len(removed_b) or any(
        (a is None) != (b is None) or (a is not None and not torch.equal(a, b))
        for a, b in zip(removed_a, removed_b)
    ):
        raise RuntimeError('native replay CE removal differs')


def fork_tracker(tracker):
    # The native tracker has a fixed immutable template and assigns a NEW
    # state list every track(). Network is stateless eval; branches share only
    # immutable template/network objects. Copy the mutable current state.
    branch = copy(tracker)
    branch.state = list(tracker.state)
    return branch


def prepare_cases(dataset, plan=None):
    if len(dataset) != 152:
        raise ValueError('requires the fixed 152-sequence training development split')
    by_name = {sequence.name: sequence for sequence in dataset}
    if len(by_name) != len(dataset):
        raise ValueError('duplicate development sequence names')
    cases = []
    for name, bounds in (AUDIT_PLAN if plan is None else plan).items():
        if name not in by_name:
            raise ValueError('missing registered development case: ' + name)
        sequence = by_name[name]
        gt = np.asarray(sequence.ground_truth_rect, dtype=float)
        start, stop = bounds
        if (gt.ndim != 2 or gt.shape[1] != 4 or len(gt) < stop
                or len(sequence.frames) != len(gt) or not np.isfinite(gt).all()
                or (gt[:, 2:] < 0).any() or (gt[0, 2:] <= 0).any()):
            raise ValueError('invalid image/GT prefix: ' + name)
        folder = Path(sequence.frames[0]).parent
        cover = np.loadtxt(folder / 'cover.label', ndmin=1)
        absence = np.loadtxt(folder / 'absence.label', ndmin=1)
        if cover.shape != (len(gt),):
            raise ValueError('visibility/GT length mismatch: ' + name)
        visibility_protocol(cover, absence)
        # Complete history, not just the selected window, must be readable.
        for frame in sequence.frames[:stop]:
            read_rgb(frame)
        cases.append((sequence, gt, cover, absence, start, stop))
    return cases


def npz_bytes(arrays):
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    return stream.getvalue()


def float_row(tracker, snapshot, anchor, gt, frame, captured):
    out, box = snapshot['output'], list(tracker.state)
    decoded = decode_native(tracker, out, anchor, snapshot['resize'], snapshot['shape'])
    if box != decoded:
        raise RuntimeError('native float bbox differs from captured-output decode')
    return dict(frame_1based=frame, anchor_xywh=list(anchor), bbox_xywh=box,
        iou=bbox_iou_xywh(box, gt) if (gt[2:] > 0).all() else None,
        response_raw=response_statistics(out['score_map']),
        response_windowed=response_statistics(tracker.output_window * out['score_map']),
        input_sha256=snapshot['input_sha256'], resize_factor=snapshot['resize'],
        exact_replay=True, actual_injected_residual=dict(captured.last))


def native_step(tracker, image, gt, frame, capture):
    anchor = list(tracker.state)
    with capture_native(tracker) as snapshot:
        result = tracker.track(image)
    snapshot['shape'] = image.shape
    if list(result['target_bbox']) != list(tracker.state) or tracker.frame_id != frame - 1:
        raise RuntimeError('native state/frame-id contract violated')
    # Replay uses the EXACT normalized inputs captured from this native call,
    # not inputs reconstructed from a prediction file or a rounded anchor.
    replay = tracker.network.forward(**snapshot['inputs'])
    assert_exact(snapshot['output'], replay)
    if input_hashes(snapshot['inputs']) != snapshot['input_sha256']:
        raise RuntimeError('self replay modified actual native input tensors')
    row = float_row(tracker, snapshot, anchor, gt, frame, capture)
    return snapshot, row


def label_output(tracker, out, snapshot, anchor, gt, cfg, capture):
    # All GT-dependent computation is AFTER inference, never model kwargs.
    device = out['score_map'].device
    target = transform_image_to_crop(torch.tensor(gt, dtype=torch.float32),
        torch.tensor(anchor, dtype=torch.float32), snapshot['resize'],
        torch.tensor([cfg.TEST.SEARCH_SIZE] * 2), normalize=True).to(device)[None]
    padding = torch.from_numpy(snapshot['padding']).to(device=device, dtype=torch.bool)[None]
    metrics, arrays = summarize_output(tracker.network, out, cfg, anchor, snapshot['resize'],
        snapshot['shape'], gt, target, padding, capture)
    decoded = decode_native(tracker, out, anchor, snapshot['resize'], snapshot['shape'])
    # The older integer-audit helper groups float additions differently in
    # map_box_back_from_anchor. Reuse its LABELS/MAPS, never its decoded box.
    # Even a one-ULP mapping difference must not enter a native float trace.
    metrics['bbox_xywh'] = decoded
    metrics['iou'] = bbox_iou_xywh(decoded, gt) if (gt[2:] > 0).all() else None
    return metrics, arrays


def save_inputs(payloads, prefix, snapshot):
    member = prefix + '_inputs.npz'
    payloads[member] = npz_bytes({k: v.detach().cpu().numpy() for k, v in snapshot['inputs'].items()
                                if v is not None})
    return member


def same_crop_audit(trackers, snapshots, frame_rows, captures, cfg, gt, name, frame, payloads):
    records = []
    for anchor_arm in CONFIGS:
        snapshot = snapshots[anchor_arm]
        anchor = frame_rows[anchor_arm]['anchor_xywh']
        inputs, outputs, arrays = snapshot['inputs'], {}, {}
        reference = None

        def evaluate(arm, label):
            out = trackers[arm].network.forward(**inputs)
            if any(not torch.isfinite(out[k]).all() for k in REPLAY_KEYS):
                raise RuntimeError('non-finite counterfactual output: ' + label)
            if reference is not None and not torch.equal(reference['search_global_index'], out['search_global_index']):
                raise RuntimeError('pre-residual CE indices differ on shared native input')
            if arm == anchor_arm and label == arm:
                assert_exact(snapshot['output'], out)
            metrics, maps = label_output(trackers[arm], out, snapshot, anchor, gt, cfg, captures[arm])
            outputs[label] = metrics
            arrays.update({label + '_' + key: value for key, value in maps.items()})
            return out

        reference = evaluate('Tclean', 'Tclean')
        evaluate('Rfreeze', 'Rfreeze')
        evaluate('Rdisc', 'Rdisc')
        with intervention(trackers['Rdisc'].network, 'route0'):
            evaluate('Rdisc', 'Rdisc_route0')
        with intervention(trackers['Rdisc'].network, 'alpha0'):
            zero = evaluate('Rdisc', 'Rdisc_alpha0')
            # Route logits still differ: disabling residual injection does
            # not remove the learned route head. Only VISUAL outputs must
            # equal Tclean; the pre-residual CE indices must also match.
            assert_exact(reference, zero, keys=EXACT_KEYS + ('search_global_index',))
        if input_hashes(inputs) != snapshot['input_sha256']:
            raise RuntimeError('counterfactual modified actual native input tensors')
        if outputs[anchor_arm]['bbox_xywh'] != frame_rows[anchor_arm]['bbox_xywh']:
            raise RuntimeError('same-crop replay differs from native float bbox')
        prefix = f'crops/{name}/{frame:06d}_{anchor_arm}'
        crop_member, maps_member = prefix + '.png', prefix + '.npz'
        payloads[crop_member] = png_bytes(snapshot['patch'])
        payloads[maps_member] = npz_bytes(arrays)
        records.append(dict(sequence=name, frame_1based=frame, anchor=anchor_arm,
            anchor_xywh=anchor, resize_factor=snapshot['resize'], input_sha256=snapshot['input_sha256'],
            inputs_member=save_inputs(payloads, prefix, snapshot), crop_member=crop_member,
            maps_member=maps_member, image_member=f'images/{name}/{frame:06d}.png',
            native_float_bbox=frame_rows[anchor_arm]['bbox_xywh'], native_replay_exact=True,
            alpha0_exact=True, crop_geometry=crop_geometry(anchor, cfg.TEST.SEARCH_FACTOR, snapshot['shape'], gt),
            outputs=outputs))
    return records


def collect(cases, models, cfg):
    captures = {arm: SpatialCapture() for arm in CONFIGS}
    hooks = [models[arm].backbone.vdrm.register_forward_hook(captures[arm], with_kwargs=True) for arm in CONFIGS]
    payloads, records, forks, trajectories = {}, [], [], []
    try:
        with torch.no_grad():
            for sequence, gt, cover, absence, start, stop in cases:
                name = sequence.name
                trackers = {arm: NativeTracker(model, config_for(CONFIGS[arm])) for arm, model in models.items()}
                initial = read_rgb(sequence.frames[0])
                payloads[f'images/{name}/000001.png'] = png_bytes(initial)
                for tracker in trackers.values():
                    tracker.initialize(initial, {'init_bbox': gt[0].tolist()})
                templates = {arm: dict(template=tracker.z_dict1.tensors, template_bbox=tracker.vdrm_template_bbox,
                                      ce_template_mask=tracker.box_mask_z) for arm, tracker in trackers.items()}
                if any(input_hashes(v) != input_hashes(templates['Tclean']) for v in templates.values()):
                    raise RuntimeError('native initialization tensors differ across frozen arms')
                traces = {arm: [dict(frame_1based=1, bbox_xywh=gt[0].tolist())] for arm in CONFIGS}
                branches = {}
                for frame in range(2, stop + 1):
                    if frame == start:
                        branches = {mode: fork_tracker(trackers['Rdisc']) for mode in FORKS}
                    image = read_rgb(sequence.frames[frame - 1])
                    snapshots, rows = {}, {}
                    for arm, tracker in trackers.items():
                        snapshots[arm], rows[arm] = native_step(tracker, image, gt[frame - 1], frame, captures[arm])
                        traces[arm].append(rows[arm])
                    if frame >= start:
                        payloads[f'images/{name}/{frame:06d}.png'] = png_bytes(image)
                        # Forks never update the three main trackers. Control
                        # validates shallow-copy isolation throughout window.
                        for mode, branch in branches.items():
                            context = nullcontext() if mode == 'control' else intervention(models['Rdisc'], mode)
                            with context:
                                snapshot, row = native_step(branch, image, gt[frame - 1], frame, captures['Rdisc'])
                                if mode == 'control':
                                    assert_exact(snapshots['Rdisc']['output'], snapshot['output'])
                                    if row['bbox_xywh'] != rows['Rdisc']['bbox_xywh'] or row['anchor_xywh'] != rows['Rdisc']['anchor_xywh']:
                                        raise RuntimeError('unmodified closed-loop fork differs from native Rdisc')
                                metrics, maps = label_output(branch, snapshot['output'], snapshot,
                                    row['anchor_xywh'], gt[frame - 1], cfg, captures['Rdisc'])
                                prefix = f'forks/{name}/{frame:06d}_{mode}'
                                payloads[prefix + '.png'] = png_bytes(snapshot['patch'])
                                payloads[prefix + '.npz'] = npz_bytes(maps)
                                forks.append(dict(sequence=name, mode=mode, fork_frame_1based=start,
                                    **row, outputs=metrics, control_exact=(mode == 'control'),
                                    crop_geometry=crop_geometry(row['anchor_xywh'], cfg.TEST.SEARCH_FACTOR, image.shape, gt[frame - 1]),
                                    crop_member=prefix + '.png', maps_member=prefix + '.npz',
                                    inputs_member=save_inputs(payloads, prefix, snapshot)))
                        before = {arm: (list(t.state), t.frame_id) for arm, t in trackers.items()}
                        records.extend(same_crop_audit(trackers, snapshots, rows, captures, cfg,
                                                     gt[frame - 1], name, frame, payloads))
                        if before != {arm: (list(t.state), t.frame_id) for arm, t in trackers.items()}:
                            raise RuntimeError('counterfactual audit modified closed-loop tracker state')
                    if frame % 10 == 0 or frame == stop:
                        print(f'{name}: native float prefix {frame}/{stop}; all self replays exact', flush=True)
                trajectories.append(dict(sequence=name, fork_frame_1based=start, stop_frame_1based=stop,
                    source_frames=sequence.frames[:stop], gt_xywh=gt[:stop].tolist(),
                    cover=cover[:stop].tolist(), absence=absence[:stop].tolist(), native=traces))
    finally:
        for hook in hooks:
            hook.remove()
    payloads.update({'records.json': dump_json(records), 'forks.json': dump_json(forks),
                     'trajectories.json': dump_json(trajectories)})
    return payloads, records, forks


def run(save_dir, output, device):
    if Path(output).exists():
        raise FileExistsError('refusing to overwrite audit bundle: ' + str(output))
    paths = checkpoint_paths(save_dir)
    cases = prepare_cases(get_dataset('got10k_vdrm_dev'))
    print('Preflight passed: 3 checkpoints; 2 complete native prefixes; no prediction TXT required', flush=True)
    models, cfg, provenance = load_models(paths, torch.device(device))
    try:
        payloads, records, forks = collect(cases, models, cfg)
    finally:
        for arm, model in models.items():
            after = state_hash(model)
            if (after != provenance[arm]['state_sha256_before'] or any(p.grad is not None for p in model.parameters())
                    or any(module.training for module in model.modules())):
                raise RuntimeError('audit changed model state/mode or produced gradients: ' + arm)
            provenance[arm]['state_sha256_after'] = after
    manifest = dict(version=1, kind='vdrm_native_float', dataset='got10k_vdrm_dev',
        configurations=CONFIGS, plan={name: list(bounds) for name, bounds in AUDIT_PLAN.items()},
        selected_frames=sum(stop - start + 1 for start, stop in AUDIT_PLAN.values()),
        native_tracking_steps=sum(stop - 1 for _, stop in AUDIT_PLAN.values()) * 3,
        records=len(records), fork_records=len(forks), optimizer_steps=0, provenance=provenance,
        environment=dict(python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
            device=str(device), gpu=torch.cuda.get_device_name() if torch.device(device).type == 'cuda' else None,
            cudnn_benchmark=torch.backends.cudnn.benchmark, cudnn_deterministic=torch.backends.cudnn.deterministic,
            matmul_tf32=torch.backends.cuda.matmul.allow_tf32, cudnn_tf32=torch.backends.cudnn.allow_tf32),
        protocol='Two posthoc development cases only; native initialize/track and actual float previous state; '
                 'exact actual-input self replay and unmodified closed-loop control required; '
                 'route0 zeros learned correction only; alpha0 temporarily disables whole residual; '
                 'fork at fixed pre-event frame, not GT reset; GT for initialization/posthoc labels only; '
                 'no optimizer, training, benchmark or automatic identity claim.')
    payloads['README.txt'] = manifest['protocol'].encode('utf-8')
    write_bundle(output, manifest, payloads)
    verified = check_bundle(output)
    print(f'NATIVE FLOAT AUDIT PASSED: {verified["native_tracking_steps"]} native steps, '
          f'{verified["records"]} same-crop records, {verified["fork_records"]} fork records', flush=True)
    print('Bundle:', Path(output).resolve(), flush=True)
    return verified


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--save_dir', default='./output')
    parser.add_argument('--output', help='New .tar.gz path; existing files are never overwritten')
    parser.add_argument('--device', choices=['cuda'], default='cuda')
    parser.add_argument('--check_bundle', help='Standalone bundle check; no data, checkpoints or GPU required')
    args = parser.parse_args()
    if args.check_bundle:
        manifest = check_bundle(args.check_bundle)
        print(f'BUNDLE CHECK PASSED: {manifest["records"]} same-crop records, {manifest["fork_records"]} fork records')
        return
    if not torch.cuda.is_available():
        parser.error('native Preprocessor requires CUDA for tracking; bundle check does not')
    # Inference only; no user-facing seed flag, no changed training recipe.
    torch.manual_seed(42)
    output = args.output or str(Path(args.save_dir) / 'analysis/frozen_vdrm/native_float_audit.tar.gz')
    run(args.save_dir, output, args.device)


if __name__ == '__main__':
    main()
