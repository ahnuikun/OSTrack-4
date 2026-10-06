"""Read-only, same-crop counterfactuals for five registered frozen VDRM cases.

This is NOT a tracker run or an unbiased benchmark. All anchors are saved
previous predictions (integer TXT precision), never GT or new predictions.
Current GT is used only AFTER inference for diagnostic labels and metrics.
"""

import argparse
from contextlib import contextmanager
import io
import math
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from lib.config.ostrack.config import default_config, update_config_from_file
from lib.models.ostrack import build_ostrack
from lib.test.analysis.frozen_failure_diagnostics import (
    CONFIGS, PLAN, check_bundle, dump_json, make_window_plan, overlapping_fraction, write_bundle)
from lib.test.evaluation import get_dataset
from lib.test.evaluation.environment import env_settings
from lib.test.evaluation.result_paths import resolve_result_bbox_path
from lib.test.utils.hann import hann2d
from lib.train.actors.discriminative_route_loss import compute_discriminative_route_loss
from lib.train.data.processing_utils import sample_target, transform_image_to_crop
from lib.train.frozen_vdrm import (
    PREFIX, file_hash, validate_frozen_config, validate_frozen_checkpoint, visual_hash)
from lib.utils.box_ops import clip_box
from lib.utils.ce_utils import generate_mask_cond
from tracking.vdrm_diagnostics.backend_pair_metrics import bbox_iou_xywh, response_statistics
from tracking.vdrm_diagnostics.diagnose_vdrm_backend_pairs import (
    ResidualCapture, map_box_back_from_anchor, preprocess_patch, read_rgb, template_bbox_in_crop)

EXACT_KEYS = ('pred_boxes', 'score_map', 'size_map', 'offset_map', 'backbone_feat')


class SpatialCapture(ResidualCapture):
    """Keep per-retained-token injected norms/energy as well as summaries."""

    def __call__(self, module, args, kwargs, output):
        super().__call__(module, args, kwargs, output)
        tokens = args[0] if args else kwargs['tokens']
        start = int(kwargs['template_length'])
        search = tokens[:, start:]
        delta = output[0][:, start:] - search
        self.maps = dict(injected_relative_norm=(delta.norm(dim=-1) / search.norm(dim=-1).clamp_min(1e-6))
                         .detach().cpu().numpy(), injected_energy=delta.square().sum(-1).detach().cpu().numpy())


def config_for(name):
    cfg = default_config()
    update_config_from_file(PROJECT / 'experiments/ostrack' / (name + '.yaml'), cfg)
    return cfg


def checkpoint_paths(save_dir):
    paths = {a: Path(save_dir) / 'checkpoints/train/ostrack' / n / 'OSTrack_ep0300.pth.tar'
             for a, n in CONFIGS.items()}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError('missing epoch-300 checkpoint: ' + str(path))
    return paths


def prepare_cases(results_root, dataset):
    if len(dataset) != 152:
        raise ValueError('requires the registered 152-sequence GOT training development split')
    by_name = {s.name: s for s in dataset}
    cases = []
    for name in PLAN:
        sequence = by_name[name]
        gt = sequence.ground_truth_rect
        if len(sequence.frames) != len(gt):
            raise ValueError('GT/image length mismatch: ' + name)
        folder = Path(sequence.frames[0]).parent
        cover = np.loadtxt(folder / 'cover.label', ndmin=1)
        absence = np.loadtxt(folder / 'absence.label', ndmin=1)
        paths = {a: Path(resolve_result_bbox_path(Path(results_root) / n, sequence))
                 for a, n in CONFIGS.items()}
        predictions = {a: np.loadtxt(p, ndmin=2) for a, p in paths.items()}
        plan = make_window_plan(name, gt, cover, absence, predictions)
        # Decode all requested images before loading any GPU model.
        for frame in [1, *plan['selected_frames_1based']]:
            read_rgb(sequence.frames[frame - 1])
        plan['result_sources'] = {a: dict(path=str(p.resolve()), sha256=file_hash(p))
                                  for a, p in paths.items()}
        cases.append((sequence, predictions, plan))
    return cases


def state_hash(model):
    # Include original VDRM scalars AND the learned route head, not just vision.
    return visual_hash({'probe.' + n: v for n, v in model.state_dict().items()})


def load_models(paths, device, arms=None):
    # Tclean must be loaded first: every candidate is checked against its
    # exact source checkpoint. Default remains the original three-arm audit.
    arms = tuple(CONFIGS) if arms is None else tuple(arms)
    if not arms or arms[0] != 'Tclean' or len(set(arms)) != len(arms) or any(a not in CONFIGS for a in arms):
        raise ValueError('model arms must be unique registered names, with Tclean first')
    models, configs, provenance = {}, {}, {}
    source_hash = file_hash(paths['Tclean'])
    source_visual = None
    for arm in arms:
        name = CONFIGS[arm]
        cfg = config_for(name)
        checkpoint = torch.load(paths[arm], map_location='cpu', weights_only=False)
        model = build_ostrack(cfg, training=False)
        if arm == 'Tclean':
            settings = checkpoint.get('settings')
            identity = settings.get('config_name') if isinstance(settings, dict) else getattr(settings, 'config_name', None)
            if (identity != name or checkpoint.get('net_type') != 'OSTrack'
                    or checkpoint.get('epoch') != 300 or checkpoint['net'][PREFIX + 'alpha'].item() != 0.):
                raise ValueError('source is not the registered zero-alpha Tclean epoch-300 checkpoint')
            source_visual = visual_hash(checkpoint['net'])
            contract = dict(source_sha256=source_hash, visual_sha256=source_visual)
        else:
            validate_frozen_config(cfg)
            contract = validate_frozen_checkpoint(model, checkpoint, cfg=cfg, expected_epoch=300)
            if (contract['source_sha256'] != source_hash or contract['visual_sha256'] != source_visual
                    or contract.get('config_name') != name):
                raise ValueError('candidate does not share this exact Tclean checkpoint: ' + arm)
        model.load_state_dict(checkpoint['net'], strict=True)
        models[arm] = model.requires_grad_(False).to(device).eval()
        configs[arm] = cfg
        provenance[arm] = dict(checkpoint=str(paths[arm].resolve()),
            checkpoint_sha256=source_hash if arm == 'Tclean' else file_hash(paths[arm]),
            contract=contract, state_sha256_before=state_hash(model))
        del checkpoint
    cfg = configs['Tclean']
    if any(c.TEST != cfg.TEST or c.DATA.MEAN != cfg.DATA.MEAN or c.DATA.STD != cfg.DATA.STD
           or c.MODEL.BACKBONE.STRIDE != cfg.MODEL.BACKBONE.STRIDE for c in configs.values()):
        raise ValueError('same-crop models must share native test geometry/normalization')
    return models, cfg, provenance


@contextmanager
def intervention(model, mode):
    """Restore temporary hooks/alpha even when inference raises."""
    module = model.backbone.vdrm
    alpha = module.alpha.detach().clone()
    hook = None
    try:
        if mode == 'route0':
            if module.route_head is None:
                raise ValueError('route0 requires Rdisc route head')
            hook = module.route_head.register_forward_hook(lambda m, args, output: torch.zeros_like(output))
        elif mode == 'alpha0':
            with torch.no_grad():
                module.alpha.zero_()
        else:
            raise ValueError('unknown counterfactual: ' + mode)
        yield
    finally:
        if hook is not None:
            hook.remove()
        with torch.no_grad():
            module.alpha.copy_(alpha)


def crop_geometry(anchor, factor, image_shape, gt):
    side = math.ceil(math.sqrt(anchor[2] * anchor[3]) * factor)
    x, y = [round(anchor[i] + .5 * anchor[i + 2] - .5 * side) for i in (0, 1)]
    height, width = image_shape[:2]
    # sample_target deliberately excludes the final source row/column when
    # touching the boundary (+1 padding rule); report actual visual support.
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + side, width - 1), min(y + side, height - 1)
    return dict(square_xywh=[x, y, side, side],
        valid_source_xywh=[x0, y0, max(0, x1 - x0), max(0, y1 - y0)],
        gt_fraction_in_square=overlapping_fraction(gt, [x, y, side, side]),
        gt_fraction_in_visual_support=overlapping_fraction(gt, [x0, y0, max(0, x1-x0), max(0, y1-y0)]))


def summarize_output(model, out, cfg, anchor, resize, image_shape, gt, target, padding, capture):
    grid = cfg.TEST.SEARCH_SIZE // cfg.MODEL.BACKBONE.STRIDE
    window = hann2d(torch.tensor([grid, grid]).long(), centered=True).to(out['score_map'].device)
    response = window * out['score_map']
    predicted = model.box_head.cal_bbox(response, out['size_map'], out['offset_map']).view(-1, 4)
    box = (predicted.mean(0) * cfg.TEST.SEARCH_SIZE / resize).tolist()
    mapped = clip_box(map_box_back_from_anchor(box, anchor, cfg.TEST.SEARCH_SIZE, resize),
                      *image_shape[:2], margin=10)
    logits, indices = out['part_route_logits'], out['search_global_index']
    _, _, masks = compute_discriminative_route_loss(logits, indices, target, grid, grid,
        part_valid=out['part_valid'], padding_mask=padding,
        negative_guard=cfg.TRAIN.VDRM_ROUTE_NEGATIVE_GUARD, return_masks=True)
    # Count full-grid positives with the identical part/padding protocol. A
    # tiny target with no grid center is not mistaken for CE removing GT.
    _, _, full = compute_discriminative_route_loss(
        logits.new_zeros(1, 4, grid * grid), torch.arange(grid * grid, device=logits.device)[None],
        target, grid, grid, part_valid=out['part_valid'], padding_mask=padding,
        negative_guard=cfg.TRAIN.VDRM_ROUTE_NEGATIVE_GUARD, return_masks=True)
    final_kept = torch.ones(1, grid * grid, dtype=torch.bool, device=logits.device)
    for removed in out['removed_indexes_s']:
        if removed is not None:
            final_kept.scatter_(1, removed.long(), False)
    final_indices = final_kept[0].nonzero().flatten()[None]
    final_gt_cells = (full['positive'] & final_kept[:, None]).sum(-1)[0]
    peak = logits.argmax(-1)
    eligible = masks['eligible'][0]
    positive_peak = masks['positive'].gather(2, peak[..., None]).squeeze(-1)[0]
    pos, neg = masks['positive'], masks['negative']
    gap = logits.masked_fill(~pos, -torch.inf).amax(-1) - logits.masked_fill(~neg, -torch.inf).amax(-1)
    gate = out['part_route_gate']
    module = model.backbone.vdrm
    prior = F.softplus(module.part_route_log_match_scale) * out['part_similarity'] + module.part_route_match_bias
    correction = logits - prior
    valid_gt = bool((np.asarray(gt)[2:] > 0).all())
    metrics = dict(bbox_xywh=mapped, iou=bbox_iou_xywh(mapped, gt) if valid_gt else None,
        response_raw=response_statistics(out['score_map']), response_windowed=response_statistics(response),
        q_parts=out['part_reliability'][0].tolist(), valid_parts=out['part_valid'][0].tolist(),
        ce_retained_tokens=int(indices.numel()), gt_cells_full=full['positive'].sum(-1)[0].tolist(),
        gt_cells_retained=pos.sum(-1)[0].tolist(), final_ce_retained_tokens=int(final_indices.numel()),
        gt_cells_final_ce=final_gt_cells.tolist(), eligible_parts=eligible.tolist(),
        route_peak_in_gt_part=positive_peak.tolist(),
        route_peak_global_index=indices[:, None].expand(-1, 4, -1).gather(2, peak[..., None])[0, :, 0].tolist(),
        route_gt_background_gap=[float(gap[0, i]) if eligible[i] else None for i in range(4)],
        gate_fraction_below_001=float((gate < .01).float().mean()),
        gate_fraction_above_099=float((gate > .99).float().mean()),
        correction_mean=float(correction.mean()), correction_abs_max=float(correction.abs().max()),
        actual_injected_residual=dict(capture.last),
        scalars={k: float(out[k].mean()) for k in ('vdrm_alpha', 'vdrm_alpha_raw',
            'vdrm_raw_delta_relative_norm', 'vdrm_delta_relative_norm')})
    arrays = {k: out[k].detach().cpu().numpy() for k in ('score_map', 'size_map', 'offset_map',
        'part_route_logits', 'part_route_gate', 'part_reliability', 'part_valid', 'search_global_index')}
    arrays.update(positive_mask=pos.cpu().numpy(), negative_mask=neg.cpu().numpy(),
                  windowed_score=response.detach().cpu().numpy(), route_correction=correction.cpu().numpy(),
                  final_search_global_index=final_indices.cpu().numpy(),
                  **capture.maps)
    return metrics, arrays


def png_bytes(rgb):
    success, image = cv2.imencode('.png', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not success:
        raise RuntimeError('PNG encoding failed')
    return image.tobytes()


def collect_cases(cases, models, cfg):
    device = next(models['Tclean'].parameters()).device
    captures = {a: SpatialCapture() for a in CONFIGS}
    hooks = [models[a].backbone.vdrm.register_forward_hook(captures[a], with_kwargs=True) for a in CONFIGS]
    payloads, records = {}, []
    try:
        with torch.no_grad():
            for sequence, predictions, plan in cases:
                name, gt = sequence.name, sequence.ground_truth_rect
                initial = read_rgb(sequence.frames[0])
                payloads[f'images/{name}/000001.png'] = png_bytes(initial)
                patch, resize, mask = sample_target(initial, gt[0], cfg.TEST.TEMPLATE_FACTOR, cfg.TEST.TEMPLATE_SIZE)
                template, _ = preprocess_patch(patch, mask, cfg, device)
                template_box = template_bbox_in_crop(gt[0], resize, cfg.TEST.TEMPLATE_SIZE, device)
                template_mask = generate_mask_cond(cfg, 1, device, template_box)
                for frame in plan['selected_frames_1based']:
                    index = frame - 1
                    image = read_rgb(sequence.frames[index])
                    image_member = f'images/{name}/{frame:06d}.png'
                    payloads[image_member] = png_bytes(image)
                    for anchor_arm in CONFIGS:
                        anchor = predictions[anchor_arm][index - 1]
                        patch, resize, mask = sample_target(image, anchor, cfg.TEST.SEARCH_FACTOR, cfg.TEST.SEARCH_SIZE)
                        search, padding = preprocess_patch(patch, mask, cfg, device)
                        # Posthoc target labels; this tensor NEVER goes into model(**inputs).
                        target = transform_image_to_crop(torch.tensor(gt[index], dtype=torch.float32),
                            torch.tensor(anchor, dtype=torch.float32), resize,
                            torch.tensor([cfg.TEST.SEARCH_SIZE] * 2), normalize=True).to(device)[None]
                        inputs = dict(template=template, search=search,
                                      template_bbox=template_box, ce_template_mask=template_mask)
                        outputs, maps = {}, {}

                        def forward(arm, label):
                            out = models[arm](**inputs)
                            if any(not torch.isfinite(out[k]).all() for k in EXACT_KEYS):
                                raise RuntimeError('non-finite native output: ' + label)
                            if label != 'Tclean' and not torch.equal(reference['search_global_index'], out['search_global_index']):
                                raise RuntimeError('shared pre-residual CE indices differ: ' + label)
                            metrics, arrays = summarize_output(models[arm], out, cfg, anchor, resize,
                                image.shape, gt[index], target, padding, captures[arm])
                            outputs[label] = metrics
                            for key, value in arrays.items():
                                map_name = {'score_map': 'score', 'part_route_logits': 'route_logits'}.get(key, key)
                                maps[label + '_' + map_name] = value
                            return out

                        reference = forward('Tclean', 'Tclean')
                        forward('Rfreeze', 'Rfreeze')
                        forward('Rdisc', 'Rdisc')
                        with intervention(models['Rdisc'], 'route0'):
                            forward('Rdisc', 'Rdisc_route0')
                        with intervention(models['Rdisc'], 'alpha0'):
                            zero = forward('Rdisc', 'Rdisc_alpha0')
                            if any(not torch.equal(reference[k], zero[k]) for k in EXACT_KEYS):
                                raise RuntimeError('alpha=0 is not exactly Tclean on the shared real crop')
                        prefix = f'crops/{name}/{frame:06d}_{anchor_arm}'
                        crop_member, maps_member = prefix + '.png', prefix + '.npz'
                        payloads[crop_member] = png_bytes(patch)
                        stream = io.BytesIO()
                        np.savez_compressed(stream, **maps)
                        payloads[maps_member] = stream.getvalue()
                        records.append(dict(sequence=name, frame_1based=frame, anchor=anchor_arm,
                            anchor_frame_1based=frame - 1, anchor_xywh=anchor.tolist(),
                            target_in_crop_xywh=target[0].tolist(), crop_geometry=crop_geometry(anchor,
                                cfg.TEST.SEARCH_FACTOR, image.shape, gt[index]),
                            cover=float(plan['trajectory']['cover'][index]),
                            absence=float(plan['trajectory']['absence'][index]),
                            original_frame_path=str(sequence.frames[index]), image_member=image_member,
                            crop_member=crop_member, maps_member=maps_member, alpha0_exact=True,
                            replay_vs_saved_bbox_max_abs=float(np.abs(np.asarray(outputs[anchor_arm]['bbox_xywh'])
                                - predictions[anchor_arm][index]).max()), outputs=outputs))
                    print(f'{name}: frame {frame}/{len(gt)}, all three anchors checked', flush=True)
    finally:
        for hook in hooks:
            hook.remove()
    payloads['records.json'] = dump_json(records)
    payloads['windows.json'] = dump_json([p for _, _, p in cases])
    return payloads, records


def run(save_dir, results_root, output, device):
    if Path(output).exists():
        raise FileExistsError('refusing to overwrite diagnostic bundle: ' + str(output))
    paths = checkpoint_paths(save_dir)
    cases = prepare_cases(results_root, get_dataset('got10k_vdrm_dev'))
    print('Preflight passed: three checkpoints, five full trajectories, labels and selected images complete', flush=True)
    models, cfg, provenance = load_models(paths, torch.device(device))
    try:
        payloads, records = collect_cases(cases, models, cfg)
    finally:
        for arm, model in models.items():
            after = state_hash(model)
            if after != provenance[arm]['state_sha256_before'] or any(p.grad is not None for p in model.parameters()):
                raise RuntimeError('diagnostic changed model state or produced gradients: ' + arm)
            provenance[arm]['state_sha256_after'] = after
    manifest = dict(version=1, dataset='got10k_vdrm_dev', configurations=CONFIGS,
        selected_frames=sum(len(p['selected_frames_1based']) for _, _, p in cases),
        records=len(records), provenance=provenance, optimizer_steps=0,
        protocol='Posthoc selected cases; identical crop per saved integer previous-box anchor; '
                 'open-loop not closed-loop replay; current GT for labels only; native Hann decoder; '
                 'route0 zeros only the learned correction, keeps Rdisc scalars; no benchmark/identity claims.')
    payloads['README.txt'] = manifest['protocol'].encode('utf-8')
    write_bundle(output, manifest, payloads)
    verified = check_bundle(output)
    print(f'DIAGNOSTIC BUNDLE CHECK PASSED: {verified["selected_frames"]} frames, {len(records)} crop records', flush=True)
    print('Bundle:', Path(output).resolve(), flush=True)
    return verified


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--save_dir', default='./output')
    parser.add_argument('--results_root', help='Parent of config directories; default evaluation results_path/ostrack')
    parser.add_argument('--output', help='New .tar.gz path; existing bundles are never overwritten')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--check_bundle', help='Verify downloaded bundle only; no datasets/checkpoints/GPU required')
    args = parser.parse_args()
    if args.check_bundle:
        manifest = check_bundle(args.check_bundle)
        print(f'BUNDLE CHECK PASSED: {manifest["selected_frames"]} frames, {manifest["records"]} crop records')
        return
    results = args.results_root or str(Path(env_settings().results_path) / 'ostrack')
    output = args.output or str(Path(args.save_dir) / 'analysis/frozen_vdrm/failure_diagnostics.tar.gz')
    run(args.save_dir, results, output, args.device)


if __name__ == '__main__':
    main()
