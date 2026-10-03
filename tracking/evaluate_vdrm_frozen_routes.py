"""Paired offline GT-region vs background routing check on fixed GOT dev.

Both models receive exactly the same saved Rfreeze-anchored search crop.
Ground truth only labels scores after inference; it never enters the route
head or chooses its output. This is a spatial discrimination proxy, not an
identity-switch count or q_vis/q_id calibration experiment.
"""

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from lib.config.ostrack.config import default_config, update_config_from_file
from lib.models.ostrack import build_ostrack
from lib.test.evaluation import get_dataset, trackerlist
from lib.test.evaluation.result_paths import resolve_result_bbox_path
from lib.train.actors.discriminative_route_loss import compute_discriminative_route_loss
from lib.train.data.processing_utils import sample_target, transform_image_to_crop
from lib.train.frozen_vdrm import validate_frozen_checkpoint, validate_frozen_config
from lib.utils.ce_utils import generate_mask_cond


def ranking_statistics(logits, masks):
    pos = logits.masked_fill(~masks['positive'], -torch.inf).amax(-1)
    neg = logits.masked_fill(~masks['negative'], -torch.inf).amax(-1)
    eligible = masks['eligible']
    count = int(eligible.sum())
    return dict(eligible_parts=count,
        rank_correct=int(((pos > neg) & eligible).sum()),
        logit_gap_sum=float((pos[eligible] - neg[eligible]).sum()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tracker_params', nargs=2, required=True, metavar=('RFREEZE', 'RDISC'))
    parser.add_argument('--save_dir', default='./output')
    parser.add_argument('--frames_per_sequence', type=int, default=4)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    if args.frames_per_sequence < 1:
        parser.error('--frames_per_sequence must be positive')
    device = torch.device(args.device)
    models, configs, contracts = [], [], []
    for expected_arm, name in zip(('rfreeze', 'rdisc'), args.tracker_params):
        cfg = default_config()
        update_config_from_file(PROJECT / 'experiments/ostrack' / (name + '.yaml'), cfg)
        validate_frozen_config(cfg)
        if cfg.TRAIN.VDRM_EXPERIMENT_ARM != expected_arm:
            raise ValueError('provide Rfreeze then Rdisc configurations')
        path = Path(args.save_dir) / 'checkpoints/train/ostrack' / name / 'OSTrack_ep0300.pth.tar'
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        model = build_ostrack(cfg, training=False)
        contract = validate_frozen_checkpoint(model, checkpoint, cfg=cfg, expected_epoch=cfg.TEST.EPOCH)
        model.load_state_dict(checkpoint['net'], strict=True)
        models.append(model.requires_grad_(False).to(device).eval())
        configs.append(cfg)
        contracts.append(contract)
    if any(contracts[0][k] != contracts[1][k] for k in ('source_sha256', 'visual_sha256')):
        raise ValueError('offline pair does not share the exact same Tclean source')
    dataset = get_dataset('got10k_vdrm_dev')
    if len(dataset) != 152:
        raise ValueError('offline routing check requires the full fixed 152-sequence dev')
    cfg = configs[0]
    mean = torch.tensor(cfg.DATA.MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(cfg.DATA.STD, device=device).view(1, 3, 1, 1)
    anchor_tracker = trackerlist('ostrack', args.tracker_params[0], dataset_name='got10k_vdrm_dev')[0]
    prepared = []
    for sequence in dataset:
        predictions = np.loadtxt(resolve_result_bbox_path(anchor_tracker.results_dir, sequence), ndmin=2)
        gt = sequence.ground_truth_rect
        if predictions.shape != (len(sequence.frames), 4) or gt.shape != predictions.shape:
            raise ValueError('incomplete anchor predictions/GT: ' + sequence.name)
        if not np.isfinite(predictions).all() or (predictions[:, 2:] <= 0).any():
            raise ValueError('invalid anchor predictions: ' + sequence.name)
        folder = Path(sequence.frames[0]).parent
        cover = np.loadtxt(folder / 'cover.label', ndmin=1)
        absence = np.loadtxt(folder / 'absence.label', ndmin=1)
        if cover.shape != (len(gt),) or absence.shape != cover.shape:
            raise ValueError('visibility annotation length mismatch: ' + sequence.name)
        visible = np.flatnonzero((cover >= 7) & (absence == 0) & (np.arange(len(gt)) > 0))
        if len(visible):
            indices = np.unique(np.linspace(0, len(visible) - 1,
                min(args.frames_per_sequence, len(visible)), dtype=int))
            frames = visible[indices].tolist()
        else:
            frames = []
        prepared.append((sequence, predictions, frames))

    def crop(frame_path, anchor_box, target_box, size, factor):
        image = cv2.imread(str(frame_path))
        if image is None:
            raise FileNotFoundError(frame_path)
        patch, resize, padding = sample_target(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), anchor_box, factor, size)
        image_tensor = torch.from_numpy(patch).to(device).float().permute(2, 0, 1)[None] / 255.
        target = transform_image_to_crop(torch.tensor(target_box, dtype=torch.float32),
            torch.tensor(anchor_box, dtype=torch.float32), resize,
            torch.tensor([size, size], dtype=torch.float32), normalize=True).to(device)[None]
        return (image_tensor - mean) / std, target, torch.from_numpy(padding).to(device)[None]

    result = dict(dataset='got10k_vdrm_dev', total_sequences=152,
                  frames_per_sequence=args.frames_per_sequence, anchor_param=args.tracker_params[0],
                  protocol='same integer-result-anchored crop; cover>=7; GT only for offline labeling; not q_id evaluation',
                  contracts=contracts, sequences=[])
    for sequence, predictions, frames in prepared:
        totals = [dict(eligible_parts=0, rank_correct=0, logit_gap_sum=0.) for _ in models]
        if frames:
            template, template_box, _ = crop(sequence.frames[0], sequence.ground_truth_rect[0],
                sequence.ground_truth_rect[0], cfg.DATA.TEMPLATE.SIZE, cfg.DATA.TEMPLATE.FACTOR)
            template_mask = generate_mask_cond(cfg, 1, device, template_box)
            for frame in frames:
                search, target, padding = crop(sequence.frames[frame], predictions[frame - 1],
                    sequence.ground_truth_rect[frame], cfg.DATA.SEARCH.SIZE, cfg.DATA.SEARCH.FACTOR)
                shared_masks = None
                for i, model in enumerate(models):
                    with torch.no_grad():
                        out = model(template, search, template_bbox=template_box, ce_template_mask=template_mask)
                        logits = out['part_route_logits']
                        _, _, masks = compute_discriminative_route_loss(logits, out['search_global_index'], target,
                            cfg.DATA.SEARCH.SIZE // cfg.MODEL.BACKBONE.STRIDE,
                            cfg.DATA.SEARCH.SIZE // cfg.MODEL.BACKBONE.STRIDE,
                            part_valid=out['part_valid'], padding_mask=padding,
                            negative_guard=cfg.TRAIN.VDRM_ROUTE_NEGATIVE_GUARD, return_masks=True)
                        if shared_masks is not None and any(not torch.equal(masks[k], shared_masks[k])
                                for k in ('positive', 'negative', 'eligible')):
                            raise RuntimeError('paired pre-residual CE/labels differ on identical input')
                        shared_masks = masks
                        stats = ranking_statistics(logits, masks)
                    for key in totals[i]:
                        totals[i][key] += stats[key]
        row = dict(sequence=sequence.name, sampled_frames_1based=[f + 1 for f in frames])
        for name, stats in zip(args.tracker_params, totals):
            n = stats['eligible_parts']
            row[name] = dict(**stats, rank_accuracy=stats['rank_correct'] / n if n else None,
                            mean_logit_gap=stats['logit_gap_sum'] / n if n else None)
        result['sequences'].append(row)
        print(f'{sequence.name}: sampled {len(frames)}, eligible parts {totals[0]["eligible_parts"]}')
    valid = [r for r in result['sequences'] if r[args.tracker_params[0]]['eligible_parts'] > 0]
    if not valid:
        raise RuntimeError('offline check has no eligible target/background pair')
    result['eligible_sequences'] = len(valid)
    result['summary'] = {}
    for name in args.tracker_params:
        result['summary'][name] = dict(
            sequence_mean_rank_accuracy=float(np.mean([r[name]['rank_accuracy'] for r in valid])),
            sequence_mean_logit_gap=float(np.mean([r[name]['mean_logit_gap'] for r in valid])))
    delta = np.array([r[args.tracker_params[1]]['rank_accuracy'] - r[args.tracker_params[0]]['rank_accuracy'] for r in valid])
    rng = np.random.default_rng(42)
    samples = rng.integers(0, len(delta), size=(5000, len(delta)))
    result['paired_rank_delta_pp'] = float(100 * delta.mean())
    result['paired_rank_delta_ci95_pp'] = (100 * np.quantile(delta[samples].mean(1), [.025, .975])).tolist()
    path = Path(args.save_dir) / 'analysis/frozen_vdrm/route_offline.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print('Offline routing summary:', json.dumps(result['summary']))
    print('Eligible sequences:', len(valid), '/152; report:', path.resolve())


if __name__ == '__main__':
    main()
