"""Rank actual frozen CENTER boxes; search GT labels never enter the model."""

import torch
import torch.nn.functional as F

from .ostrack import OSTrackActor
from lib.test.utils.hann import hann2d


def candidate_iou(boxes, targets):
    boxes, targets = boxes.detach(), targets.detach()
    lower = boxes[..., :2] - boxes[..., 2:] / 2
    upper = boxes[..., :2] + boxes[..., 2:] / 2
    intersection = (torch.minimum(upper, targets[:, None, :2] + targets[:, None, 2:]) -
                    torch.maximum(lower, targets[:, None, :2])).clamp_min(0).prod(-1)
    union = boxes[..., 2:].clamp_min(0).prod(-1) + targets[:, None, 2:].clamp_min(0).prod(-1) - intersection
    return (intersection / union.clamp_min(1e-12)).clamp(0, 1)


def target_has_evidence(targets, padding=None, occlusion=None, size=256):
    """Pixel support, not a nearest search-token fallback for tiny targets."""
    points = (torch.arange(size, device=targets.device, dtype=targets.dtype) + .5) / size
    yy, xx = torch.meshgrid(points, points, indexing='ij')
    inside = ((xx[None] >= targets[:, 0, None, None]) & (xx[None] < (targets[:, 0]+targets[:, 2])[:, None, None]) &
              (yy[None] >= targets[:, 1, None, None]) & (yy[None] < (targets[:, 1]+targets[:, 3])[:, None, None]))
    for mask in (padding, occlusion):
        if mask is not None:
            if mask.shape != inside.shape:
                raise ValueError('candidate evidence masks must match search pixels')
            inside &= ~mask.bool()
    return inside.flatten(1).any(1) & (targets[:, 2:] > 0).all(1)


def quality_losses(output, targets, config, padding=None, occlusion=None):
    logits = output['candidate_quality_logits']
    targets = targets.detach().to(logits)
    if targets.shape != (logits.shape[0], 4) or not torch.isfinite(targets).all():
        raise ValueError('invalid candidate-quality GT boxes')
    quality = candidate_iou(output['candidate_boxes'], targets)
    evidence = target_has_evidence(targets, padding, occlusion)
    valid = output['candidate_valid_mask'] & evidence[:, None]
    base = output['response_base_score_map'].detach()
    hann = hann2d(torch.tensor(base.shape[-2:]), centered=True).to(base).flatten()
    # Native score normalization is a common factor and cancels in pairwise differences.
    energy = base.flatten(1).log() + hann.log()[None] + output['response_delta_energy'].flatten(1)
    base_energy = base.flatten(1).log() + hann.log()[None]
    groups = [], [], [], []
    qualities, ranks, protections, abstentions = groups
    rank_count = protect_count = no_good_count = 0
    for row in valid.any(1).nonzero(as_tuple=False).flatten().tolist():
        allowed, q = valid[row], quality[row]
        good = allowed & (q >= config.POSITIVE_IOU)
        bad = allowed & (q <= config.NEGATIVE_IOU)
        middle = allowed & ~good & ~bad
        # Balance quality ranges per image; 200 background boxes cannot drown
        # the target's few high-IoU boxes. Soft IoU is the calibration target.
        losses = [F.binary_cross_entropy_with_logits(logits[row, mask], q[mask])
                  for mask in (good, middle, bad) if mask.any()]
        qualities.append(torch.stack(losses).mean())
        if good.any() and bad.any():
            positive = q.masked_fill(~good, -1).argmax()
            hard = energy[row, bad].topk(min(config.HARD_NEGATIVE_TOPK, int(bad.sum()))).values
            ranks.append(F.softplus(hard - energy[row, positive] + config.RANK_MARGIN).mean())
            rank_count += 1
        original = base_energy[row].argmax()
        if allowed[original] and q[original] >= config.POSITIVE_IOU:
            worse = allowed & (q < q[original] - config.PROTECT_IOU_GAP)
            if worse.any():
                hard = energy[row, worse].topk(min(config.HARD_NEGATIVE_TOPK, int(worse.sum()))).values
                protections.append(F.softplus(hard - energy[row, original] + config.RANK_MARGIN).mean())
                protect_count += 1
        if not good.any():
            abstentions.append(output['response_delta_energy'].flatten(1)[row, allowed].square().mean())
            no_good_count += 1
    zero = logits.sum() * 0
    components = {name: torch.stack(values).mean() if values else zero for name, values in zip(
        ('candidate_quality', 'candidate_rank', 'candidate_preserve', 'candidate_abstain'), groups)}
    stats = dict(eligible_samples=int(valid.any(1).sum()), ranking_samples=rank_count,
        preservation_samples=protect_count, no_good_candidate_samples=no_good_count,
        no_visual_evidence_samples=int((~evidence).sum()),
        high_quality_candidates=int((valid & (quality >= config.POSITIVE_IOU)).sum()))
    return components, stats


class CandidateQualityActor(OSTrackActor):
    def compute_losses(self, pred_dict, gt_dict, return_status=True, return_components=False):
        config = self.cfg.TRAIN.RESPONSE_RESIDUAL
        components, stats = quality_losses(pred_dict, gt_dict['search_anno'][-1], config,
            gt_dict.get('search_att', [None])[-1], pred_dict.get('disc_route_occlusion_mask'))
        total = sum(components[key] * config[weight] for key, weight in (
            ('candidate_quality', 'QUALITY_WEIGHT'), ('candidate_rank', 'RANK_WEIGHT'),
            ('candidate_preserve', 'PRESERVE_WEIGHT'), ('candidate_abstain', 'ABSTAIN_WEIGHT')))
        if not torch.isfinite(total):
            raise ValueError('non-finite candidate quality loss')
        status = {'Loss/total': total.item(), **{'Loss/'+k:v.item() for k,v in components.items()},
                  **{'Response/'+k:v for k,v in stats.items()}}
        if return_components:
            return total, status, components
        return (total, status) if return_status else total
