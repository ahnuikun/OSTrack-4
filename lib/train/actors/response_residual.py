"""Supervise the final (Hann-windowed) response, never box regression."""

import torch
import torch.nn.functional as F

from lib.test.utils.hann import hann2d
from .ostrack import OSTrackActor


def response_losses(logits, score_map, boxes, config, removed_indexes=(),
                    padding_mask=None, occlusion_mask=None, return_masks=False):
    if logits.ndim != 4 or logits.shape[1] != 1 or logits.shape != score_map.shape:
        raise ValueError('response maps must have matching [B,1,H,W] shapes')
    batch, _, height, width = logits.shape
    boxes = boxes.to(logits)
    if boxes.shape != (batch, 4) or not torch.isfinite(boxes).all():
        raise ValueError('response GT must be finite [B,4]')
    y, x = torch.meshgrid((torch.arange(height, device=logits.device) + .5) / height,
                         (torch.arange(width, device=logits.device) + .5) / width, indexing='ij')
    xy = torch.stack((x, y), -1).reshape(1, -1, 2)
    valid = torch.ones(batch, height * width, dtype=torch.bool, device=logits.device)
    for indices in removed_indexes:
        if indices is not None:
            if (indices.shape[0] != batch or not torch.isfinite(indices).all()
                    or (indices != indices.long()).any() or (indices < 0).any()
                    or (indices >= height * width).any()):
                raise ValueError('invalid CE-removed indices')
            valid.scatter_(1, indices.long(), False)

    def pooled_mask(mask):
        if mask.ndim != 3 or mask.shape[0] != batch:
            raise ValueError('response pixel masks must be [B,H,W]')
        return F.adaptive_max_pool2d(mask.to(logits)[:, None], (height, width)).flatten(1).bool()

    if padding_mask is not None:
        valid &= ~pooled_mask(padding_mask)
    inside = ((xy >= boxes[:, None, :2]) & (xy < (boxes[:, :2] + boxes[:, 2:])[:, None])).all(-1)
    inside &= (boxes[:, 2:] > 0).all(1)[:, None]
    positives = inside & valid
    if occlusion_mask is not None:
        positives &= ~pooled_mask(occlusion_mask)
    guard = logits.new_tensor([config.NEGATIVE_GUARD_CELLS / width,
                               config.NEGATIVE_GUARD_CELLS / height])
    neighborhood = ((xy >= boxes[:, None, :2] - guard)
                    & (xy < (boxes[:, :2] + boxes[:, 2:])[:, None] + guard)).all(-1)
    negatives = valid & ~neighborhood
    # No search-GT nearest-cell fallback. If CE/crop/occlusion removes all
    # evidence, skip this sample instead of training a fabricated positive.
    eligible = positives.any(1) & negatives.any(1)
    flat_logits, flat_score = logits.flatten(1), score_map.flatten(1)
    focal_rows, rank_rows = [], []
    hann = hann2d(torch.tensor([height, width]), centered=True).to(logits).flatten()
    final_response = flat_score.log() + hann.log()[None]
    for row in eligible.nonzero(as_tuple=False).flatten().tolist():
        pos, neg = positives[row], negatives[row]
        # Balanced sigmoid focal classification of actual target-support cells.
        # Unlike a Gaussian at an occluded/removed center, these labels retain
        # only visual target support that survived CE and augmentation.
        pos_focal = -((1 - flat_logits[row, pos].sigmoid()).square()
                      * F.logsigmoid(flat_logits[row, pos])).mean()
        neg_focal = -(flat_logits[row, neg].sigmoid().square()
                      * F.logsigmoid(-flat_logits[row, neg])).mean()
        focal_rows.append(.5 * (pos_focal + neg_focal))
        positive_peak = final_response[row, pos].max()
        k = min(int(config.HARD_NEGATIVE_TOPK), int(neg.sum()))
        hard = final_response[row, neg].topk(k).values
        rank_rows.append(F.softplus(hard - positive_peak + config.RANK_MARGIN).mean())
    zero = logits.sum() * 0
    focal = torch.stack(focal_rows).mean() if focal_rows else zero
    rank = torch.stack(rank_rows).mean() if rank_rows else zero
    stats = dict(eligible_samples=int(eligible.sum()), skipped_samples=int((~eligible).sum()),
                 positive_cells=int((positives & eligible[:, None]).sum()),
                 negative_cells=int((negatives & eligible[:, None]).sum()))
    if return_masks:
        stats.update(positive=positives, negative=negatives, eligible=eligible, final_response=final_response)
    return focal, rank, stats


class ResponseResidualActor(OSTrackActor):
    def compute_losses(self, pred_dict, gt_dict, return_status=True, return_components=False):
        config = self.cfg.TRAIN.RESPONSE_RESIDUAL
        focal, rank, stats = response_losses(pred_dict['score_logits'], pred_dict['score_map'],
            gt_dict['search_anno'][-1], config, pred_dict.get('removed_indexes_s', ()),
            padding_mask=gt_dict.get('search_att', [None])[-1],
            occlusion_mask=pred_dict.get('disc_route_occlusion_mask'))
        total = config.FOCAL_WEIGHT * focal + config.RANK_WEIGHT * rank
        if not torch.isfinite(total):
            raise ValueError('non-finite response loss')
        status = {'Loss/total': total.item(), 'Loss/response_focal': focal.item(),
                  'Loss/response_hann_rank': rank.item(),
                  **{'Response/' + key: value for key, value in stats.items()}}
        if return_components:
            return total, status, {'response_focal': focal, 'response_hann_rank': rank}
        return (total, status) if return_status else total
