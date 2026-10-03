"""Strict part routing labels: no fabricated positive or CE fallback labels."""

import math

import torch
import torch.nn.functional as F

from lib.models.layers.discriminative_route import retained_coordinates


def compute_discriminative_route_loss(logits, global_index, target_box, height, width,
                                      part_valid=None, part_visibility=None,
                                      occlusion_mask=None, padding_mask=None,
                                      distractor_boxes=None, distractor_applied=None,
                                      hard_topk=4, negative_guard=.5, return_masks=False):
    """Balance positive parts against hard negatives and ordinary background.

    Positive cell centers must actually lie inside the corresponding GT part.
    Other target parts, GT boundary neighborhoods, padded cells and positive
    cells overlapping synthetic occlusion are ignored. Missing CE positives
    disable a part; tiny parts are never assigned an arbitrary positive.
    """
    if logits.ndim != 3 or hard_topk < 1 or negative_guard < 0:
        raise ValueError('invalid discriminative routing loss arguments')
    b, k, length = logits.shape
    part_grid = math.isqrt(k)
    if k != part_grid ** 2:
        raise ValueError('parts must form a square grid')
    index, x, y = retained_coordinates(global_index, b, length, height, width,
                                        logits.dtype, logits.device)
    box = target_box.to(logits).reshape(-1, 4)
    if box.shape != (b, 4) or not torch.isfinite(box).all():
        raise ValueError('target boxes must be finite [B,4]')
    x0, y0, w, h = box.unbind(-1)
    valid_box = (w > 0) & (h > 0) & (x0 < 1) & (y0 < 1) & (x0 + w > 0) & (y0 + h > 0)
    valid_cell = torch.ones_like(x, dtype=torch.bool)

    def masked_cells(mask):
        mask = mask.to(device=logits.device, dtype=logits.dtype)
        if mask.ndim != 3 or mask.shape[0] != b:
            raise ValueError('pixel masks must have shape [B,H,W]')
        pooled = F.adaptive_max_pool2d(mask[:, None], (height, width)).flatten(1)
        return pooled.gather(1, index).bool()

    if padding_mask is not None:
        valid_cell &= ~masked_cells(padding_mask)
    positive_cell = valid_cell.clone()
    if occlusion_mask is not None:
        positive_cell &= ~masked_cells(occlusion_mask)
    ids = torch.arange(k, device=logits.device)
    col = ids.remainder(part_grid).to(logits.dtype)
    row = ids.div(part_grid, rounding_mode='floor').to(logits.dtype)
    px0 = x0[:, None] + w[:, None] * col / part_grid
    px1 = x0[:, None] + w[:, None] * (col + 1) / part_grid
    py0 = y0[:, None] + h[:, None] * row / part_grid
    py1 = y0[:, None] + h[:, None] * (row + 1) / part_grid
    positive = ((x[:, None] >= px0[:, :, None]) & (x[:, None] < px1[:, :, None])
                & (y[:, None] >= py0[:, :, None]) & (y[:, None] < py1[:, :, None])
                & positive_cell[:, None] & valid_box[:, None, None])
    near_target = ((x >= (x0 - negative_guard / width)[:, None])
                   & (x < (x0 + w + negative_guard / width)[:, None])
                   & (y >= (y0 - negative_guard / height)[:, None])
                   & (y < (y0 + h + negative_guard / height)[:, None]))
    negative = (~near_target & valid_cell & valid_box[:, None])[:, None].expand(-1, k, -1)
    valid_part = torch.ones(b, k, dtype=torch.bool, device=logits.device)
    for value, name in ((part_valid, 'part_valid'), (part_visibility, 'part_visibility')):
        if value is not None:
            if value.shape != (b, k):
                raise ValueError(name + ' must have shape [B,K]')
            valid_part &= value.to(logits.device) > 0
    eligible = valid_part & positive.any(-1) & negative.any(-1)

    # Select only true retained in-box distractor cells. A removed paste must
    # never be replaced by an arbitrary neighboring cell with a false label.
    pasted = torch.zeros_like(negative)
    if distractor_boxes is not None and distractor_applied is not None:
        db = distractor_boxes.to(logits).reshape(-1, 4)
        applied = distractor_applied.to(logits.device).reshape(-1).bool()
        if db.shape != (b, 4) or applied.shape != (b,) or not torch.isfinite(db).all():
            raise ValueError('distractor metadata batch/values mismatch')
        inside = ((x >= db[:, 0, None]) & (x < (db[:, 0] + db[:, 2])[:, None])
                  & (y >= db[:, 1, None]) & (y < (db[:, 1] + db[:, 3])[:, None])
                  & applied[:, None] & (db[:, 2:] > 0).all(-1)[:, None])
        pasted = negative & inside[:, None]
    count = min(hard_topk, length)
    top = logits.detach().masked_fill(~negative, -torch.inf).topk(count, dim=-1).indices
    mined = torch.zeros_like(negative).scatter_(2, top, True) & negative
    hard = torch.where(pasted.any(-1, keepdim=True), pasted, mined)
    ordinary = negative & ~hard

    def masked_mean(values, mask):
        return (values * mask).sum(-1) / mask.sum(-1).clamp_min(1)

    pos_loss = masked_mean(-F.logsigmoid(logits), positive)
    neg_element = -F.logsigmoid(-logits)
    hard_loss = masked_mean(neg_element, hard)
    ordinary_loss = masked_mean(neg_element, ordinary)
    grouped = .5 * (hard_loss + ordinary_loss)
    neg_loss = torch.where(hard.any(-1) & ordinary.any(-1), grouped,
                           masked_mean(neg_element, negative))
    per_part = .5 * (pos_loss + neg_loss)
    loss = (per_part * eligible).sum() / eligible.sum().clamp_min(1)
    probability = logits.sigmoid().detach()

    def average(mask):
        mask = mask & eligible[:, :, None]
        return (probability * mask).sum() / mask.sum().clamp_min(1)

    pos_p, hard_p = average(positive), average(hard)
    diagnostics = {
        'part_route_positive_probability': pos_p,
        'part_route_background_probability': average(negative),
        'disc_route_hard_negative_probability': hard_p,
        'disc_route_positive_hard_gap': pos_p - hard_p,
        'disc_route_eligible_parts': eligible.sum().to(logits.dtype),
        'disc_route_pasted_parts': (pasted.any(-1) & eligible).sum().to(logits.dtype),
    }
    if return_masks:
        return loss, diagnostics, dict(positive=positive, negative=negative,
                                      hard_negative=hard, eligible=eligible)
    return loss, diagnostics
