"""Frozen-box, template-conditioned candidate quality for the Rqual screen."""

import math

import torch
from torch import nn
import torch.nn.functional as F


def dense_center_boxes(size_map, offset_map):
    """All CENTER candidates as normalized cxcywh; same formula as cal_bbox."""
    batch, _, height, width = size_map.shape
    y, x = torch.meshgrid(torch.arange(height, device=size_map.device),
                          torch.arange(width, device=size_map.device), indexing='ij')
    centers = torch.stack((x / width, y / height), 0).to(size_map)[None]
    centers = centers + offset_map / offset_map.new_tensor([width, height])[None, :, None, None]
    return torch.cat((centers, size_map), 1).flatten(2).transpose(1, 2).detach()


def quality_delta(logits, valid, bound):
    """Remove the common component BEFORE bounding; never alter invalid cells."""
    weight = valid.to(logits)
    mean = (logits * weight).sum(1, keepdim=True) / weight.sum(1, keepdim=True).clamp_min(1)
    return bound * torch.tanh((logits - mean) / bound) * weight


def apply_energy_correction(base_score, delta):
    """Native Hann is applied later. Common scale keeps scores <=1, no clipping."""
    delta = delta.view_as(base_score)
    shift = delta.flatten(1).amax(1)[:, None, None, None]
    # Zero initialization/bypass must be bit-exact Tclean, including tiny scores.
    return base_score.detach() * torch.exp(delta - shift)


class CandidateQuality(nn.Module):
    def __init__(self, embed_dim, projection_dim=32, hidden_dim=64, roi_grid_size=3, max_delta_logit=2.):
        super().__init__()
        if min(embed_dim, projection_dim, hidden_dim, roi_grid_size) <= 0 or not math.isfinite(max_delta_logit) or max_delta_logit <= 0:
            raise ValueError('invalid candidate-quality dimensions/bound')
        self.roi_grid_size, self.bound = roi_grid_size, float(max_delta_logit)
        self.project = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, projection_dim))
        # Product, absolute difference, cosine, base logit, box w/h/offset, support.
        self.correct = nn.Sequential(nn.Linear(2 * projection_dim + 7, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        nn.init.zeros_(self.correct[-1].weight)
        nn.init.zeros_(self.correct[-1].bias)

    @torch.no_grad()
    def pool(self, features, boxes, support):
        """Pool frozen features, so CUDA grid_sample backward is never invoked."""
        batch, count, _ = boxes.shape
        axis = (torch.arange(self.roi_grid_size, device=features.device, dtype=features.dtype) + .5) / self.roi_grid_size
        y, x = torch.meshgrid(axis, axis, indexing='ij')
        points = torch.stack((x, y), -1).reshape(1, 1, -1, 2)
        grid = boxes[:, :, None, :2] + (points - .5) * boxes[:, :, None, 2:]
        grid = grid * 2 - 1
        samples = F.grid_sample(features * support, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
        weights = F.grid_sample(support, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
        pooled = samples.sum(-1) / weights.sum(-1).clamp_min(1e-6)
        return pooled.transpose(1, 2), weights.mean(-1).reshape(batch, count)

    def forward(self, features, base_logits, size_map, offset_map, template_bbox, retained):
        features, base_logits = features.detach(), base_logits.detach()
        batch, _, height, width = base_logits.shape
        count = height * width
        template = features[:, :-count]
        side = math.isqrt(template.shape[1])
        if height != width or side * side != template.shape[1] or template_bbox is None:
            raise ValueError('candidate head requires square grids and an initialization template box')
        template_bbox = template_bbox.detach().to(features)
        if template_bbox.shape != (batch, 4) or not torch.isfinite(template_bbox).all() or (template_bbox[:, 2:] <= 0).any():
            raise ValueError('invalid candidate template initialization box')
        boxes = dense_center_boxes(size_map.detach(), offset_map.detach())
        search = features[:, -count:].transpose(1, 2).reshape(batch, -1, height, width)
        template = template.transpose(1, 2).reshape(batch, -1, side, side)
        template_box = torch.cat((template_bbox[:, :2] + template_bbox[:, 2:] / 2, template_bbox[:, 2:]), 1)[:, None]
        template_roi, template_support = self.pool(template, template_box, template.new_ones(batch, 1, side, side))
        search_roi, support = self.pool(search, boxes, retained.to(search).reshape(batch, 1, height, width))
        # A CE-removed center may still have a usable decoded box supported by
        # neighboring retained tokens. Validity is based on its ROI, not center.
        valid = (support > 1e-6) & (template_support > 1e-6)
        candidate = F.normalize(self.project(search_roi), dim=-1)
        prototype = F.normalize(self.project(template_roi), dim=-1)
        geometry = torch.cat((boxes[:, :, 2:], offset_map.detach().flatten(2).transpose(1, 2)), -1)
        cues = torch.cat((candidate * prototype, (candidate - prototype).abs(),
            (candidate * prototype).sum(-1, keepdim=True), base_logits.flatten(1)[:, :, None],
            geometry, support[:, :, None]), -1)
        logits = self.correct(cues).squeeze(-1)
        delta = quality_delta(logits, valid, self.bound)
        return dict(candidate_quality_logits=logits, candidate_boxes=boxes, candidate_valid_mask=valid,
                    candidate_roi_support=support, response_delta_energy=delta.view_as(base_logits))
