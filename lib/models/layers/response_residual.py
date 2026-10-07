"""Template-conditioned, zero-initialized correction of final CENTER logits."""

import math
import torch
from torch import nn
import torch.nn.functional as F


class ResponseResidual(nn.Module):
    def __init__(self, embed_dim, projection_dim=32, hidden_dim=64, max_delta_logit=2.0):
        super().__init__()
        self.max_delta_logit = float(max_delta_logit)
        if min(embed_dim, projection_dim, hidden_dim) <= 0 or not math.isfinite(self.max_delta_logit) or self.max_delta_logit <= 0:
            raise ValueError('invalid response residual dimensions/bound')
        self.project = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, projection_dim))
        self.correct = nn.Sequential(nn.Conv2d(2 * projection_dim + 2, hidden_dim, 3, padding=1),
            nn.GELU(), nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden_dim, 1, 1))
        nn.init.zeros_(self.correct[-1].weight)
        nn.init.zeros_(self.correct[-1].bias)

    def forward(self, features, base_logits, template_bbox):
        features, base_logits = features.detach(), base_logits.detach()
        batch, _, height, width = base_logits.shape
        if height != width or template_bbox is None:
            raise ValueError('response head requires a square search grid and initialization template box')
        search_length = height * width
        template = features[:, :-search_length]
        side = math.isqrt(template.shape[1])
        if side * side != template.shape[1] or side == 0:
            raise ValueError('invalid template token grid')
        box = template_bbox.detach().to(features)
        if box.shape != (batch, 4) or not torch.isfinite(box).all() or (box[:, 2:] <= 0).any():
            raise ValueError('invalid template box')
        y, x = torch.meshgrid((torch.arange(side, device=features.device) + .5) / side,
                             (torch.arange(side, device=features.device) + .5) / side, indexing='ij')
        xy = torch.stack((x, y), -1).reshape(1, -1, 2)
        inside = ((xy >= box[:, None, :2]) & (xy < (box[:, :2] + box[:, 2:])[:, None])).all(-1)
        # A tiny INITIALIZATION target can miss token centers. This fallback
        # uses only its known template box, never current/search-frame GT.
        missing = ~inside.any(1)
        nearest = (xy - (box[:, :2] + .5 * box[:, 2:])[:, None]).square().sum(-1).argmin(1)
        inside = inside.clone()
        inside[missing, nearest[missing]] = True
        projected_template = self.project(template)
        prototype = (projected_template * inside[..., None]).sum(1) / inside.sum(1, keepdim=True)
        prototype = F.normalize(prototype, dim=-1)[:, None]
        search = F.normalize(self.project(features[:, -search_length:]), dim=-1)
        cues = torch.cat((search * prototype, search - prototype,
                          (search * prototype).sum(-1, keepdim=True)), -1)
        cues = cues.transpose(1, 2).reshape(batch, -1, height, width)
        raw = self.correct(torch.cat((cues, base_logits), 1))
        return self.max_delta_logit * torch.tanh(raw / self.max_delta_logit)
