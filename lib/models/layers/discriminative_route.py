"""Template-conditioned spatial routing; not a temporal reliability head."""

import math

import torch
from torch import nn
import torch.nn.functional as F


def retained_coordinates(index, batch, length, height, width, dtype, device):
    """Resolve retained CE tokens in the original search grid, not a new grid."""
    if height < 1 or width < 1 or index is None or index.shape != (batch, length):
        raise ValueError('route requires [B,L] CE indices and a positive search grid')
    index = index.to(device=device)
    if not torch.isfinite(index).all() or not torch.equal(index, index.long().to(index.dtype)):
        raise ValueError('CE indices must be finite integers')
    index = index.long()
    if length == 0 or index.min() < 0 or index.max() >= height * width:
        raise ValueError('CE indices outside the original search grid')
    if length > 1 and (index.sort(dim=1).values.diff(dim=1) == 0).any():
        raise ValueError('duplicate retained CE indices')
    x = (index.remainder(width).to(dtype) + .5) / width
    y = (index.div(width, rounding_mode='floor').to(dtype) + .5) / height
    return index, x, y


class DiscriminativePartRoute(nn.Module):
    """Small learned correction to V8 logits, zero-initialized at the output.

    All input features are detached. Tracking loss can still optimize this
    head via the residual; auxiliary route loss cannot optimize visual inputs
    or V8's calibration scalars. No GT position is an inference input.
    """

    def __init__(self, embed_dim, projection_dim=32, hidden_dim=64):
        super().__init__()
        if min(embed_dim, projection_dim, hidden_dim) < 1:
            raise ValueError('route feature dimensions must be positive')
        self.projection = nn.Linear(embed_dim, projection_dim, bias=False)
        self.classifier = nn.Sequential(
            nn.Linear(2 * projection_dim + 5, hidden_dim),
            nn.GELU(), nn.Linear(hidden_dim, 1),
        )
        self.zero_output()

    def zero_output(self):
        nn.init.zeros_(self.classifier[-1].weight)
        nn.init.zeros_(self.classifier[-1].bias)

    def forward(self, prototypes, search, global_index, grid_size):
        if grid_size is None or not isinstance(grid_size, int) or grid_size < 1:
            raise ValueError('discriminative routing requires the original integer search grid size')
        b, k, c = prototypes.shape
        if search.ndim != 3 or search.shape[0] != b or search.shape[2] != c:
            raise ValueError('prototype/search feature shape mismatch')
        part_grid = math.isqrt(k)
        if part_grid ** 2 != k:
            raise ValueError('parts must form a square grid')
        _, sx, sy = retained_coordinates(
            global_index, b, search.shape[1], grid_size, grid_size,
            search.dtype, search.device,
        )
        p = F.normalize(prototypes.detach(), dim=-1)
        s = F.normalize(search.detach(), dim=-1)
        cosine = torch.einsum('bkc,blc->bkl', p, s)
        p = F.normalize(self.projection(p), dim=-1)[:, :, None, :]
        s = F.normalize(self.projection(s), dim=-1)[:, None, :, :]
        ids = torch.arange(k, device=search.device)
        px = (ids.remainder(part_grid).to(search.dtype) + .5) / part_grid
        py = (ids.div(part_grid, rounding_mode='floor').to(search.dtype) + .5) / part_grid
        shape = (b, k, search.shape[1])
        features = torch.cat((
            p - s, p * s, cosine.unsqueeze(-1),
            sx[:, None, :, None].expand(*shape, 1),
            sy[:, None, :, None].expand(*shape, 1),
            px[None, :, None, None].expand(*shape, 1),
            py[None, :, None, None].expand(*shape, 1),
        ), dim=-1)
        return self.classifier(features).squeeze(-1)
