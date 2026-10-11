"""Diagnostic-only replay of the actual frozen layers after VDRM insertion."""

from contextlib import contextmanager
from copy import deepcopy
import hashlib

import torch
from torch import nn

from lib.models.ostrack.ostrack import OSTrack


@contextmanager
def numeric_backend(precision):
    """Keep native settings, or disable TF32; always restore global settings."""
    if precision not in ('native32', 'strict32', 'float64'):
        raise ValueError('unknown numeric precision')
    old = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    try:
        if precision != 'native32':
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = old


class CachedSuffix(nn.Module):
    """Use real CEBlock, norm and CENTER head modules, with a fixed prefix.

    This path must match saved native outputs and gradients before it is used
    for higher-precision experiments. It is never used by the actual tracker.
    """

    def __init__(self, model):
        super().__init__()
        b = model.backbone
        if (b.cat_mode != 'direct' or b.add_cls_token or model.head_type != 'CENTER'
                or model.aux_loss or model.frozen_response or not model.vdrm_enabled
                or b.vdrm is None):
            raise ValueError('numeric suffix only supports original Rdisc CENTER/direct')
        start = b.vdrm_insert_index + 1
        self.ce_after = tuple(i - start for i in b.ce_loc if i >= start)
        if sum(i < start for i in b.ce_loc) != 1:
            raise ValueError('expected exactly one CE stage before insertion')
        self.blocks = deepcopy(nn.ModuleList(list(b.blocks)[start:]))
        self.norm = deepcopy(b.norm)
        self.box_head = deepcopy(model.box_head)
        self.feat_sz_s = model.feat_sz_s
        self.feat_len_s = model.feat_len_s
        self.head_type = model.head_type
        self.vdrm_enabled = model.vdrm_enabled
        self.eval().requires_grad_(False)

    def forward(self, template, search, indices, ce_template_mask, delta):
        if template.shape[0] != 1 or search.shape != delta.shape:
            raise ValueError('suffix expects one frame and an exact-shape update')
        x = torch.cat((template, search + delta), dim=1)
        ti = torch.arange(template.shape[1], device=x.device, dtype=indices.dtype)[None]
        si = indices.clone()
        # Pre-insertion removed cells are completely determined by the cached
        # retained indices. Their ordering does not affect zero restoration.
        initial = torch.ones((1, self.feat_len_s), dtype=torch.bool, device=x.device)
        initial.scatter_(1, si.long(), False)
        removed_masks = [initial]
        removed_indices = [torch.nonzero(initial[0], as_tuple=False).reshape(1, -1).to(indices.dtype)]
        for i, block in enumerate(self.blocks):
            x, ti, si, removed, _ = block(x, ti, si, None, ce_template_mask, None)
            if i in self.ce_after:
                mask = torch.zeros_like(initial)
                if removed is not None:
                    mask.scatter_(1, removed.long(), True)
                    removed_indices.append(removed)
                removed_masks.append(mask)
        x = self.norm(x)
        z, retained = x[:, :ti.shape[1]], x[:, ti.shape[1]:]
        # Match production restoration, including its gradient layout. A
        # shorter direct scatter has equal outputs but may select a different
        # floating-point backward kernel due to different gradient strides.
        padding = x.new_zeros((1, self.feat_len_s - si.shape[1], x.shape[-1]))
        padded = torch.cat((retained, padding), dim=1)
        index_all = torch.cat((si, torch.cat(removed_indices, dim=1)), dim=1)
        restored = torch.zeros_like(padded).scatter_(1, index_all.long().unsqueeze(-1).expand_as(padded), padded)
        out = OSTrack.forward_head(self, torch.cat((z, restored), dim=1), None)
        out['ce_removed'] = torch.stack(removed_masks)
        out['final_kept'] = ~out['ce_removed'].any(dim=0)
        return out


@contextmanager
def activation_signature(module, score_branch=False):
    """Hash relevant ReLU signs without modifying activations.

    Derivatives target the score objective. Size/offset branch boundaries do
    not affect this derivative and must not disqualify its finite differences.
    """
    digest = hashlib.sha256()
    values = {'relu_modules': 0}

    def capture(_module, args):
        a = (args[0].detach() > 0).to(torch.uint8).cpu().numpy()
        digest.update(str(a.shape).encode('ascii'))
        digest.update(a.tobytes())
        values['relu_modules'] += 1

    handles = [m.register_forward_pre_hook(capture) for n, m in module.named_modules()
               if isinstance(m, nn.ReLU) and (not score_branch or '_ctr' in n.split('.')[0])]
    try:
        yield values
    finally:
        for h in handles:
            h.remove()
        values['relu_sha256'] = digest.hexdigest()
