"""GT-location proxies for diagnosis, never a deployable tracking policy."""

from contextlib import contextmanager

import torch
import torch.nn.functional as F

from .module1_gate_interventions import gate_intervention


@contextmanager
def location_intervention(model, mode, target_mask):
    base = mode[:-5] if mode.endswith('_norm') else mode
    if base not in ('q_target', 'route_target', 'joint_target'):
        raise ValueError('unknown location proxy: ' + mode)
    module = model.backbone.vdrm
    with gate_intervention(model, 'control') as values:
        original = module._part_aligned_statistics

        def statistics(similarity, prototypes, q, valid, route_logits_override=None):
            result = original(similarity, prototypes, q, valid, route_logits_override=route_logits_override)
            logits, route, original_residual, diagnostics = result
            mask = torch.as_tensor(target_mask, device=q.device, dtype=torch.bool)
            if similarity.shape[0] != 1 or mask.shape != (similarity.shape[-1],) or mask.sum() < 4:
                raise ValueError('location proxy needs batch one and four supported target tokens')
            effective_q, effective_route = q, route
            scale = F.softplus(module.log_match_scale)
            if base in ('q_target', 'joint_target'):
                evidence = similarity[..., mask].topk(4, dim=-1).values.mean(-1)
                effective_q = torch.sigmoid(scale * evidence + module.match_bias) * valid.to(q.dtype)
            if base in ('route_target', 'joint_target'):
                effective_route = mask[None, None, :].to(route.dtype) * valid[..., None].to(route.dtype)
            count = valid.sum(-1, keepdim=True).clamp_min(1)
            residual = torch.einsum('bkl,bkc->blc', effective_q[..., None] * effective_route, prototypes)
            residual = residual / count[..., None].to(q.dtype)
            gain = torch.ones((1, 1, 1), device=q.device, dtype=q.dtype)
            if mode.endswith('_norm'):
                norm, reference = residual.norm(), original_residual.norm()
                if norm == 0 and reference > 0:
                    raise RuntimeError('cannot norm-match zero location residual')
                if norm > 0:
                    gain = (reference / norm).reshape(1, 1, 1)
                    residual = residual * gain
            values.update(effective_q=effective_q, effective_route=effective_route, norm_gain=gain,
                          residual_norm=residual.norm(dim=-1), similarity=similarity,
                          match_scale=scale, match_bias=module.match_bias, target_mask=mask[None])
            return logits, route, residual, diagnostics

        try:
            module._part_aligned_statistics = statistics
            yield values
        finally:
            module._part_aligned_statistics = original
