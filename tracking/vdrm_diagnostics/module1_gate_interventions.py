"""Temporary V8 gate interventions; raw diagnostics remain untouched."""

from contextlib import contextmanager

import torch


@contextmanager
def gate_intervention(model, mode):
    """Flatten gates before residual construction; restore even on exceptions.

    Norm controls match the *pre-injection* Frobenius norm to the original
    residual on the very same input. They do not use GT or change alpha.
    """
    module = model.backbone.vdrm
    norm_match = mode.endswith('_norm')
    base = mode[:-5] if norm_match else mode
    if base not in ('control', 'qflat', 'routeflat', 'bothflat', 'alpha0') or (
            norm_match and base not in ('qflat', 'routeflat', 'bothflat')):
        raise ValueError('unknown gate intervention: ' + mode)
    if (module.spatial_gate_mode != 'part_aligned' or module.residual_max_ratio != 0
            or module.inference_ablation is not None):
        raise ValueError('requires original unbounded V8 part_aligned without other ablations')
    original = module._part_aligned_statistics
    had_override = '_part_aligned_statistics' in module.__dict__
    alpha = module.alpha.detach().clone()
    values = {}

    def statistics(similarity, prototypes, q, valid, route_logits_override=None):
        count = valid.sum(-1, keepdim=True).clamp_min(1)
        effective_q = q
        if base in ('qflat', 'bothflat'):
            effective_q = q.sum(-1, keepdim=True) / count * valid.to(q.dtype)
        logits, route, residual, diagnostics = original(
            similarity, prototypes, effective_q, valid, route_logits_override=route_logits_override)
        effective_route = route
        if base in ('routeflat', 'bothflat'):
            effective_route = route.mean(-1, keepdim=True).expand_as(route)
        gain = torch.ones((q.shape[0], 1, 1), device=q.device, dtype=q.dtype)
        if norm_match:
            reference = torch.einsum('bkl,bkc->blc', route * q[..., None], prototypes)
            reference = reference / count[..., None].to(q.dtype)
            target = reference.norm(dim=(1, 2), keepdim=True)
            current = residual.norm(dim=(1, 2), keepdim=True)
            if ((current == 0) & (target > 0)).any():
                raise RuntimeError('cannot norm-match a zero residual to a nonzero reference')
            gain = torch.where(current > 0, target / current.clamp_min(1e-30), torch.ones_like(current))
            residual = residual * gain
        values.update(raw_q=q, effective_q=effective_q, raw_route=route,
            effective_route=effective_route, part_valid=valid,
            prototype_gram=prototypes @ prototypes.transpose(-1, -2), norm_gain=gain,
            residual_norm=residual.norm(dim=-1))
        return logits, route, residual, diagnostics

    try:
        module.inference_ablation = 'route_spatial_mean' if base in ('routeflat', 'bothflat') else None
        module._part_aligned_statistics = statistics
        if base == 'alpha0':
            with torch.no_grad():
                module.alpha.zero_()
        yield values
    finally:
        module.inference_ablation = None
        if had_override:
            module._part_aligned_statistics = original
        else:
            del module._part_aligned_statistics
        with torch.no_grad():
            module.alpha.copy_(alpha)
