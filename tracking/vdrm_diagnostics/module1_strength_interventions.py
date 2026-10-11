"""Signed residual gain for fixed-input diagnosis; never writes parameters."""

from contextlib import contextmanager

import torch

from .module1_gate_interventions import gate_intervention
from .module1_location_interventions import location_intervention


@contextmanager
def strength_intervention(model, family, strength, target_mask=None):
    if family not in ('original', 'route_target_norm') or strength not in (-4., -1., 0., 1., 4., 16.):
        raise ValueError('unregistered residual strength intervention')
    module = model.backbone.vdrm
    original_alpha = module._effective_alpha
    had_alpha_override = '_effective_alpha' in module.__dict__
    reference_alpha = original_alpha().detach().clone()
    context = (gate_intervention(model, 'control') if family == 'original' else
               location_intervention(model, 'route_target_norm', target_mask))
    with context as values:
        original_statistics = module._part_aligned_statistics
        residual = {}

        def statistics(*args, **kwargs):
            result = original_statistics(*args, **kwargs)
            residual['value'] = result[2]
            return result

        def capture_update(mod, args, kwargs, output):
            tokens = args[0] if args else kwargs['tokens']
            start = int(kwargs['template_length'])
            search = tokens[:, start:]
            delta = output[0][:, start:] - search
            r = residual['value']
            actual_alpha = output[1]['vdrm_alpha']
            expected = actual_alpha * r
            # This signed projection plus the norm distinguishes opposite updates.
            values.update(reference_alpha=reference_alpha,
                          injected_residual_dot=(delta * r).sum(-1),
                          injection_roundoff_max=(delta - expected).abs().amax().reshape(1),
                          search_norm=search.norm(dim=-1))
            # The actual subtraction must agree up to float32 addition rounding.
            if not torch.allclose(delta, expected, rtol=5e-3, atol=1e-4):
                raise RuntimeError('actual residual injection differs from requested signed gain')

        hook = None
        try:
            module._part_aligned_statistics = statistics
            # Multiply the effective value, not the raw parameter inside tanh.
            # Gains >1 may exceed the trained alpha bound; this is a stress test.
            module._effective_alpha = lambda: original_alpha() * strength
            hook = module.register_forward_hook(capture_update, with_kwargs=True)
            yield values
        finally:
            if hook is not None:
                hook.remove()
            module._part_aligned_statistics = original_statistics
            if had_alpha_override:
                module._effective_alpha = original_alpha
            else:
                del module._effective_alpha
