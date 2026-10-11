"""Replace the search update at the original VDRM insertion, for GT diagnosis."""

from contextlib import contextmanager

import torch


@contextmanager
def replace_search_update(model, reference, delta=None, gradient=False):
    """Parameters and template tokens are untouched; gradients target one leaf.

    delta=None: original update. gradient=True: start at zero update and expose
    its leaf. Diagnostics in the model output describe the original module;
    this experiment records the actual replacement separately.
    """
    if gradient and delta is not None:
        raise ValueError('gradient baseline must start at zero update')
    module = model.backbone.vdrm
    if module.residual_max_ratio != 0 or module.inference_ablation is not None:
        raise ValueError('requires the unchanged Rdisc insertion path')
    values = {}

    def hook(mod, args, kwargs, output):
        tokens = args[0] if args else kwargs['tokens']
        start = int(kwargs['template_length'])
        search = tokens[:, start:]
        if (not torch.equal(search, reference['search_tokens'])
                or not torch.equal(tokens[:, :start], reference['template_tokens'])):
            raise RuntimeError('direction experiment changed the insertion prefix')
        if not torch.equal(output[0][:, :start], tokens[:, :start]):
            raise RuntimeError('VDRM unexpectedly changed template tokens')
        requested = delta
        if gradient:
            requested = torch.zeros_like(search, requires_grad=True)
            values['leaf'] = requested
        if requested is None:
            result = output
        else:
            if requested.shape != search.shape or not torch.isfinite(requested).all():
                raise ValueError('invalid replacement update')
            result = (torch.cat((tokens[:, :start].detach(), search.detach() + requested), dim=1), output[1])
        values['actual_delta'] = (result[0][:, start:] - search).detach().clone()
        return result

    handle = module.register_forward_hook(hook, with_kwargs=True)
    try:
        yield values
    finally:
        handle.remove()
