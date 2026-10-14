"""Shared residual adapter; per-image total-update budget, no GT at inference."""
import math
import torch
from torch import nn
import torch.nn.functional as F

ARMS = ("prototype", "full")

def insertion_features(module,tokens,kwargs,output):
    start=int(kwargs['template_length'])
    template,search=tokens[:,:start],tokens[:,start:]
    masks,valid=module._build_part_masks(kwargs['template_bbox'],start)
    masks=masks.to(template)
    prototypes=torch.einsum('bkl,blc->bkc',masks,template)/masks.sum(-1,keepdim=True).clamp_min(1)
    prototypes=prototypes*valid[...,None]
    diag=output[1]
    mixture=torch.einsum('bkn,bkc->bnc',diag['part_route_gate'],prototypes*diag['part_reliability'][...,None])
    mixture=mixture/valid.sum(-1).clamp_min(1)[:,None,None]
    return search,prototypes,valid,mixture,output[0][:,start:]-search


class ResidualAdapter(nn.Module):
    """Bounded correction with a SHARED output subspace of at most ``width``.

    ``full`` only disables projection onto the four template prototypes. It
    does not mean unrestricted 768-dimensional output: value(h) lies in the
    column space of value.weight, shared by every token and training frame.
    Keep width=32 for exact historical replay; capacity fixes must opt in.
    """
    def __init__(self,dimension=768,width=32,space='full'):
        super().__init__()
        if space not in ARMS:raise ValueError('unknown residual space')
        self.space=space
        self.search=nn.Linear(dimension,width,bias=False)
        self.template=nn.Linear(dimension,width,bias=False)
        self.value=nn.Linear(width,dimension,bias=False)
        nn.init.zeros_(self.value.weight)

    def forward(self,search,prototypes,valid,mixture,original):
        if self.space == 'prototype' and search.shape[0] != 1:
            raise ValueError('historical prototype projection requires batch size one')
        width=search.shape[-1]
        h=F.gelu(self.search(F.layer_norm(search,(width,)))+self.template(F.layer_norm(mixture,(width,))))
        correction=self.value(h)
        if self.space=='prototype':
            p=prototypes[0,valid[0]].detach().double()
            if len(p):
                _,s,vh=torch.linalg.svd(p,full_matrices=False)
                rank=int((s>max(float(s[0])*1e-6,1e-12)).sum())
                basis=vh[:rank].to(correction)
                correction=(correction@basis.T)@basis
            else:correction=correction*0
        reference = original.norm() if len(original) == 1 else original.flatten(1).norm(dim=1)[:,None,None]
        correction=correction*(reference/math.sqrt(original[0].numel()))
        combined=original+correction
        # One original-update norm for BOTH arms, including the original path.
        combined_norm = combined.norm() if len(combined) == 1 else combined.flatten(1).norm(dim=1)[:,None,None]
        scale=(reference/combined_norm.clamp_min(1e-12)).clamp(max=1.)
        # At scratch initialization alpha=0 and value=0. A zero scale here
        # would kill the only gradient that can turn the original residual on.
        scale=torch.where((reference == 0) & (combined_norm == 0), torch.ones_like(scale), scale)
        return combined*scale
