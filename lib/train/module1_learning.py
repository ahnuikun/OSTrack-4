"""Step08 pilot: paired, bounded residual learning at the original insertion.

Only the final projection differs between arms. No labels enter the adapter.
Existing Rdisc q, route, alpha, backbone and heads remain frozen.
"""

from contextlib import contextmanager
import math

import torch
from torch import nn
import torch.nn.functional as F

from lib.utils.box_ops import box_cxcywh_to_xyxy,box_xywh_to_xyxy,giou_loss

ARMS=('prototype','full')


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
        if search.shape[0]!=1:raise ValueError('pilot uses batch size one')
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
        reference=original.norm()
        correction=correction*(reference/math.sqrt(original.numel()))
        combined=original+correction
        # One original-update norm for BOTH arms, including the original path.
        scale=(reference/combined.norm().clamp_min(1e-12)).clamp(max=1.)
        return combined*scale


@contextmanager
def adapter_insertion(model,adapter):
    if adapter is None:
        yield {}
        return
    module=model.backbone.vdrm
    if module.spatial_gate_mode!='part_aligned' or module.residual_max_ratio!=0 or module.inference_ablation is not None:
        raise ValueError('pilot requires original Rdisc insertion')
    values={}
    def hook(mod,args,kwargs,output):
        tokens=args[0] if args else kwargs['tokens']
        search,p,valid,mix,original=insertion_features(mod,tokens,kwargs,output)
        delta=adapter(search,p,valid,mix,original)
        start=int(kwargs['template_length'])
        modified=torch.cat((tokens[:,:start],search+delta),dim=1)
        actual=modified[:,start:]-search
        ratio=float(actual.norm()/original.norm().clamp_min(1e-12))
        if not torch.isfinite(modified).all() or ratio>1.005:raise RuntimeError('invalid learned residual or exceeded norm budget')
        values.update(actual_norm_ratio=ratio)
        return modified,output[1]
    handle=module.register_forward_hook(hook,with_kwargs=True)
    try:yield values
    finally:handle.remove()


def dense_boxes(size,offset):
    """Differentiable CENTER decode; older ROI helper deliberately detaches."""
    h,w=size.shape[-2:]
    yy,xx=torch.meshgrid(torch.arange(h,device=size.device),torch.arange(w,device=size.device),indexing='ij')
    grid=torch.stack((xx,yy)).to(size)[None]
    centers=(grid+offset)/size.new_tensor([w,h])[None,:,None,None]
    return torch.cat((centers,size),dim=1).flatten(2).transpose(1,2)


def learning_loss(out,base,good,target,hann,protect):
    """All competing cells, differentiable localization, healthy-output anchor.

    'good' is a fixed baseline geometry label, NOT an instance-identity label.
    Frames with no supported good box use output preservation only.
    """
    raw=out['score_map'].flatten();good=good.flatten().bool()
    zero=raw.sum()*0
    ranking,localization=zero,zero
    if good.any():
        for score,temperature in ((raw,.1),((out['score_map']*hann).flatten(),.05)):
            logits=score/temperature
            ranking=ranking+torch.logsumexp(logits,0)-torch.logsumexp(logits[good],0)
        boxes=dense_boxes(out['size_map'],out['offset_map'])[0,good]
        target=target.reshape(1,4).expand(len(boxes),4)
        gt_cxcywh=torch.cat((target[:,:2]+target[:,2:]/2,target[:,2:]),1)
        giou,_=giou_loss(box_cxcywh_to_xyxy(boxes),box_xywh_to_xyxy(target))
        localization=2*giou+5*F.l1_loss(boxes,gt_cxcywh)
    preservation=zero
    if protect or not good.any():
        preservation=sum(F.mse_loss(out[k],base[k]) for k in ('score_map','size_map','offset_map'))
    total=ranking+localization+10*preservation
    if not torch.isfinite(total):raise RuntimeError('nonfinite residual learning loss')
    return total,dict(rank=float(ranking.detach()),localization=float(localization.detach()),
                      preservation=float(preservation.detach()),total=float(total.detach()))
