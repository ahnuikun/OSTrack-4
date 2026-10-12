"""Budget, diagnostic selection, leaf-only oracle and evidence corruption."""

from copy import deepcopy
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pytest
import torch
from torch import nn

from lib.train.module1_learning import ResidualAdapter
from lib.test.analysis.module1_fit_audit import oracle_selection,projected_proposal,check_bundle
from lib.test.analysis.frozen_failure_diagnostics import dump_json,write_bundle
from tracking.diagnose_vdrm_native_float import npz_bytes
from tracking.diagnose_vdrm_module1_fit import audit_case


def test_projection_budget_direction_and_zero_radius():
    original=torch.zeros(1,4,8);original[...,0]=1
    grad=torch.ones_like(original);grad[...,1]=4
    prototypes=torch.eye(2,8)[None];valid=torch.ones(1,2,dtype=torch.bool)
    a=projected_proposal(original,grad,original,prototypes,valid,'prototype')
    b=projected_proposal(original,grad,original,prototypes,valid,'full')
    assert a.norm()<=original.norm() and b.norm()<=original.norm()
    assert not a[...,2:].any() and b[...,2:].abs().sum()>0
    assert ((a-original)*grad).sum()<0 and ((b-original)*grad).sum()<0
    assert not projected_proposal(original*0,grad,original*0,prototypes,valid,'full').any()


def test_selection_covers_supported_failures_without_selecting_no_good():
    rows=[dict(good_cells=1,baseline_iou=.2,protect=False,cover=8) for _ in range(15)]
    rows += [dict(good_cells=0,baseline_iou=0,protect=False,cover=1)]*5
    rows += [dict(good_cells=1,baseline_iou=.8,protect=True,cover=2)]*7
    rows += [dict(good_cells=1,baseline_iou=.8,protect=True,cover=8)]*7
    ids=oracle_selection(rows)
    assert len(ids)==24 and len(set(ids))==24 and not set(ids)&set(range(15,20))
    assert sum(i<15 for i in ids)==12 and len(oracle_selection(rows,True))==3


class TinySuffix(nn.Module):
    def __init__(self):
        super().__init__();self.constant=nn.Parameter(torch.tensor(1.),requires_grad=False)
    def forward(self,template,search,indices,mask,delta):
        x=search+delta
        return dict(score_map=(x[...,0]+x[...,1]*20).sigmoid().reshape(1,1,16,16),
            size_map=(.3+x[...,:2].transpose(1,2)*.01).reshape(1,2,16,16),
            offset_map=(.5+x[...,2:4].transpose(1,2)*.01).reshape(1,2,16,16),
            final_kept=torch.ones(1,256,dtype=torch.bool))


def fixture(steps=2):
    torch.manual_seed(4);suffix=TinySuffix()
    x=torch.zeros(1,256,8);x[...,0]=torch.linspace(-1,1,256)
    delta=torch.zeros_like(x);delta[...,1]=.01
    a={k:v.numpy() for k,v in dict(template_tokens=torch.zeros(1,64,8),search_tokens=x,delta=delta,
        indices=torch.arange(256)[None].float(),ce_template_mask=torch.ones(1,64,dtype=torch.bool),
        prototypes=torch.eye(4,8)[None],part_valid=torch.ones(1,4,dtype=torch.bool),mixture=torch.ones_like(x),
        hann=torch.ones(16,16),good=torch.arange(256)==136,target=torch.tensor([.35,.35,.3,.3])).items()}
    a.update({k:v.numpy() for k,v in suffix(torch.zeros(1,64,8),x,None,None,delta).items()})
    c=dict(case_id='train_0000',population='train',sequence='fixture',frame_1based=2,good_cells=1,protect=False,
           baseline_iou=0.,oracle_selected=True,gt_xywh=[95.,95.,20.,20.],anchor_xywh=[80.,80.,40.,40.],
           resize_factor=4.,image_shape_hw=[200,200])
    adapters={arm:ResidualAdapter(8,4,arm).eval().requires_grad_(False) for arm in ('prototype','full')}
    with torch.no_grad():
        for adapter in adapters.values():adapter.value.weight.normal_(std=.1)
    payloads={};case,rows=audit_case(a,c,suffix,adapters,steps,payloads)
    assert all(p.grad is None for p in suffix.parameters())
    manifest=dict(kind='vdrm_module1_step09_fit_audit',version=1,parameter_updates=0,deployable_oracle=False,
        total_norm_budget=1.,step_size=.25,oracle_steps=steps,smoke_only=steps==2,cases=1,train_cases=1,regression_cases=0,
        suffix_before='x',suffix_after='x',adapter_before={'full':'a'},adapter_after={'full':'a'})
    payloads.update({'cases.json':dump_json([case]),'rows.json':dump_json(rows)})
    return manifest,payloads,case,rows


@pytest.mark.parametrize('steps',[2,8])
def test_actual_leaf_oracle_and_cpu_round_trip(tmp_path,steps):
    m,p,c,rows=fixture(steps);path=tmp_path/'fit.tar.gz';write_bundle(path,m,p)
    summary,_,_=check_bundle(path)
    assert summary['parameter_updates']==0 and len(rows)==6
    baseline=rows[0]['loss']['total']
    assert all(r['loss']['total']<=baseline+1e-7 for r in rows if r['mode'].startswith('oracle_'))
    assert any(r['linear_loss_change']!=0 for r in rows if r['mode'] in ('prototype','full'))


@pytest.mark.parametrize('fault',['loss','iou','budget','trace','alignment','missing','weights','population','support'])
def test_cpu_checker_rejects_corruption(tmp_path,fault):
    m,p,c,rows=fixture()
    if fault=='loss':rows[2]['loss']['total']+=.1
    elif fault=='iou':rows[2]['hann_iou']+=.1
    elif fault=='budget':rows[2]['geometry']['norm_ratio']=2
    elif fault=='trace':rows[-1]['trace'][0]['accepted']=not rows[-1]['trace'][0]['accepted']
    elif fault=='alignment':rows[2]['linear_loss_change']+=1
    elif fault=='missing':rows.pop()
    elif fault=='weights':m['suffix_after']='changed'
    elif fault=='population':m['train_cases']=2
    elif fault=='support':c['good_cells']=2
    p['cases.json']=dump_json([c]);p['rows.json']=dump_json(rows)
    path=tmp_path/'bad.tar.gz';write_bundle(path,m,p)
    with pytest.raises(ValueError):check_bundle(path)


def test_no_good_case_is_reported_without_fake_oracle():
    # The source loss test covers preservation-only gradients; the selection
    # itself must never pretend an unsupported failure is an oracle success.
    assert oracle_selection([dict(good_cells=0,baseline_iou=0.,protect=False,cover=1)])==[]
