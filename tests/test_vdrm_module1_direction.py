"""Direction-space mathematics, actual leaf gradients and evidence contracts."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import sys
from unittest.mock import patch

import numpy as np
import pytest
import torch

from lib.models.layers.vdrm import VisibilityDrivenRepresentationModule
from lib.test.analysis.frozen_failure_diagnostics import dump_json,write_bundle
from lib.test.analysis.module1_gates import ARMS,CONFIGURATIONS
from lib.test.analysis.module1_direction import (
    OBJECTIVES,SPACES,BUDGETS,FD_EPS,frame_plan,choose_pair,project_gradient,normalized_update,modes,check_bundle)
from tracking import diagnose_vdrm_module1_direction as entry
from tracking.diagnose_vdrm_native_float import npz_bytes
from tracking.vdrm_diagnostics.module1_direction_interventions import replace_search_update
from test_vdrm_module1_location import proxy_arrays


def test_projection_is_orthogonal_and_handles_dependent_invalid_prototypes():
    p=np.array([[[1.,0.,0.],[2.,0.,0.],[0.,1.,0.],[0.,0.,1.]]])
    g=np.array([[[3.,4.,5.],[1.,2.,3.]]])
    projected,rank=project_gradient(g,p,[[1,1,1,0]])
    assert rank==2
    np.testing.assert_allclose(projected,[[[3,4,0],[1,2,0]]],atol=1e-12)
    assert np.isclose((projected*(g-projected)).sum(),0)
    zero,rank=project_gradient(g,p,[[0,0,0,0]])
    assert rank==0 and not zero.any()
    for budget in (1.,4.,-.001):
        d=normalized_update(projected,2*budget)
        assert np.isclose(np.linalg.norm(d),2*abs(budget))
        assert np.sign((g*d).sum())==np.sign(budget)
    assert not normalized_update(np.zeros_like(g),2).any()


def fixture_module():
    torch.manual_seed(36)
    module=VisibilityDrivenRepresentationModule(spatial_gate_mode='part_aligned').eval().requires_grad_(False)
    module.alpha.fill_(-.07)
    tokens=torch.randn(1,20,8)
    reference={'template_tokens':tokens[:,:4].clone(),'search_tokens':tokens[:,4:].clone()}
    kwargs=dict(template_length=4,template_bbox=torch.tensor([[0.,0.,1.,1.]]),
                search_global_index=torch.arange(16)[None].float(),search_grid_size=4)
    return SimpleNamespace(backbone=SimpleNamespace(vdrm=module)),tokens,reference,kwargs


def test_real_module_leaf_gradient_no_weight_or_template_update_and_failure_cleanup():
    model,tokens,reference,kwargs=fixture_module();module=model.backbone.vdrm
    before={k:v.clone() for k,v in module.state_dict().items()}
    with replace_search_update(model,reference,gradient=True) as v:
        out=module(tokens,**kwargs)[0]
        score=out[:,4:].square().sum()
        gradient,=torch.autograd.grad(score,v['leaf'])
        torch.testing.assert_close(gradient,2*reference['search_tokens'])
        assert not v['actual_delta'].any()
        torch.testing.assert_close(out[:,:4],tokens[:,:4],atol=0,rtol=0)
    d=torch.ones_like(reference['search_tokens'])*.04
    with torch.no_grad(),replace_search_update(model,reference,d) as v:
        out=module(tokens,**kwargs)[0]
        torch.testing.assert_close(out[:,4:],reference['search_tokens']+d)
    with pytest.raises(RuntimeError,match='sentinel'):
        with replace_search_update(model,reference):raise RuntimeError('sentinel')
    assert not module._forward_hooks
    assert all(torch.equal(value,module.state_dict()[k]) for k,value in before.items())
    assert all(p.grad is None and not p.requires_grad for p in module.parameters())


def test_insertion_prefix_mismatch_is_not_silently_accepted():
    model,tokens,ref,kwargs=fixture_module();ref['search_tokens']+=1
    with pytest.raises(RuntimeError,match='prefix'):
        with replace_search_update(model,ref,gradient=True):model.backbone.vdrm(tokens,**kwargs)
    assert not model.backbone.vdrm._forward_hooks


def fixture_prefix():
    a=proxy_arrays('control')
    a={k:a[k] for k in ('score_map','size_map','offset_map','final_kept')}
    a['actual_delta']=np.full((1,12,8),.01,np.float32)
    a['ce_removed']=np.zeros((3,1,256),bool)
    p=dict(prototypes=np.eye(4,8,dtype=np.float32)[None],part_valid=np.ones((1,4),bool),
        indices=np.array([[102,103,104,105,118,119,120,121,134,135,136,137]]),
        reference_delta=a['actual_delta'].copy(),padding=np.zeros((256,256),bool),hann=np.ones((1,1,16,16),np.float32))
    for label in ('original','zero'):
        for k in ('score_map','size_map','offset_map'):p[label+'_'+k]=a[k].copy()
    for objective in OBJECTIVES:p['gradient_'+objective]=np.ones((1,12,8),np.float32)
    return a,p


def test_pair_requires_both_baselines_valid_retained_nonpadding_candidates():
    a,p=fixture_prefix();case=dict(gt_xywh=[60.,60.,40.,40.],anchor_xywh=[60.,60.,40.,40.],resize_factor=1.6,image_shape_hw=[160,160],cover=8,absence=0)
    pair,reason=choose_pair(a,a,p,case)
    assert reason is None and pair[0]==136 and pair[1]!=136
    z=deepcopy(a);z['final_kept'][:]=False
    assert choose_pair(a,z,p,case)[1]=='no_common_good_candidate'
    p['padding'][:]=True
    assert choose_pair(a,a,p,case)[1]=='no_common_good_candidate'
    case['cover']=0
    assert choose_pair(a,a,p,case)[1]=='not_visible_or_invalid'


def bundle_fixture(path,corrupt=None):
    a,p=fixture_prefix();cases,rows,payloads=[],[],{}
    for n,fs in frame_plan(True).items():
        for f in fs:
            box=[60.,60.,40.,40.]
            c=dict(sequence=n,frame_1based=f,gt_xywh=box,anchor_xywh=box,source_bbox_xywh=box,
                image_shape_hw=[160,160],resize_factor=1.6,cover=8,absence=0,input_sha256={'search':'e'*64},
                reference_norm=float(np.linalg.norm(p['reference_delta'].astype(float))),prototype_rank=4,prefix_member=f'prefix/{n}_{f}.npz')
            pair,reason=choose_pair(a,a,p,c);c.update(pair=pair,exclusion_reason=reason)
            cases.append(c);payloads[c['prefix_member']]=npz_bytes(p)
            for mode,obj,space,budget in modes(True):
                aa=deepcopy(a)
                if mode=='zero':aa['actual_delta'][:]=0
                elif mode!='original':
                    g=p['gradient_'+obj];pg,_=project_gradient(g,p['prototypes'],p['part_valid'])
                    aa['actual_delta']=normalized_update(pg if space=='prototype' else g,c['reference_norm']*budget)
                member=f'arrays/{n}_{f}_{mode}_{obj}_{space}_{budget}.npz'
                payloads[member]=npz_bytes(aa)
                rows.append(dict(sequence=n,frame_1based=f,mode=mode,objective=obj,space=space,budget=budget,
                    arrays_member=member,input_sha256=c['input_sha256'],bbox_xywh=box))
    if corrupt:corrupt(cases,rows,payloads)
    provenance={arm:dict(checkpoint_sha256='a'*64,state_sha256_before='b'*64,state_sha256_after='b'*64) for arm in ARMS}
    manifest=dict(kind='vdrm_module1_step06_direction',version=1,smoke_only=True,frame_plan={k:list(v) for k,v in frame_plan(True).items()},
        objectives=list(OBJECTIVES),spaces=list(SPACES),budgets=list(BUDGETS),fd_epsilon=FD_EPS,optimizer_steps=0,
        configurations={arm:CONFIGURATIONS[arm] for arm in ARMS},source_verified=True,gt_gradient_diagnosis_only=True,
        source_sha256='c'*64,provenance=provenance,source_provenance=provenance,forward_calls=len(rows),backward_calls=len(cases)*2)
    write_bundle(path,manifest,{'cases.json':dump_json(cases),'rows.json':dump_json(rows),**payloads})


def test_cpu_bundle_recomputes_vectors_and_outputs(tmp_path):
    p=tmp_path/'good.tar.gz';bundle_fixture(p)
    s,f,g=check_bundle(p)
    assert s['cases']==10 and s['forward_calls']==140 and s['backward_calls']==20
    assert len(g)==20 and all(np.isclose(d['projected_energy_fraction'],.5) for d in g)
    assert not s['tracking_performance_passed'] and not s['global_capacity_bound']
    # These constant-map bundle fixtures intentionally do not implement their
    # stored gradients. Structural checks alone must not mark derivatives valid.
    assert not s['derivative_audit_passed']


@pytest.mark.parametrize('kind',['missing','pair','input','sign','norm','gradient','control'])
def test_invalid_bundle_rejected(tmp_path,kind):
    def corrupt(cases,rows,payloads):
        import io
        if kind=='missing':rows.pop()
        elif kind=='pair':cases[0]['pair']=cases[0]['pair'][::-1]
        elif kind=='input':rows[2]['input_sha256']={'search':'f'*64}
        else:
            name=cases[0]['prefix_member'] if kind in ('gradient','control') else rows[2]['arrays_member']
            with np.load(io.BytesIO(payloads[name])) as raw:arr={k:raw[k] for k in raw.files}
            if kind=='sign':arr['actual_delta']*=-1
            elif kind=='norm':arr['actual_delta']*=2
            elif kind=='gradient':arr['gradient_raw'][:]=0
            else:arr['original_score_map']+=.01
            payloads[name]=npz_bytes(arr)
    p=tmp_path/'bad.tar.gz';bundle_fixture(p,corrupt)
    with pytest.raises(ValueError):check_bundle(p)


def test_cli_fixed_frames_and_output_protection(tmp_path):
    assert sum(map(len,frame_plan(False).values()))==25
    for smoke in (False,True):
        with patch.object(sys,'argv',['diagnose_vdrm_module1_direction.py','--save_dir','./output']+(['--smoke'] if smoke else [])),patch.object(entry,'run') as run:
            entry.main()
        args=run.call_args.args
        assert args[3]==smoke and Path(args[1]).name=='module1_evidence.tar.gz'
        assert Path(args[2]).name==('module1_direction_smoke.tar.gz' if smoke else 'module1_direction.tar.gz')
    out=tmp_path/'direction.tar.gz';out.write_bytes(b'keep')
    with pytest.raises(FileExistsError):entry.run(tmp_path,'missing',out)
    assert out.read_bytes()==b'keep'
