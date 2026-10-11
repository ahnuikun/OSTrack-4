"""Test signed update semantics, restoration, artifact contracts and readouts."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import sys
from unittest.mock import patch

import numpy as np
import pytest
import torch

from lib.models.layers.vdrm import VisibilityDrivenRepresentationModule
from lib.test.analysis.frozen_failure_diagnostics import dump_json, write_bundle
from lib.test.analysis.module1_gates import ARMS, CONFIGURATIONS, protocol_plan
from lib.test.analysis.module1_location import SCALES
from lib.test.analysis.module1_strength import GAINS, schedule, check_bundle, selection_metrics, summarize
from tracking import diagnose_vdrm_module1_strength as entry
from tracking.diagnose_vdrm_native_float import npz_bytes
from tracking.vdrm_diagnostics.module1_strength_interventions import strength_intervention
from tracking.vdrm_diagnostics.module1_location_interventions import location_intervention
from test_vdrm_module1_location import proxy_arrays


@pytest.mark.parametrize('family',['original','route_target_norm'])
@pytest.mark.parametrize('gain',GAINS)
def test_real_module_signed_output_and_restoration(family,gain):
    torch.manual_seed(12)
    m=VisibilityDrivenRepresentationModule(spatial_gate_mode='part_aligned',alpha_max=.1).eval()
    model=SimpleNamespace(backbone=SimpleNamespace(vdrm=m))
    tokens=torch.randn(1,20,16)
    mask=np.arange(16)<5
    kwargs=dict(template_length=4,template_bbox=torch.tensor([[0.,0.,1.,1.]]),
                search_global_index=torch.arange(16)[None].float(),search_grid_size=4)
    with torch.no_grad():
        m.alpha.fill_(-.073)
        before={k:v.clone() for k,v in m.state_dict().items()}
        reference=m(tokens,**kwargs)[0]
        if family=='route_target_norm':
            with location_intervention(model,family,mask): reference=m(tokens,**kwargs)[0]
        with strength_intervention(model,family,gain,mask) as values:
            out,diagnostics=m(tokens,**kwargs)
            torch.testing.assert_close(out[:,4:]-tokens[:,4:],gain*(reference[:,4:]-tokens[:,4:]),atol=4e-6,rtol=3e-4)
            torch.testing.assert_close(out[:,:4],tokens[:,:4],atol=0,rtol=0)
            assert np.sign(diagnostics['vdrm_alpha'].item())==np.sign(-gain)
            if gain!=0: assert torch.sign(values['injected_residual_dot'].sum()).item()==np.sign(-gain)
        assert all(torch.equal(v,m.state_dict()[k]) for k,v in before.items())
        assert '_effective_alpha' not in m.__dict__ and '_part_aligned_statistics' not in m.__dict__
        assert not m._forward_hooks


def test_exception_restores_existing_override_and_hooks():
    m=VisibilityDrivenRepresentationModule(spatial_gate_mode='part_aligned')
    model=SimpleNamespace(backbone=SimpleNamespace(vdrm=m))
    original=m._effective_alpha
    m._effective_alpha=original
    with pytest.raises(RuntimeError,match='sentinel'):
        with strength_intervention(model,'original',-1): raise RuntimeError('sentinel')
    assert m._effective_alpha is original and not m._forward_hooks
    assert '_part_aligned_statistics' not in m.__dict__


def test_raw_vs_hann_selection_and_fixed_reference_masks():
    a=proxy_arrays('control')
    row=dict(anchor_xywh=[60.,60.,40.,40.],gt_xywh=[60.,60.,40.,40.],
             image_shape_hw=[160,160],resize_factor=1.6,cover=8,absence=0)
    a['score_map'][0,0,0,0]=.8
    a['hann'][0,0,8,8]=.1
    result=selection_metrics(a,row,a)
    assert result['raw_peak']==136 and result['raw_peak_iou']==1
    assert result['fixed_raw_margin']>0 and result['fixed_hann_margin']<0
    changed=deepcopy(a)
    changed['offset_map'][0,:,8,8]=100
    result2=selection_metrics(changed,row,a)
    assert result2['raw_peak_iou']==0
    assert result2['fixed_raw_margin']==result['fixed_raw_margin']


def strength_arrays(family,gain):
    a=proxy_arrays('control' if family=='original' else 'route_target_norm')
    a['reference_alpha']=a['alpha'].copy()
    a['alpha']=a['alpha']*np.float32(gain)
    a['injected_energy']=(a['residual_norm']*a['alpha'])**2
    a['injected_relative_norm']=a['residual_norm']*abs(a['alpha'])/2
    a['search_norm']=np.full_like(a['residual_norm'],2)
    a['injected_residual_dot']=a['alpha']*a['residual_norm']**2
    a['injection_roundoff_max']=np.zeros(1,np.float32)
    if gain==0:
        for k in ('score_map','size_map','offset_map'): a['source_'+k]=a[k].copy()
    return a


def bundle_fixture(path,corrupt=None):
    plan=protocol_plan(True)
    cases,rows,payloads=[],[],{}
    box=[60.,60.,40.,40.]
    for name,(start,stop) in plan.items():
        for frame in range(start,stop+1):
            case=dict(sequence=name,frame_1based=frame,anchor_xywh=box,gt_xywh=box,source_bbox_xywh=box,
                image_shape_hw=[160,160],resize_factor=1.6,cover=8.,absence=0.,input_sha256={'search':'e'*64},
                eligibility={str(s):dict(reason=None,target_tokens=12) for s in SCALES})
            cases.append(case)
            for family,gain,scale in schedule(case['eligibility']):
                member=f'arrays/{name}/{frame}_{family}_{gain}_{scale}.npz'
                payloads[member]=npz_bytes(strength_arrays(family,gain))
                rows.append(dict(case,family=family,strength=gain,roi_scale=scale,
                    mode='control' if family=='original' else 'route_target_norm',scope='shared_residual_strength',
                    arrays_member=member,bbox_xywh=box,iou=1.,source_maps_exact=True,saved_Tclean_exact=True))
    if corrupt: corrupt(cases,rows,payloads)
    provenance={a:dict(checkpoint_sha256='a'*64,state_sha256_before='b'*64,state_sha256_after='b'*64) for a in ARMS}
    manifest=dict(kind='vdrm_module1_step05_strength',version=1,smoke_only=True,plan={k:list(v) for k,v in plan.items()},
        gains=list(GAINS),scales=list(SCALES),optimizer_steps=0,fixed_input_only=True,source_verified=True,
        configurations={a:CONFIGURATIONS[a] for a in ARMS},provenance=provenance,source_provenance=provenance,
        source_sha256='c'*64,forward_calls=len(rows))
    write_bundle(path,manifest,{'cases.json':dump_json(cases),'rows.json':dump_json(rows),**payloads})


def test_complete_bundle(tmp_path):
    p=tmp_path/'good.tar.gz'
    bundle_fixture(p)
    summary,frames=check_bundle(p)
    assert summary['cases']==10 and summary['forward_calls']==160 and len(frames)==160
    assert summary['fixed_input_only'] and not summary['tracking_performance_passed']
    assert len(summary['groups'])==16


def test_same_gain_contrast_uses_location_eligible_population():
    cases=[]
    frames=[]
    for frame,eligible in [(2,True),(3,False)]:
        case=dict(sequence='fixture',frame_1based=frame,gt_xywh=[0,0,10,10],cover=8,absence=0,
            eligibility={str(s):dict(reason=None if eligible else 'insufficient_gt_tokens',target_tokens=4 if eligible else 0) for s in SCALES})
        cases.append(case)
        for family,gain,scale in schedule(case['eligibility']):
            iou=.2 if family=='original' else .4
            if frame==3: iou=.9
            frames.append(dict(sequence='fixture',frame_1based=frame,family=family,strength=gain,roi_scale=scale,iou=iou,raw_peak_iou=iou,peak=0))
    result=summarize(frames,cases,True)
    d=result['groups']['route_target_norm/1.0/4']['per_sequence']['fixture']
    assert d['frames']==1 and d['mean_delta_vs_original_same_gain']==pytest.approx(.2)
    assert d['mean_delta_vs_family_gain_one']==0


@pytest.mark.parametrize('kind',['sign','alpha','missing','gain','support','maps','baseline','input'])
def test_tampered_evidence_rejected(tmp_path,kind):
    def corrupt(cases,rows,payloads):
        if kind=='missing': rows.pop()
        elif kind=='gain': rows[2]['strength']=2
        elif kind=='support': cases[0]['eligibility']=deepcopy(cases[0]['eligibility']);cases[0]['eligibility']['1.0']['target_tokens']=11
        elif kind=='baseline': rows[0]['source_maps_exact']=False
        elif kind=='input': rows[2]['input_sha256']={'search':'x'*64}
        else:
            i=1 if kind=='maps' else 2
            a=strength_arrays('original',rows[i]['strength'])
            if kind=='sign': a['injected_residual_dot']*=-1
            elif kind=='alpha': a['alpha']*=-1
            else: a['source_score_map']+=.01
            payloads[rows[i]['arrays_member']]=npz_bytes(a)
    p=tmp_path/'bad.tar.gz'
    bundle_fixture(p,corrupt)
    with pytest.raises(ValueError): check_bundle(p)


def test_real_count_and_cli_defaults_and_output_protection(tmp_path):
    eligible={str(s):dict(reason=None,target_tokens=4) for s in SCALES}
    excluded={str(s):dict(reason='insufficient_gt_tokens',target_tokens=3) for s in SCALES}
    assert len(schedule(eligible))==16 and len(schedule(excluded))==6
    assert 499*6+(405+396)*5==6999
    for smoke in (False,True):
        with patch.object(sys,'argv',['diagnose_vdrm_module1_strength.py','--save_dir','./output']+(['--smoke'] if smoke else [])),patch.object(entry,'run') as launch:
            entry.main()
        args=launch.call_args.args
        assert args[3]==smoke and Path(args[1]).name=='module1_evidence.tar.gz'
        assert Path(args[2]).name==('module1_strength_smoke.tar.gz' if smoke else 'module1_strength.tar.gz')
    out=tmp_path/'strength.tar.gz'
    out.write_bytes(b'keep')
    with pytest.raises(FileExistsError): entry.run(tmp_path,'missing',out)
    assert out.read_bytes()==b'keep'
