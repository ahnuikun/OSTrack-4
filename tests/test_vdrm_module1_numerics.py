"""Numerical convergence, precision isolation and evidence corruption checks."""

from copy import deepcopy
import io
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pytest
import torch

from lib.test.analysis.frozen_failure_diagnostics import dump_json,write_bundle
from lib.test.analysis.module1_direction import OBJECTIVES,SPACES,choose_pair
from lib.test.analysis.module1_gates import ARMS,CONFIGURATIONS
from lib.test.analysis.module1_numerics import (
    PRECISIONS,EPSILONS,HEAD_KEYS,selected_cases,schedule,requested_update,actual_update,
    finite_difference_diagnostic,convergence,check_bundle)
from tracking.diagnose_vdrm_native_float import npz_bytes
from tracking import diagnose_vdrm_module1_numerics as entry
from tracking.vdrm_diagnostics.module1_numeric_suffix import numeric_backend,activation_signature
from test_vdrm_module1_direction import fixture_prefix


def test_backend_restored_after_error_and_relu_signature_does_not_change_values():
    old=(torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)
    with pytest.raises(RuntimeError,match='sentinel'):
        with numeric_backend('strict32'):
            assert not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32
            raise RuntimeError('sentinel')
    assert old==(torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)
    module=torch.nn.Sequential(torch.nn.ReLU(inplace=True))
    x=torch.tensor([-2.,3.],requires_grad=True)
    with activation_signature(module) as sig:result=module(x*1)
    result.sum().backward()
    torch.testing.assert_close(result,torch.tensor([0.,3.]))
    torch.testing.assert_close(x.grad,torch.tensor([0.,1.]))
    assert sig['relu_modules']==1 and len(sig['relu_sha256'])==64
    assert not module[0]._forward_pre_hooks
    with pytest.raises(ValueError):
        with numeric_backend('half'):pass


def test_convergence_requires_adjacent_steps_and_stable_ce_and_relu():
    rows=[dict(sequence='case',frame_1based=1,precision='float64',objective='raw',space='full',
               epsilon=e,stable_pass=ok) for e,ok in [(1e-6,True),(1e-5,False),(1e-4,True),(.001,False)]]
    assert not convergence(rows)[0]['converged']
    rows[1]['stable_pass']=True
    assert convergence(rows)[0]['adjacent_passing_epsilons']==[(1e-6,1e-5),(1e-5,1e-4)]


def linear_fixture():
    a,p=fixture_prefix()
    p['search_tokens']=np.ones_like(p['reference_delta'])
    p['template_tokens']=np.ones((1,4,8),np.float32)
    for obj in OBJECTIVES:
        p['source_gradient_'+obj]=np.full_like(p['reference_delta'],.01)
        for precision in PRECISIONS:
            p['gradient_'+precision+'_'+obj]=p['source_gradient_'+obj].astype(np.float64 if precision=='float64' else np.float32)
    case=dict(sequence='GOT-10k_Train_003240',frame_1based=93,gt_xywh=[60.,60.,40.,40.],
        anchor_xywh=[60.,60.,40.,40.],resize_factor=1.6,image_shape_hw=[160,160],cover=8,absence=0,
        exclusion_reason=None,input_sha256={'search':'e'*64})
    case['pair'],reason=choose_pair(a,a,p,case)
    assert reason is None
    def forward(delta,precision):
        dtype=np.float64 if precision=='float64' else np.float32
        out={k:a[k].astype(dtype) if k in HEAD_KEYS[:3] else a[k].copy() for k in HEAD_KEYS}
        actual=actual_update(p,delta,precision)
        out['score_map'].reshape(-1)[case['pair'][0]]+=(actual*p['gradient_'+precision+'_raw']).sum()
        return out
    return p,case,forward


def test_float64_derivatives_detect_bad_gradient_and_activation_boundary():
    p,c,f=linear_fixture();precision='float64';eps=.001
    z=f(np.zeros_like(p['search_tokens'],dtype=float),precision)
    plus=f(requested_update(p,'raw','full',eps,precision),precision)
    minus=f(requested_update(p,'raw','full',-eps,precision),precision)
    def diagnose(signatures):return finite_difference_diagnostic(p,c,precision,'raw','full',eps,z,plus,minus,signatures)
    good=diagnose(['a']*3)
    assert good['stable_pass'] and good['relative_error']<1e-8
    assert not diagnose(['a','b','a'])['stable_pass']
    plus['ce_removed'][0,0,1]=True
    assert not diagnose(['a']*3)['stable_pass']
    plus['ce_removed'][0,0,1]=False
    p['gradient_float64_raw']*=2
    bad=diagnose(['a']*3)
    assert not bad['stable_pass'] and np.isclose(bad['relative_error'],.5)


def test_actual_addition_rounding_and_signed_update_norm():
    p,_,_=linear_fixture();p['search_tokens'][:]=1e5
    for precision in PRECISIONS:
        plus=requested_update(p,'raw','prototype',.001,precision)
        minus=requested_update(p,'raw','prototype',-.001,precision)
        np.testing.assert_array_equal(plus,-minus)
        assert np.isclose(np.linalg.norm(plus),.001*np.linalg.norm(p['reference_delta']),rtol=1e-6)
        if precision=='float64':assert actual_update(p,plus,precision).any()
        else:assert not actual_update(p,plus,precision).any()


def bundle_fixture(path,corrupt=None):
    prefix,c,forward=linear_fixture();cases=[c];rows=[];payloads={}
    for precision in PRECISIONS:
        for mode,obj,space,budget in schedule(precision):
            dtype=np.float64 if precision=='float64' else np.float32
            if mode in ('zero','repeat'):delta=np.zeros_like(prefix['search_tokens'],dtype=dtype)
            elif mode=='original':delta=prefix['reference_delta']
            else:delta=requested_update(prefix,obj,space,budget,precision)
            a=forward(delta,precision)
            if precision=='native32' and mode in ('original','zero','oracle'):
                stem=mode if mode!='oracle' else f'oracle_{obj}_{space}_{budget:g}'
                for k in HEAD_KEYS:prefix['source_'+stem+'_'+k]=a[k].copy()
            if precision=='native32' and mode=='fd' and space=='full' and abs(budget)==.001:
                for k in HEAD_KEYS:prefix[f'source_fd_{obj}_{budget:g}_'+k]=a[k].copy()
            member=f'arrays/{precision}_{mode}_{obj}_{space}_{budget}.npz'
            payloads[member]=npz_bytes(a)
            rows.append(dict(sequence=c['sequence'],frame_1based=c['frame_1based'],precision=precision,mode=mode,objective=obj,
                space=space,budget=budget,arrays_member=member,input_sha256=c['input_sha256'],relu_sha256='a'*64,relu_modules=4,
                actual_update_norm=float(np.linalg.norm(actual_update(prefix,delta,precision).astype(float)))))
    source_cases=deepcopy(cases)
    c['prefix_member']='prefix/case.npz';payloads[c['prefix_member']]=npz_bytes(prefix)
    prov={a:dict(checkpoint_sha256='a'*64,state_sha256_before='b'*64,state_sha256_after='b'*64) for a in ARMS}
    m=dict(kind='vdrm_module1_step07_numerics',version=1,smoke_only=True,precisions=list(PRECISIONS),
        epsilons={p:list(e) for p,e in EPSILONS.items()},optimizer_steps=0,source_verified=True,
        configurations={a:CONFIGURATIONS[a] for a in ARMS},source_sha256='c'*64,direction_sha256='d'*64,
        provenance=prov,source_provenance=prov,suffix_states_before={'test':'b'*64},suffix_states_after={'test':'b'*64},
        backend_before={'test':False},backend_after={'test':False},forward_calls=len(rows),backward_calls=6)
    if corrupt:corrupt(m,cases,rows,payloads)
    write_bundle(path,m,{'source_cases.json':dump_json(source_cases),'cases.json':dump_json(cases),'rows.json':dump_json(rows),**payloads})


def test_bundle_checks_real_linear_derivatives_and_no_performance_claim(tmp_path):
    path=tmp_path/'good.tar.gz';bundle_fixture(path)
    s,f,d,g,c=check_bundle(path)
    assert s['cases']==1 and s['forward_calls']==143 and s['backward_calls']==6
    assert s['float64_derivative_audit_passed'] and not s['tracking_performance_passed']
    assert len(d)==56 and len(g)==6 and len(c)==12


@pytest.mark.parametrize('kind',['schedule','input','norm','native','gradient','repeat','dtype','backend','weights','pair'])
def test_corrupt_numeric_evidence_rejected(tmp_path,kind):
    def corrupt(m,cases,rows,payloads):
        if kind=='schedule':rows.pop()
        elif kind=='input':rows[0]['input_sha256']={'search':'f'*64}
        elif kind=='norm':rows[0]['actual_update_norm']*=2
        elif kind=='backend':m['backend_after']={'test':True}
        elif kind=='weights':m['suffix_states_after']={'test':'f'*64}
        elif kind=='pair':cases[0]['pair']=cases[0]['pair'][::-1]
        else:
            if kind=='gradient':member=cases[0]['prefix_member']
            elif kind=='repeat':member=next(r['arrays_member'] for r in rows if r['mode']=='repeat')
            else:member=rows[0]['arrays_member']
            with np.load(io.BytesIO(payloads[member])) as raw:a={k:raw[k] for k in raw.files}
            if kind=='gradient':a['gradient_native32_raw']*=2
            elif kind=='dtype':a['score_map']=a['score_map'].astype(float)
            else:a['score_map']+=.01
            payloads[member]=npz_bytes(a)
    p=tmp_path/'bad.tar.gz';bundle_fixture(p,corrupt)
    with pytest.raises(ValueError):check_bundle(p)


def test_cli_real_paths_smoke_selection_and_no_overwrite(tmp_path):
    for smoke in (False,True):
        with patch.object(sys,'argv',['diagnose_vdrm_module1_numerics.py']+(['--smoke'] if smoke else [])),patch.object(entry,'run') as run:
            entry.main()
        args=run.call_args.args
        assert Path(args[2]).name=='module1_direction.tar.gz' and args[-1]==smoke
        assert Path(args[3]).name==('module1_numerics_smoke.tar.gz' if smoke else 'module1_numerics.tar.gz')
    cases=[dict(sequence='GOT-10k_Train_'+seq,frame_1based=f,exclusion_reason=None) for seq,f in [('003240',88),('003240',93),('008333',61),('008311',93)]]
    assert selected_cases(cases,True)==cases[1:]
    out=tmp_path/'existing.tar.gz';out.write_bytes(b'keep')
    with pytest.raises(FileExistsError):entry.run(tmp_path,'missing','missing',out)
    assert out.read_bytes()==b'keep'
