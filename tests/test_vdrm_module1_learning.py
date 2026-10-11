"""Paired residual geometry, loss gradients, split isolation and output contracts."""

from copy import deepcopy
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pytest
import torch

from lib.train.module1_learning import ResidualAdapter,adapter_insertion,insertion_features,dense_boxes,learning_loss
from lib.test.analysis.module1_learning import protocol,sequence_partition,training_frames,MODES,check_bundle
from lib.test.analysis.module1_gates import protocol_plan
from lib.test.analysis.frozen_failure_diagnostics import write_bundle,dump_json
from tracking.diagnose_vdrm_frozen_failures import state_hash
from tracking.diagnose_vdrm_native_float import npz_bytes
from tracking import train_vdrm_module1_learning as entry
from test_vdrm_module1_direction import fixture_module,fixture_prefix


def test_equal_parameters_zero_identity_and_projected_vs_full_directions():
    torch.manual_seed(8)
    proto=ResidualAdapter(8,4,'prototype');full=ResidualAdapter(8,4,'full');full.load_state_dict(proto.state_dict())
    x=torch.randn(1,12,8);p=torch.eye(4,8)[None];valid=torch.tensor([[1,1,0,0]],dtype=torch.bool)
    base=torch.zeros_like(x);base[...,0]=.03;mix=torch.randn_like(x)
    assert sum(v.numel() for v in proto.parameters())==sum(v.numel() for v in full.parameters())
    for model in (proto,full):torch.testing.assert_close(model(x,p,valid,mix,base),base,atol=0,rtol=0)
    with torch.no_grad():
        proto.value.weight.normal_();full.load_state_dict(proto.state_dict())
    a,b=proto(x,p,valid,mix,base),full(x,p,valid,mix,base)
    assert a[...,2:].abs().max()<1e-7 and b[...,2:].abs().max()>1e-5
    assert a.norm()<=base.norm()*1.00001 and b.norm()<=base.norm()*1.00001
    torch.testing.assert_close(proto(x,p,torch.zeros_like(valid),mix,base),base)
    assert not proto(x,p,valid,mix,torch.zeros_like(base)).any()
    a.sum().backward();assert proto.value.weight.grad.abs().sum()>0


def test_real_insertion_hook_matches_direct_adapter_and_restores_after_error():
    model,tokens,ref,kwargs=fixture_module();module=model.backbone.vdrm;before=state_hash(module)
    adapter=ResidualAdapter(8,4,'full').eval()
    with torch.no_grad():
        original=module(tokens,**kwargs)
        with adapter_insertion(model,adapter):zero=module(tokens,**kwargs)
        torch.testing.assert_close(zero[0],original[0],atol=0,rtol=0)
        adapter.value.weight.fill_(.4)
        search,p,valid,mix,delta=insertion_features(module,tokens,kwargs,original)
        expected=search+adapter(search,p,valid,mix,delta)
        with adapter_insertion(model,adapter) as values:out=module(tokens,**kwargs)
        torch.testing.assert_close(out[0][:,4:],expected,atol=0,rtol=0)
        torch.testing.assert_close(out[0][:,:4],tokens[:,:4],atol=0,rtol=0)
        assert values['actual_norm_ratio']<=1.005
    with pytest.raises(RuntimeError,match='sentinel'):
        with adapter_insertion(model,adapter):raise RuntimeError('sentinel')
    assert not module._forward_hooks and state_hash(module)==before


def test_regression_decode_and_loss_really_propagate_to_size_and_offset():
    score=torch.full((1,1,4,4),.4,requires_grad=True)
    size=torch.full((1,2,4,4),.3,requires_grad=True)
    offset=torch.full((1,2,4,4),.5,requires_grad=True)
    out=dict(score_map=score,size_map=size,offset_map=offset);base={k:v.detach().clone() for k,v in out.items()}
    good=torch.zeros(16,dtype=torch.bool);good[10]=True
    loss,parts=learning_loss(out,base,good,torch.tensor([.35,.35,.35,.35]),torch.ones_like(score),True)
    assert dense_boxes(size,offset).requires_grad and parts['localization']>0
    loss.backward()
    assert size.grad.abs().sum()>0 and offset.grad.abs().sum()>0
    assert score.grad.flatten()[10]<0 and (score.grad.flatten()[~good]>0).all()


def test_no_good_box_uses_preservation_only_not_fake_positive():
    out={k:torch.full(shape,.4,requires_grad=True) for k,shape in [('score_map',(1,1,4,4)),('size_map',(1,2,4,4)),('offset_map',(1,2,4,4))]}
    base={k:torch.zeros_like(v) for k,v in out.items()}
    loss,parts=learning_loss(out,base,torch.zeros(16,dtype=torch.bool),torch.zeros(4),torch.ones_like(out['score_map']),False)
    assert parts['rank']==0 and parts['localization']==0 and parts['preservation']>0
    loss.backward();assert all(v.grad.abs().sum()>0 for v in out.values())


def test_partition_and_visibility_sampling_are_fixed_without_audit_tuning():
    formal,excluded=sequence_partition(False);smoke,_=sequence_partition(True)
    assert len(formal['train'])==64 and len(formal['audit'])==16 and len(excluded)==856
    assert not set(formal['train'])&set(formal['audit'])
    assert not (set(formal['train'])|set(formal['audit']))&set(excluded)
    for split in formal:assert set(smoke[split])<=set(formal[split])
    cover=np.full(30,8);cover[5:8]=2;absence=np.zeros(30)
    frames=training_frames(cover,absence,30,8)
    assert len(frames)==8 and frames==sorted(set(frames)) and all(2<=f<=30 for f in frames)
    assert set(range(6,9))<=set(frames)
    assert training_frames([8,8],[0,0],2,8)==[2]


def bundle_fixture(path,corrupt=None):
    spec=protocol(True);partition,excluded=sequence_partition(True)
    a,_=fixture_prefix();a['hann']=np.ones_like(a['score_map'])
    arrays={k:a[k] for k in ('score_map','size_map','offset_map','hann')}
    box=[60.,60.,40.,40.]
    plans=[dict(population='audit',sequence='audit',sequence_id=partition['audit'][0],stop=2,initial_box=box)]
    plans += [dict(population='regression',sequence=n,sequence_id=None,stop=b[1],initial_box=box) for n,b in protocol_plan(True).items()]
    rows=[];payloads={}
    for plan in plans:
        for f in range(2,plan['stop']+1):
            for scope in ('shared','closed'):
                for arm in MODES:
                    member=f'arrays/{plan["sequence"]}_{f}_{scope}_{arm}.npz';payloads[member]=npz_bytes(arrays)
                    rows.append(dict(population=plan['population'],sequence=plan['sequence'],frame_1based=f,arm=arm,scope=scope,
                        anchor_xywh=box,bbox_xywh=box,gt_xywh=box,cover=8.,absence=0.,visible=True,iou=1.,
                        input_sha256={'search':'a'*64},actual_norm_ratio=1.,arrays_member=member,resize_factor=1.6,image_shape_hw=[160,160]))
    model=ResidualAdapter()
    with torch.no_grad():model.value.weight.fill_(.1)
    final=state_hash(model);training={}
    for arm in ('prototype','full'):
        payloads['weights_'+arm+'.npz']=npz_bytes({k:v.detach().numpy() for k,v in model.state_dict().items()})
        training[arm]=dict(initial_sha256='a'*64,final_sha256=final,sample_order_sha256='b'*64,parameter_count=73728,
            updates=4,train_cache_sha256='c'*64,frozen_suffix_before='d'*64,frozen_suffix_after='d'*64,
            maximum_update_ratio=1.,epochs=[dict(epoch=1,total=1.)])
    index=dict(rows=[dict(population='train',sequence_id=i) for i in partition['train'] for _ in range(2)])
    manifest=dict(kind='vdrm_module1_step08_learning',version=1,protocol=spec,partition=partition,forbidden_ids=excluded,
        checkpoint_selection='fixed_last_epoch',production_model_modified=False,automatic_performance_pass=False,
        provenance={'Rdisc':dict(state_sha256_before='e'*64,state_sha256_after='e'*64)},train_frames=4,native_preflight_passed=True)
    if corrupt:corrupt(manifest,rows,plans,training,index,payloads)
    payloads.update({'rows.json':dump_json(rows),'evaluation_plan.json':dump_json(plans),'training.json':dump_json(training),'train_index.json':dump_json(index),'visual_cases.json':dump_json([])})
    write_bundle(path,manifest,payloads)


def test_cpu_bundle_recomputes_native_decode_and_metrics(tmp_path):
    path=tmp_path/'good.tar.gz';bundle_fixture(path)
    s,rows,t=check_bundle(path)
    assert s['train_frames']==4 and s['training_updates_per_arm']==4 and s['max_decode_error_pixels']==0
    assert not s['tracking_performance_passed'] and not s['independent_benchmark']


@pytest.mark.parametrize('kind',['missing','train_leak','budget','initial','order','frozen','closed','shared','iou','weights','audit_partition'])
def test_corrupt_learning_evidence_rejected(tmp_path,kind):
    def corrupt(m,rows,plans,t,index,payloads):
        if kind=='missing':rows.pop()
        elif kind=='train_leak':index['rows'][0]['sequence_id']=m['partition']['audit'][0]
        elif kind=='budget':rows[0]['actual_norm_ratio']=2.
        elif kind=='initial':t['full']['initial_sha256']='f'*64
        elif kind=='order':t['full']['sample_order_sha256']='f'*64
        elif kind=='frozen':t['full']['frozen_suffix_after']='f'*64
        elif kind=='closed':next(r for r in rows if r['scope']=='closed')['anchor_xywh']=[1.,1.,40.,40.]
        elif kind=='shared':next(r for r in rows if r['scope']=='shared')['input_sha256']={'search':'f'*64}
        elif kind=='iou':rows[0]['iou']=.1
        elif kind=='weights':payloads['weights_full.npz']=npz_bytes({'wrong':np.zeros(1)})
        else:plans[0]['sequence_id']=m['partition']['train'][0]
    p=tmp_path/'bad.tar.gz';bundle_fixture(p,corrupt)
    with pytest.raises(ValueError):check_bundle(p)


def test_cli_paths_and_overwrite_protection(tmp_path):
    for smoke in (True,False):
        with patch.object(sys,'argv',['train_vdrm_module1_learning.py','--data_root','/data/train']+(['--smoke'] if smoke else [])),patch.object(entry,'run') as run:
            entry.main()
        args=run.call_args.args
        assert Path(args[3]).name=='module1_numerics.tar.gz' and args[-1]==smoke
        assert Path(args[4]).name==('module1_step08_smoke' if smoke else 'module1_step08')
    with pytest.raises(FileExistsError):entry.run('missing','missing','missing','missing',tmp_path)
