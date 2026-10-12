"""Isolated weighting, untouched control, split and CLI contracts."""

import sys
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest
import torch

from lib.test.analysis.module1_reweight import frame_weights,smoke_indices,confirmation_ids,group,native_checks,MODES
from lib.test.analysis.module1_learning import sequence_partition
from tracking import train_vdrm_module1_reweight as entry


def samples():
    return ([dict(good_cells=1,protect=False,baseline_iou=.2)]*12+
            [dict(good_cells=1,protect=True,baseline_iou=.8)]*484+
            [dict(good_cells=0,protect=False,baseline_iou=0.)]*16)


def test_exact_group_mass_and_mean_one():
    rows=samples();weights,counts=frame_weights(rows)
    assert counts==dict(supported_failure=12,healthy=484,no_good=16)
    assert np.isclose(np.mean(weights),1.)
    for label,mass in [('supported_failure',.5),('healthy',.4),('no_good',.1)]:
        assert np.isclose(sum(w for r,w in zip(rows,weights) if group(r)==label)/512,mass)
    assert np.isclose(weights[0],21.3333333333333)
    assert np.isclose(weights[-1],3.2)


def test_loss_weight_changes_only_gradient_scale():
    rows=samples();weights,_=frame_weights(rows)
    x=torch.tensor([.2,.4,.6],requires_grad=True)
    loss=x.square().sum();a,=torch.autograd.grad(loss,x,retain_graph=True)
    b,=torch.autograd.grad(loss*weights[0],x)
    torch.testing.assert_close(b,a*weights[0])


def test_smoke_covers_available_classes_and_does_not_fake_absent_failures():
    rows=samples();ids=smoke_indices(rows)
    assert len(ids)==4 and {group(rows[i]) for i in ids}=={'supported_failure','healthy','no_good'}
    healthy=rows[12:16];weights,counts=frame_weights(healthy,True)
    assert weights==[1.]*4 and counts['supported_failure']==0
    with pytest.raises(ValueError):frame_weights(healthy,False)


def test_confirmation_is_disjoint_from_all_prior_pools_and_smoke_is_subset():
    old,forbidden=sequence_partition(False);new=confirmation_ids()
    assert len(new)==16 and len(set(new))==16 and new==sorted(new)
    assert not set(new)&(set(forbidden)|set(old['train'])|set(old['audit']))
    assert confirmation_ids(True)==new[:1]


def test_actual_cli_defaults_use_formal_sources_and_separate_outputs():
    for smoke in (False,True):
        with patch.object(sys,'argv',['train_vdrm_module1_reweight.py','--data_root','/data/train']+(['--smoke'] if smoke else [])),patch.object(entry,'run') as run:
            entry.main()
        args=run.call_args.args
        assert args[1]=='/data/train' and args[2]==Path('output/analysis/module1_step08')
        assert args[3]==Path('output/analysis/module1_step09/module1_fit.tar.gz')
        assert args[-1]==smoke
        assert args[-2]==Path('output/analysis/module1_step10_smoke' if smoke else 'output/analysis/module1_step10')


def test_output_refusal_precedes_cuda_data_or_training(tmp_path):
    with patch.object(torch.cuda,'is_available',side_effect=AssertionError('must not access CUDA')):
        with pytest.raises(FileExistsError):entry.run(tmp_path,None,None,None,None,tmp_path)


def test_control_parity_requires_matching_source_float_state():
    import json
    row=dict(population='audit',sequence='x',frame_1based=2,scope='closed',arm='uniform',
        anchor_xywh=[0,0,10,10],bbox_xywh=[1,1,10,10],input_sha256='x',arrays_member='a')
    old=dict(row,arm='full',bbox_xywh=[2,1,10,10])
    with pytest.raises(RuntimeError,match='trajectory'):
        entry.verify_uniform_control([row],{}, {'rows.json':json.dumps([old]).encode()})


def native_fixture():
    from lib.test.analysis.module1_evidence import decode_candidates
    from tracking.diagnose_vdrm_native_float import npz_bytes
    a=dict(score_map=np.zeros((1,1,16,16),np.float32),size_map=np.full((1,2,16,16),.3,np.float32),
           offset_map=np.full((1,2,16,16),.5,np.float32),hann=np.ones((16,16),np.float32))
    a['score_map'].reshape(-1)[136]=.8
    base=dict(population='audit',sequence='fixture',frame_1based=2,anchor_xywh=[50.,50.,40.,40.],
        resize_factor=4.,image_shape_hw=[200,200],cover=8.,absence=0.,visible=True,input_sha256='input',
        actual_norm_ratio=1.,arrays_member='a')
    box=decode_candidates(a,base,base['image_shape_hw'])[136].tolist()
    base.update(bbox_xywh=box,gt_xywh=box,iou=1.)
    rows=[dict(base,scope=s,arm=arm) for s in ('shared','closed') for arm in MODES]
    plans=[dict(population='audit',sequence='fixture',stop=2,initial_box=base['anchor_xywh'])]
    return rows,plans,{'a':npz_bytes(a)}


def test_native_roundtrip_and_shared_feedback_checks():
    rows,plans,data=native_fixture();assert native_checks(rows,plans,data)==0


@pytest.mark.parametrize('fault',['missing','anchor','input','box','label','iou','norm'])
def test_native_corruptions_are_rejected(fault):
    rows,plans,data=native_fixture()
    if fault=='missing':rows.pop()
    elif fault=='anchor':rows[-1]['anchor_xywh']=[1.,2.,3.,4.]
    elif fault=='input':rows[0]['input_sha256']='different'
    elif fault=='box':rows[0]['bbox_xywh']=[1.,2.,3.,4.]
    elif fault=='label':rows[0]['cover']=0
    elif fault=='iou':rows[0]['iou']=0
    elif fault=='norm':rows[0]['actual_norm_ratio']=float('nan')
    with pytest.raises(ValueError):native_checks(rows,plans,data)
