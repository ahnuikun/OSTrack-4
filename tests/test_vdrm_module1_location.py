"""Location-proxy semantics, exclusion rules, artifact and CLI contracts."""

from copy import deepcopy
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from test_vdrm_module1_gates import module_fixture, array_fixture
from lib.test.analysis.frozen_failure_diagnostics import dump_json, write_bundle
from lib.test.analysis.module1_gates import ARMS, CONFIGURATIONS, protocol_plan
from lib.test.analysis.module1_location import PROXIES, SCALES, support_mask, validate_proxy_arrays, check_bundle
from tracking import diagnose_vdrm_module1_location as entry
from tracking import analyze_vdrm_module1_location as analysis
from tracking.diagnose_vdrm_native_float import npz_bytes
from tracking.vdrm_diagnostics.module1_location_interventions import location_intervention


@pytest.mark.parametrize('mode',PROXIES)
def test_proxy_formula_and_restoration(mode):
    model,args = module_fixture()
    m = model.backbone.vdrm
    mask = np.array([True]*5+[False]*7)
    with torch.no_grad():
        baseline = m._part_aligned_statistics(*args)
        alpha = m.alpha.clone()
        with location_intervention(model,mode,mask) as values:
            out = m._part_aligned_statistics(*args)
            q = args[2]
            if mode.startswith(('q_','joint_')):
                q = torch.sigmoid(F.softplus(m.log_match_scale)*args[0][...,mask].sort(-1).values[...,-4:].mean(-1)+m.match_bias)*args[3]
            g = baseline[1] if mode.startswith('q_') else torch.tensor(mask)[None,None,:]*args[3][...,None]
            expected = sum(q[0,i]*g[0,i,:,None]*args[1][0,i] for i in range(4))/3
            if mode.endswith('_norm'):
                expected *= baseline[2].norm()/expected.norm()
            torch.testing.assert_close(out[2][0],expected)
            torch.testing.assert_close(out[1],baseline[1],rtol=0,atol=0)
        assert m.inference_ablation is None and '_part_aligned_statistics' not in m.__dict__
        assert torch.equal(alpha,m.alpha)


@pytest.mark.parametrize('count',[0,1,2,3])
def test_proxy_rejects_missing_support_and_restores_on_failure(count):
    model,args = module_fixture()
    mask=np.arange(12)<count
    with pytest.raises(ValueError,match='four supported'):
        with location_intervention(model,'q_target',mask):
            model.backbone.vdrm._part_aligned_statistics(*args)
    assert '_part_aligned_statistics' not in model.backbone.vdrm.__dict__
    assert model.backbone.vdrm.inference_ablation is None


def test_bbox_support_scales_padding_visibility_and_no_parts():
    a = dict(indices=np.array([[102,103,104,105,118,119,120,121,134,135,136,137]]),
             padding=np.zeros((256,256),bool),part_valid=np.ones((1,4),bool))
    row = dict(anchor_xywh=[60.,60.,40.,40.],gt_xywh=[60.,60.,40.,40.],image_shape_hw=[160,160],cover=8,absence=0)
    assert support_mask(a,row,1.)[0].sum()==12
    assert support_mask(a,row,.8)[0].sum()==12
    row['gt_xywh']=[58.,60.,40.,40.]
    assert support_mask(a,row,1.)[0].sum()==12
    assert support_mask(a,row,.8)[0].sum()==9
    row['gt_xywh']=[400,400,20,20]
    assert support_mask(a,row,1.)[1]=='insufficient_gt_tokens'
    row['gt_xywh']=[60.,60.,40.,40.]
    a['padding'][:]=True
    assert support_mask(a,row,1.)[1]=='insufficient_gt_tokens'
    a['padding'][:]=False
    a['part_valid'][:]=False
    assert support_mask(a,row,1.)[1]=='no_valid_template_part'
    row['cover']=0
    assert support_mask(a,row,1.)[1]=='not_visible_or_invalid'


def proxy_arrays(mode):
    a = array_fixture(mode if mode in ('control','alpha0') else 'control')
    model,args = module_fixture()
    m = model.backbone.vdrm
    with torch.no_grad():
        if mode in PROXIES:
            with location_intervention(model,mode,np.ones(12,bool)) as values:
                m._part_aligned_statistics(*args)
                a.update({k:v.numpy().copy() for k,v in values.items()})
        else:
            a.update(similarity=args[0].numpy().copy(),match_scale=F.softplus(m.log_match_scale).numpy().copy(),
                     match_bias=m.match_bias.numpy().copy(),target_mask=np.zeros((1,12),bool))
    a['indices']=np.array([[102,103,104,105,118,119,120,121,134,135,136,137]])
    a['final_kept'][:]=False
    a['final_kept'][0,a['indices'][0]]=True
    a['injected_energy']=(a['residual_norm']*a['alpha'])**2
    a['injected_relative_norm']=a['residual_norm']*abs(a['alpha'])/2
    if mode=='alpha0':
        for key in ('score_map','size_map','offset_map'): a['source_'+key]=a[key].copy()
    return a


@pytest.mark.parametrize('mode',PROXIES)
def test_cpu_formula_checks_and_gate_tamper(mode):
    a = proxy_arrays(mode)
    validate_proxy_arrays(a,mode)
    a['effective_q'][0,0] += .03
    with pytest.raises(ValueError,match='declared formula'): validate_proxy_arrays(a,mode)


def bundle_fixture(path,corrupt=None):
    plan=protocol_plan(True)
    cases,rows,payloads=[],[],{}
    packed={m:npz_bytes(proxy_arrays(m)) for m in ('control','alpha0',*PROXIES)}
    box=[60.,60.,40.,40.]
    for name,(start,stop) in plan.items():
        for frame in range(start,stop+1):
            case=dict(sequence=name,frame_1based=frame,anchor_xywh=box,gt_xywh=box,source_bbox_xywh=box,
                image_shape_hw=[160,160],resize_factor=1.6,cover=8.,absence=0.,input_sha256={'search':'e'*64},
                eligibility={str(s):dict(reason=None,target_tokens=12) for s in SCALES})
            cases.append(case)
            schedule=[('control',None),('alpha0',None)]+[(m,s) for s in SCALES for m in PROXIES]
            for mode,scale in schedule:
                member=f'arrays/{name}/{frame}_{mode}_{scale}.npz'
                payloads[member]=packed[mode]
                rows.append(dict(case,mode=mode,roi_scale=scale,scope='shared_GT_location_proxy',arrays_member=member,
                                 bbox_xywh=box,iou=1.,source_maps_exact=True,saved_Tclean_exact=True))
    if corrupt: corrupt(cases,rows,payloads)
    provenance={a:dict(checkpoint_sha256='a'*64,state_sha256_before='b'*64,state_sha256_after='b'*64) for a in ARMS}
    manifest=dict(kind='vdrm_module1_step04_location',version=1,smoke_only=True,plan={k:list(v) for k,v in plan.items()},
        proxies=list(PROXIES),scales=list(SCALES),optimizer_steps=0,gt_location_proxy_only=True,source_verified=True,
        configurations={a:CONFIGURATIONS[a] for a in ARMS},provenance=provenance,source_provenance=provenance,
        source_sha256='c'*64,forward_calls=len(rows))
    write_bundle(path,manifest,{'cases.json':dump_json(cases),'rows.json':dump_json(rows),**payloads})


def test_actual_cpu_entry_and_output_protection(tmp_path):
    source,output=tmp_path/'source.tar.gz',tmp_path/'analysis.tar.gz'
    bundle_fixture(source)
    with patch.object(sys,'argv',['analyze_vdrm_module1_location.py','--bundle',str(source),'--output',str(output)]):
        analysis.main()
    summary,_=check_bundle(source)
    assert summary['forward_calls']==140 and summary['cases']==10
    assert summary['scales']['1.0']['eligible_frames']==10
    assert summary['gt_location_proxy_only'] and not summary['true_part_visibility_verified']
    previous=output.read_bytes()
    with pytest.raises(FileExistsError): analysis.run(source,output)
    assert previous==output.read_bytes()


@pytest.mark.parametrize('kind',['missing','eligibility','mask','alpha','baseline'])
def test_invalid_artifacts_rejected(tmp_path,kind):
    def corrupt(cases,rows,payloads):
        if kind=='missing': rows.pop()
        elif kind=='eligibility': cases[0]['eligibility']={'1.0':dict(reason=None,target_tokens=11),'0.8':dict(reason=None,target_tokens=12)}
        elif kind=='baseline': rows[0]['source_maps_exact']=False
        else:
            a=proxy_arrays('q_target')
            if kind=='mask': a['target_mask'][0,0]=False
            else: a['alpha'] *= -1
            payloads[rows[2]['arrays_member']]=npz_bytes(a)
    path=tmp_path/'bad.tar.gz'
    bundle_fixture(path,corrupt)
    with pytest.raises(ValueError): check_bundle(path)


def test_diagnostic_cli_defaults():
    for smoke in (False,True):
        with patch.object(sys,'argv',['diagnose_vdrm_module1_location.py','--save_dir','./output']+(['--smoke'] if smoke else [])), \
                patch.object(entry,'run') as launch:
            entry.main()
        args=launch.call_args.args
        assert args[3]==smoke
        assert Path(args[1]).as_posix()=='output/analysis/module1_step01/module1_evidence.tar.gz'
        assert Path(args[2]).name==('module1_location_smoke.tar.gz' if smoke else 'module1_location.tar.gz')
