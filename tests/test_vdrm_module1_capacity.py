"""Width correction budgets, artifact checks and rejection of false successes."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pytest
import torch
from test_vdrm_module1_learnability import prepared_case
from lib.train.module1_learning import ResidualAdapter
from lib.test.analysis.frozen_failure_diagnostics import write_bundle,dump_json
from tracking.diagnose_vdrm_native_float import npz_bytes
from tracking.diagnose_vdrm_frozen_failures import state_hash
from tracking.diagnose_vdrm_module1_learnability import record,fit_run
from tracking.verify_vdrm_module1_capacity import protocol,check_bundle,CONTROLS


def fixture():
    c,suffix,_,_=prepared_case();spec=protocol(True);payloads={};controls=[]
    for arm in CONTROLS:
        delta=c['vectors']['Rdisc' if arm=='width32_task' else arm]
        controls.append(record(c,suffix,torch.from_numpy(delta),arm,0,payloads))
    torch.manual_seed(42);initial_model=ResidualAdapter(width=128);initial=state_hash(initial_model);before=state_hash(suffix)
    initial_member='weights/initial.npz';payloads[initial_member]=npz_bytes({k:v.detach().numpy() for k,v in initial_model.state_dict().items()})
    training,rows=fit_run([c],suffix,'shared_task',spec,payloads,initial)
    payloads[c['c']['cache_member']]=npz_bytes(c['a']);payloads[c['c']['source_vectors_member']]=npz_bytes(c['vectors'])
    payloads.update({'cases.json':dump_json([c['c']]),'rows.json':dump_json(rows),'controls.json':dump_json(controls),'training.json':dump_json(training)})
    m=dict(kind='vdrm_module1_capacity_fix',version=2,initial_weights_member=initial_member,protocol=spec,source_smoke_only=True,source_control_parity=True,
           suffix_before=before,suffix_after=state_hash(suffix),selected_source_case_ids=[c['c']['case_id']])
    return m,payloads


def test_capacity_smoke_cannot_claim_performance(tmp_path):
    m,p=fixture();path=tmp_path/'capacity.tar.gz';write_bundle(path,m,p);s=check_bundle(path)
    assert s['optimizer_updates']==2 and not s['capacity_repair_passed'] and not s['tracking_performance_passed']
    assert s['initial_weights_verification']=='saved_parameters_hash_verified'


def test_legacy_initial_hash_is_explicitly_unverified(tmp_path):
    m,p=fixture();m.pop('version');p.pop(m.pop('initial_weights_member'))
    t=json.loads(p['training.json']);t['initial_sha256']='a'*64;p['training.json']=dump_json(t)
    path=tmp_path/'legacy.tar.gz';write_bundle(path,m,p);s=check_bundle(path)
    assert s['initial_weights_verification']=='unavailable_legacy_hash_only'
    assert not s['tracking_performance_passed']


@pytest.mark.parametrize('fault',['width','loss','budget','iou','missing','weights','updates','selection','frozen','initial','initial_missing'])
def test_capacity_checker_rejects_corruption(tmp_path,fault):
    m,p=fixture();r=json.loads(p['rows.json']);t=json.loads(p['training.json'])
    if fault=='width':m['protocol']['width']=32
    elif fault=='loss':r[-1]['loss']['total']+=.1
    elif fault=='budget':r[-1]['geometry']['norm_ratio']=2.
    elif fault=='iou':r[-1]['hann_iou']+=.1
    elif fault=='missing':r.pop()
    elif fault=='weights':t['final_sha256']='wrong'
    elif fault=='updates':t['updates']=500
    elif fault=='selection':m['selected_source_case_ids']=[]
    elif fault=='frozen':m['suffix_after']='changed'
    elif fault=='initial':t['initial_sha256']='a'*64
    elif fault=='initial_missing':p.pop(m['initial_weights_member'])
    p['rows.json']=dump_json(r);p['training.json']=dump_json(t)
    path=tmp_path/'bad.tar.gz';write_bundle(path,m,p)
    with pytest.raises((ValueError,KeyError)):check_bundle(path)
