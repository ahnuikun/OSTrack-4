"""One targeted correction: width 128 shared task fitting versus saved width 32.

Reuses Step11 inputs and its completed control. No teacher training, parameter
sweep, new sequence replay, or production checkpoint replacement. Passing the
four training frames establishes memorization only, not tracking robustness.
"""
import argparse
import json
import re
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
import numpy as np
import torch
from lib.train.module1_learning import ResidualAdapter
from lib.train.frozen_vdrm import file_hash, visual_hash
from lib.test.analysis.module1_numerics import read_archive, load_np
from lib.test.analysis.module1_learnability import check_bundle as check_source, protocol as old_protocol, teacher_error
from lib.test.analysis.module1_fit_audit import HEADS, geometry, head_loss
from lib.test.analysis.module1_direction import decode_metrics
from lib.test.analysis.frozen_failure_diagnostics import CONFIGS, dump_json, write_bundle
from tracking.diagnose_vdrm_frozen_failures import load_models, state_hash
from tracking.diagnose_vdrm_module1_numerics import backend_state
from tracking.diagnose_vdrm_native_float import npz_bytes
from tracking.diagnose_vdrm_module1_learnability import prepare, record, fit_run, adapter_delta
from tracking.vdrm_diagnostics.module1_numeric_suffix import CachedSuffix

CONTROLS = ('Rdisc', 'full', 'oracle_full', 'width32_task')


def protocol(smoke=False):
    return dict(old_protocol(smoke), width=128, changed_factor='shared_adapter_width_32_to_128',
                objective='original_task_loss_only', control='saved_step11_shared_task')


def check_bundle(path):
    m, d = read_archive(path); spec = m['protocol']
    if m['kind'] != 'vdrm_module1_capacity_fix' or spec != protocol(spec['smoke_only']):
        raise ValueError('wrong capacity correction protocol')
    if m['suffix_before'] != m['suffix_after'] or m['source_control_parity'] is not True:
        raise ValueError('frozen suffix or control replay changed')
    cases = json.loads(d['cases.json']); rows = json.loads(d['rows.json']); controls = json.loads(d['controls.json'])
    t = json.loads(d['training.json']); ids = [c['case_id'] for c in cases]
    if len(set(ids)) != len(ids) or not ids or any(c['population'] != 'train' for c in cases):
        raise ValueError('invalid training selection')
    if ids != m['selected_source_case_ids'] or not spec['smoke_only'] and (m['source_smoke_only'] or len(ids) != 4):
        raise ValueError('formal selection requires the four Step11 training cases')
    expected = {(i, s) for i in ids for s in spec['checkpoints']}
    if len(rows) != len(expected) or {(r['case_id'], r['step']) for r in rows} != expected or any(r['arm'] != 'shared_task' for r in rows):
        raise ValueError('incomplete capacity curve')
    if len(controls) != len(ids)*4 or {(r['case_id'], r['arm']) for r in controls} != {(i,a) for i in ids for a in CONTROLS}:
        raise ValueError('incomplete controls')
    if (t['case_ids'] != ids or t['updates'] != spec['steps'] or t['frame_exposures'] != len(ids)*spec['steps']
            or t['parameter_count'] != 3*768*128 or len(t['trace']) != spec['steps']):
        raise ValueError('changed training budget')
    # A seed is not a portable serialization of initial parameters. Historical
    # v1 archives only recorded their hash; never claim independent initial-
    # weight verification by rebuilding a random model on a different runtime.
    version=m.get('version',1)
    if version not in (1,2): raise ValueError('unsupported capacity evidence version')
    if not re.fullmatch('[0-9a-f]{64}',t['initial_sha256']): raise ValueError('invalid initial hash')
    if version==2:
        initial_arrays=load_np(d[m['initial_weights_member']])
        with torch.random.fork_rng(devices=[]):
            probe=ResidualAdapter(width=128)
        probe.load_state_dict({k:torch.from_numpy(v) for k,v in initial_arrays.items()},strict=True)
        if any(not np.isfinite(v).all() for v in initial_arrays.values()) or initial_arrays['value.weight'].any():
            raise ValueError('invalid zero-correction initialization')
        if state_hash(probe)!=t['initial_sha256']: raise ValueError('saved initial weights mismatch')
        initial_verification='saved_parameters_hash_verified'
    else:
        initial_verification='unavailable_legacy_hash_only'
    weights = load_np(d[t['weights_member']])
    final = visual_hash({'probe.'+k:torch.from_numpy(v) for k,v in weights.items()})
    if final != t['final_sha256'] or final == t['initial_sha256']:
        raise ValueError('wrong final weights or no learning')
    for step, tr in enumerate(t['trace'], 1):
        if tr['step'] != step or any(not np.isfinite(v) or v < 0 for k,v in tr.items() if k!='step'):
            raise ValueError('invalid optimizer trace')
    allowed = {'manifest.json','cases.json','rows.json','controls.json','training.json',t['weights_member']}
    if version==2: allowed.add(m['initial_weights_member'])
    ci = {c['case_id']:c for c in cases}
    for c in cases: allowed.update((c['cache_member'],c['source_vectors_member']))
    bases = {i:load_np(d[ci[i]['cache_member']]) for i in ids}
    vectors = {i:load_np(d[ci[i]['source_vectors_member']]) for i in ids}
    for r in rows+controls:
        c=ci[r['case_id']]; a=load_np(d[r['arrays_member']]); base=bases[r['case_id']]
        allowed.add(r['arrays_member'])
        if any(not np.isfinite(v).all() for v in a.values()): raise ValueError('nonfinite output')
        metrics,_=decode_metrics(a,c,base['hann']); loss=head_loss(a,base,base['good'],base['target'],base['hann'],c['protect'])
        for k,v in metrics.items():
            if (v is None)!=(r[k] is None) or v is not None and not np.isclose(v,r[k],atol=1e-7): raise ValueError('wrong decode')
        if any(not np.isclose(v,r['loss'][k],atol=2e-5,rtol=1e-5) for k,v in loss.items()): raise ValueError('wrong loss')
        g=geometry(a['actual_delta'],base['delta'])
        if g['norm_ratio']>1.005 or any(not np.isclose(v,r['geometry'][k],atol=1e-7) for k,v in g.items()): raise ValueError('wrong norm geometry')
        error=teacher_error(a['actual_delta'],vectors[r['case_id']]['oracle_full'],base['delta'])
        if not np.isclose(error,r['teacher_error'],atol=1e-7): raise ValueError('wrong teacher diagnostic')
        if r['arm']=='Rdisc' or r['arm']=='shared_task' and r['step']==0:
            if any(not np.array_equal(a[k],base[k]) for k in HEADS): raise ValueError('native identity failed')
    if set(d)!=allowed: raise ValueError('missing or extra evidence')
    last=[r for r in rows if r['step']==spec['steps']]
    repaired=sum(ci[r['case_id']]['baseline_iou']<.5<=r['hann_iou'] for r in last)
    return dict(smoke_only=spec['smoke_only'],cases=len(cases),optimizer_updates=t['updates'],
                initial_weights_verification=initial_verification,
                capacity_repair_passed=not spec['smoke_only'] and repaired==len(cases),
                tracking_performance_passed=False,repairs=repaired,
                mean_iou=float(np.mean([r['hann_iou'] for r in last])),
                per_case=[dict(case_id=r['case_id'],iou=r['hann_iou'],
                              width32_iou=next(x['hann_iou'] for x in controls if x['case_id']==r['case_id'] and x['arm']=='width32_task')) for r in last])


def run(save_dir, source, output, smoke=False):
    output=Path(output)
    if output.exists(): raise FileExistsError('refusing to overwrite capacity correction')
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    print('Checking Step11 evidence and checkpoint identity...',flush=True)
    check_source(source); old,data=read_archive(source)
    if not smoke and old['protocol']['smoke_only']: raise ValueError('formal correction requires formal Step11')
    if backend_state()!=old['backend']: raise ValueError('backend differs from Step11')
    cases=json.loads(data['cases.json']); cases=cases[:2] if smoke else cases
    if not smoke and len(cases)!=4: raise ValueError('expected four formal source training cases')
    paths={a:Path(save_dir)/'checkpoints/train/ostrack'/CONFIGS[a]/'OSTrack_ep0300.pth.tar' for a in ('Tclean','Rdisc')}
    models,_,provenance=load_models(paths,'cuda',arms=('Tclean','Rdisc'))
    for arm in models:
        for k in ('checkpoint_sha256','state_sha256_before'):
            if provenance[arm][k]!=old['provenance'][arm][k]: raise ValueError('checkpoint mismatch')
    suffix=CachedSuffix(models['Rdisc']).cuda();del models
    before=state_hash(suffix)
    if before!=old['suffix_before']: raise ValueError('suffix mismatch')
    spec=protocol(smoke);payloads={};prepared=[];controls=[]
    old_rows={(r['case_id'],r['arm'],r['step']):r for r in json.loads(data['rows.json'])}
    old_training=next(t for t in json.loads(data['training.json']) if t['run_id']=='shared_task')
    control=ResidualAdapter(width=32).cuda().eval().requires_grad_(False)
    control.load_state_dict({k:torch.from_numpy(v).cuda() for k,v in load_np(data[old_training['weights_member']]).items()})
    for c in cases:
        a=load_np(data[c['cache_member']]);v=load_np(data[c['source_vectors_member']]);p=prepare(a,c,v,'cuda');prepared.append(p)
        for key in ('cache_member','source_vectors_member'): payloads[c[key]]=data[c[key]]
        for arm in CONTROLS:
            if arm=='width32_task':
                delta=adapter_delta(control,p); previous=old_rows[c['case_id'],'shared_task',old['protocol']['steps']]
            else:
                delta=torch.from_numpy(v[arm]).cuda();previous=old_rows[c['case_id'],arm,0]
            controls.append(record(p,suffix,delta,arm,0,payloads,load_np(data[previous['arrays_member']])))
    torch.manual_seed(spec['seed']);initial_model=ResidualAdapter(width=128);initial=state_hash(initial_model)
    initial_member='weights/initial.npz'
    payloads[initial_member]=npz_bytes({k:v.detach().numpy() for k,v in initial_model.state_dict().items()})
    print(f'CAPACITY FIX PRECHECK PASSED: {len(cases)} frames; one width128 task fit; {spec["steps"]} updates',flush=True)
    training,rows=fit_run(prepared,suffix,'shared_task',spec,payloads,initial)
    after=state_hash(suffix)
    if before!=after or any(p.grad is not None or p.requires_grad for p in suffix.parameters()): raise RuntimeError('frozen suffix changed')
    m=dict(kind='vdrm_module1_capacity_fix',version=2,initial_weights_member=initial_member,
           initializer_runtime=dict(torch_version=str(torch.__version__),cpu_capability=torch.backends.cpu.get_cpu_capability()),
           protocol=spec,source_sha256=file_hash(source),source_smoke_only=old['protocol']['smoke_only'],
           selected_source_case_ids=[c['case_id'] for c in cases],source_control_parity=True,suffix_before=before,suffix_after=after,
           provenance=provenance,backend=backend_state())
    payloads.update({'cases.json':dump_json(cases),'rows.json':dump_json(rows),'controls.json':dump_json(controls),'training.json':dump_json(training)})
    write_bundle(output,m,payloads);summary=check_bundle(output)
    print(json.dumps(summary,indent=2),flush=True)
    print('CAPACITY EVIDENCE CHECK PASSED; training repair is NOT a tracking-performance pass',flush=True)
    print('Download:',output.resolve(),flush=True)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--save_dir',default='./output')
    p.add_argument('--source');p.add_argument('--output');p.add_argument('--smoke',action='store_true')
    args=p.parse_args();root=Path(args.save_dir)/'analysis';torch.set_num_threads(2)
    run(args.save_dir,args.source or root/'module1_step11/module1_learnability.tar.gz',
        args.output or root/('module1_capacity_fix_smoke' if args.smoke else 'module1_capacity_fix')/'module1_capacity.tar.gz',args.smoke)


if __name__=='__main__': main()
