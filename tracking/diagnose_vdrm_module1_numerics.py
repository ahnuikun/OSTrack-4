"""Step07: audit Step06 derivatives at multiple steps and precisions.

No training or tracker change. Replay real frozen suffix layers on the exact
cached prefix; require native output AND gradient parity before diagnostics.
"""

import argparse
import io
import json
from pathlib import Path
import platform
import sys
import tarfile

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))

import numpy as np
import torch

from lib.test.analysis.frozen_failure_diagnostics import CONFIGS,dump_json,write_bundle
from lib.test.analysis.module1_gates import ARMS,protocol_plan,read_source
from lib.test.analysis.module1_direction import check_bundle as check_direction,OBJECTIVES
from lib.test.analysis.module1_numerics import (
    PRECISIONS,EPSILONS,HEAD_KEYS,selected_cases,schedule,requested_update,actual_update,
    read_archive,load_np,check_bundle,gradient_relative_error,NATIVE_GRADIENT_RTOL)
from tracking.diagnose_vdrm_frozen_failures import config_for,load_models,state_hash
from tracking.diagnose_vdrm_module1_location import cached_inputs,selected_records
from tracking.diagnose_vdrm_native_float import input_hashes,npz_bytes
from tracking.vdrm_diagnostics.module1_numeric_suffix import CachedSuffix,numeric_backend,activation_signature


def backend_state():
    return dict(matmul_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_tf32=torch.backends.cudnn.allow_tf32,
                cudnn_benchmark=torch.backends.cudnn.benchmark,cudnn_deterministic=torch.backends.cudnn.deterministic)


def collect(source_path,source,direction_manifest,direction_data,models,cfg,smoke):
    source_cases=json.loads(direction_data['cases.json'])
    chosen=selected_cases(source_cases,smoke)
    if not chosen:raise ValueError('Step06 has no eligible cases')
    by_key={(c['sequence'],c['frame_1based']):c for c in chosen}
    selected={k:c for k,c in selected_records(source,protocol_plan(direction_manifest['smoke_only'])).items()
              if (c['sequence'],c['frame_1based']) in by_key}
    if len(selected)!=len(chosen):raise ValueError('source lacks selected numeric cases')
    old_rows={(r['sequence'],r['frame_1based'],r['mode'],r['objective'],r['space'],r['budget']):r
              for r in json.loads(direction_data['rows.json'])}
    model=models['Rdisc'];device=next(model.parameters()).device
    suffix32=CachedSuffix(model).to(device)
    suffix64=CachedSuffix(model).to(device=device,dtype=torch.float64)
    suffixes={'native32':suffix32,'strict32':suffix32,'float64':suffix64}
    before={p:state_hash(s) for p,s in suffixes.items()}
    prefixes={}
    # One sequential read, not one decompression per selected frame.
    with tarfile.open(source_path,'r|gz') as t:
        for member in t:
            if member.name not in selected:continue
            a=load_np(t.extractfile(member).read());record=selected[member.name]
            key=(record['sequence'],record['frame_1based']);case=by_key[key]
            for k,v in record.items():
                if case[k]!=v:raise ValueError('Step01/Step06 metadata mismatch: '+k)
            inputs=cached_inputs(a,cfg,device)
            if input_hashes(inputs)!=case['input_sha256']:raise ValueError('cached input mismatch')
            p=load_np(direction_data[case['prefix_member']])
            for k in ('prototypes','part_valid','indices','hann','padding'):
                if not np.array_equal(a[k],p[k]):raise ValueError('Step01/Step06 prefix mismatch: '+k)
            if not np.array_equal(a['delta'],p['reference_delta']):raise ValueError('reference update mismatch')
            prefix={k:p[k] for k in ('prototypes','part_valid','indices','hann','reference_delta')}
            prefix.update({k:a[k] for k in ('template_tokens','search_tokens')})
            prefix['ce_template_mask']=inputs['ce_template_mask'].cpu().numpy().copy()
            for obj in OBJECTIVES:prefix['source_gradient_'+obj]=p['gradient_'+obj]
            for mode,obj,space,budget in schedule('native32'):
                if mode=='repeat' or (mode=='fd' and (space!='full' or abs(budget)!=.001)):continue
                old_mode='finite_difference' if mode=='fd' else mode
                old_budget=None if mode in ('original','zero') else budget
                r=old_rows[key[0],key[1],old_mode,obj,space,old_budget]
                a_old=load_np(direction_data[r['arrays_member']])
                stem=mode if mode in ('original','zero') else (f'oracle_{obj}_{space}_{budget:g}' if mode=='oracle' else f'fd_{obj}_{budget:g}')
                for k in HEAD_KEYS:prefix['source_'+stem+'_'+k]=a_old[k]
            prefixes[key]=prefix
    cases,rows,payloads=[],[],{}
    calls=backwards=0
    for chosen_case in chosen:
        case=dict(chosen_case);key=(case['sequence'],case['frame_1based']);prefix=prefixes.pop(key)
        for precision in PRECISIONS:
            suffix=suffixes[precision];dtype=torch.float64 if precision=='float64' else torch.float32
            template=torch.from_numpy(prefix['template_tokens']).to(device=device,dtype=dtype)
            search=torch.from_numpy(prefix['search_tokens']).to(device=device,dtype=dtype)
            indices=torch.from_numpy(prefix['indices']).to(device)
            mask=torch.from_numpy(prefix['ce_template_mask']).to(device)
            hann=torch.from_numpy(prefix['hann']).to(device=device,dtype=dtype)
            with numeric_backend(precision):
                for mode,obj,space,budget in schedule(precision):
                    if mode in ('zero','repeat'):delta=np.zeros_like(prefix['search_tokens'],dtype=np.float64 if precision=='float64' else np.float32)
                    elif mode=='original':delta=prefix['reference_delta']
                    else:delta=requested_update(prefix,obj,space,budget,precision)
                    leaf=torch.from_numpy(delta).to(device=device,dtype=dtype).requires_grad_(mode=='zero')
                    with torch.set_grad_enabled(mode=='zero'),activation_signature(suffix.box_head,score_branch=True) as signature:
                        out=suffix(template,search,indices,mask,leaf)
                    arrays={k:out[k].detach().cpu().numpy().copy() for k in HEAD_KEYS}
                    actual=((search+leaf)-search).detach().cpu().numpy()
                    if not np.array_equal(actual,actual_update(prefix,delta,precision)):
                        raise RuntimeError('actual suffix update differs from CPU arithmetic')
                    if precision=='native32' and mode not in ('repeat','fd'):
                        stem=mode if mode in ('zero','original') else f'oracle_{obj}_{space}_{budget:g}'
                        for k in HEAD_KEYS:
                            if not np.array_equal(arrays[k],prefix['source_'+stem+'_'+k]):
                                raise RuntimeError('native suffix parity failed: '+str(key)+' '+stem+' '+k)
                    if precision=='native32' and mode=='fd' and space=='full' and abs(budget)==.001:
                        for k in HEAD_KEYS:
                            if not np.array_equal(arrays[k],prefix[f'source_fd_{obj}_{budget:g}_'+k]):
                                raise RuntimeError('native finite difference replay mismatch: '+str(key)+' '+k)
                    name=f'arrays/{key[0]}_{key[1]}_{precision}_{mode}_{obj}_{space}_{budget}.npz'
                    payloads[name]=npz_bytes(arrays)
                    rows.append(dict(sequence=key[0],frame_1based=key[1],precision=precision,mode=mode,objective=obj,
                        space=space,budget=budget,arrays_member=name,input_sha256=case['input_sha256'],
                        actual_update_norm=float(np.linalg.norm(actual.astype(float))),**signature))
                    calls+=1
                    if mode=='zero':
                        for j,objective in enumerate(OBJECTIVES):
                            score=out['score_map'] if objective=='raw' else out['score_map']*hann
                            score=score.reshape(-1);margin=score[case['pair'][0]]-score[case['pair'][1]]
                            g,=torch.autograd.grad(margin,leaf,retain_graph=j==0)
                            g=g.detach().cpu().numpy().copy();prefix['gradient_'+precision+'_'+objective]=g
                            # Forward maps must be bit exact. Permit only tiny
                            # backward roundoff, recorded again by the CPU check.
                            if precision=='native32' and gradient_relative_error(g,prefix['source_gradient_'+objective])>NATIVE_GRADIENT_RTOL:
                                old=prefix['source_gradient_'+objective].astype(float)
                                relative=float(np.linalg.norm(g.astype(float)-old)/max(np.linalg.norm(old),1e-30))
                                raise RuntimeError('native suffix gradient differs from Step06: '+str(key)+
                                    f' objective={objective} relative_l2={relative:.9g} max_abs={np.abs(g-old).max():.9g}')
                            backwards+=1
                        if leaf.grad is not None:raise RuntimeError('unexpected leaf gradient accumulation')
                    del out,leaf
            print(f'NUMERICS: {len(cases)+1}/{len(chosen)} {key[0]}/{key[1]} {precision}; {calls} forwards',flush=True)
        case['prefix_member']=f'prefix/{key[0]}_{key[1]}.npz'
        payloads[case['prefix_member']]=npz_bytes(prefix);cases.append(case)
    after={p:state_hash(s) for p,s in suffixes.items()}
    if before!=after or any(p.grad is not None or p.requires_grad for s in suffixes.values() for p in s.parameters()):
        raise RuntimeError('numeric suffix weights or gradient state changed')
    payloads.update({'source_cases.json':dump_json(source_cases),'cases.json':dump_json(cases),'rows.json':dump_json(rows)})
    return payloads,calls,backwards,before,after


def run(save_dir,source_path,direction_path,output,smoke=False):
    output=Path(output);analysis_output=output.with_name(output.name.removesuffix('.tar.gz')+'_analysis.tar.gz')
    for p in (output,analysis_output):
        if p.exists():raise FileExistsError('refusing to overwrite numeric artifact: '+str(p))
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    print('Checking Step01 and Step06 evidence on CPU...',flush=True)
    direction_summary,*_=check_direction(direction_path)
    dm,dd=read_archive(direction_path)
    source=read_source(source_path)
    if source['sha256']!=dm['source_sha256']:raise ValueError('Step06 used another Step01 source')
    paths={a:Path(save_dir)/'checkpoints/train/ostrack'/CONFIGS[a]/'OSTrack_ep0300.pth.tar' for a in ARMS}
    models,_,provenance=load_models(paths,'cuda',arms=ARMS)
    for arm in ARMS:
        for k in ('checkpoint_sha256','state_sha256_before'):
            if provenance[arm][k]!=dm['provenance'][arm][k]:raise ValueError('numeric checkpoint differs from Step06')
    count=len(selected_cases(json.loads(dd['cases.json']),smoke))
    forwards=sum(len(schedule(p)) for p in PRECISIONS)*count
    print(f'STEP07 PRECHECK PASSED: smoke={smoke}, {count} eligible frames, {forwards} suffix forwards, {6*count} gradients',flush=True)
    backend_before=backend_state()
    try:
        payloads,calls,backwards,sb,sa=collect(source_path,source,dm,dd,models,config_for(CONFIGS['Rdisc']),smoke)
    finally:
        for arm,model in models.items():
            provenance[arm]['state_sha256_after']=state_hash(model)
            if (provenance[arm]['state_sha256_after']!=provenance[arm]['state_sha256_before']
                    or any(p.grad is not None or p.requires_grad for p in model.parameters())
                    or any(m.training for m in model.modules())):
                raise RuntimeError('numeric audit changed original model')
    manifest=dict(kind='vdrm_module1_step07_numerics',version=1,smoke_only=smoke,precisions=list(PRECISIONS),
        epsilons={p:list(v) for p,v in EPSILONS.items()},optimizer_steps=0,source_verified=True,
        configurations={a:CONFIGS[a] for a in ARMS},source_sha256=source['sha256'],
        direction_sha256=direction_summary['evidence_sha256'],source_provenance=dm['provenance'],provenance=provenance,
        suffix_states_before=sb,suffix_states_after=sa,backend_before=backend_before,backend_after=backend_state(),
        forward_calls=calls,backward_calls=backwards,
        environment=dict(python=platform.python_version(),torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name()))
    write_bundle(output,manifest,payloads)
    summary,frames,diagnostics,gradients,conv=check_bundle(output)
    write_bundle(analysis_output,dict(kind='vdrm_module1_step07_numerics_analysis',version=1,evidence_sha256=summary['evidence_sha256']),
        {'summary.json':dump_json(summary),'per_frame.json':dump_json(frames),'derivatives.json':dump_json(diagnostics),
         'gradients.json':dump_json(gradients),'convergence.json':dump_json(conv)})
    print('STEP07 ARTIFACT CHECK PASSED; native suffix parity passed; no training or performance pass',flush=True)
    print('Float64 derivative audit passed:',summary['float64_derivative_audit_passed'],flush=True)
    print('Adjacent-step convergence:',summary['convergence_by_precision'],flush=True)
    print('Download:',output.resolve(),flush=True);print('Download:',analysis_output.resolve(),flush=True)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--save_dir',default='./output');parser.add_argument('--source');parser.add_argument('--direction')
    parser.add_argument('--output');parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();root=Path(args.save_dir)/'analysis'
    run(args.save_dir,args.source or str(root/'module1_step01/module1_evidence.tar.gz'),
        args.direction or str(root/'module1_step06/module1_direction.tar.gz'),
        args.output or str(root/'module1_step07'/('module1_numerics_smoke.tar.gz' if args.smoke else 'module1_numerics.tar.gz')),args.smoke)


if __name__=='__main__':main()
