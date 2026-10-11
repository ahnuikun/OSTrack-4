"""Step06: GT-assisted local direction capacity on 25 declared cached frames.

This is an oracle diagnostic, not training or deployable inference. Compare
unrestricted signed prototype-span directions against all feature directions
at equal update norms. Gradients target only an insertion leaf, never weights.
"""

import argparse
import io
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
from lib.test.analysis.module1_direction import (
    BUDGETS,FD_EPS,OBJECTIVES,SPACES,check_bundle,choose_pair,frame_plan,modes,normalized_update,project_gradient)
from tracking.diagnose_vdrm_frozen_failures import config_for,load_models,state_hash
from tracking.diagnose_vdrm_module1_gates import final_kept
from tracking.diagnose_vdrm_module1_location import cached_inputs,selected_records
from tracking.diagnose_vdrm_native_float import NativeTracker,decode_native,input_hashes,npz_bytes
from tracking.vdrm_diagnostics.module1_direction_interventions import replace_search_update


def head_arrays(out,actual_delta):
    values={k:out[k] for k in ('score_map','size_map','offset_map')}
    values.update(final_kept=final_kept(out),actual_delta=actual_delta)
    removed=[]
    for index in out['removed_indexes_s']:
        mask=torch.zeros((1,256),device=actual_delta.device,dtype=torch.bool)
        if index is not None:mask.scatter_(1,index.long(),True)
        removed.append(mask)
    values['ce_removed']=torch.stack(removed) if removed else torch.zeros((0,1,256),device=actual_delta.device,dtype=torch.bool)
    return {k:v.detach().cpu().numpy().copy() for k,v in values.items()}


def collect(bundle,selected,models,cfg):
    model=models['Rdisc'];tracker=NativeTracker(model,cfg)
    if any(p.requires_grad for p in model.parameters()):raise RuntimeError('all model weights must remain frozen')
    device=next(model.parameters()).device
    cases,rows,payloads=[],[],{}
    backwards=0
    with tarfile.open(bundle,'r|gz') as archive:
        for member in archive:
            if member.name not in selected:continue
            with np.load(io.BytesIO(archive.extractfile(member).read()),allow_pickle=False) as a:
                source={k:a[k] for k in a.files}
            case=dict(selected[member.name]);inputs=cached_inputs(source,cfg,device)
            if input_hashes(inputs)!=case['input_sha256']:raise RuntimeError('direction input hash mismatch')
            if not np.array_equal(tracker.output_window.cpu().numpy(),source['hann']):raise RuntimeError('direction Hann mismatch')
            reference={k:torch.from_numpy(source[k]).to(device) for k in ('template_tokens','search_tokens')}
            prefix={k:source[k].copy() for k in ('prototypes','part_valid','indices','padding','hann')}
            prefix['reference_delta']=source['delta'].copy()
            case['reference_norm']=float(np.linalg.norm(source['delta'].astype(float)))
            for label,pre in [('original',''),('zero','base_')]:
                for k in ('score_map','size_map','offset_map'):prefix[label+'_'+k]=source[pre+k].copy()

            def store(out,values,mode,objective=None,space=None,budget=None):
                arrays=head_arrays(out,values['actual_delta'])
                box=decode_native(tracker,out,case['anchor_xywh'],case['resize_factor'],case['image_shape_hw'])
                if input_hashes(inputs)!=case['input_sha256']:raise RuntimeError('direction forward modified inputs')
                name=f'arrays/{case["sequence"]}/{case["frame_1based"]:06d}_{mode}_{objective}_{space}_{budget}.npz'
                payloads[name]=npz_bytes(arrays)
                rows.append(dict(sequence=case['sequence'],frame_1based=case['frame_1based'],mode=mode,
                    objective=objective,space=space,budget=budget,input_sha256=case['input_sha256'],
                    arrays_member=name,bbox_xywh=box))
                return arrays,box

            with torch.no_grad(),replace_search_update(model,reference) as values:
                out=model.forward(**inputs)
                original,box=store(out,values,'original')
            for k in ('score_map','size_map','offset_map','final_kept'):
                if not np.array_equal(original[k],source[k]):raise RuntimeError('direction original differs from source: '+k)
            if not np.array_equal(original['actual_delta'],source['delta']):raise RuntimeError('source actual delta mismatch')
            if box!=case['source_bbox_xywh']:raise RuntimeError('direction original float box mismatch')

            with torch.enable_grad(),replace_search_update(model,reference,gradient=True) as values:
                zero_out=model.forward(**inputs)
                leaf=values['leaf']
                with torch.no_grad():zero,_=store(zero_out,values,'zero')
            for k in ('score_map','size_map','offset_map'):
                if not np.array_equal(zero[k],source['base_'+k]):raise RuntimeError('zero replacement differs from saved Tclean: '+k)
            pair,reason=choose_pair(original,zero,prefix,case)
            case.update(pair=pair,exclusion_reason=reason)
            directions={}
            if reason is None:
                for i,objective in enumerate(OBJECTIVES):
                    with torch.enable_grad():
                        score=zero_out['score_map'] if objective=='raw' else zero_out['score_map']*tracker.output_window
                        score=score.reshape(-1)
                        margin=score[pair[0]]-score[pair[1]]
                        gradient,=torch.autograd.grad(margin,leaf,retain_graph=i==0)
                    g=gradient.detach().cpu().numpy().copy()
                    if not np.isfinite(g).all():raise RuntimeError('nonfinite insertion gradient')
                    prefix['gradient_'+objective]=g
                    pg,rank=project_gradient(g,prefix['prototypes'],prefix['part_valid'])
                    case['prototype_rank']=rank
                    directions[objective]=(g,pg);backwards+=1
                if leaf.grad is not None or any(p.grad is not None for p in model.parameters()):
                    raise RuntimeError('autograd.grad unexpectedly accumulated gradients')
            del zero_out,leaf

            for mode,objective,space,budget in modes(reason is None)[2:]:
                g,pg=directions[objective]
                delta=normalized_update(pg if space=='prototype' else g,case['reference_norm']*budget)
                with torch.no_grad(),replace_search_update(model,reference,torch.from_numpy(delta).to(device)) as values:
                    out=model.forward(**inputs)
                    store(out,values,mode,objective,space,budget)
            case['prefix_member']=f'prefix/{case["sequence"]}_{case["frame_1based"]:06d}.npz'
            payloads[case['prefix_member']]=npz_bytes(prefix)
            cases.append(case)
            print(f'DIRECTION: {len(cases)}/{len(selected)} frames; {len(rows)} forwards; {backwards} leaf gradients; exclusion={reason}',flush=True)
    if len(cases)!=len(selected):raise RuntimeError('missing direction cached frames')
    return {'cases.json':dump_json(cases),'rows.json':dump_json(rows),**payloads},len(rows),backwards


def run(save_dir,source_path,output,smoke=False):
    output=Path(output);analysis_output=output.with_name(output.name.removesuffix('.tar.gz')+'_analysis.tar.gz')
    for p in (output,analysis_output):
        if p.exists():raise FileExistsError('refusing to overwrite direction artifact: '+str(p))
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    paths={a:Path(save_dir)/'checkpoints/train/ostrack'/CONFIGS[a]/'OSTrack_ep0300.pth.tar' for a in ARMS}
    for p in paths.values():
        if not p.is_file():raise FileNotFoundError('missing checkpoint: '+str(p))
    print('Checking step01 cached evidence on CPU...',flush=True)
    source=read_source(source_path)
    plan=frame_plan(smoke)
    selected={k:c for k,c in selected_records(source,protocol_plan(smoke)).items() if c['frame_1based'] in plan[c['sequence']]}
    if len(selected)!=sum(map(len,plan.values())):raise ValueError('source lacks declared direction frames')
    models,_,provenance=load_models(paths,'cuda',arms=ARMS)
    cfg=config_for(CONFIGS['Rdisc'])
    for arm in ARMS:
        if any(provenance[arm][k]!=source['manifest']['provenance'][arm][k] for k in ('checkpoint_sha256','state_sha256_before')):
            raise ValueError('direction checkpoint differs from source')
    print(f'STEP06 PRECHECK PASSED: smoke={smoke}, {len(selected)} frames; GT-assisted direction diagnosis',flush=True)
    try:
        payloads,count,backwards=collect(source_path,selected,models,cfg)
    finally:
        for arm,model in models.items():
            after=state_hash(model)
            if (after!=provenance[arm]['state_sha256_before'] or any(p.grad is not None or p.requires_grad for p in model.parameters())
                    or any(m.training for m in model.modules())):
                raise RuntimeError('direction diagnosis changed model state, mode or gradients')
            provenance[arm]['state_sha256_after']=after
    manifest=dict(kind='vdrm_module1_step06_direction',version=1,smoke_only=smoke,
        frame_plan={k:list(v) for k,v in plan.items()},objectives=list(OBJECTIVES),spaces=list(SPACES),budgets=list(BUDGETS),fd_epsilon=FD_EPS,
        configurations={a:CONFIGS[a] for a in ARMS},optimizer_steps=0,gt_gradient_diagnosis_only=True,source_verified=True,
        source_sha256=source['sha256'],source_provenance={a:source['manifest']['provenance'][a] for a in ARMS},
        provenance=provenance,forward_calls=count,backward_calls=backwards,
        environment=dict(python=platform.python_version(),torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name()))
    write_bundle(output,manifest,payloads)
    summary,frames,gradients=check_bundle(output)
    write_bundle(analysis_output,dict(kind='vdrm_module1_step06_direction_analysis',version=1,
        source_sha256=source['sha256'],evidence_sha256=summary['evidence_sha256'],optimizer_steps=0),
        {'summary.json':dump_json(summary),'per_frame.json':dump_json(frames),'gradients.json':dump_json(gradients)})
    marker='STEP06 SMOKE PASSED' if smoke else 'STEP06 DIRECTION CHECK PASSED'
    print(f'{marker}: {len(selected)} frames; {count} forwards; {backwards} leaf gradients; no training or performance pass',flush=True)
    print('Gradient finite-difference audit:',summary['finite_difference_checks'],
          'passed='+str(summary['derivative_audit_passed']),flush=True)
    print('Download:',output.resolve(),flush=True);print('Download:',analysis_output.resolve(),flush=True)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--save_dir',default='./output');parser.add_argument('--source');parser.add_argument('--output')
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();root=Path(args.save_dir)/'analysis'
    run(args.save_dir,args.source or str(root/'module1_step01/module1_evidence.tar.gz'),
        args.output or str(root/'module1_step06'/('module1_direction_smoke.tar.gz' if args.smoke else 'module1_direction.tar.gz')),args.smoke)


if __name__=='__main__':main()
