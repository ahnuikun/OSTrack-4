"""Step09: fixed Step08 weights, full training-cache audit, bounded GT oracle.

No model training and no new closed-loop performance claim. Oracle updates use
GT on declared diagnostic frames only; they are not deployable predictions.
"""

import argparse
import json
from pathlib import Path
import sys
import tarfile

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
import numpy as np
import torch

from lib.train.module1_learning import ResidualAdapter,learning_loss
from lib.train.frozen_vdrm import file_hash
from lib.train.data.processing_utils import transform_image_to_crop
from lib.test.analysis.module1_learning import check_bundle as check_learning
from lib.test.analysis.module1_numerics import read_archive,load_np
from lib.test.analysis.module1_gates import read_source,protocol_plan
from lib.test.analysis.module1_direction import frame_plan,decode_metrics
from lib.test.analysis.module1_evidence import decode_candidates,ious
from lib.test.analysis.module1_fit_audit import HEADS,MODES,oracle_selection,geometry,projected_proposal,check_bundle
from lib.test.analysis.frozen_failure_diagnostics import CONFIGS,dump_json,write_bundle
from tracking.diagnose_vdrm_frozen_failures import load_models,config_for,state_hash
from tracking.diagnose_vdrm_native_float import npz_bytes,input_hashes
from tracking.diagnose_vdrm_module1_location import selected_records,cached_inputs
from tracking.diagnose_vdrm_module1_numerics import backend_state
from tracking.train_vdrm_module1_learning import torch_prefix,suffix_forward
from tracking.vdrm_diagnostics.module1_numeric_suffix import CachedSuffix


def prepare_case(a,c,cfg,device):
    """Reconstruct the exact Step08 labels on old, fixed diagnostic inputs."""
    inputs=cached_inputs(a,cfg,device)
    if input_hashes(inputs)!=c['input_sha256']:raise ValueError('diagnostic input hash mismatch')
    a['ce_template_mask']=inputs['ce_template_mask'].cpu().numpy()
    p,q,route,valid=[torch.from_numpy(a[k]).to(device) for k in ('prototypes','q','route','part_valid')]
    a['mixture']=(torch.einsum('bkn,bkc->bnc',route,p*q[...,None])/valid.sum(-1).clamp_min(1)[:,None,None]).cpu().numpy()
    boxes=decode_candidates(a,c,c['image_shape_hw'])
    overlap=ious(boxes,c['gt_xywh']);peak=int((a['score_map']*a['hann']).argmax())
    visible=c['cover']>0 and c['absence']==0 and min(c['gt_xywh'][2:])>0
    padding=a['padding'].reshape(16,16,16,16).any(axis=(1,3)).reshape(-1)
    a['good']=(overlap>=.5)&a['final_kept'].reshape(-1)&~padding if visible else np.zeros(256,bool)
    a['target']=transform_image_to_crop(torch.tensor(c['gt_xywh'],dtype=torch.float32),torch.tensor(c['anchor_xywh'],dtype=torch.float32),
        c['resize_factor'],torch.tensor([256,256]),normalize=True).numpy()
    c.update(baseline_iou=float(overlap[peak]),protect=bool(visible and overlap[peak]>=.5),good_cells=int(a['good'].sum()))
    return a,c


def audit_case(a,c,suffix,adapters,steps,payloads,reference_heads=None):
    device=next(suffix.parameters()).device;p=torch_prefix(a,device)
    base={k:torch.from_numpy(a[k]).to(device) for k in HEADS[:3]}
    good=torch.from_numpy(a['good']).to(device);target=torch.from_numpy(a['target']).to(device)
    c=dict(c,oracle_eligible=bool(c['oracle_selected'] and good.any()))
    case_id=c['case_id'];c['base_member']=f'base/{case_id}.npz'
    payloads[c['base_member']]=npz_bytes({k:a[k] for k in HEADS+('good','target','hann')})
    vectors={};rows=[]
    def forward(delta):
        out=suffix_forward(suffix,p,delta)
        loss,parts=learning_loss(out,base,good,target,p['hann'],c['protect'])
        return out,loss,parts
    def save(mode,delta,out,parts,trace=None):
        arrays={k:out[k].detach().cpu().numpy() for k in HEADS}
        metrics,_=decode_metrics(arrays,c,a['hann'])
        actual=((p['search_tokens']+delta)-p['search_tokens']).detach().cpu().numpy()
        geom=geometry(actual,a['delta'])
        if geom['norm_ratio']>1.005:raise RuntimeError('actual residual exceeded budget')
        member=f'heads/{case_id}_{mode}.npz';payloads[member]=npz_bytes(arrays)
        r=dict(case_id=case_id,mode=mode,arrays_member=member,loss=parts,geometry=geom,**metrics)
        if trace is not None:r['trace']=trace
        if c['oracle_selected']:vectors[mode]=actual
        rows.append(r)
        return arrays
    with torch.no_grad():
        updates={'Rdisc':p['delta'],'zero':torch.zeros_like(p['delta'])}
        for arm,adapter in adapters.items():updates[arm]=adapter(p['search_tokens'],p['prototypes'],p['part_valid'],p['mixture'],p['delta'])
        for mode in MODES:
            delta=updates[mode];out,_,parts=forward(delta);arrays=save(mode,delta,out,parts)
            if mode=='Rdisc' and any(not np.array_equal(arrays[k],a[k]) for k in HEADS):raise RuntimeError('cached native suffix parity failed')
            if mode=='zero' and 'base_score_map' in a and any(not np.array_equal(arrays[k],a['base_'+k]) for k in HEADS[:3]):
                raise RuntimeError('zero suffix differs from saved Tclean')
            if reference_heads and mode in reference_heads:
                if any(not np.array_equal(arrays[k],reference_heads[mode][k]) for k in HEADS[:3]):raise RuntimeError('learned suffix differs from Step08 shared output')
    if c['oracle_eligible']:
        original=p['delta'].detach().clone().requires_grad_(True)
        _,loss,_=forward(original);gradient,=torch.autograd.grad(loss,original)
        if not torch.isfinite(gradient).all():raise RuntimeError('nonfinite oracle gradient')
        vectors['gradient']=gradient.detach().cpu().numpy()
        for r in rows:
            if r['mode'] in ('prototype','full'):
                difference=vectors[r['mode']].astype(float)-vectors['Rdisc'].astype(float)
                r['linear_loss_change']=float(np.sum(difference*vectors['gradient'].astype(float)))
        for space in ('prototype','full'):
            current=p['delta'].detach().clone();trace=[]
            for iteration in range(steps):
                leaf=current.detach().requires_grad_(True);_,loss,parts=forward(leaf)
                g,=torch.autograd.grad(loss,leaf)
                if not torch.isfinite(g).all():raise RuntimeError('nonfinite oracle gradient')
                with torch.no_grad():
                    proposal=projected_proposal(current,g,p['delta'],p['prototypes'],p['part_valid'],space)
                    out,next_loss,next_parts=forward(proposal)
                    accepted=next_parts['total']<=parts['total']
                    trace.append(dict(iteration=iteration+1,before=parts['total'],proposal=next_parts['total'],accepted=accepted))
                    if accepted:current=proposal
            with torch.no_grad():
                out,_,parts=forward(current);save('oracle_'+space,current,out,parts,trace)
    if c['oracle_selected']:
        c['vectors_member']=f'vectors/{case_id}.npz';payloads[c['vectors_member']]=npz_bytes(vectors)
    return c,rows


def run(save_dir,learning_dir,source_path,output,smoke=False):
    output=Path(output);analysis=output.with_name(output.name.removesuffix('.tar.gz')+'_analysis.tar.gz')
    if output.exists() or analysis.exists():raise FileExistsError('refusing to overwrite fit audit')
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    learning_dir=Path(learning_dir);learning_path=learning_dir/'module1_learning.tar.gz'
    print('Checking Step08 evidence, selected cache hashes and Step01 source...',flush=True)
    check_learning(learning_path);m,data=read_archive(learning_path);source=read_source(source_path)
    if not smoke and m['protocol']['smoke_only']:raise ValueError('formal audit requires formal Step08 learning evidence')
    if source['sha256']!=m['source_sha256']:raise ValueError('Step01 source differs from learning evidence')
    if backend_state()!=m['backend']:raise ValueError('backend differs from Step08')
    index=json.loads(data['train_index.json']);train=index['rows'][:4] if smoke else index['rows']
    if file_hash(learning_dir/'cache/index.json')!=json.loads(data['training.json'])['full']['train_cache_sha256']:
        raise ValueError('cache index mismatch')
    for row in train:
        if file_hash(learning_dir/'cache'/row['member'])!=row['sha256']:raise ValueError('cache file mismatch')
    cfg=config_for(CONFIGS['Rdisc']);paths={a:Path(save_dir)/'checkpoints/train/ostrack'/CONFIGS[a]/'OSTrack_ep0300.pth.tar' for a in ('Tclean','Rdisc')}
    models,_,provenance=load_models(paths,'cuda',arms=('Tclean','Rdisc'))
    for arm in ('Tclean','Rdisc'):
        for k in ('checkpoint_sha256','state_sha256_before'):
            if provenance[arm][k]!=m['provenance'][arm][k]:raise ValueError('checkpoint mismatch: '+arm)
    suffix=CachedSuffix(models['Rdisc']).cuda();del models
    adapters={}
    for arm in ('prototype','full'):
        adapter=ResidualAdapter(space=arm).cuda().eval().requires_grad_(False)
        adapter.load_state_dict({k:torch.from_numpy(v).cuda() for k,v in load_np(data['weights_'+arm+'.npz']).items()})
        adapters[arm]=adapter
    before=state_hash(suffix);adapter_before={a:state_hash(v) for a,v in adapters.items()}
    diagnostic=frame_plan(smoke);selected={k:c for k,c in selected_records(source,protocol_plan(smoke)).items() if c['frame_1based'] in diagnostic[c['sequence']]}
    old_rows={(r['sequence'],r['frame_1based'],r['arm']):r for r in json.loads(data['rows.json']) if r['population']=='regression' and r['scope']=='shared'}
    chosen=set(oracle_selection(train,smoke));steps=2 if smoke else 8;payloads={};cases=[];rows=[]
    print(f'STEP09 PRECHECK PASSED: {len(train)} training caches, {len(selected)} fixed regression frames; no parameter updates',flush=True)
    def collect(a,c,references=None):
        case,result=audit_case(a,c,suffix,adapters,steps,payloads,references);cases.append(case);rows.extend(result)
        if len(cases)%32==0 or c['population']=='regression':print(f'FIT AUDIT: {len(cases)} cases, {len(rows)} final outputs',flush=True)
    for i,row in enumerate(train):
        a=load_np((learning_dir/'cache'/row['member']).read_bytes())
        c=dict(row,case_id=f'train_{i:04d}',oracle_selected=i in chosen)
        collect(a,c)
    with tarfile.open(source_path,'r|gz') as archive:
        for member in archive:
            if member.name not in selected:continue
            a=load_np(archive.extractfile(member).read());c=dict(selected[member.name],population='regression',oracle_selected=True)
            c['case_id']=f'regression_{c["sequence"]}_{c["frame_1based"]:06d}'
            a,c=prepare_case(a,c,cfg,'cuda')
            references={mode:load_np(data[old_rows[c['sequence'],c['frame_1based'],mode]['arrays_member']]) for mode in ('Rdisc','prototype','full')}
            collect(a,c,references)
    after=state_hash(suffix);adapter_after={a:state_hash(v) for a,v in adapters.items()}
    if before!=after or adapter_before!=adapter_after or any(p.grad is not None or p.requires_grad for p in suffix.parameters()):
        raise RuntimeError('fit audit changed frozen weights')
    if len(cases)!=len(train)+len(selected):raise RuntimeError('missing diagnostic cases')
    manifest=dict(kind='vdrm_module1_step09_fit_audit',version=1,smoke_only=smoke,cases=len(cases),
        train_cases=len(train),regression_cases=len(selected),learning_sha256=file_hash(learning_path),source_sha256=source['sha256'],
        parameter_updates=0,deployable_oracle=False,total_norm_budget=1.,step_size=.25,oracle_steps=steps,
        suffix_before=before,suffix_after=after,adapter_before=adapter_before,adapter_after=adapter_after,provenance=provenance,
        backend=backend_state(),source_training_smoke=m['protocol']['smoke_only'])
    payloads.update({'cases.json':dump_json(cases),'rows.json':dump_json(rows)})
    write_bundle(output,manifest,payloads);summary,_,_=check_bundle(output)
    write_bundle(analysis,dict(kind='vdrm_module1_step09_fit_analysis',version=1,evidence_sha256=file_hash(output)),{'summary.json':dump_json(summary)})
    print('STEP09 AUDIT CHECK PASSED; no training or performance approval',flush=True)
    print('Download:',output.resolve(),flush=True);print('Download:',analysis.resolve(),flush=True)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--save_dir',default='./output')
    p.add_argument('--learning_dir');p.add_argument('--source');p.add_argument('--output');p.add_argument('--smoke',action='store_true')
    args=p.parse_args();root=Path(args.save_dir)/'analysis';torch.set_num_threads(2)
    run(args.save_dir,args.learning_dir or root/'module1_step08',args.source or root/'module1_step01/module1_evidence.tar.gz',
        args.output or root/('module1_step09_smoke' if args.smoke else 'module1_step09')/'module1_fit.tar.gz',args.smoke)


if __name__=='__main__':main()
