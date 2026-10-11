"""Step08 paired residual pilot: native cache, fitting, shared and closed audit.

All audit GT is read for labels only after forward. Training uses disjoint
sequences; checkpoints are fixed at the last epoch, never chosen by audit.
"""

import argparse
from copy import deepcopy
import json
import platform
from pathlib import Path
import re
import sys

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))

import numpy as np
import torch

from lib.train.module1_learning import ARMS,ResidualAdapter,adapter_insertion,learning_loss
from lib.test.analysis.module1_learning import protocol,sequence_partition,training_frames,hash_json,MODES,check_bundle
from lib.test.analysis.module1_numerics import check_bundle as check_numerics,read_archive,load_np
from lib.test.analysis.module1_gates import read_source,protocol_plan
from lib.test.analysis.module1_evidence import decode_candidates,ious
from lib.test.analysis.frozen_failure_diagnostics import CONFIGS,dump_json,write_bundle
from lib.test.evaluation.environment import env_settings
from lib.train.data.processing_utils import transform_image_to_crop
from lib.train.frozen_vdrm import file_hash
from tracking.diagnose_vdrm_frozen_failures import load_models,config_for,state_hash,png_bytes
from tracking.diagnose_vdrm_native_float import NativeTracker,native_step,read_rgb,capture_native,decode_native,npz_bytes,input_hashes
from tracking.diagnose_vdrm_module1_evidence import PartCapture
from tracking.diagnose_vdrm_module1_direction import head_arrays
from tracking.vdrm_diagnostics.backend_pair_metrics import bbox_iou_xywh
from tracking.vdrm_diagnostics.module1_numeric_suffix import CachedSuffix
from tracking.diagnose_vdrm_module1_numerics import backend_state


def dataset_plans(root,spec,partition,source):
    root=Path(root)
    names=(root/'list.txt').read_text(encoding='utf-8').splitlines()
    if len(names)!=len(set(names)) or any(not re.fullmatch(r'GOT-10k_Train_\d{6}',n) for n in names):
        raise ValueError('invalid GOT training list')
    if max(partition['train']+partition['audit'])>=len(names):raise ValueError('GOT train list shorter than registered IDs')
    refs={t['sequence']:t for t in source['trajectories']}
    requested=[(split,i,names[i],None) for split,ids in partition.items() for i in ids]
    requested += [('regression',None,n,b[1]) for n,b in protocol_plan(spec['smoke_only']).items()]
    plans=[]
    for population,identity,name,limit in requested:
        folder=root/name
        gt=np.loadtxt(folder/'groundtruth.txt',delimiter=',',ndmin=2)
        cover=np.loadtxt(folder/'cover.label',ndmin=1);absence=np.loadtxt(folder/'absence.label',ndmin=1)
        if (gt.shape!=(len(cover),4) or absence.shape!=cover.shape or len(gt)<2
                or not np.isfinite(gt).all() or not np.isfinite(cover).all() or not np.isfinite(absence).all()
                or (gt[:,2:]<0).any() or (gt[0,2:]<=0).any() or ((cover<0)|(cover>8)).any()
                or not np.isin(absence,[0,1]).all()):raise ValueError('invalid sequence annotations: '+name)
        stop=min(len(gt),spec['train_prefix']) if population=='train' else (limit or spec['audit_prefix'] or len(gt))
        stop=min(stop,len(gt))
        paths=[folder/f'{f:08d}.jpg' for f in range(1,stop+1)]
        for p in paths:
            if not p.is_file():raise FileNotFoundError(p)
        if population=='regression':
            if stop!=limit:raise ValueError('regression prefix too short')
            ref=refs[name]
            for k,v in [('gt_xywh',gt),('cover',cover),('absence',absence)]:
                if not np.array_equal(v[:stop],ref[k][:stop]):raise ValueError('regression labels differ from Step01')
            for f,p in enumerate(paths,1):
                import hashlib
                if hashlib.sha256(png_bytes(read_rgb(str(p)))).hexdigest()!=source['manifest']['members_sha256'][f'images/{name}/{f:06d}.png']:
                    raise ValueError('regression image differs from Step01')
        plans.append(dict(population=population,sequence_id=identity,sequence=name,frames=[str(p) for p in paths],gt=gt[:stop],
            cover=cover[:stop],absence=absence[:stop],stop=stop,
            cache_frames=training_frames(cover,absence,stop,spec['frames_per_sequence']) if population=='train' else []))
    return plans,file_hash(root/'list.txt')


@torch.no_grad()
def preflight_native(plans,models,cfg,source):
    refs={t['sequence']:t for t in source['trajectories']}
    for plan in plans:
        if plan['population']!='regression':continue
        for arm in ('Rdisc','Tclean'):
            tracker=NativeTracker(models[arm],cfg)
            tracker.initialize(read_rgb(plan['frames'][0]),dict(init_bbox=plan['gt'][0].tolist()))
            anchor=list(tracker.state)
            with capture_native(tracker) as snapshot:tracker.track(read_rgb(plan['frames'][1]))
            reference=refs[plan['sequence']]['native'][arm][1]
            if (anchor!=reference['anchor_xywh'] or list(tracker.state)!=reference['bbox_xywh']
                    or snapshot['input_sha256']!=reference['input_sha256']):
                raise RuntimeError('native preflight differs from Step01 before training: '+plan['sequence']+' '+arm)


def torch_prefix(a,device):
    return {k:torch.from_numpy(a[k]).to(device) for k in
            ('template_tokens','search_tokens','indices','ce_template_mask','prototypes','part_valid','mixture','delta','hann')}


def suffix_forward(suffix,p,delta):
    return suffix(p['template_tokens'],p['search_tokens'],p['indices'],p['ce_template_mask'],delta)


def cache_training(plans,model,cfg,directory,suffix):
    directory.mkdir(parents=True,exist_ok=False)
    rows=[];capture=PartCapture();device=next(model.parameters()).device
    with torch.no_grad():
        for number,plan in enumerate([p for p in plans if p['population']=='train']):
            tracker=NativeTracker(model,cfg);tracker.initialize(read_rgb(plan['frames'][0]),dict(init_bbox=plan['gt'][0].tolist()))
            for frame in range(2,plan['stop']+1):
                image=read_rgb(plan['frames'][frame-1])
                if frame not in plan['cache_frames']:
                    tracker.track(image);continue
                hook=model.backbone.vdrm.register_forward_hook(capture,with_kwargs=True)
                try:snapshot,native=native_step(tracker,image,plan['gt'][frame-1],frame,capture)
                finally:hook.remove()
                a=dict(capture.arrays);out=snapshot['output']
                a.update(head_arrays(out,torch.from_numpy(a['delta']).to(device)))
                a.update(hann=tracker.output_window.cpu().numpy(),padding=np.asarray(snapshot['padding']),
                         ce_template_mask=snapshot['inputs']['ce_template_mask'].cpu().numpy())
                # Use the same einsum/order as live insertion for bitwise parity.
                p=torch.from_numpy(a['prototypes']).to(device);q=torch.from_numpy(a['q']).to(device)
                route=torch.from_numpy(a['route']).to(device);valid=torch.from_numpy(a['part_valid']).to(device)
                a['mixture']=(torch.einsum('bkn,bkc->bnc',route,p*q[...,None])/valid.sum(-1).clamp_min(1)[:,None,None]).cpu().numpy()
                record=dict(population='train',sequence=plan['sequence'],sequence_id=plan['sequence_id'],frame_1based=frame,
                    gt_xywh=plan['gt'][frame-1].tolist(),cover=float(plan['cover'][frame-1]),absence=float(plan['absence'][frame-1]),
                    image_shape_hw=list(image.shape[:2]),anchor_xywh=native['anchor_xywh'],resize_factor=native['resize_factor'],
                    input_sha256=native['input_sha256'],baseline_iou=native['iou'])
                visible=record['cover']>0 and record['absence']==0 and min(record['gt_xywh'][2:])>0
                overlaps=ious(decode_candidates(a,record,record['image_shape_hw']),record['gt_xywh']) if visible else np.zeros(256)
                padding=a['padding'].reshape(16,16,16,16).any(axis=(1,3)).reshape(-1)
                a['good']=(overlaps>=.5)&a['final_kept'].reshape(-1)&~padding if visible else np.zeros(256,bool)
                a['target']=transform_image_to_crop(torch.tensor(record['gt_xywh'],dtype=torch.float32),
                    torch.tensor(record['anchor_xywh'],dtype=torch.float32),record['resize_factor'],torch.tensor([256,256]),normalize=True).numpy()
                record.update(protect=bool(visible and native['iou']>=.5),good_cells=int(a['good'].sum()))
                tensors=torch_prefix(a,device)
                replay=suffix_forward(suffix,tensors,tensors['delta'])
                for k in ('score_map','size_map','offset_map','final_kept','ce_removed'):
                    if not np.array_equal(replay[k].cpu().numpy(),a[k]):raise RuntimeError('training suffix differs from native: '+k)
                member=f'{plan["sequence"]}_{frame:06d}.npz';(directory/member).write_bytes(npz_bytes(a))
                record.update(member=member,sha256=file_hash(directory/member));rows.append(record)
            print(f'CACHE {number+1}: {plan["sequence"]}, {len(plan["cache_frames"])} samples',flush=True)
    index=dict(rows=rows,good_frames=sum(r['good_cells']>0 for r in rows),protected_frames=sum(r['protect'] for r in rows))
    if not rows or not index['good_frames']:raise ValueError('no supported training frames; do not train a null pilot')
    (directory/'index.json').write_bytes(dump_json(index))
    return index


def train_adapters(cache_dir,index,suffix,spec,output_dir):
    device=next(suffix.parameters()).device
    torch.manual_seed(spec['seed'])
    initial=ResidualAdapter(width=spec['width']).to(device)
    initial_state=deepcopy(initial.state_dict());initial_hash=state_hash(initial)
    order=[np.random.default_rng(spec['seed']+epoch).permutation(len(index['rows'])).tolist() for epoch in range(spec['epochs'])]
    cache_hash=file_hash(cache_dir/'index.json')
    trained,training={},{}
    for arm in ARMS:
        adapter=ResidualAdapter(width=spec['width'],space=arm).to(device);adapter.load_state_dict(initial_state)
        optimizer=torch.optim.AdamW(adapter.parameters(),lr=spec['learning_rate'],weight_decay=spec['weight_decay'])
        before=state_hash(suffix);epochs=[];updates=0;maximum_ratio=0.
        for epoch,indices in enumerate(order):
            log=[]
            for i in indices:
                row=index['rows'][i]
                if row['population']!='train':raise ValueError('audit row reached optimizer')
                path=cache_dir/row['member']
                if file_hash(path)!=row['sha256']:raise ValueError('training cache changed')
                a=load_np(path.read_bytes());p=torch_prefix(a,device)
                base={k:torch.from_numpy(a[k]).to(device) for k in ('score_map','size_map','offset_map')}
                optimizer.zero_grad(set_to_none=True)
                delta=adapter(p['search_tokens'],p['prototypes'],p['part_valid'],p['mixture'],p['delta'])
                ratio=float(((p['search_tokens']+delta)-p['search_tokens']).norm()/p['delta'].norm().clamp_min(1e-12))
                if ratio>1.005:raise RuntimeError('training update exceeded original norm')
                maximum_ratio=max(maximum_ratio,ratio)
                out=suffix_forward(suffix,p,delta)
                if updates==0:
                    for k in base:
                        if not torch.equal(out[k],base[k]):raise RuntimeError('adapter zero initialization changed native outputs')
                loss,metrics=learning_loss(out,base,torch.from_numpy(a['good']).to(device),
                    torch.from_numpy(a['target']).to(device),p['hann'],row['protect'])
                loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(adapter.parameters(),spec['gradient_clip'],error_if_nonfinite=True)
                optimizer.step();updates+=1
                metrics['gradient_norm']=float(norm);log.append(metrics)
            epochs.append(dict(epoch=epoch+1,**{k:float(np.mean([r[k] for r in log])) for k in log[0]}))
            print(f'TRAIN {arm} epoch {epoch+1}/{spec["epochs"]}: loss={epochs[-1]["total"]:.6f}',flush=True)
        after=state_hash(suffix)
        if before!=after or any(p.grad is not None or p.requires_grad for p in suffix.parameters()):raise RuntimeError('training changed frozen suffix')
        adapter.eval().requires_grad_(False)
        report=dict(initial_sha256=initial_hash,final_sha256=state_hash(adapter),sample_order_sha256=hash_json(order),
            parameter_count=sum(p.numel() for p in adapter.parameters()),updates=updates,epochs=epochs,
            maximum_update_ratio=maximum_ratio,train_cache_sha256=cache_hash,frozen_suffix_before=before,frozen_suffix_after=after)
        if report['initial_sha256']==report['final_sha256'] or not adapter.value.weight.count_nonzero():
            raise RuntimeError('adapter did not learn a nonzero correction')
        torch.save(dict(kind='module1_step08_adapter',arm=arm,protocol=spec,training=report,
                        net={k:v.cpu() for k,v in adapter.state_dict().items()}),output_dir/f'adapter_{arm}.pth')
        trained[arm]=adapter;training[arm]=report
    return trained,training


def evaluate(plans,models,cfg,adapters,source):
    rows=[];payloads={};eval_plan=[];visuals=[]
    refs={t['sequence']:t for t in source['trajectories']}
    with torch.no_grad():
        for number,plan in enumerate([p for p in plans if p['population']!='train']):
            name=plan['sequence'];population=plan['population']
            trackers={a:NativeTracker(models['Tclean'] if a=='Tclean' else models['Rdisc'],cfg) for a in MODES}
            initial=read_rgb(plan['frames'][0])
            for t in trackers.values():t.initialize(initial,dict(init_bbox=plan['gt'][0].tolist()))
            eval_plan.append(dict(population=population,sequence=name,sequence_id=plan['sequence_id'],stop=plan['stop'],initial_box=plan['gt'][0].tolist()))
            for frame in range(2,plan['stop']+1):
                image=read_rgb(plan['frames'][frame-1]);gt=plan['gt'][frame-1]
                common=dict(population=population,sequence=name,frame_1based=frame,gt_xywh=gt.tolist(),
                    cover=float(plan['cover'][frame-1]),absence=float(plan['absence'][frame-1]),image_shape_hw=list(image.shape[:2]))
                common['visible']=bool(min(gt[2:])>0 and common['cover']>0 and common['absence']==0)
                def save(arm,scope,out,snapshot,anchor,box,ratio):
                    member=f'arrays/{population}/{name}_{frame}_{scope}_{arm}.npz'
                    a={k:out[k].cpu().numpy() for k in ('score_map','size_map','offset_map')}
                    a['hann']=trackers['Rdisc'].output_window.cpu().numpy()
                    payloads[member]=npz_bytes(a)
                    row=dict(common,arm=arm,scope=scope,anchor_xywh=list(anchor),bbox_xywh=list(box),
                        resize_factor=float(snapshot['resize']),input_sha256=snapshot['input_sha256'],arrays_member=member,
                        actual_norm_ratio=ratio,iou=bbox_iou_xywh(box,gt) if min(gt[2:])>0 else None)
                    rows.append(row);return row
                for arm,t in trackers.items():
                    anchor=list(t.state)
                    with adapter_insertion(t.network,adapters.get(arm)) as values,capture_native(t) as snapshot:
                        prediction=t.track(image)
                    if list(prediction['target_bbox'])!=list(t.state) or t.frame_id!=frame-1:raise RuntimeError('broken native tracker state')
                    ratio=values.get('actual_norm_ratio',0. if arm=='Tclean' else 1.)
                    r=save(arm,'closed',snapshot['output'],snapshot,anchor,t.state,ratio)
                    if population=='regression' and arm in ('Rdisc','Tclean'):
                        ref=refs[name]['native'][arm][frame-1]
                        if any(r[k]!=ref[k] for k in ('anchor_xywh','bbox_xywh','input_sha256')):
                            raise RuntimeError('regression baseline differs from Step01: '+name+'/'+str(frame)+' '+arm)
                    if arm=='Rdisc':base_snapshot=snapshot;base_row=r
                before={a:(list(t.state),t.frame_id) for a,t in trackers.items()}
                for arm in MODES:
                    if arm=='Rdisc':
                        row=dict(base_row,scope='shared');row['arrays_member']=base_row['arrays_member'];rows.append(row);continue
                    net=models['Tclean'] if arm=='Tclean' else models['Rdisc']
                    with adapter_insertion(net,adapters.get(arm)) as values:
                        out=net.forward(**base_snapshot['inputs'])
                    box=decode_native(trackers['Rdisc'],out,base_row['anchor_xywh'],base_snapshot['resize'],image.shape)
                    save(arm,'shared',out,base_snapshot,base_row['anchor_xywh'],box,values.get('actual_norm_ratio',0.))
                if before!={a:(list(t.state),t.frame_id) for a,t in trackers.items()}:raise RuntimeError('shared evaluation advanced tracker')
                if input_hashes(base_snapshot['inputs'])!=base_snapshot['input_sha256']:raise RuntimeError('shared inputs mutated')
            if population=='audit':
                # Keep a small set of actual scene images for the local review.
                rr={r['frame_1based']:r for r in rows if r['sequence']==name and r['scope']=='closed' and r['arm']=='Rdisc'}
                learned={arm:{r['frame_1based']:r for r in rows if r['sequence']==name and r['scope']=='closed' and r['arm']==arm} for arm in ARMS}
                visible=[f for f,r in rr.items() if r['visible']]
                selected={1:['initial_template']}
                if visible:
                    for label,frame in [('largest_full_gain',max(visible,key=lambda f:learned['full'][f]['iou']-rr[f]['iou'])),
                            ('largest_full_loss',min(visible,key=lambda f:learned['full'][f]['iou']-rr[f]['iou'])),
                            ('largest_arm_difference',max(visible,key=lambda f:abs(learned['full'][f]['iou']-learned['prototype'][f]['iou'])))]:
                        selected.setdefault(frame,[]).append(label)
                for frame,labels in selected.items():
                    member=f'images/{name}_{frame:06d}.png';payloads[member]=png_bytes(read_rgb(plan['frames'][frame-1]))
                    visuals.append(dict(sequence=name,frame_1based=frame,gt_xywh=plan['gt'][frame-1].tolist(),reasons=labels,image_member=member))
            print(f'AUDIT {number+1}: {population} {name}, {plan["stop"]-1} native steps per arm',flush=True)
    return rows,eval_plan,visuals,payloads


def run(data_root,save_dir,source_path,numerics_path,output_dir,smoke=False):
    output_dir=Path(output_dir)
    if output_dir.exists():raise FileExistsError('refusing to overwrite learning pilot: '+str(output_dir))
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    spec=protocol(smoke);partition,forbidden=sequence_partition(smoke)
    print('Checking Step01/Step07 evidence and sequence isolation...',flush=True)
    ns,*_=check_numerics(numerics_path)
    if not ns['float64_derivative_audit_passed']:raise ValueError('Step07 float64 derivative audit must pass before learning')
    nm,_=read_archive(numerics_path);source=read_source(source_path)
    if source['sha256']!=nm['source_sha256']:raise ValueError('Step07 source mismatch')
    if backend_state()!=nm['backend_before']:raise ValueError('backend settings differ from validated Step07 experiment')
    plans,list_hash=dataset_plans(data_root,spec,partition,source)
    paths={a:Path(save_dir)/'checkpoints/train/ostrack'/CONFIGS[a]/'OSTrack_ep0300.pth.tar' for a in ('Tclean','Rdisc')}
    models,_,provenance=load_models(paths,'cuda',arms=('Tclean','Rdisc'))
    for a in models:
        for k in ('checkpoint_sha256','state_sha256_before'):
            if provenance[a][k]!=nm['provenance'][a][k]:raise ValueError('learning checkpoint differs from numerical audit')
    cfg=config_for(CONFIGS['Rdisc'])
    preflight_native(plans,models,cfg,source)
    suffix=CachedSuffix(models['Rdisc']).cuda()
    output_dir.mkdir(parents=True,exist_ok=False)
    (output_dir/'protocol.json').write_bytes(dump_json(dict(protocol=spec,partition=partition,forbidden_ids=forbidden)))
    print(f'STEP08 PRECHECK PASSED: {len(partition["train"])} train, {len(partition["audit"])} audit, 5 regression sequences; two paired arms',flush=True)
    index=cache_training(plans,models['Rdisc'],cfg,output_dir/'cache',suffix)
    adapters,training=train_adapters(output_dir/'cache',index,suffix,spec,output_dir)
    del suffix;torch.cuda.empty_cache()
    rows,eval_plan,visuals,payloads=evaluate(plans,models,cfg,adapters,source)
    for a,model in models.items():
        provenance[a]['state_sha256_after']=state_hash(model)
        if provenance[a]['state_sha256_before']!=provenance[a]['state_sha256_after'] or any(p.grad is not None or p.requires_grad for p in model.parameters()):
            raise RuntimeError('pilot changed original frozen checkpoint')
    for a,adapter in adapters.items():
        if state_hash(adapter)!=training[a]['final_sha256']:raise RuntimeError('evaluation changed learned checkpoint')
        payloads['weights_'+a+'.npz']=npz_bytes({k:v.cpu().numpy() for k,v in adapter.state_dict().items()})
    manifest=dict(kind='vdrm_module1_step08_learning',version=1,protocol=spec,partition=partition,forbidden_ids=forbidden,
        provenance=provenance,source_sha256=source['sha256'],numerics_sha256=ns['evidence_sha256'],data_list_sha256=list_hash,
        train_frames=len(index['rows']),good_train_frames=index['good_frames'],protected_train_frames=index['protected_frames'],
        checkpoint_selection='fixed_last_epoch',production_model_modified=False,automatic_performance_pass=False,
        native_preflight_passed=True,backend=backend_state(),
        environment=dict(python=platform.python_version(),torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name()))
    payloads.update({'rows.json':dump_json(rows),'evaluation_plan.json':dump_json(eval_plan),'training.json':dump_json(training),
                     'train_index.json':dump_json(index),'visual_cases.json':dump_json(visuals)})
    target=output_dir/'module1_learning.tar.gz';write_bundle(target,manifest,payloads)
    summary,_,_=check_bundle(target)
    write_bundle(output_dir/'module1_learning_analysis.tar.gz',dict(kind='vdrm_module1_step08_learning_analysis',version=1,evidence_sha256=file_hash(target)),
        {'summary.json':dump_json(summary),'training.json':dump_json(training)})
    print('STEP08 PILOT CHECK PASSED; results require analysis, no performance approval',flush=True)
    print('Download:',target.resolve(),flush=True);print('Download:',(output_dir/'module1_learning_analysis.tar.gz').resolve(),flush=True)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data_root');p.add_argument('--save_dir',default='./output');p.add_argument('--source');p.add_argument('--numerics')
    p.add_argument('--output_dir');p.add_argument('--smoke',action='store_true');args=p.parse_args()
    root=Path(args.save_dir)/'analysis';torch.set_num_threads(2)
    run(args.data_root or str(Path(env_settings().got10k_path)/'train'),args.save_dir,
        args.source or str(root/'module1_step01/module1_evidence.tar.gz'),args.numerics or str(root/'module1_step07/module1_numerics.tar.gz'),
        args.output_dir or str(root/('module1_step08_smoke' if args.smoke else 'module1_step08')),args.smoke)


if __name__=='__main__':main()
