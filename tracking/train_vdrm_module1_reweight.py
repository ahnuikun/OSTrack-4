"""Step10: change ONLY frame loss weights, reuse Step08 uniform control.

Same full-space adapter, initialization, cache, order and update count. Add
fresh confirmation sequences; never select checkpoints on evaluation results.
"""

import argparse
import json
from pathlib import Path
import sys

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
import numpy as np
import torch

from lib.train.module1_learning import ResidualAdapter,learning_loss
from lib.train.frozen_vdrm import file_hash
from lib.test.evaluation.environment import env_settings
from lib.test.analysis.module1_learning import check_bundle as check_learning,hash_json,protocol as learning_protocol
from lib.test.analysis.module1_fit_audit import check_bundle as check_fit,HEADS,geometry
from lib.test.analysis.module1_numerics import read_archive,load_np
from lib.test.analysis.module1_gates import read_source
from lib.test.analysis.module1_evidence import decode_candidates,ious
from lib.test.analysis.module1_reweight import protocol,frame_weights,smoke_indices,confirmation_ids,check_bundle
from lib.test.analysis.frozen_failure_diagnostics import CONFIGS,dump_json,write_bundle
from tracking.diagnose_vdrm_frozen_failures import load_models,config_for,state_hash
from tracking.diagnose_vdrm_native_float import npz_bytes
from tracking.diagnose_vdrm_module1_numerics import backend_state
from tracking.train_vdrm_module1_learning import dataset_plans,preflight_native,torch_prefix,suffix_forward,evaluate
from tracking.vdrm_diagnostics.module1_numeric_suffix import CachedSuffix


def fit_balanced(cache_dir,index,suffix,spec,source_training):
    device=next(suffix.parameters()).device;torch.manual_seed(spec['seed'])
    adapter=ResidualAdapter(width=spec['width'],space='full').to(device)
    initial=state_hash(adapter)
    if initial!=source_training['initial_sha256']:raise ValueError('initialization differs from Step08')
    order=[np.random.default_rng(spec['seed']+epoch).permutation(len(index['rows'])).tolist() for epoch in range(spec['epochs'])]
    order_hash=hash_json(order)
    if not spec['smoke_only'] and order_hash!=source_training['sample_order_sha256']:raise ValueError('training order differs from Step08')
    weights,counts=frame_weights(index['rows'],spec['smoke_only'])
    optimizer=torch.optim.AdamW(adapter.parameters(),lr=spec['learning_rate'],weight_decay=spec['weight_decay'])
    before=state_hash(suffix);updates=0;epochs=[]
    for epoch,indices in enumerate(order):
        metrics=[]
        for i in indices:
            row=index['rows'][i];path=cache_dir/row['member']
            if file_hash(path)!=row['sha256']:raise ValueError('cache changed during training')
            a=load_np(path.read_bytes());p=torch_prefix(a,device)
            base={k:torch.from_numpy(a[k]).to(device) for k in HEADS[:3]}
            optimizer.zero_grad(set_to_none=True)
            delta=adapter(p['search_tokens'],p['prototypes'],p['part_valid'],p['mixture'],p['delta'])
            actual=(p['search_tokens']+delta)-p['search_tokens']
            if float(actual.norm()/p['delta'].norm().clamp_min(1e-12))>1.005:raise RuntimeError('training norm budget exceeded')
            out=suffix_forward(suffix,p,delta)
            if updates==0 and any(not torch.equal(out[k],base[k]) for k in base):raise RuntimeError('zero initialization differs from native cache')
            loss,parts=learning_loss(out,base,torch.from_numpy(a['good']).to(device),torch.from_numpy(a['target']).to(device),p['hann'],row['protect'])
            weighted=loss*weights[i];weighted.backward()
            norm=torch.nn.utils.clip_grad_norm_(adapter.parameters(),spec['gradient_clip'],error_if_nonfinite=True)
            optimizer.step();updates+=1
            parts.update(weighted_total=float(weighted.detach()),gradient_norm=float(norm));metrics.append(parts)
        epochs.append(dict(epoch=epoch+1,**{k:float(np.mean([r[k] for r in metrics])) for k in metrics[0]}))
        print(f'REWEIGHT TRAIN {epoch+1}/{spec["epochs"]}: raw loss={epochs[-1]["total"]:.6f}, weighted={epochs[-1]["weighted_total"]:.6f}',flush=True)
    adapter.zero_grad(set_to_none=True);adapter.eval().requires_grad_(False);after=state_hash(suffix)
    if before!=after or any(p.grad is not None or p.requires_grad for p in suffix.parameters()):raise RuntimeError('changed frozen suffix')
    final=state_hash(adapter)
    if final==initial or not adapter.value.weight.count_nonzero():raise RuntimeError('no learned correction')
    return adapter,dict(initial_sha256=initial,final_sha256=final,sample_order_sha256=order_hash,parameter_count=sum(p.numel() for p in adapter.parameters()),
        updates=updates,epochs=epochs,frame_weights=weights,group_counts=counts,suffix_before=before,suffix_after=after)


@torch.no_grad()
def evaluate_cache(cache_dir,index,suffix,adapters,old_fit_data):
    device=next(suffix.parameters()).device;rows=[];payloads={}
    old_cases={c['member']:c for c in json.loads(old_fit_data['cases.json']) if c['population']=='train'}
    old_rows={(r['case_id'],r['mode']):r for r in json.loads(old_fit_data['rows.json'])}
    for i,case in enumerate(index['rows']):
        a=load_np((cache_dir/case['member']).read_bytes());p=torch_prefix(a,device)
        base={k:torch.from_numpy(a[k]).to(device) for k in HEADS[:3]}
        updates={'Rdisc':p['delta']}
        for arm,adapter in adapters.items():updates[arm]=adapter(p['search_tokens'],p['prototypes'],p['part_valid'],p['mixture'],p['delta'])
        for arm,delta in updates.items():
            out=suffix_forward(suffix,p,delta);arrays={k:out[k].cpu().numpy() for k in HEADS};arrays['hann']=a['hann']
            if arm=='Rdisc':
                if any(not np.array_equal(arrays[k],a[k]) for k in HEADS):raise RuntimeError('fixed train native parity failed')
                arrays.update(good=a['good'],target=a['target'])
            if arm in ('Rdisc','uniform'):
                old=old_rows[old_cases[case['member']]['case_id'],'Rdisc' if arm=='Rdisc' else 'full']
                reference=load_np(old_fit_data[old['arrays_member']])
                if any(not np.array_equal(arrays[k],reference[k]) for k in HEADS):raise RuntimeError('fixed control differs from Step09')
            _,loss=learning_loss(out,base,torch.from_numpy(a['good']).to(device),torch.from_numpy(a['target']).to(device),p['hann'],case['protect'])
            boxes=decode_candidates(arrays,case,case['image_shape_hw']);peak=int((arrays['score_map']*a['hann']).argmax())
            actual=((p['search_tokens']+delta)-p['search_tokens']).cpu().numpy();ratio=geometry(actual,a['delta'])['norm_ratio']
            if ratio>1.005:raise RuntimeError('fixed train norm exceeded')
            member=f'fit/{i:04d}_{arm}.npz';payloads[member]=npz_bytes(arrays)
            rows.append(dict(cache_index=i,arm=arm,arrays_member=member,loss=loss,iou=float(ious(boxes,case['gt_xywh'])[peak]),norm_ratio=ratio))
        if (i+1)%128==0:print(f'FIXED TRAIN AUDIT: {i+1}/{len(index["rows"])}',flush=True)
    return rows,payloads


def verify_uniform_control(rows,payloads,old_data):
    old_rows={(r['population'],r['sequence'],r['frame_1based'],r['scope'],r['arm']):r for r in json.loads(old_data['rows.json'])}
    checked=0
    for r in rows:
        if r['population']=='confirmation' or r['arm']=='balanced':continue
        arm='full' if r['arm']=='uniform' else r['arm']
        previous=old_rows[r['population'],r['sequence'],r['frame_1based'],r['scope'],arm]
        if any(r[k]!=previous[k] for k in ('anchor_xywh','bbox_xywh','input_sha256')):raise RuntimeError('control native trajectory differs from Step08')
        a=load_np(payloads[r['arrays_member']]);b=load_np(old_data[previous['arrays_member']])
        if any(not np.array_equal(a[k],b[k]) for k in HEADS[:3]):raise RuntimeError('control head differs from Step08')
        checked+=1
    if not checked:raise RuntimeError('no source control checked')
    return checked


def run(save_dir,data_root,learning_dir,fit_path,source_path,output_dir,smoke=False):
    output_dir=Path(output_dir)
    if output_dir.exists():raise FileExistsError('refusing to overwrite reweight experiment')
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    print('Checking Step08/09 provenance, cached training and new confirmation split...',flush=True)
    learning_dir=Path(learning_dir);learning_path=learning_dir/'module1_learning.tar.gz';cache_dir=learning_dir/'cache'
    check_learning(learning_path);old,old_data=read_archive(learning_path)
    check_fit(fit_path);fit_manifest,fit_data=read_archive(fit_path);source=read_source(source_path)
    if fit_manifest['learning_sha256']!=file_hash(learning_path) or source['sha256']!=old['source_sha256'] or source['sha256']!=fit_manifest['source_sha256']:
        raise ValueError('Step01/08/09 evidence chain mismatch')
    if not smoke and (old['protocol']['smoke_only'] or fit_manifest['smoke_only']):raise ValueError('formal experiment requires formal Step08 and Step09')
    if backend_state()!=old['backend'] or backend_state()!=fit_manifest['backend']:raise ValueError('backend differs from source experiments')
    source_training=json.loads(old_data['training.json'])['full'];original_index=json.loads(old_data['train_index.json'])
    if file_hash(cache_dir/'index.json')!=source_training['train_cache_sha256']:raise ValueError('training cache index mismatch')
    indices=smoke_indices(original_index['rows']) if smoke else list(range(len(original_index['rows'])))
    index={'rows':[original_index['rows'][i] for i in indices]}
    for row in index['rows']:
        if file_hash(cache_dir/row['member'])!=row['sha256']:raise ValueError('training cache hash mismatch')
    spec=protocol(smoke);weights,counts=frame_weights(index['rows'],smoke)
    partition=dict(train=[],audit=old['partition']['audit'][:1] if smoke else old['partition']['audit'],confirmation=confirmation_ids(smoke))
    plans,list_hash=dataset_plans(data_root,learning_protocol(smoke),partition,source)
    if list_hash!=old['data_list_sha256']:raise ValueError('GOT training sequence list differs from Step08')
    paths={a:Path(save_dir)/'checkpoints/train/ostrack'/CONFIGS[a]/'OSTrack_ep0300.pth.tar' for a in ('Tclean','Rdisc')}
    models,_,provenance=load_models(paths,'cuda',arms=('Tclean','Rdisc'));cfg=config_for(CONFIGS['Rdisc'])
    for arm in models:
        for k in ('checkpoint_sha256','state_sha256_before'):
            if provenance[arm][k]!=old['provenance'][arm][k]:raise ValueError('base checkpoint mismatch')
    preflight_native(plans,models,cfg,source)
    suffix=CachedSuffix(models['Rdisc']).cuda();uniform=ResidualAdapter(space='full').cuda().eval().requires_grad_(False)
    uniform.load_state_dict({k:torch.from_numpy(v).cuda() for k,v in load_np(old_data['weights_full.npz']).items()})
    if state_hash(uniform)!=source_training['final_sha256']:raise ValueError('uniform control weight mismatch')
    output_dir.mkdir(parents=True,exist_ok=False)
    (output_dir/'protocol.json').write_bytes(dump_json(dict(protocol=spec,partition=partition,group_counts=counts,frame_weights=weights)))
    print(f'STEP10 PRECHECK PASSED: train groups={counts}; confirmation={len(partition["confirmation"])}; one reweighted fit',flush=True)
    balanced,training=fit_balanced(cache_dir,index,suffix,spec,source_training)
    torch.save(dict(kind='module1_step10_balanced',protocol=spec,training=training,source_checkpoint_sha256=provenance['Rdisc']['checkpoint_sha256'],
        net={k:v.cpu() for k,v in balanced.state_dict().items()}),output_dir/'adapter_balanced.pth')
    adapters={'uniform':uniform,'balanced':balanced}
    fit_rows,payloads=evaluate_cache(cache_dir,index,suffix,adapters,fit_data)
    del suffix;torch.cuda.empty_cache()
    rows,eval_plan,visuals,native_payloads=evaluate(plans,models,cfg,adapters,source,adapter_modes=('uniform','balanced'))
    checked=verify_uniform_control(rows,native_payloads,old_data);payloads.update(native_payloads)
    for arm,model in models.items():
        provenance[arm]['state_sha256_after']=state_hash(model)
        if provenance[arm]['state_sha256_before']!=provenance[arm]['state_sha256_after']:raise RuntimeError('frozen base changed')
    for arm,adapter in adapters.items():
        expected=source_training['final_sha256'] if arm=='uniform' else training['final_sha256']
        if state_hash(adapter)!=expected or any(p.grad is not None or p.requires_grad for p in adapter.parameters()):
            raise RuntimeError('evaluation changed learned weights or accumulated gradients')
        payloads['weights_'+arm+'.npz']=npz_bytes({k:v.cpu().numpy() for k,v in adapter.state_dict().items()})
    manifest=dict(kind='vdrm_module1_step10_reweight',version=1,protocol=spec,provenance=provenance,
        source_sha256=source['sha256'],learning_sha256=file_hash(learning_path),fit_sha256=file_hash(fit_path),
        source_training_smoke=old['protocol']['smoke_only'],source_initial_sha256=source_training['initial_sha256'],
        source_sample_order_sha256=source_training['sample_order_sha256'],source_full_sha256=source_training['final_sha256'],
        data_list_sha256=list_hash,source_control_parity=True,source_control_rows=checked,automatic_performance_pass=False,backend=backend_state())
    payloads.update({'rows.json':dump_json(rows),'evaluation_plan.json':dump_json(eval_plan),'visual_cases.json':dump_json(visuals),
        'training.json':dump_json(training),'train_index.json':dump_json(index),'fit_rows.json':dump_json(fit_rows)})
    target=output_dir/'module1_reweight.tar.gz';write_bundle(target,manifest,payloads);summary,_=check_bundle(target)
    write_bundle(output_dir/'module1_reweight_analysis.tar.gz',dict(kind='vdrm_module1_step10_reweight_analysis',version=1,evidence_sha256=file_hash(target)),{'summary.json':dump_json(summary)})
    print('STEP10 REWEIGHT CHECK PASSED; results require analysis, no performance approval',flush=True)
    print('Download:',target.resolve(),flush=True);print('Download:',(output_dir/'module1_reweight_analysis.tar.gz').resolve(),flush=True)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--save_dir',default='./output');p.add_argument('--data_root')
    p.add_argument('--learning_dir');p.add_argument('--fit');p.add_argument('--source');p.add_argument('--output_dir');p.add_argument('--smoke',action='store_true')
    args=p.parse_args();root=Path(args.save_dir)/'analysis';torch.set_num_threads(2)
    run(args.save_dir,args.data_root or str(Path(env_settings().got10k_path)/'train'),args.learning_dir or root/'module1_step08',
        args.fit or root/'module1_step09/module1_fit.tar.gz',args.source or root/'module1_step01/module1_evidence.tar.gz',
        args.output_dir or root/('module1_step10_smoke' if args.smoke else 'module1_step10'),args.smoke)


if __name__=='__main__':main()
