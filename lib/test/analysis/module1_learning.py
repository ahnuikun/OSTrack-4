"""Predeclared sequence partition and CPU evidence checks for the Step08 pilot."""

import hashlib
import json
from pathlib import Path

import numpy as np

from lib.train.identity_offline import registered_ids,load_protocol
from lib.test.analysis.identity_guard import partition as guard_partition
from lib.test.evaluation.got10kdataset import _vdrm_dev_sequence_ids
from .module1_numerics import read_archive,load_np
from .module1_evidence import ious,decode_candidates
from .module1_gates import protocol_plan

PROJECT=Path(__file__).resolve().parents[3]
MODES=('Rdisc','Tclean','prototype','full')


def protocol(smoke=False):
    return dict(version=1,smoke_only=bool(smoke),seed=42,train_sequences=2 if smoke else 64,
        audit_sequences=1 if smoke else 16,train_prefix=8 if smoke else 128,frames_per_sequence=2 if smoke else 8,
        audit_prefix=8 if smoke else None,epochs=1 if smoke else 5,width=32,learning_rate=.001,weight_decay=.0001,
        total_update_budget=1.,gradient_clip=1.,checkpoint_selection='fixed_last_epoch',
        precision='native_fp32',same_input_and_native_closed_loop=True,independent_benchmark=False)


def sequence_partition(smoke=False):
    old_train,_=registered_ids(load_protocol());guard=guard_partition(old_train)
    forbidden=set(old_train)|set(guard['fit'])|set(guard['audit'])|set(_vdrm_dev_sequence_ids())
    available=sorted({int(v) for v in (PROJECT/'lib/train/data_specs/got10k_vot_train_split.txt').read_text().split()}-forbidden)
    # Smoke is a strict subset of the fixed formal partition.
    choice=np.random.default_rng(81042).choice(available,80,replace=False).tolist()
    spec=protocol(smoke)
    result=dict(train=sorted(choice[:64])[:spec['train_sequences']],audit=sorted(choice[64:])[:spec['audit_sequences']])
    if set(result['train'])&set(result['audit']) or (set(result['train'])|set(result['audit']))&forbidden:
        raise ValueError('overlapping learning partition')
    return result,sorted(forbidden)


def training_frames(cover,absence,stop,count):
    candidates=np.arange(1,stop)
    hard=candidates[(np.asarray(cover)[1:stop]>0)&(np.asarray(cover)[1:stop]<=4)&(np.asarray(absence)[1:stop]==0)]
    selected=[]
    if len(hard):selected=hard[np.unique(np.linspace(0,len(hard)-1,min(count//2,len(hard)),dtype=int))].tolist()
    remaining=[int(i) for i in candidates if i not in selected]
    if remaining:
        selected += [remaining[i] for i in np.unique(np.linspace(0,len(remaining)-1,min(count-len(selected),len(remaining)),dtype=int))]
    return sorted(f+1 for f in selected)


def hash_json(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def summarize(rows):
    result={}
    for population in ('audit','regression'):
        for scope in ('shared','closed'):
            selected=[r for r in rows if r['population']==population and r['scope']==scope]
            baseline={(r['sequence'],r['frame_1based']):r for r in selected if r['arm']=='Rdisc'}
            for arm in MODES:
                rr=[r for r in selected if r['arm']==arm]
                sequences=sorted({r['sequence'] for r in rr});deltas=[];normal=[];partial=[]
                repairs=losses=harms=gains=0
                for name in sequences:
                    sr=[r for r in rr if r['sequence']==name and r['visible']]
                    def delta(r):return r['iou']-baseline[r['sequence'],r['frame_1based']]['iou']
                    if sr:deltas.append(float(np.mean([delta(r) for r in sr])*100))
                    for values,pred in [(normal,lambda r:r['cover']>=7),(partial,lambda r:0<r['cover']<=4)]:
                        ss=[r for r in sr if pred(r)]
                        if ss:values.append(float(np.mean([delta(r) for r in ss])*100))
                    for r in sr:
                        b=baseline[r['sequence'],r['frame_1based']]['iou'];v=r['iou']
                        repairs+=int(b<.5<=v);losses+=int(v<.5<=b);harms+=int(v-b<-.05);gains+=int(v-b>.05)
                interval=None
                if len(deltas)>=2:
                    rng=np.random.default_rng(42);boot=np.asarray(deltas)[rng.integers(0,len(deltas),(2000,len(deltas)))].mean(1)
                    interval=np.quantile(boot,[.025,.975]).tolist()
                result[f'{population}/{scope}/{arm}']=dict(sequences=len(sequences),visible_sequences=len(deltas),
                    macro_delta_pp=float(np.mean(deltas)) if deltas else None,sequence_bootstrap_95=interval,
                    normal_macro_delta_pp=float(np.mean(normal)) if normal else None,
                    partial_occlusion_macro_delta_pp=float(np.mean(partial)) if partial else None,
                    repairs=repairs,losses=losses,harms_gt5pp=harms,gains_gt5pp=gains)
    return result


def check_bundle(path):
    m,data=read_archive(path)
    p=m.get('protocol',{})
    if m.get('kind')!='vdrm_module1_step08_learning' or m.get('version')!=1 or p!=protocol(p.get('smoke_only',False)):
        raise ValueError('wrong learning protocol')
    partition,forbidden=sequence_partition(p['smoke_only'])
    if m['partition']!=partition or m['forbidden_ids']!=forbidden:raise ValueError('learning partition changed')
    if (m['checkpoint_selection']!='fixed_last_epoch' or m['production_model_modified'] or m['automatic_performance_pass']
            or m.get('native_preflight_passed') is not True):
        raise ValueError('invalid learning claims')
    for a,v in m['provenance'].items():
        if v['state_sha256_before']!=v['state_sha256_after']:raise ValueError('frozen base changed')
    training=json.loads(data['training.json']);rows=json.loads(data['rows.json']);plans=json.loads(data['evaluation_plan.json'])
    train_index=json.loads(data['train_index.json'])
    if (len(train_index['rows'])!=m['train_frames']
            or {r['sequence_id'] for r in train_index['rows']}!=set(partition['train'])
            or any(r['population']!='train' for r in train_index['rows'])):raise ValueError('invalid training membership')
    audit=[v for v in plans if v['population']=='audit'];regression=[v for v in plans if v['population']=='regression']
    if ({v['sequence_id'] for v in audit}!=set(partition['audit']) or len(audit)!=len(partition['audit'])
            or {v['sequence'] for v in regression}!=set(protocol_plan(p['smoke_only']))
            or len(regression)!=5 or len(plans)!=len(audit)+len(regression)):
        raise ValueError('invalid audit or regression membership')
    for v in regression:
        if v['stop']!=protocol_plan(p['smoke_only'])[v['sequence']][1]:raise ValueError('wrong regression prefix')
    if set(training)!={'prototype','full'}:raise ValueError('missing paired training arm')
    a,b=training['prototype'],training['full']
    for k in ('initial_sha256','sample_order_sha256','parameter_count','updates','train_cache_sha256'):
        if a[k]!=b[k]:raise ValueError('unequal paired training: '+k)
    if a['parameter_count']!=3*768*p['width'] or a['updates']!=p['epochs']*m['train_frames']:
        raise ValueError('wrong learning budget')
    for arm,t in training.items():
        if t['initial_sha256']==t['final_sha256'] or t['frozen_suffix_before']!=t['frozen_suffix_after']:
            raise ValueError('missing learning or changed frozen suffix')
        if len(t['epochs'])!=p['epochs'] or t['maximum_update_ratio']>1.005:raise ValueError('invalid training trace')
        import torch
        from lib.train.frozen_vdrm import visual_hash
        weights=load_np(data['weights_'+arm+'.npz'])
        if visual_hash({'probe.'+k:torch.from_numpy(v) for k,v in weights.items()})!=t['final_sha256']:
            raise ValueError('learned weights hash mismatch')
    ri={(r['population'],r['sequence'],r['frame_1based'],r['scope'],r['arm']):r for r in rows}
    expected={(v['population'],v['sequence'],f,s,a) for v in plans for f in range(2,v['stop']+1) for s in ('shared','closed') for a in MODES}
    if len(rows)!=len(expected) or set(ri)!=expected:raise ValueError('incomplete learning evaluation')
    visuals=json.loads(data['visual_cases.json'])
    for v in visuals:
        matched=[plan for plan in audit if plan['sequence']==v['sequence'] and 1<=v['frame_1based']<=plan['stop']]
        if len(matched)!=1:raise ValueError('invalid learning visual case')
    if set(data)!={'manifest.json','training.json','rows.json','evaluation_plan.json','train_index.json','visual_cases.json','weights_prototype.npz','weights_full.npz'}|{r['arrays_member'] for r in rows}|{v['image_member'] for v in visuals}:
        raise ValueError('unexpected or missing learning evidence')
    max_decode_error=0.
    for r in rows:
        a=load_np(data[r['arrays_member']]);boxes=decode_candidates(a,r,r['image_shape_hw'])
        peak=int((a['score_map']*a['hann']).reshape(-1).argmax())
        error=float(np.abs(boxes[peak]-np.asarray(r['bbox_xywh'])).max())
        if error>.002:raise ValueError('learning output does not match native head decode')
        max_decode_error=max(max_decode_error,error)
    for plan in plans:
        key=(plan['population'],plan['sequence'])
        for arm in MODES:
            previous=plan['initial_box']
            for frame in range(2,plan['stop']+1):
                r=ri[key[0],key[1],frame,'closed',arm]
                if r['anchor_xywh']!=previous:raise ValueError('broken closed-loop feedback')
                previous=r['bbox_xywh']
        for frame in range(2,plan['stop']+1):
            base=ri[key[0],key[1],frame,'closed','Rdisc']
            for scope in ('shared','closed'):
                for arm in MODES:
                    r=ri[key[0],key[1],frame,scope,arm]
                    if r['gt_xywh']!=base['gt_xywh'] or r['cover']!=base['cover'] or r['absence']!=base['absence']:
                        raise ValueError('mismatched evaluation labels')
                    if scope=='shared' and (r['input_sha256']!=base['input_sha256'] or r['anchor_xywh']!=base['anchor_xywh']):
                        raise ValueError('shared evaluation used different inputs')
                    expected_iou=float(ious(np.asarray([r['bbox_xywh']]),r['gt_xywh'])[0]) if min(r['gt_xywh'][2:])>0 else None
                    if expected_iou is not None and not np.isclose(expected_iou,r['iou'],atol=1e-7):raise ValueError('wrong learning IoU')
                    visible=min(r['gt_xywh'][2:])>0 and r['cover']>0 and r['absence']==0
                    if r['visible']!=visible:raise ValueError('wrong learning visibility')
                    if r['actual_norm_ratio']>1.005:raise ValueError('evaluation exceeded residual budget')
    result=dict(smoke_only=p['smoke_only'],training_updates_per_arm=training['prototype']['updates'],train_frames=m['train_frames'],
        sequence_isolated=True,independent_benchmark=False,tracking_performance_passed=False,
        max_decode_error_pixels=max_decode_error,groups=summarize(rows))
    return result,rows,training
