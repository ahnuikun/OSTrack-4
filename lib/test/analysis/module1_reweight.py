"""Fixed Step10 class weighting and fresh sequence confirmation protocol."""

import json
import numpy as np
import torch

from lib.train.frozen_vdrm import visual_hash
from .module1_learning import PROJECT,sequence_partition,summarize
from .module1_numerics import read_archive,load_np
from .module1_gates import protocol_plan
from .module1_evidence import decode_candidates,ious
from .module1_fit_audit import head_loss

MODES=('Rdisc','Tclean','uniform','balanced')
MASS={'supported_failure':.5,'healthy':.4,'no_good':.1}


def protocol(smoke=False):
    return dict(version=1,smoke_only=smoke,seed=42,epochs=1 if smoke else 5,width=32,
        learning_rate=.001,weight_decay=.0001,gradient_clip=1.,total_norm_budget=1.,
        group_mass=MASS,confirmation_sequences=1 if smoke else 16,
        checkpoint_selection='fixed_last_epoch',changed_factor='per_frame_loss_weight',
        independent_benchmark=False)


def group(row):
    if not row['good_cells']:return 'no_good'
    if row['protect']:return 'healthy'
    if row['baseline_iou']<.5:return 'supported_failure'
    raise ValueError('inconsistent training frame label')


def frame_weights(rows,smoke=False):
    labels=[group(r) for r in rows];counts={k:labels.count(k) for k in MASS}
    if not rows or not smoke and any(v==0 for v in counts.values()):raise ValueError('formal reweighting requires all three training groups')
    active=sum(MASS[k] for k,n in counts.items() if n)
    weights=[MASS[k]/active*len(rows)/counts[k] for k in labels]
    return weights,counts


def smoke_indices(rows):
    selected=[]
    for label in MASS:
        matching=[i for i,r in enumerate(rows) if group(r)==label]
        if matching:selected.append(matching[0])
    selected += [i for i in range(len(rows)) if i not in selected][:max(0,4-len(selected))]
    return sorted(selected)


def confirmation_ids(smoke=False):
    old,forbidden=sequence_partition(False)
    excluded=set(forbidden)|set(old['train'])|set(old['audit'])
    available=sorted({int(v) for v in (PROJECT/'lib/train/data_specs/got10k_vot_train_split.txt').read_text().split()}-excluded)
    ids=sorted(np.random.default_rng(101042).choice(available,16,replace=False).tolist())
    return ids[:1] if smoke else ids


def native_checks(rows,plans,data):
    keys={(r['population'],r['sequence'],r['frame_1based'],r['scope'],r['arm']):r for r in rows}
    expected={(p['population'],p['sequence'],f,s,a) for p in plans for f in range(2,p['stop']+1) for s in ('shared','closed') for a in MODES}
    if len(rows)!=len(expected) or set(keys)!=expected:raise ValueError('incomplete native reweight evaluation')
    max_error=0.
    for p in plans:
        for arm in MODES:
            previous=p['initial_box']
            for f in range(2,p['stop']+1):
                r=keys[p['population'],p['sequence'],f,'closed',arm]
                if r['anchor_xywh']!=previous:raise ValueError('broken native feedback')
                previous=r['bbox_xywh']
        for f in range(2,p['stop']+1):
            base=keys[p['population'],p['sequence'],f,'closed','Rdisc']
            for scope in ('shared','closed'):
                for arm in MODES:
                    r=keys[p['population'],p['sequence'],f,scope,arm]
                    if any(r[k]!=base[k] for k in ('gt_xywh','cover','absence','visible')):raise ValueError('inconsistent native labels')
                    if scope=='shared' and any(r[k]!=base[k] for k in ('anchor_xywh','input_sha256')):raise ValueError('shared input changed')
                    a=load_np(data[r['arrays_member']])
                    if any(not np.isfinite(v).all() for v in a.values()):raise ValueError('nonfinite head')
                    boxes=decode_candidates(a,r,r['image_shape_hw']);peak=int((a['score_map']*a['hann']).argmax())
                    error=float(np.abs(boxes[peak]-np.asarray(r['bbox_xywh'])).max());max_error=max(max_error,error)
                    if error>.002:raise ValueError('native decode mismatch')
                    valid=min(r['gt_xywh'][2:])>0
                    if r['visible']!=(valid and r['cover']>0 and r['absence']==0):raise ValueError('invalid visibility')
                    if valid and not np.isclose(ious(np.asarray([r['bbox_xywh']]),r['gt_xywh'])[0],r['iou'],atol=1e-7):raise ValueError('wrong IoU')
                    if not np.isfinite(r['actual_norm_ratio']) or not 0<=r['actual_norm_ratio']<=1.005:raise ValueError('native norm exceeded')
    return max_error


def check_bundle(path):
    m,data=read_archive(path);spec=m.get('protocol',{})
    if m.get('kind')!='vdrm_module1_step10_reweight' or m.get('version')!=1 or spec!=protocol(spec.get('smoke_only',False)):
        raise ValueError('wrong reweight protocol')
    if m.get('source_control_parity') is not True or m.get('automatic_performance_pass') is not False:raise ValueError('invalid reweight claims')
    for v in m['provenance'].values():
        if v['state_sha256_before']!=v['state_sha256_after']:raise ValueError('frozen base changed')
    training=json.loads(data['training.json']);index=json.loads(data['train_index.json']);rows=json.loads(data['rows.json'])
    plans=json.loads(data['evaluation_plan.json']);fit=json.loads(data['fit_rows.json']);visuals=json.loads(data['visual_cases.json'])
    weights,counts=frame_weights(index['rows'],spec['smoke_only'])
    if weights!=training['frame_weights'] or counts!=training['group_counts']:raise ValueError('wrong frame weights')
    if training['updates']!=len(weights)*spec['epochs'] or training['parameter_count']!=73728:raise ValueError('wrong training budget')
    if training['initial_sha256']!=m['source_initial_sha256'] or training['final_sha256']==training['initial_sha256']:raise ValueError('wrong initialization or missing learning')
    if not spec['smoke_only'] and training['sample_order_sha256']!=m['source_sample_order_sha256']:raise ValueError('sample order changed')
    if training['suffix_before']!=training['suffix_after'] or len(training['epochs'])!=spec['epochs']:raise ValueError('training contract changed')
    old,_=sequence_partition(m['source_training_smoke'])
    expected_train=set(old['train'])
    actual_train={r['sequence_id'] for r in index['rows']}
    if (any(r['population']!='train' for r in index['rows']) or not actual_train<=expected_train
            or not spec['smoke_only'] and actual_train!=expected_train):raise ValueError('training partition changed')
    confirmation=[p for p in plans if p['population']=='confirmation'];audit=[p for p in plans if p['population']=='audit'];regression=[p for p in plans if p['population']=='regression']
    expected_audit=old['audit'][:1] if spec['smoke_only'] else old['audit']
    if sorted(p['sequence_id'] for p in confirmation)!=confirmation_ids(spec['smoke_only']) or sorted(p['sequence_id'] for p in audit)!=expected_audit:
        raise ValueError('confirmation or audit partition changed')
    if len(plans)!=len(audit)+len(confirmation)+5 or len(regression)!=5 or {p['sequence'] for p in regression}!=set(protocol_plan(spec['smoke_only'])):
        raise ValueError('wrong regression set')
    for p in regression:
        if p['stop']!=protocol_plan(spec['smoke_only'])[p['sequence']][1]:raise ValueError('wrong regression length')
    for arm,expected_hash in [('uniform',m['source_full_sha256']),('balanced',training['final_sha256'])]:
        arrays=load_np(data['weights_'+arm+'.npz'])
        if visual_hash({'probe.'+k:torch.from_numpy(v) for k,v in arrays.items()})!=expected_hash:raise ValueError('wrong learned weights')
    if len(fit)!=len(weights)*3 or {(r['cache_index'],r['arm']) for r in fit}!={(i,a) for i in range(len(weights)) for a in ('Rdisc','uniform','balanced')}:
        raise ValueError('incomplete fixed training evaluation')
    bases={r['cache_index']:load_np(data[r['arrays_member']]) for r in fit if r['arm']=='Rdisc'}
    for r in fit:
        case=index['rows'][r['cache_index']];base=bases[r['cache_index']];a=load_np(data[r['arrays_member']])
        loss=head_loss(a,base,base['good'],base['target'],base['hann'],case['protect'])
        if any(not np.isclose(loss[k],r['loss'][k],atol=2e-5,rtol=1e-5) for k in loss):raise ValueError('fixed train loss mismatch')
        boxes=decode_candidates(a,case,case['image_shape_hw']);peak=int((a['score_map']*base['hann']).argmax())
        iou=float(ious(boxes,case['gt_xywh'])[peak])
        if not np.isclose(iou,r['iou'],atol=1e-7):raise ValueError('fixed train IoU mismatch')
        if not np.isfinite(r['norm_ratio']) or not 0<=r['norm_ratio']<=1.005:raise ValueError('fit norm exceeded')
    allowed={'manifest.json','training.json','train_index.json','rows.json','evaluation_plan.json','fit_rows.json','visual_cases.json','weights_uniform.npz','weights_balanced.npz'}
    allowed|={r['arrays_member'] for r in rows+fit}|{v['image_member'] for v in visuals}
    if set(data)!=allowed:raise ValueError('extra or missing reweight evidence')
    error=native_checks(rows,plans,data)
    fits={}
    for label in ('all',*MASS):
        ids={i for i,c in enumerate(index['rows']) if label=='all' or group(c)==label}
        for arm in ('Rdisc','uniform','balanced'):
            rr=[r for r in fit if r['cache_index'] in ids and r['arm']==arm]
            if rr:fits[label+'/'+arm]=dict(frames=len(rr),loss_mean=float(np.mean([r['loss']['total'] for r in rr])),iou_mean=float(np.mean([r['iou'] for r in rr])),
                repairs=sum(index['rows'][r['cache_index']]['baseline_iou']<.5<=r['iou'] for r in rr),losses=sum(r['iou']<.5<=index['rows'][r['cache_index']]['baseline_iou'] for r in rr))
    return dict(smoke_only=spec['smoke_only'],training_updates=training['updates'],group_counts=counts,max_decode_error_pixels=error,
        tracking_performance_passed=False,independent_benchmark=False,fit=fits,
        versus_Rdisc=summarize(rows,('audit','confirmation','regression'),MODES),
        versus_uniform=summarize(rows,('audit','confirmation','regression'),MODES,'uniform')),rows
