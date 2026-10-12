"""Step09: fixed-checkpoint fitting audit and bounded GT-assisted diagnostics."""

import json
import numpy as np
import torch

from lib.train.module1_learning import learning_loss
from .module1_numerics import read_archive, load_np
from .module1_direction import decode_metrics

HEADS = ('score_map', 'size_map', 'offset_map', 'final_kept')
MODES = ('Rdisc', 'zero', 'prototype', 'full')


def oracle_selection(rows, smoke=False):
    """Only training labels choose training diagnostics; old cases are fixed."""
    failures = [i for i,r in enumerate(rows) if r['good_cells'] and r['baseline_iou'] < .5]
    healthy = [i for i,r in enumerate(rows) if r['good_cells'] and r['protect']]
    partial = [i for i in healthy if 0 < rows[i]['cover'] <= 4]
    normal = [i for i in healthy if i not in partial]
    def sample(values,n):
        return [values[i] for i in np.unique(np.linspace(0,len(values)-1,min(n,len(values)),dtype=int))] if values else []
    return sorted(set(sample(failures,1 if smoke else 12)+sample(partial,1 if smoke else 6)+sample(normal,1 if smoke else 6)))


def geometry(delta, original):
    a=np.asarray(delta,dtype=np.float64).reshape(-1);b=np.asarray(original,dtype=np.float64).reshape(-1)
    bn=max(float(np.linalg.norm(b)),1e-30);an=float(np.linalg.norm(a))
    return dict(norm_ratio=an/bn,change_ratio=float(np.linalg.norm(a-b))/bn,
                cosine=float(a@b)/max(an*bn,1e-30))


def projected_proposal(current, gradient, original, prototypes, valid, space, step=.25):
    """Fixed-length descent proposal, capped by TOTAL original-update norm."""
    if space not in ('prototype','full'): raise ValueError('unknown oracle space')
    direction=-gradient.detach()
    if space=='prototype':
        p=prototypes[0,valid[0]].double()
        if len(p):
            _,s,vh=torch.linalg.svd(p,full_matrices=False)
            basis=vh[:int((s>max(float(s[0])*1e-6,1e-12)).sum())].to(direction)
            direction=(direction@basis.T)@basis
        else: direction=direction*0
    radius=original.norm()
    proposal=current.detach()+step*radius*direction/direction.norm().clamp_min(1e-30)
    return proposal*(radius/proposal.norm().clamp_min(1e-30)).clamp(max=1.)


def head_loss(arrays, base, good, target, hann, protect):
    out={k:torch.from_numpy(arrays[k]) for k in HEADS[:3]}
    baseline={k:torch.from_numpy(base[k]) for k in HEADS[:3]}
    _,parts=learning_loss(out,baseline,torch.from_numpy(good),torch.from_numpy(target),torch.from_numpy(hann),protect)
    return parts


def summarize(cases,rows):
    groups={}
    for population in ('train','regression'):
        for group in ('all','supported_failure','healthy','no_good'):
            selected=[c for c in cases if c['population']==population and
                (group=='all' or group=='supported_failure' and c['good_cells']>0 and c['baseline_iou']<.5
                 or group=='healthy' and c['protect'] or group=='no_good' and c['good_cells']==0)]
            ids={c['case_id'] for c in selected}
            for mode in MODES+('oracle_prototype','oracle_full'):
                ss=[r for r in rows if r['case_id'] in ids and r['mode']==mode]
                if not ss:continue
                base={r['case_id']:r for r in rows if r['mode']=='Rdisc'}
                groups[f'{population}/{group}/{mode}']=dict(frames=len(ss),
                    loss_mean=float(np.mean([r['loss']['total'] for r in ss])),
                    loss_delta_mean=float(np.mean([r['loss']['total']-base[r['case_id']]['loss']['total'] for r in ss])),
                    hann_delta_pp=float(np.mean([r['hann_iou']-base[r['case_id']]['hann_iou'] for r in ss if r['hann_iou'] is not None])*100) if any(r['hann_iou'] is not None for r in ss) else None,
                    repairs=sum(base[r['case_id']]['hann_iou']<.5<=r['hann_iou'] for r in ss if r['hann_iou'] is not None),
                    losses=sum(r['hann_iou']<.5<=base[r['case_id']]['hann_iou'] for r in ss if r['hann_iou'] is not None))
    return groups


def check_bundle(path):
    m,data=read_archive(path)
    if (m.get('kind')!='vdrm_module1_step09_fit_audit' or m.get('version')!=1
            or m.get('parameter_updates')!=0 or m.get('deployable_oracle') is not False
            or m.get('total_norm_budget')!=1. or m.get('step_size')!=.25
            or m.get('oracle_steps')!=(2 if m.get('smoke_only') else 8)):
        raise ValueError('wrong fit audit protocol')
    if m['suffix_before']!=m['suffix_after'] or m['adapter_before']!=m['adapter_after']:
        raise ValueError('fit audit changed frozen weights')
    cases=json.loads(data['cases.json']);rows=json.loads(data['rows.json'])
    ci={c['case_id']:c for c in cases}
    if len(ci)!=len(cases) or len(cases)!=m['cases']:raise ValueError('duplicate or missing cases')
    if (sum(c['population']=='train' for c in cases)!=m['train_cases']
            or sum(c['population']=='regression' for c in cases)!=m['regression_cases']
            or m['train_cases']+m['regression_cases']!=len(cases)):
        raise ValueError('wrong case populations')
    expected={(c['case_id'],a) for c in cases for a in MODES+(('oracle_prototype','oracle_full') if c['oracle_eligible'] else ())}
    if len(rows)!=len(expected) or {(r['case_id'],r['mode']) for r in rows}!=expected:raise ValueError('missing fit audit mode')
    members={'manifest.json','cases.json','rows.json'}|{c['base_member'] for c in cases}|{r['arrays_member'] for r in rows}
    members|={c['vectors_member'] for c in cases if c.get('vectors_member')}
    if set(data)!=members:raise ValueError('missing or extra fit evidence')
    for c in cases:
        a=load_np(data[c['base_member']])
        if int(a['good'].sum())!=c['good_cells']:raise ValueError('wrong positive support')
        if c['oracle_eligible']!=(c['oracle_selected'] and c['good_cells']>0):raise ValueError('wrong oracle eligibility')
    for r in rows:
        c=ci[r['case_id']];base=load_np(data[c['base_member']]);a=load_np(data[r['arrays_member']])
        if any(not np.isfinite(v).all() for v in a.values()):raise ValueError('nonfinite head')
        metrics,_=decode_metrics(a,c,base['hann']);loss=head_loss(a,base,base['good'],base['target'],base['hann'],c['protect'])
        for k,v in metrics.items():
            if (v is None)!=(r[k] is None) or v is not None and not np.isclose(v,r[k],atol=1e-7):raise ValueError('wrong decode metrics')
        for k,v in loss.items():
            if not np.isclose(v,r['loss'][k],atol=2e-5,rtol=1e-5):raise ValueError('wrong loss')
        if any(not np.isfinite(v) for v in r['geometry'].values()) or not 0<=r['geometry']['norm_ratio']<=1.005:
            raise ValueError('norm budget exceeded')
        if r['mode']=='Rdisc' and any(not np.array_equal(a[k],base[k]) for k in HEADS):raise ValueError('native parity failed')
        if c.get('vectors_member'):
            vectors=load_np(data[c['vectors_member']]);g=geometry(vectors[r['mode']],vectors['Rdisc'])
            if any(not np.isclose(g[k],r['geometry'][k],atol=1e-7) for k in g):raise ValueError('wrong geometry')
            if 'gradient' in vectors and r['mode'] in ('prototype','full'):
                change=vectors[r['mode']].astype(float)-vectors['Rdisc'].astype(float);grad=vectors['gradient'].astype(float)
                dot=float(np.sum(change*grad))
                if not np.isclose(dot,r['linear_loss_change'],atol=1e-7,rtol=1e-5):raise ValueError('wrong gradient alignment')
        if r['mode'].startswith('oracle_'):
            trace=r['trace']
            if len(trace)!=m['oracle_steps']:raise ValueError('wrong oracle iteration count')
            previous=head_loss(base,base,base['good'],base['target'],base['hann'],c['protect'])['total']
            for step in trace:
                if abs(step['before']-previous)>3e-5:raise ValueError('broken oracle trace')
                if step['accepted']!=(step['proposal']<=step['before']):raise ValueError('wrong oracle acceptance')
                previous=step['proposal'] if step['accepted'] else step['before']
            if abs(previous-r['loss']['total'])>3e-5:raise ValueError('oracle final loss mismatch')
    return dict(smoke_only=m['smoke_only'],cases=len(cases),rows=len(rows),parameter_updates=0,
                tracking_performance_passed=False,groups=summarize(cases,rows)),cases,rows
