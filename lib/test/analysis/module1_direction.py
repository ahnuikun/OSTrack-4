"""CPU protocol and vector checks for a bounded, GT-assisted capacity probe."""

import hashlib
import io
import json
from pathlib import PurePosixPath
import tarfile

import numpy as np

from .module1_evidence import _file_hash, decode_candidates, ious
from .module1_gates import ARMS, CONFIGURATIONS

FRAME_PLAN = {
    'GOT-10k_Train_003240': (88,92,93,94),
    'GOT-10k_Train_001664': (27,32,65,70,74,75,93),
    'GOT-10k_Train_007878': (22,64,72,75,87),
    'GOT-10k_Train_008333': (13,59,61,63),
    'GOT-10k_Train_008311': (38,54,67,80,93),
}
OBJECTIVES = ('raw','hann')
SPACES = ('prototype','full')
BUDGETS = (1.,4.)
FD_EPS = .001


def frame_plan(smoke=False):
    return {n:(2,3) if smoke else frames for n,frames in FRAME_PLAN.items()}


def project_gradient(gradient, prototypes, valid):
    """Orthogonal projection, with unrestricted signed coefficients per token.

    This space is strictly more permissive than the original q * route gates.
    SVD is calculated in CPU float64 and independently reproducible.
    """
    p = np.asarray(prototypes, dtype=np.float64)[0, np.asarray(valid)[0].astype(bool)]
    gradient = np.asarray(gradient, dtype=np.float64)
    if len(p) == 0:
        return np.zeros_like(gradient),0
    _,s,vh = np.linalg.svd(p, full_matrices=False)
    rank = int((s > max(float(s[0])*1e-6,1e-12)).sum())
    basis = vh[:rank]
    return (gradient @ basis.T) @ basis,rank


def normalized_update(direction, norm):
    direction = np.asarray(direction,dtype=np.float64)
    size = np.linalg.norm(direction)
    if size == 0:
        return np.zeros_like(direction,dtype=np.float32)
    return (direction * (norm/size)).astype(np.float32)


def choose_pair(original, zero, prefix, case):
    """Require a good and a low-IoU candidate retained in BOTH baselines.

    A low-IoU candidate is not automatically a different instance. All pairs
    are geometry labels; identity conclusions need the separate reviewed ROIs.
    """
    if case['cover']<=0 or case['absence']!=0 or min(case['gt_xywh'][2:])<=0:
        return None,'not_visible_or_invalid'
    if not prefix['part_valid'].any(): return None,'no_valid_parts'
    if np.linalg.norm(prefix['reference_delta'].astype(float))==0:return None,'zero_reference_update'
    boxes = [decode_candidates(a,case,case['image_shape_hw']) for a in (original,zero)]
    overlaps = [ious(b,case['gt_xywh']) for b in boxes]
    # Any pixel padding disqualifies that cell, as in the earlier evidence.
    padding = prefix['padding'].reshape(16,16,16,16).any(axis=(1,3)).reshape(-1)
    retained = original['final_kept'].reshape(-1).astype(bool) & zero['final_kept'].reshape(-1).astype(bool) & ~padding
    good = retained & (overlaps[0]>=.5) & (overlaps[1]>=.5)
    bad = retained & (overlaps[0]<.2) & (overlaps[1]<.2)
    if not good.any(): return None,'no_common_good_candidate'
    if not bad.any(): return None,'no_common_low_iou_candidate'
    scores = (original['score_map']*prefix['hann']).reshape(-1)
    def best(mask):
        cells=np.flatnonzero(mask)
        return int(cells[np.argmax(scores[cells])])
    return [best(good),best(bad)],None


def objective_value(a,pair,objective,hann):
    score = a['score_map'] if objective=='raw' else a['score_map']*hann
    score = score.reshape(-1)
    return float(score[pair[0]]-score[pair[1]])


def decode_metrics(a,case,hann):
    boxes = decode_candidates(a,case,case['image_shape_hw'])
    valid = min(case['gt_xywh'][2:])>0
    overlap = ious(boxes,case['gt_xywh']) if valid else None
    result={}
    for name,scores in [('hann',a['score_map']*hann),('raw',a['score_map'])]:
        peak=int(scores.reshape(-1).argmax())
        result[name+'_peak']=peak
        result[name+'_iou']=float(overlap[peak]) if valid else None
    result['best_candidate_iou']=float(overlap.max()) if valid else None
    return result,boxes[result['hann_peak']]


def modes(eligible):
    result=[('original',None,None,None),('zero',None,None,None)]
    if eligible:
        result += [('oracle',o,s,b) for o in OBJECTIVES for s in SPACES for b in BUDGETS]
        result += [('finite_difference',o,'full',b) for o in OBJECTIVES for b in (-FD_EPS,FD_EPS)]
    return result


def summarize(frames,cases,smoke):
    groups={}
    for objective in OBJECTIVES:
        for space in SPACES:
            for budget in BUDGETS:
                selected=[f for f in frames if (f['mode'],f['objective'],f['space'],f['budget'])==('oracle',objective,space,budget)]
                groups[f'{objective}/{space}/{budget:g}']=dict(cases=len(selected),
                    pair_margin_improved=sum(f['pair_margin']>f['zero_pair_margin'] for f in selected),
                    pair_margin_crossed_zero=sum(f['zero_pair_margin']<=0<f['pair_margin'] for f in selected),
                    hann_repairs=sum(f['original_hann_iou']<.5<=f['hann_iou'] for f in selected),
                    hann_losses=sum(f['hann_iou']<.5<=f['original_hann_iou'] for f in selected),
                    raw_repairs=sum(f['original_raw_iou']<.5<=f['raw_iou'] for f in selected),
                    raw_losses=sum(f['raw_iou']<.5<=f['original_raw_iou'] for f in selected))
    return dict(smoke_only=smoke,gt_gradient_diagnosis_only=True,tracking_performance_passed=False,
                optimizer_steps=0,global_capacity_bound=False,groups=groups,
                eligible_cases=sum(c['exclusion_reason'] is None for c in cases))


def check_bundle(path):
    contents,seen={},set()
    with tarfile.open(path,'r|gz') as t:
        for m in t:
            n=m.name
            if not m.isfile() or n in seen or PurePosixPath(n).is_absolute() or '..' in PurePosixPath(n).parts or '\\' in n:
                raise ValueError('unsafe/duplicate direction member')
            seen.add(n);data=t.extractfile(m).read()
            if len(seen)==1:
                if n!='manifest.json':raise ValueError('manifest must be first')
                manifest=json.loads(data)
            else:
                if hashlib.sha256(data).hexdigest()!=manifest['members_sha256'].get(n):raise ValueError('direction checksum mismatch')
                contents[n]=data
    if seen!=set(manifest['members_sha256'])|{'manifest.json'}:raise ValueError('incomplete direction bundle')
    smoke=manifest.get('smoke_only')
    if (type(smoke) is not bool or manifest.get('kind')!='vdrm_module1_step06_direction' or manifest.get('version')!=1
            or manifest.get('frame_plan')!={k:list(v) for k,v in frame_plan(smoke).items()}
            or manifest.get('objectives')!=list(OBJECTIVES) or manifest.get('budgets')!=list(BUDGETS)
            or manifest.get('spaces')!=list(SPACES) or manifest.get('fd_epsilon')!=FD_EPS
            or manifest.get('optimizer_steps')!=0 or manifest.get('gt_gradient_diagnosis_only') is not True
            or manifest.get('source_verified') is not True or len(manifest.get('source_sha256',''))!=64
            or manifest.get('configurations')!={a:CONFIGURATIONS[a] for a in ARMS}):
        raise ValueError('wrong direction protocol')
    for arm in ARMS:
        p,old=manifest['provenance'][arm],manifest['source_provenance'][arm]
        if p['state_sha256_before']!=p['state_sha256_after'] or any(p[k]!=old[k] for k in ('checkpoint_sha256','state_sha256_before')):
            raise ValueError('wrong/changed direction checkpoint')
    cases,rows=json.loads(contents['cases.json']),json.loads(contents['rows.json'])
    ci={(c['sequence'],c['frame_1based']):c for c in cases}
    expected={(n,f) for n,fs in frame_plan(smoke).items() for f in fs}
    if len(cases)!=len(expected) or set(ci)!=expected:raise ValueError('missing/duplicate direction case')
    ri={(r['sequence'],r['frame_1based'],r['mode'],r['objective'],r['space'],r['budget']):r for r in rows}
    expected_rows={(c['sequence'],c['frame_1based'],*m) for c in cases for m in modes(c['exclusion_reason'] is None)}
    if len(rows)!=len(expected_rows) or set(ri)!=expected_rows or manifest['forward_calls']!=len(rows):raise ValueError('wrong direction schedule')
    if manifest['backward_calls']!=sum(c['exclusion_reason'] is None for c in cases)*2:raise ValueError('wrong backward count')
    used={'cases.json','rows.json'}|{c['prefix_member'] for c in cases}|{r['arrays_member'] for r in rows}
    if used!=set(contents):raise ValueError('unexpected/missing direction data')
    def load(member):
        with np.load(io.BytesIO(contents[member]),allow_pickle=False) as a:
            result={k:a[k] for k in a.files}
        if any(not np.isfinite(v).all() for v in result.values()):raise ValueError('nonfinite direction evidence')
        return result
    frames,diagnostics=[],[]
    for case in cases:
        name,fr=case['sequence'],case['frame_1based']
        prefix=load(case['prefix_member'])
        original_row=ri[name,fr,'original',None,None,None];zero_row=ri[name,fr,'zero',None,None,None]
        original,zero=load(original_row['arrays_member']),load(zero_row['arrays_member'])
        if original_row['bbox_xywh']!=case['source_bbox_xywh']:raise ValueError('direction original bbox mismatch')
        for label,a in [('original',original),('zero',zero)]:
            for k in ('score_map','size_map','offset_map'):
                if not np.array_equal(a[k],prefix[label+'_'+k]):raise ValueError('direction control map mismatch')
        pair,reason=choose_pair(original,zero,prefix,case)
        if pair!=case['pair'] or reason!=case['exclusion_reason']:raise ValueError('wrong direction candidate pair')
        reference_norm=float(np.linalg.norm(prefix['reference_delta'].astype(float)))
        if not np.isclose(reference_norm,case['reference_norm'],rtol=1e-6,atol=1e-8):raise ValueError('wrong reference norm')
        om,_=decode_metrics(original,case,prefix['hann'])
        per_case={}
        gradients={}
        if reason is None:
            for objective in OBJECTIVES:
                g=prefix['gradient_'+objective]
                pg,rank=project_gradient(g,prefix['prototypes'],prefix['part_valid'])
                if rank!=case['prototype_rank']:raise ValueError('wrong prototype rank')
                gradients[objective]=(g,pg)
                energy=float(np.square(g.astype(float)).sum());pe=float(np.square(pg).sum())
                diagnostics.append(dict(sequence=name,frame_1based=fr,objective=objective,
                    gradient_norm=float(np.sqrt(energy)),projected_gradient_norm=float(np.sqrt(pe)),
                    projected_energy_fraction=pe/energy if energy>0 else None,prototype_rank=rank))
        for mode,objective,space,budget in modes(reason is None):
            row=ri[name,fr,mode,objective,space,budget];a=load(row['arrays_member'])
            if row['input_sha256']!=case['input_sha256']:raise ValueError('direction input hashes changed')
            if a['actual_delta'].shape!=prefix['reference_delta'].shape:raise ValueError('wrong update shape')
            if mode=='original':expected_delta=prefix['reference_delta']
            elif mode=='zero':expected_delta=np.zeros_like(prefix['reference_delta'])
            else:
                g,pg=gradients[objective]
                expected_delta=normalized_update(pg if space=='prototype' else g,reference_norm*budget)
            if not np.allclose(a['actual_delta'],expected_delta,rtol=5e-3,atol=1e-4):raise ValueError('wrong direction/budget update')
            metrics,box=decode_metrics(a,case,prefix['hann'])
            error=float(np.abs(box-row['bbox_xywh']).max())
            if error>.002:raise ValueError('direction native decode mismatch')
            f=dict(sequence=name,frame_1based=fr,mode=mode,objective=objective,space=space,budget=budget,
                **metrics,original_hann_iou=om['hann_iou'],original_raw_iou=om['raw_iou'],decode_error_pixels=error,
                actual_update_norm=float(np.linalg.norm(a['actual_delta'].astype(float))),
                final_kept_changes_vs_zero=int((a['final_kept']!=zero['final_kept']).sum()))
            if objective:
                f.update(pair_margin=objective_value(a,pair,objective,prefix['hann']),
                    zero_pair_margin=objective_value(zero,pair,objective,prefix['hann']))
            frames.append(f);per_case[mode,objective,space,budget]=(a,f)
        if reason is None:
            for d in diagnostics[-2:]:
                objective=d['objective'];g=gradients[objective][0]
                plus,pf=per_case['finite_difference',objective,'full',FD_EPS]
                minus,mf=per_case['finite_difference',objective,'full',-FD_EPS]
                direction=normalized_update(g,reference_norm)
                analytic=float(np.sum(g.astype(float)*direction))
                numeric=(pf['pair_margin']-mf['pair_margin'])/(2*FD_EPS)
                signal=2*FD_EPS*abs(analytic)
                same_ce=np.array_equal(plus['ce_removed'],zero['ce_removed']) and np.array_equal(minus['ce_removed'],zero['ce_removed'])
                d.update(analytic_directional_derivative=analytic,numeric_directional_derivative=numeric,
                    finite_difference_signal=signal,finite_difference_same_ce=same_ce,
                    finite_difference_resolvable=signal>=1e-5,
                    finite_difference_relative_error=abs(numeric-analytic)/max(abs(analytic),1e-12))
    summary=summarize(frames,cases,smoke)
    checkable=[d for d in diagnostics if d['finite_difference_same_ce'] and d['finite_difference_resolvable']]
    summary['finite_difference_checks']=dict(total=len(diagnostics),stable_and_resolvable=len(checkable),
        within_10_percent=sum(d['finite_difference_relative_error']<=.1 for d in checkable))
    summary['derivative_audit_passed']=bool(checkable) and all(d['finite_difference_relative_error']<=.1 for d in checkable)
    summary.update(cases=len(cases),forward_calls=len(rows),backward_calls=manifest['backward_calls'],
        max_decode_error_pixels=max(f['decode_error_pixels'] for f in frames),
        source_sha256=manifest['source_sha256'],evidence_sha256=_file_hash(path))
    return summary,frames,diagnostics
