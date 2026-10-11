"""CPU checks for Step07; numerical validation is distinct from artifact validity."""

import hashlib
import io
import json
from pathlib import PurePosixPath
import tarfile

import numpy as np

from .module1_direction import OBJECTIVES, SPACES, BUDGETS, project_gradient, decode_metrics, objective_value
from .module1_evidence import _file_hash
from .module1_gates import ARMS, CONFIGURATIONS

PRECISIONS = ('native32', 'strict32', 'float64')
EPSILONS = {'native32': (.00025, .001, .004, .016),
            'strict32': (.00025, .001, .004, .016),
            'float64': (.00000001, .0000001, .000001, .00001, .0001, .001)}
HEAD_KEYS = ('score_map', 'size_map', 'offset_map', 'final_kept', 'ce_removed')
NATIVE_GRADIENT_RTOL = 1e-5


def gradient_relative_error(actual, reference):
    return float(np.linalg.norm(actual.astype(float)-reference.astype(float)) /
                 max(np.linalg.norm(reference.astype(float)),1e-30))


def selected_cases(cases, smoke):
    eligible = [c for c in cases if c['exclusion_reason'] is None]
    if not smoke:
        return eligible
    # Include the key repair and two failing Step06 derivatives, if available.
    keys = [('GOT-10k_Train_003240', 93), ('GOT-10k_Train_008333', 61), ('GOT-10k_Train_008311', 93)]
    priority = [c for key in keys for c in eligible if (c['sequence'], c['frame_1based']) == key]
    return (priority + [c for c in eligible if c not in priority])[:3]


def schedule(precision):
    rows = [('zero', None, None, 0.), ('repeat', None, None, 0.)]
    rows += [('oracle', o, s, b) for o in OBJECTIVES for s in SPACES for b in BUDGETS]
    rows += [('fd', o, s, sign * eps) for o in OBJECTIVES for s in SPACES
             for eps in EPSILONS[precision] for sign in (-1., 1.)]
    if precision == 'native32':
        rows.insert(0, ('original', None, None, 1.))
    return rows


def requested_update(prefix, objective, space, budget, precision):
    """Same Step06 directions at all precisions; only arithmetic changes."""
    g = prefix['source_gradient_' + objective].astype(np.float64)
    if space == 'prototype':
        g, _ = project_gradient(g, prefix['prototypes'], prefix['part_valid'])
    norm = float(np.linalg.norm(prefix['reference_delta'].astype(np.float64)))
    length = float(np.linalg.norm(g))
    dtype = np.float64 if precision == 'float64' else np.float32
    return (g * (norm * budget / length)).astype(dtype) if length else np.zeros_like(g, dtype=dtype)


def actual_update(prefix, delta, precision):
    dtype = np.float64 if precision == 'float64' else np.float32
    search = prefix['search_tokens'].astype(dtype)
    return (search + delta) - search


def finite_difference_diagnostic(prefix, case, precision, objective, space, eps, zero, plus, minus, signatures):
    g = prefix['gradient_' + precision + '_' + objective].astype(np.float64)
    direction = requested_update(prefix, objective, space, 1., precision)
    plus_delta = actual_update(prefix, requested_update(prefix, objective, space, eps, precision), precision)
    minus_delta = actual_update(prefix, requested_update(prefix, objective, space, -eps, precision), precision)
    analytic = float((g * direction).sum())
    # Account separately for rounding of the actual addition to the cached search.
    rounded = float((g * ((plus_delta.astype(float) - minus_delta.astype(float)) / (2 * eps))).sum())
    margins = [objective_value(a, case['pair'], objective, prefix['hann'].astype(a['score_map'].dtype)) for a in (zero, plus, minus)]
    numeric = (margins[1] - margins[2]) / (2 * eps)
    scale = max(abs(rounded), 1e-15)
    same_ce = all(np.array_equal(a['ce_removed'], zero['ce_removed']) for a in (plus, minus))
    same_relu = len(set(signatures)) == 1
    def clipping(a):
        score=a['score_map'].reshape(-1)[case['pair']]
        return (score<=np.asarray(1e-4,dtype=score.dtype),score>=np.asarray(1-1e-4,dtype=score.dtype))
    same_clip=all(np.array_equal(clipping(a),clipping(zero)) for a in (plus,minus))
    threshold = 1e-10 if precision == 'float64' else 1e-5
    resolvable = 2 * eps * abs(rounded) >= threshold
    error = abs(numeric - rounded) / scale
    return dict(sequence=case['sequence'], frame_1based=case['frame_1based'], precision=precision,
                objective=objective, space=space, epsilon=eps, analytic=analytic, rounded_analytic=rounded,
                numeric=numeric, relative_error=error, ideal_relative_error=abs(numeric-analytic)/max(abs(analytic),1e-15),
                plus_slope=(margins[1]-margins[0])/eps, minus_slope=(margins[0]-margins[2])/eps,
                same_ce=same_ce, same_relu=same_relu, same_clip=same_clip, resolvable=resolvable,
                stable_pass=same_ce and same_relu and same_clip and resolvable and error <= .1)


def convergence(diagnostics):
    result=[]
    keys=sorted({(r['sequence'],r['frame_1based'],r['precision'],r['objective'],r['space']) for r in diagnostics})
    for key in keys:
        rs=sorted([r for r in diagnostics if (r['sequence'],r['frame_1based'],r['precision'],r['objective'],r['space'])==key],key=lambda r:r['epsilon'])
        adjacent=[(a['epsilon'],b['epsilon']) for a,b in zip(rs,rs[1:]) if a['stable_pass'] and b['stable_pass']]
        result.append(dict(sequence=key[0],frame_1based=key[1],precision=key[2],objective=key[3],space=key[4],
                           converged=bool(adjacent),adjacent_passing_epsilons=adjacent))
    return result


def read_archive(path):
    contents={}
    with tarfile.open(path,'r|gz') as t:
        manifest=None
        for member in t:
            n=member.name
            if (not member.isfile() or n in contents or PurePosixPath(n).is_absolute()
                    or '..' in PurePosixPath(n).parts or '\\' in n):
                raise ValueError('unsafe/duplicate numerics archive member')
            data=t.extractfile(member).read()
            if manifest is None:
                if n!='manifest.json':raise ValueError('manifest must be first')
                manifest=json.loads(data)
            elif hashlib.sha256(data).hexdigest()!=manifest['members_sha256'].get(n):
                raise ValueError('numerics checksum mismatch')
            contents[n]=data
    if manifest is None or set(contents)!=set(manifest['members_sha256'])|{'manifest.json'}:
        raise ValueError('incomplete numerics archive')
    return manifest,contents


def load_np(data):
    with np.load(io.BytesIO(data),allow_pickle=False) as a:
        result={k:a[k] for k in a.files}
    if any(not np.isfinite(v).all() for v in result.values()):raise ValueError('nonfinite numerics array')
    return result


def check_bundle(path):
    m,data=read_archive(path)
    if (m.get('kind')!='vdrm_module1_step07_numerics' or m.get('version')!=1 or type(m.get('smoke_only')) is not bool
            or m.get('precisions')!=list(PRECISIONS) or m.get('epsilons')!={k:list(v) for k,v in EPSILONS.items()}
            or m.get('optimizer_steps')!=0 or m.get('source_verified') is not True
            or m.get('configurations')!={a:CONFIGURATIONS[a] for a in ARMS}
            or any(len(m.get(k,''))!=64 for k in ('source_sha256','direction_sha256'))):
        raise ValueError('wrong numeric protocol')
    for arm in ARMS:
        p,old=m['provenance'][arm],m['source_provenance'][arm]
        if p['state_sha256_before']!=p['state_sha256_after'] or any(p[k]!=old[k] for k in ('checkpoint_sha256','state_sha256_before')):
            raise ValueError('numeric checkpoint changed')
    if m.get('suffix_states_before')!=m.get('suffix_states_after'):raise ValueError('numeric suffix changed weights')
    if m.get('backend_before')!=m.get('backend_after'):raise ValueError('numeric backend was not restored')
    all_cases=json.loads(data['source_cases.json']);cases=json.loads(data['cases.json']);rows=json.loads(data['rows.json'])
    expected=selected_cases(all_cases,m['smoke_only'])
    key=lambda c:(c['sequence'],c['frame_1based'])
    if not cases or [key(c) for c in cases]!=[key(c) for c in expected]:raise ValueError('wrong numeric cases')
    expected_rows={(key(c),p,*mode) for c in cases for p in PRECISIONS for mode in schedule(p)}
    ri={(key(r),r['precision'],r['mode'],r['objective'],r['space'],r['budget']):r for r in rows}
    if (len(rows)!=len(expected_rows) or set(ri)!=expected_rows or m['forward_calls']!=len(rows)
            or m['backward_calls']!=len(cases)*6):raise ValueError('wrong numeric schedule')
    if set(data)!={'manifest.json','source_cases.json','cases.json','rows.json'}|{c['prefix_member'] for c in cases}|{r['arrays_member'] for r in rows}:
        raise ValueError('unexpected or missing numeric arrays')
    frames,diagnostics,gradients=[],[],[]
    for case,source_case in zip(cases,expected):
        for k in source_case:
            if k!='prefix_member' and case[k]!=source_case[k]:raise ValueError('numeric case changed source metadata')
        prefix=load_np(data[case['prefix_member']])
        for precision in PRECISIONS:
            local={}
            for mode,obj,space,budget in schedule(precision):
                row=ri[key(case),precision,mode,obj,space,budget]
                if row['input_sha256']!=case['input_sha256']:raise ValueError('numeric input hashes changed')
                if row.get('relu_modules')!=4 or len(row.get('relu_sha256',''))!=64:raise ValueError('missing score ReLU signature')
                a=load_np(data[row['arrays_member']]);local[mode,obj,space,budget]=(row,a)
                dtype=np.float64 if precision=='float64' else np.float32
                if any(a[k].dtype!=dtype for k in HEAD_KEYS[:3]):raise ValueError('wrong numeric output dtype')
                if mode in ('zero','repeat'):delta=np.zeros_like(prefix['search_tokens'],dtype=dtype)
                elif mode=='original':delta=prefix['reference_delta']
                else:delta=requested_update(prefix,obj,space,budget,precision)
                actual=actual_update(prefix,delta,precision)
                if not np.isclose(np.linalg.norm(actual.astype(float)),row['actual_update_norm'],rtol=1e-6,atol=1e-10):raise ValueError('wrong applied update norm')
                if precision=='native32' and mode in ('original','zero','oracle'):
                    stem=mode if mode!='oracle' else f'oracle_{obj}_{space}_{budget:g}'
                    for k in HEAD_KEYS:
                        if not np.array_equal(a[k],prefix['source_'+stem+'_'+k]):raise ValueError('cached suffix differs from native source')
                if precision=='native32' and mode=='fd' and space=='full' and abs(budget)==.001:
                    for k in HEAD_KEYS:
                        if not np.array_equal(a[k],prefix[f'source_fd_{obj}_{budget:g}_'+k]):raise ValueError('native finite difference differs from source')
                metrics,_=decode_metrics(a,case,prefix['hann'].astype(dtype))
                frames.append(dict(sequence=case['sequence'],frame_1based=case['frame_1based'],precision=precision,
                                   mode=mode,objective=obj,space=space,budget=budget,**metrics))
            zr,z=local['zero',None,None,0.];rr,repeat=local['repeat',None,None,0.]
            if any(not np.array_equal(z[k],repeat[k]) for k in HEAD_KEYS) or zr['relu_sha256']!=rr['relu_sha256']:
                raise ValueError('repeated zero update is nondeterministic')
            for obj in OBJECTIVES:
                g=prefix['gradient_'+precision+'_'+obj].astype(float);old=prefix['source_gradient_'+obj].astype(float)
                if precision=='native32' and gradient_relative_error(g,old)>NATIVE_GRADIENT_RTOL:
                    raise ValueError('suffix gradient differs from source')
                pg,rank=project_gradient(g,prefix['prototypes'],prefix['part_valid'])
                gradients.append(dict(sequence=case['sequence'],frame_1based=case['frame_1based'],precision=precision,objective=obj,
                    cosine_to_source=float((g*old).sum()/max(np.linalg.norm(g)*np.linalg.norm(old),1e-30)),
                    relative_l2_to_source=float(np.linalg.norm(g-old)/max(np.linalg.norm(old),1e-30)),
                    projected_energy_fraction=float(np.square(pg).sum()/max(np.square(g).sum(),1e-30)),prototype_rank=rank))
                for space in SPACES:
                    for eps in EPSILONS[precision]:
                        pr,plus=local['fd',obj,space,eps];mr,minus=local['fd',obj,space,-eps]
                        diagnostics.append(finite_difference_diagnostic(prefix,case,precision,obj,space,eps,z,plus,minus,
                            [zr['relu_sha256'],pr['relu_sha256'],mr['relu_sha256']]))
    conv=convergence(diagnostics)
    summary=dict(smoke_only=m['smoke_only'],cases=len(cases),forward_calls=len(rows),backward_calls=m['backward_calls'],
        optimizer_steps=0,tracking_performance_passed=False,native_suffix_parity_passed=True,
        direction_sha256=m['direction_sha256'],evidence_sha256=_file_hash(path),
        convergence_by_precision={p:dict(total=sum(r['precision']==p for r in conv),converged=sum(r['precision']==p and r['converged'] for r in conv)) for p in PRECISIONS})
    summary['float64_derivative_audit_passed']=all(r['converged'] for r in conv if r['precision']=='float64')
    return summary,frames,diagnostics,gradients,conv
