"""Step11: selected training-frame memorization, not tracking performance."""

import json
import numpy as np
import torch

from lib.train.frozen_vdrm import visual_hash
from .module1_numerics import read_archive, load_np
from .module1_fit_audit import HEADS, geometry, head_loss
from .module1_direction import decode_metrics

ARMS = ('single_task', 'single_teacher', 'shared_task', 'shared_teacher')
CONTROLS = ('Rdisc', 'full', 'oracle_full')


def protocol(smoke=False):
    return dict(version=1, smoke_only=smoke, seed=42, width=32,
                steps=2 if smoke else 500, checkpoints=[0, 2] if smoke else [0, 10, 50, 100, 200, 500],
                learning_rate=.001, weight_decay=.0001, gradient_clip=1., total_norm_budget=1.,
                shared_gradient='mean_over_all_selected_frames_per_update',
                teacher_objective='squared_residual_error_divided_by_original_squared_norm',
                checkpoint_selection='fixed_last_step', selection='training_oracle_repaired_failures',
                deployable_model=False, tracking_performance_passed=False)


def select_cases(cases, rows, smoke=False):
    by_key = {(r['case_id'], r['mode']): r for r in rows}
    eligible = [c for c in cases if c['population'] == 'train' and c['oracle_eligible']]
    repaired = [c for c in eligible if by_key[c['case_id'], 'Rdisc']['hann_iou'] < .5
                <= by_key[c['case_id'], 'oracle_full']['hann_iou']]
    # Synthetic smoke fixtures need not contain an oracle repair. This fallback
    # is deliberately forbidden for formal experiments.
    selected = (repaired + [c for c in eligible if c not in repaired])[:2] if smoke else repaired
    if not selected:
        raise ValueError('no eligible training cases; refusing to substitute evaluation cases')
    return selected


def teacher_error(delta, teacher, original):
    return float(np.sum((np.asarray(delta, dtype=float) - teacher) ** 2) /
                 max(float(np.sum(np.asarray(original, dtype=float) ** 2)), 1e-30))


def check_bundle(path):
    m, data = read_archive(path)
    spec = m.get('protocol', {})
    if (m.get('kind') != 'vdrm_module1_step11_learnability' or m.get('version') != 1
            or spec != protocol(spec.get('smoke_only', False))):
        raise ValueError('wrong learnability protocol')
    if m['suffix_before'] != m['suffix_after'] or not m['source_control_parity']:
        raise ValueError('frozen suffix or source replay changed')
    cases = json.loads(data['cases.json']); rows = json.loads(data['rows.json'])
    training = json.loads(data['training.json']); source_cases = json.loads(data['source_cases.json'])
    source_rows = json.loads(data['source_rows.json'])
    selected = select_cases(source_cases, source_rows, spec['smoke_only'])
    ids = [c['case_id'] for c in cases]
    if ids != [c['case_id'] for c in selected] or len(ids) != len(set(ids)):
        raise ValueError('selection changed')
    for c, old in zip(cases, selected):
        if any(c[k] != v for k, v in old.items()):
            raise ValueError('source case metadata changed')
    expected = {(i, a, s) for i in ids for a in ARMS for s in spec['checkpoints']}
    expected |= {(i, a, 0) for i in ids for a in CONTROLS}
    if len(rows) != len(expected) or {(r['case_id'], r['arm'], r['step']) for r in rows} != expected:
        raise ValueError('missing or duplicate checkpoint outputs')
    expected_runs = {a + '/' + i for a in ARMS[:2] for i in ids} | {a for a in ARMS[2:]}
    if len(training) != len(expected_runs) or {t['run_id'] for t in training} != expected_runs:
        raise ValueError('missing training runs')
    allowed = {'manifest.json', 'cases.json', 'rows.json', 'training.json', 'source_cases.json', 'source_rows.json'}
    source_lookup = {(r['case_id'], r['mode']): r for r in source_rows}
    cache = {c['case_id']: load_np(data[c['cache_member']]) for c in cases}
    teachers = {c['case_id']: load_np(data[c['source_vectors_member']]) for c in cases}
    for c in cases:
        allowed.update((c['cache_member'], c['source_vectors_member']))
        if c['population'] != 'train' or int(cache[c['case_id']]['good'].sum()) != c['good_cells']:
            raise ValueError('invalid training support')
    ci = {c['case_id']: c for c in cases}
    for r in rows:
        c = ci[r['case_id']]; base = cache[r['case_id']]; vectors = teachers[r['case_id']]
        a = load_np(data[r['arrays_member']]); allowed.add(r['arrays_member'])
        if any(not np.isfinite(v).all() for v in a.values()):
            raise ValueError('nonfinite output')
        g = geometry(a['actual_delta'], base['delta'])
        if not 0 <= g['norm_ratio'] <= 1.005 or any(not np.isclose(g[k], r['geometry'][k], atol=1e-7) for k in g):
            raise ValueError('wrong residual geometry')
        error = teacher_error(a['actual_delta'], vectors['oracle_full'], base['delta'])
        if not np.isclose(error, r['teacher_error'], atol=1e-7):
            raise ValueError('wrong teacher error')
        metrics, _ = decode_metrics(a, c, base['hann'])
        loss = head_loss(a, base, base['good'], base['target'], base['hann'], c['protect'])
        for k, v in metrics.items():
            if (v is None) != (r[k] is None) or v is not None and not np.isclose(v, r[k], atol=1e-7):
                raise ValueError('wrong decoded metric')
        if any(not np.isclose(v, r['loss'][k], atol=2e-5, rtol=1e-5) for k, v in loss.items()):
            raise ValueError('wrong task loss')
        if r['arm'] in CONTROLS:
            reference = source_lookup[r['case_id'], r['arm']]
            allowed.add(reference['arrays_member'])
            old = load_np(data[reference['arrays_member']])
            if any(not np.array_equal(a[k], old[k]) for k in HEADS):
                raise ValueError('source control parity failed')
            if not np.array_equal(a['actual_delta'], vectors[r['arm']]):
                raise ValueError('source control residual changed')
        elif r['step'] == 0:
            if any(not np.array_equal(a[k], base[k]) for k in HEADS):
                raise ValueError('zero initialization is not native')
    for t in training:
        shared = t['run_id'].startswith('shared_')
        fitted = ids if shared else [t['run_id'].split('/')[1]]
        if (t['case_ids'] != fitted or t['updates'] != spec['steps']
                or t['frame_exposures'] != len(fitted) * spec['steps']
                or t['initial_sha256'] != m['source_initial_sha256']
                or t['parameter_count'] != 73728 or len(t['trace']) != spec['steps']):
            raise ValueError('training budget or initialization changed')
        for i, step in enumerate(t['trace'], 1):
            if step['step'] != i or any(not np.isfinite(v) or v < 0 for k, v in step.items() if k != 'step'):
                raise ValueError('invalid optimization trace')
        weights = load_np(data[t['weights_member']]); allowed.add(t['weights_member'])
        digest = visual_hash({'probe.' + k: torch.from_numpy(v) for k, v in weights.items()})
        if digest != t['final_sha256'] or digest == t['initial_sha256']:
            raise ValueError('incorrect weights or no learning')
    if set(data) != allowed:
        raise ValueError('missing or extra evidence members')
    results = {}
    for arm in (*CONTROLS, *ARMS):
        last = [r for r in rows if r['arm'] == arm and r['step'] == (spec['steps'] if arm in ARMS else 0)]
        results[arm] = dict(frames=len(last), mean_iou=float(np.mean([r['hann_iou'] for r in last])),
                           repairs=sum(ci[r['case_id']]['baseline_iou'] < .5 <= r['hann_iou'] for r in last),
                           mean_task_loss=float(np.mean([r['loss']['total'] for r in last])),
                           mean_teacher_error=float(np.mean([r['teacher_error'] for r in last])))
    return dict(smoke_only=spec['smoke_only'], selected_training_frames=len(cases),
                training_runs=len(training), optimizer_updates=sum(t['updates'] for t in training),
                tracking_performance_passed=False, posthoc_memorization_only=True, results=results), cases, rows
