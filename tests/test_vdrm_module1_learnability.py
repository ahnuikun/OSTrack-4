"""Train-only selection, optimizer semantics, head replay and corruption checks."""

from copy import deepcopy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pytest
import torch

from test_vdrm_module1_fit import fixture, TinySuffix
from lib.train.module1_learning import ResidualAdapter
from lib.test.analysis.module1_fit_audit import HEADS
from lib.test.analysis.module1_learnability import ARMS, CONTROLS, protocol, select_cases, check_bundle
from lib.test.analysis.module1_numerics import load_np
from lib.test.analysis.frozen_failure_diagnostics import dump_json, write_bundle
from tracking.diagnose_vdrm_native_float import npz_bytes
from tracking.diagnose_vdrm_frozen_failures import state_hash
from tracking.diagnose_vdrm_module1_learnability import prepare, record, fit_run, adapter_delta


def test_selection_never_uses_evaluation_cases_or_unrepaired_formal_frames():
    cases = [dict(case_id=str(i), population=p, oracle_eligible=True) for i, p in enumerate(('train', 'train', 'regression'))]
    rows = [dict(case_id=str(i), mode=a, hann_iou=v) for i, pair in enumerate(((.2, .8), (.2, .3), (.1, .9)))
            for a, v in zip(('Rdisc', 'oracle_full'), pair)]
    assert [c['case_id'] for c in select_cases(cases, rows)] == ['0']
    assert [c['case_id'] for c in select_cases(cases[1:], rows, True)] == ['1']
    with pytest.raises(ValueError):
        select_cases(cases[1:], rows)


def prepared_case():
    _, payloads, c, old_rows = fixture()
    vectors = load_np(payloads[c['vectors_member']])
    # Expand the toy suffix's exact inputs to the production 768 channels.
    x = np.zeros((1, 256, 768), np.float32); x[0, :, 0] = np.linspace(-1, 1, 256, dtype=np.float32)
    delta = np.zeros_like(x); delta[..., 1] = .01
    a = dict(template_tokens=np.zeros((1, 64, 768), np.float32), search_tokens=x, delta=delta,
             indices=np.arange(256, dtype=np.float32)[None], ce_template_mask=np.ones((1, 64), bool),
             prototypes=np.eye(4, 768, dtype=np.float32)[None], part_valid=np.ones((1, 4), bool),
             mixture=np.ones_like(x), hann=np.ones((16, 16), np.float32), good=np.arange(256) == 136,
             target=np.array([.35, .35, .3, .3], np.float32))
    # Use the source base to preserve torch.linspace's exact float32 values.
    tx = torch.linspace(-1, 1, 256).numpy(); a['search_tokens'][0, :, 0] = tx
    for k, v in vectors.items():
        vectors[k] = np.pad(v, ((0, 0), (0, 0), (0, 760)))
    suffix = TinySuffix()
    with torch.no_grad():
        a.update({k: v.numpy() for k, v in suffix(None, torch.from_numpy(a['search_tokens']), None, None, torch.from_numpy(delta)).items()})
    c = dict(c, cache_member='cache/test.npz', source_vectors_member='source_vectors/test.npz')
    return prepare(a, c, vectors, 'cpu'), suffix, old_rows, payloads


def evidence():
    c, suffix, old_rows, old_payloads = prepared_case(); spec = protocol(True)
    torch.manual_seed(42); initial = state_hash(ResidualAdapter())
    before = state_hash(suffix); payloads = {}; rows = []; training = []
    source_case = {k: v for k, v in c['c'].items() if k not in ('cache_member', 'source_vectors_member')}
    for arm in CONTROLS:
        old = next(r for r in old_rows if r['mode'] == arm)
        payloads[old['arrays_member']] = old_payloads[old['arrays_member']]
        rows.append(record(c, suffix, torch.from_numpy(c['vectors'][arm]), arm, 0, payloads))
    for arm in ARMS:
        t, rr = fit_run([c], suffix, arm, spec, payloads, initial); training.append(t); rows.extend(rr)
    payloads[c['c']['cache_member']] = npz_bytes(c['a'])
    payloads[c['c']['source_vectors_member']] = npz_bytes(c['vectors'])
    payloads.update({'cases.json': dump_json([c['c']]), 'rows.json': dump_json(rows), 'training.json': dump_json(training),
                     'source_cases.json': dump_json([source_case]), 'source_rows.json': dump_json(old_rows)})
    m = dict(kind='vdrm_module1_step11_learnability', version=1, protocol=spec, suffix_before=before,
             suffix_after=state_hash(suffix), source_control_parity=True, source_initial_sha256=initial)
    return m, payloads, suffix


def test_real_fitting_roundtrip_and_frozen_suffix(tmp_path):
    m, p, suffix = evidence(); path = tmp_path / 'evidence.tar.gz'; write_bundle(path, m, p)
    summary, _, rows = check_bundle(path)
    assert summary['optimizer_updates'] == 8 and len(rows) == 11
    assert summary['posthoc_memorization_only'] and not summary['tracking_performance_passed']
    assert all(p.grad is None and not p.requires_grad for p in suffix.parameters())


def test_shared_teacher_uses_one_mean_gradient_step():
    first, suffix, _, _ = prepared_case(); second = deepcopy(first)
    second['teacher'] = -first['teacher']; second['vectors']['oracle_full'] = -first['vectors']['oracle_full']
    second['c']['case_id'] = 'train_0001'
    torch.manual_seed(42); reference = ResidualAdapter(); initial = state_hash(reference)
    optimizer = torch.optim.AdamW(reference.parameters(), lr=.001, weight_decay=.0001)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        losses = [(adapter_delta(reference, c) - c['teacher']).square().sum() / c['p']['delta'].square().sum() for c in (first, second)]
        (sum(losses) / 2).backward(); torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.); optimizer.step()
    payloads = {}; t, _ = fit_run([first, second], suffix, 'shared_teacher', protocol(True), payloads, initial)
    weights = load_np(payloads[t['weights_member']])
    for k, v in reference.state_dict().items():
        np.testing.assert_allclose(weights[k], v.detach().numpy(), atol=1e-7, rtol=1e-6)
    assert t['updates'] == 2 and t['frame_exposures'] == 4


@pytest.mark.parametrize('fault', ['loss', 'iou', 'teacher', 'budget', 'missing', 'weights', 'selection', 'updates', 'frozen', 'initial'])
def test_checker_rejects_corruption(tmp_path, fault):
    m, p, _ = evidence(); rows = json.loads(p['rows.json']); training = json.loads(p['training.json'])
    if fault == 'loss': rows[-1]['loss']['total'] += .1
    elif fault == 'iou': rows[-1]['hann_iou'] += .1
    elif fault == 'teacher': rows[-1]['teacher_error'] += .1
    elif fault == 'budget': rows[-1]['geometry']['norm_ratio'] = 2.
    elif fault == 'missing': rows.pop()
    elif fault == 'weights': training[0]['final_sha256'] = 'wrong'
    elif fault == 'selection':
        cases = json.loads(p['cases.json']); cases[0]['population'] = 'regression'; p['cases.json'] = dump_json(cases)
    elif fault == 'updates': training[-1]['updates'] += 1
    elif fault == 'frozen': m['suffix_after'] = 'changed'
    elif fault == 'initial': m['source_initial_sha256'] = 'changed'
    p['rows.json'] = dump_json(rows); p['training.json'] = dump_json(training)
    path = tmp_path / 'bad.tar.gz'; write_bundle(path, m, p)
    with pytest.raises(ValueError):
        check_bundle(path)
