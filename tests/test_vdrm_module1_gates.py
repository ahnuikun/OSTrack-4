"""Gate semantics, failure restoration, bundle contracts and CLI wiring."""

from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from lib.models.layers.vdrm import VisibilityDrivenRepresentationModule
from lib.test.analysis.frozen_failure_diagnostics import dump_json, write_bundle
from lib.test.analysis.module1_gates import (
    ARMS, MODES, SHARED_MODES, CONFIGURATIONS, check_bundle, protocol_plan, summarize, validate_gate_arrays)
from tracking import analyze_vdrm_module1_gates as analyzer
from tracking import diagnose_vdrm_module1_gates as entry
from tracking.diagnose_vdrm_native_float import npz_bytes
from tracking.vdrm_diagnostics.module1_gate_interventions import gate_intervention


def module_fixture():
    module = VisibilityDrivenRepresentationModule(spatial_gate_mode='part_aligned')
    model = SimpleNamespace(backbone=SimpleNamespace(vdrm=module))
    torch.manual_seed(42)
    similarity, prototypes = torch.randn(1, 4, 12), torch.randn(1, 4, 8)
    q = torch.tensor([[.2, .4, .8, 0.]])
    valid = torch.tensor([[True, True, True, False]])
    return model, (similarity, prototypes, q, valid)


@pytest.mark.parametrize('mode', SHARED_MODES)
def test_gate_semantics_and_norm_controls(mode):
    model, args = module_fixture()
    m = model.backbone.vdrm
    alpha = m.alpha.detach().clone()
    with torch.no_grad():
        baseline = m._part_aligned_statistics(*args)
        with gate_intervention(model, mode) as values:
            out = m._part_aligned_statistics(*args)
            base = mode[:-5] if mode.endswith('_norm') else mode
            q = args[2] if base not in ('qflat', 'bothflat') else torch.tensor([[1.4/3, 1.4/3, 1.4/3, 0.]])
            g = baseline[1] if base not in ('routeflat', 'bothflat') else baseline[1].mean(-1, keepdim=True).expand_as(baseline[1])
            expected = sum(q[0, i] * g[0, i, :, None] * args[1][0, i] for i in range(4)) / 3
            if mode.endswith('_norm'):
                expected *= baseline[2].norm() / expected.norm()
                torch.testing.assert_close(out[2].norm(), baseline[2].norm())
            torch.testing.assert_close(out[2][0], expected)
            torch.testing.assert_close(values['effective_q'], q)
            torch.testing.assert_close(out[1], baseline[1], rtol=0, atol=0)
            if mode == 'control':
                torch.testing.assert_close(out[2], baseline[2], rtol=0, atol=0)
            if mode == 'alpha0':
                assert m.alpha.item() == 0
        assert '_part_aligned_statistics' not in m.__dict__ and m.inference_ablation is None
        torch.testing.assert_close(m.alpha, alpha, rtol=0, atol=0)


def test_restoration_on_exception_and_preexisting_override():
    model, args = module_fixture()
    m = model.backbone.vdrm
    original = m._part_aligned_statistics
    m._part_aligned_statistics = original
    alpha = m.alpha.detach().clone()
    with pytest.raises(RuntimeError, match='sentinel'):
        with gate_intervention(model, 'alpha0'):
            raise RuntimeError('sentinel')
    assert m._part_aligned_statistics is original and m.inference_ablation is None
    assert torch.equal(m.alpha, alpha)
    m.inference_ablation = 'part_mean'
    with pytest.raises(ValueError):
        with gate_intervention(model, 'qflat'):
            pass
    assert m.inference_ablation == 'part_mean'


def array_fixture(mode):
    model, args = module_fixture()
    with torch.no_grad(), gate_intervention(model, mode) as values:
        model.backbone.vdrm._part_aligned_statistics(*args)
        a = {k: v.numpy().copy() for k, v in values.items()}
    alpha = 0 if mode == 'alpha0' else -.056
    a.update(alpha=np.array(alpha, np.float32), indices=np.arange(12)[None],
             injected_energy=(a['residual_norm'] * alpha)**2,
             injected_relative_norm=a['residual_norm'] * abs(alpha) / 2)
    a.update(score_map=np.zeros((1, 1, 16, 16), np.float32), size_map=np.full((1, 2, 16, 16), .25, np.float32),
             offset_map=np.zeros((1, 2, 16, 16), np.float32), hann=np.ones((1, 1, 16, 16), np.float32),
             final_kept=np.zeros((1, 256), bool), padding=np.zeros((256, 256), bool))
    a['score_map'][0, 0, 8, 8] = .9
    a['final_kept'][0, :12] = True
    return a


@pytest.mark.parametrize('mode', SHARED_MODES)
def test_cpu_formula_checks_and_wrong_effective_gate_rejected(mode):
    a = array_fixture(mode)
    validate_gate_arrays(a, mode)
    a['effective_q'][0, 0] += .02
    with pytest.raises(ValueError, match='effective gates'):
        validate_gate_arrays(a, mode)


def bundle_fixture(path, corrupt=None, smoke=True):
    plan = protocol_plan(smoke)
    provenance = {arm: dict(checkpoint_sha256='a'*64, state_sha256_before='b'*64, state_sha256_after='b'*64) for arm in ARMS}
    box = [60., 60., 40., 40.]
    source = dict(sha256='c'*64, provenance=provenance, initial_boxes={name: box for name in plan}, frames=[])
    rows, payloads = [], {}
    cached = {mode: array_fixture(mode) for mode in SHARED_MODES}
    for key in ('score_map', 'size_map', 'offset_map', 'final_kept'):
        cached['alpha0']['tclean_'+key] = cached['alpha0'][key].copy()
    packed = {mode: npz_bytes(a) for mode, a in cached.items()}
    for name, (start, stop) in plan.items():
        for frame in range(start, stop+1):
            common = dict(sequence=name, frame_1based=frame, gt_xywh=box, cover=8., absence=0.,
                          image_shape_hw=[160, 160], image_sha256='d'*64)
            native = dict(anchor_xywh=box, bbox_xywh=box, input_sha256={'search': 'e'*64}, resize_factor=1.6, iou=1.)
            source['frames'].append(dict(common, native={arm: native for arm in ARMS}))
            for scope, modes in [('closed', MODES), ('shared', SHARED_MODES)]:
                for mode in modes:
                    used_scope = 'closed' if mode == 'control' else scope
                    member = f'arrays/{name}/{frame}_{used_scope}_{mode}.npz'
                    payloads[member] = packed[mode]
                    rows.append(dict(common, **native, scope=scope, mode=mode, arrays_member=member,
                                     exact_replay=True, native_input_verified=True, alpha0_exact_Tclean=True))
    if corrupt:
        corrupt(rows, source, payloads)
    count = sum(stop-start+1 for start, stop in plan.values())
    manifest = dict(kind='vdrm_module1_step03_gates', version=1, dataset='got10k_vdrm_dev', smoke_only=smoke,
        modes=list(MODES), shared_modes=list(SHARED_MODES), plan={k: list(v) for k, v in plan.items()},
        configurations={a: CONFIGURATIONS[a] for a in ARMS}, optimizer_steps=0, source_verified=True,
        source_sha256=source['sha256'], provenance=provenance, native_steps=count*5, forward_calls=count*18)
    write_bundle(path, manifest, {'source_reference.json': dump_json(source), 'rows.json': dump_json(rows), **payloads})


def test_full_scheduled_bundle_and_cpu_entry(tmp_path):
    source, output = tmp_path / 'evidence.tar.gz', tmp_path / 'analysis.tar.gz'
    bundle_fixture(source, smoke=False)
    with patch.object(sys, 'argv', ['analyze_vdrm_module1_gates.py', '--bundle', str(source), '--output', str(output)]):
        analyzer.main()
    summary, _ = check_bundle(source)
    assert summary['native_steps'] == 2495 and summary['forward_calls'] == 8982
    assert summary['records'] == 6487 and summary['max_decode_error_pixels'] == 0
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        analyzer.run(source, output)
    assert output.read_bytes() == before


@pytest.mark.parametrize('kind', ['duplicate', 'chain', 'source', 'shared', 'labels', 'alpha0', 'array'])
def test_corrupted_contracts_rejected(tmp_path, kind):
    def corrupt(rows, source, payloads):
        if kind == 'duplicate': rows[-1] = deepcopy(rows[0])
        elif kind == 'chain': rows[1]['anchor_xywh'] = [61, 60, 40, 40]
        elif kind == 'source': rows[0]['input_sha256'] = {'search': 'f'*64}
        elif kind == 'shared': rows[6]['input_sha256'] = {'search': 'f'*64}
        elif kind == 'labels': rows[0]['cover'] = 0
        elif kind == 'alpha0':
            row = next(r for r in rows if r['scope'] == 'shared' and r['mode'] == 'alpha0')
            row['alpha0_exact_Tclean'] = False
        else:
            a = array_fixture('qflat')
            a['injected_energy'] *= 2
            payloads[rows[1]['arrays_member']] = npz_bytes(a)
    path = tmp_path / 'bad.tar.gz'
    bundle_fixture(path, corrupt)
    with pytest.raises(ValueError):
        check_bundle(path)


def test_no_visible_frames_are_not_nan_or_performance_pass():
    rows = [dict(sequence='x', frame_1based=2, scope=scope, mode=mode, visible=False)
            for scope, modes in [('closed', MODES), ('shared', SHARED_MODES)] for mode in modes]
    result = summarize(rows, False)
    assert result['scopes']['shared']['sequence_macro_delta_iou']['qflat'] is None
    assert result['performance_passed'] is False


def test_cli_defaults_and_smoke_are_real_parser_paths():
    for smoke in (False, True):
        args = ['diagnose_vdrm_module1_gates.py', '--save_dir', './output'] + (['--smoke'] if smoke else [])
        with patch.object(sys, 'argv', args), patch.object(entry, 'run') as launch:
            entry.main()
        actual = launch.call_args.args
        assert actual[0] == './output' and actual[3] == smoke
        assert Path(actual[1]).as_posix() == 'output/analysis/module1_step01/module1_evidence.tar.gz'
        assert Path(actual[2]).name == ('module1_gates_smoke.tar.gz' if smoke else 'module1_gates.tar.gz')


def test_source_parity_mismatch_is_actionable():
    row = dict(bbox_xywh=[1, 2, 3, 4], anchor_xywh=[1, 2, 3, 4], input_sha256={})
    reference = dict(sequence='x', frame_1based=2, native={'Rdisc': deepcopy(row)})
    row['bbox_xywh'][0] = 2
    with pytest.raises(RuntimeError, match='differs from step01'):
        entry.assert_source_native(row, reference, 'control')
