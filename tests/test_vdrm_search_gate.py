"""Causality, held-out thresholds, deterministic convex fit and safe CLI outputs."""
import contextlib
import inspect
import io
import json
from pathlib import Path
import re
import shlex
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT)); sys.path.insert(0, str(PROJECT/'tests'))
from lib.test.analysis.candidate_cues import npz_bytes
from lib.test.analysis.search_gate import (FEATURES, PROTOCOL, causal_features, fold_map,
    sequence_weights, fit_logistic, predict, calibrate_threshold, proposal_diagnostics,
    dataset_rows, cross_validate, evaluate)
from tracking import screen_vdrm_search_gate as entry
from tracking import analyze_vdrm_search_gate as analyzer
from test_vdrm_search_coverage import synthetic_coverage


def synthetic_rows():
    rng = np.random.default_rng(120)
    ids = np.repeat(np.arange(15), 20)
    x = rng.normal(size=(len(ids), len(FEATURES)))
    x[:, 0] = rng.uniform(size=len(ids))
    y = x[:, 0] < .4
    return dict(features=x, bad=y, sequence_id=ids, normal=np.ones(len(ids), dtype=bool),
                frame=np.tile(np.arange(2, 22), 15), visible=np.ones(len(ids), dtype=bool), native_iou=(~y)*.8)


class GateMathTests(unittest.TestCase):
    def test_features_have_prediction_only_signature_and_no_future_dependency(self):
        self.assertEqual(list(inspect.signature(causal_features).parameters), ['scores', 'boxes', 'anchors'])
        rng = np.random.default_rng(42)
        scores = rng.uniform(size=(20, 256))
        boxes = np.tile([2., 3., 10., 20.], (20, 1)); anchors = boxes.copy()
        full = causal_features(scores, boxes, anchors)
        for stop in (1, 2, 7, 19):
            np.testing.assert_array_equal(full[:stop], causal_features(scores[:stop], boxes[:stop], anchors[:stop]))
        scores[10:] = 0; boxes[10:, :2] += 50
        np.testing.assert_array_equal(full[:10], causal_features(scores, boxes, anchors)[:10])

    def test_gt_labels_cannot_change_features_and_invisible_is_unusable(self):
        a = synthetic_coverage(); first = dataset_rows(10, a)
        a['gt'][1:] = [200., 200., 10., 10.]; a['cover'][1:] = 8; a['absence'][1:] = 0
        changed = dataset_rows(10, a)
        np.testing.assert_array_equal(first['features'], changed['features'])
        self.assertFalse(np.array_equal(first['bad'], changed['bad']))
        a['cover'][1] = 0
        self.assertTrue(dataset_rows(10, a)['bad'][0])

    def test_degenerate_scores_are_finite_and_bad_geometry_rejected(self):
        boxes = np.tile([0., 0., 10., 10.], (4, 1))
        x = causal_features(np.zeros((4, 256)), boxes, boxes)
        self.assertEqual(x.shape, (4, len(FEATURES))); self.assertTrue(np.isfinite(x).all())
        boxes[0, 2] = 0
        with self.assertRaisesRegex(ValueError, 'invalid native'):
            causal_features(np.zeros((4, 256)), boxes, boxes)

    def test_fold_map_does_not_depend_on_labels_or_input_order(self):
        ids = list(range(128)); assignment = fold_map(ids)
        self.assertEqual(assignment, fold_map(ids[::-1]))
        self.assertEqual(sorted(np.bincount(list(assignment.values())).tolist()), [25,25,26,26,26])
        with self.assertRaises(ValueError): fold_map(ids+[0])

    def test_equal_sequence_weights_and_strict_tie_threshold(self):
        groups = np.array([1]*100+[2]*10)
        weight = sequence_weights(groups)
        self.assertAlmostEqual(weight[:100].sum(), .5)
        # 10% in each sequence; at threshold .8 exact ties must not fire.
        risks = np.array([.8]*90+[.9]*10+[.8]*9+[.9])
        result = calibrate_threshold(risks, np.zeros(110, dtype=bool), groups)
        self.assertEqual(result['threshold'], .8)
        self.assertAlmostEqual(result['calibration_fpr'], .1)
        same = calibrate_threshold(np.ones(110), np.zeros(110, dtype=bool), groups)
        self.assertEqual(same['threshold'], 1); self.assertEqual(same['calibration_fpr'], 0)
        with self.assertRaisesRegex(ValueError, 'no usable'):
            calibrate_threshold(risks, np.ones(110, dtype=bool), groups)

    def test_solver_determinism_constant_columns_and_replication_weights(self):
        rows = synthetic_rows(); x = rows['features']; x[:, -1] = 0
        model = fit_logistic(x, rows['bad'], rows['sequence_id'])
        self.assertLessEqual(model['gradient_inf'], PROTOCOL['gradient_tolerance'])
        self.assertEqual(model, fit_logistic(x, rows['bad'], rows['sequence_id']))
        # Duplicating frames of one sequence does not increase its fitting weight.
        duplicate = np.r_[np.arange(len(x)), np.flatnonzero(rows['sequence_id'] == 0)]
        other = fit_logistic(x[duplicate], rows['bad'][duplicate], rows['sequence_id'][duplicate])
        np.testing.assert_allclose(predict(model, x), predict(other, x), atol=1e-12, rtol=0)

    def test_test_labels_never_change_own_fold_model_or_threshold(self):
        rows = synthetic_rows(); predictions, models = cross_validate(rows)
        own = predictions['fold'] == 0
        modified = dict(rows, bad=rows['bad'].copy())
        modified['bad'][own] = ~modified['bad'][own]
        after, after_models = cross_validate(modified)
        self.assertEqual(models[0], after_models[0])
        np.testing.assert_array_equal(predictions['risk'][own], after['risk'][own])
        np.testing.assert_array_equal(predictions['flag'][own], after['flag'][own])
        for model in models:
            roles = [set(model['sequences'][k]) for k in ('train', 'calibration', 'test')]
            self.assertFalse(roles[0] & roles[1] or roles[0] & roles[2] or roles[1] & roles[2])
        train = np.isin(rows['sequence_id'], models[0]['sequences']['train'])
        weights = sequence_weights(rows['sequence_id'][train])
        np.testing.assert_allclose(models[0]['model']['mean'], (rows['features'][train]*weights[:, None]).sum(0))

    def test_development_gate_does_not_approve_deployment(self):
        rows = synthetic_rows(); predictions, _ = cross_validate(rows)
        result, records = evaluate(rows, predictions)
        self.assertEqual(len(records), 15)
        for key in ('independent_audit_passed', 'closed_loop_tracking_passed', 'residual_effectiveness_passed',
                    'reliability_effectiveness_passed', 'deployment_approved'):
            self.assertFalse(result[key])
        self.assertGreaterEqual(result['calibration']['brier'], 0)
        self.assertGreaterEqual(result['calibration']['ece_10_bins'], 0)

    def test_gate_requires_both_macro_targets_and_cannot_swap_to_pooled(self):
        # One long clean sequence must not hide false alarms in short sequences.
        groups = np.r_[np.repeat(np.arange(11), 10), np.full(1000, 11)]
        bad = np.r_[np.tile([False]*5+[True]*5, 11), np.zeros(1000, dtype=bool)]
        rows = dict(sequence_id=groups, bad=bad, normal=np.ones(len(bad), dtype=bool))
        flag = bad.copy()
        # 20% FPR in each short sequence; 0% in the long one.
        flag[np.arange(0, 110, 10)] = True
        predictions = dict(flag=flag, peak_flag=flag.copy(), risk=np.where(flag, .8, .1))
        result, _ = evaluate(rows, predictions)
        self.assertLess(result['arms']['logistic']['pooled']['fpr'], .1)
        self.assertGreater(result['arms']['logistic']['macro_percent']['fpr']['delta_pp'], 10)
        self.assertFalse(result['development_continuation_gate'])
        predictions.update(flag=bad.copy(), peak_flag=bad.copy(), risk=np.where(bad, .8, .1))
        self.assertTrue(evaluate(rows, predictions)[0]['development_continuation_gate'])
        predictions['flag'][:] = False
        self.assertFalse(evaluate(rows, predictions)[0]['development_continuation_gate'])

    def test_proposal_top1_does_not_invent_good_candidate(self):
        arrays = synthetic_coverage()
        result = proposal_diagnostics(arrays)
        self.assertEqual(result['all_cell_added_frames'], 0)
        self.assertEqual(result['per_crop_top1_added_frames'], 0)
        # Add a supported low-scoring good candidate to an extra tile.
        arrays['boxes'][0, 2, 1] = arrays['gt'][1]
        arrays['scores'][0, 2, 1] = 0
        result = proposal_diagnostics(arrays)
        self.assertEqual(result['all_cell_added_frames'], 1)
        self.assertEqual(result['per_crop_top1_added_frames'], 0)


class GateEntryTests(unittest.TestCase):
    def test_refuses_overwrite_before_source_read(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(entry, 'file_sha', side_effect=AssertionError('should fail before source read')):
                with self.assertRaises(FileExistsError): entry.run('missing.tar.gz', directory)

    def test_full_cpu_entry_outputs_deterministic_metadata_and_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root/'source.tar.gz'
            specs = [dict(sequence_id=i, member=f'sequences/{i}.npz') for i in range(15)]
            with tarfile.open(source, 'w:gz') as archive:
                for spec in specs:
                    blob = npz_bytes(synthetic_coverage())
                    info = tarfile.TarInfo(spec['member']); info.size = len(blob)
                    archive.addfile(info, io.BytesIO(blob))
            original = source.read_bytes()
            with patch.object(entry, 'check_bundle', return_value=(dict(sequences=specs, checkpoint_sha256='a'*64), {})):
                with contextlib.redirect_stdout(io.StringIO()):
                    result = entry.run(source, root/'output')
                    again = entry.run(source, root/'again')
            self.assertEqual(result, again)
            self.assertEqual(source.read_bytes(), original)
            checksums = json.loads((root/'output/checksums.json').read_text())
            for name, digest in checksums.items():
                self.assertEqual(entry.file_sha(root/'output'/name), digest)
                self.assertEqual((root/'output'/name).read_bytes(), (root/'again'/name).read_bytes())
            with np.load(root/'output/oof_predictions.npz', allow_pickle=False) as loaded:
                self.assertEqual(len(loaded['bad']), 60)
            with patch.object(entry, 'check_bundle', return_value=(dict(sequences=specs, checkpoint_sha256='a'*64), {})):
                with contextlib.redirect_stdout(io.StringIO()):
                    verified = analyzer.verify(source, root/'output')
                self.assertEqual(verified, result)
                # Merely updating checksums cannot hide a changed fitted coefficient.
                artifact = root/'output/manifest.json'
                metadata = json.loads(artifact.read_text())
                metadata['folds'][0]['model']['beta'][0] += .1
                artifact.write_text(json.dumps(metadata), encoding='utf-8')
                checksums['manifest.json'] = entry.file_sha(artifact)
                (root/'output/checksums.json').write_text(json.dumps(checksums), encoding='utf-8')
                with self.assertRaisesRegex(ValueError, 'numeric value differs'):
                    analyzer.verify(source, root/'output')

    def test_documented_cli_is_accepted_without_seed(self):
        doc = (PROJECT/'VDRM_SEARCH_GATE_PLAN.md').read_text(encoding='utf-8')
        command = re.search(r'```css\n(.*?)\n```', doc, re.S).group(1).replace('\\\n', '')
        args = shlex.split(command)
        self.assertNotIn('--seed', args)
        with patch.object(sys, 'argv', args[1:]), patch.object(entry, 'run') as run:
            entry.main()
        run.assert_called_once_with('output/analysis/search_coverage_v1/search_coverage_results.tar.gz',
                                    'output/analysis/search_gate_v1')
        command = re.findall(r'```css\n(.*?)\n```', doc, re.S)[1].replace('\\\n', '')
        args = shlex.split(command)
        with patch.object(sys, 'argv', args[1:]), patch.object(analyzer, 'verify') as verify:
            analyzer.main()
        verify.assert_called_once_with('output/analysis/search_coverage_v1/search_coverage_results.tar.gz',
                                       'output/analysis/search_gate_v1')


if __name__ == '__main__':
    unittest.main()
