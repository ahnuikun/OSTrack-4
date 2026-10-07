"""Early-prefix coverage, native invariance, independent labels, and real CLI paths."""

import contextlib
from copy import deepcopy
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
import torch

from test_vdrm_response import response_model
from test_vdrm_response_probe import response_fixture
from test_vdrm_native_float import rewrite_bundle
from lib.test.analysis.response_probe import CONFIGS, FOCUS_PLAN, EARLY_PLAN, check_bundle, protocol
from lib.test.analysis.response_failure_audit import analyze_bundle, analyze_frame, energy_peak, _sigmoid
from lib.train.frozen_vdrm import file_hash
from tracking import diagnose_vdrm_response as entry
from tracking import analyze_vdrm_response as analyzer

PROJECT = Path(__file__).resolve().parents[1]


class EarlyProtocolTests(unittest.TestCase):
    def test_profile_preserves_endpoints_and_existing_focus(self):
        self.assertEqual(protocol('focus'), (FOCUS_PLAN, 179, 1996))
        plan, count, steps = protocol('early_failure')
        self.assertEqual((count, steps), (298, 1996))
        self.assertEqual(plan['GOT-10k_Train_007878'], (2, 130))
        self.assertEqual(plan['GOT-10k_Train_008311'], (2, 100))
        self.assertEqual({n: b for n, (_, b) in plan.items()}, {n: b for n, (_, b) in FOCUS_PLAN.items()})
        with self.assertRaises(ValueError):
            protocol('arbitrary')

    def test_documented_commands_parse_and_resolve_separate_paths(self):
        document = (PROJECT / 'VDRM_RRESP_EARLY_AUDIT.md').read_text(encoding='utf-8')
        commands = 0
        for block in re.findall(r'```\w+\n(.*?)```', document, re.S):
            if 'python tracking/' not in block:
                continue
            tokens = shlex.split(block.replace('\\\n', ''))
            args = tokens[tokens.index('python') + 1:]
            self.assertNotIn('--seed', args)
            commands += 1
            with patch.object(sys, 'argv', args):
                if args[0] == 'tracking/diagnose_vdrm_response.py':
                    with patch.object(torch.cuda, 'is_available', return_value=True), patch.object(entry, 'run') as run:
                        entry.main()
                    self.assertEqual(run.call_args.args, ('./output', str(Path(
                        'output/analysis/frozen_response_early/rresp_early_probe.tar.gz')), 'cuda'))
                    self.assertEqual(run.call_args.kwargs, {'profile': 'early_failure'})
                else:
                    self.assertEqual(args[0], 'tracking/analyze_vdrm_response.py')
                    with patch.object(analyzer, 'run') as run:
                        analyzer.main()
                    self.assertEqual(tuple(Path(a).as_posix() for a in run.call_args.args), (
                        'output/analysis/frozen_response_early/rresp_early_probe.tar.gz',
                        'output/analysis/frozen_response_early/rresp_early_analysis.tar.gz'))
        self.assertEqual(commands, 2)

    def test_uniform_logit_shift_can_change_hann_but_constant_energy_cannot(self):
        logits = np.array([2., 0.], dtype=np.float32)
        hann = np.array([.6, 1.], dtype=np.float32)
        base = _sigmoid(logits)
        self.assertEqual(int((base * hann).argmax()), 0)
        self.assertEqual(int((_sigmoid(logits + 2.) * hann).argmax()), 1)
        for shift in (-2., 0., 2.):
            self.assertEqual(energy_peak(base, hann, np.full_like(base, shift)), 0)


class EarlyRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)
        cls.temp = tempfile.TemporaryDirectory(prefix='response_early_')
        cls.root = Path(cls.temp.name)
        cls.dataset = response_fixture(cls.root)
        cls.output = cls.root / 'early.tar.gz'
        cls.paths = {a: cls.root / 'checkpoints/train/ostrack' / n / 'OSTrack_ep0300.pth.tar'
                     for a, n in CONFIGS.items()}
        cls.hashes = {a: file_hash(p) for a, p in cls.paths.items()}
        with patch('tracking.diagnose_vdrm_frozen_failures.build_ostrack', side_effect=response_model), \
                patch('lib.models.ostrack.build_ostrack', side_effect=response_model), \
                patch.object(entry, 'get_dataset', return_value=cls.dataset), \
                patch.object(torch.Tensor, 'cuda', lambda t, *a, **kw: t), contextlib.redirect_stdout(io.StringIO()):
            cls.manifest = entry.run(cls.root, cls.output, 'cpu', profile='early_failure')
            # Same checkpoints/images: adding early replay/saving must not alter ANY native row.
            cls.focus_path = cls.root / 'focus.tar.gz'
            entry.run(cls.root, cls.focus_path, 'cpu')
        _, cls.payloads, cls.shapes = check_bundle(cls.output, include_payloads=True)
        cls.traces = json.loads(cls.payloads['trajectories.json'])
        cls.records = json.loads(cls.payloads['records.json'])
        cls.summary, cls.frames, cls.report = analyze_bundle(cls.output)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()
        torch.set_num_threads(cls.threads)

    def test_early_replays_preserve_complete_native_histories_and_old_focus_ao(self):
        self.assertEqual(self.hashes, {a: file_hash(p) for a, p in self.paths.items()})
        with tarfile.open(self.focus_path, 'r:gz') as archive:
            original = json.load(archive.extractfile('trajectories.json'))
            focus = json.load(archive.extractfile('focus_summary.json'))
        self.assertEqual(self.traces, original)
        self.assertEqual(self.summary['original_focus_summary'], focus)
        self.assertEqual(self.manifest['records'], 298)
        self.assertEqual(self.manifest['optimizer_steps'], 0)
        for name, (start, stop) in EARLY_PLAN.items():
            self.assertEqual([r['frame_1based'] for r in self.records if r['sequence'] == name],
                             list(range(start, stop+1)))
        self.assertFalse((self.root / 'test').exists())

    def test_analysis_needs_no_models_dataset_or_cuda_and_refuses_overwrite(self):
        output = self.root / 'analysis.tar.gz'
        with patch.object(entry, 'get_dataset', side_effect=AssertionError('dataset accessed')), \
                patch.object(entry, 'load_models', side_effect=AssertionError('weights accessed')), \
                patch.object(torch.Tensor, 'cuda', side_effect=AssertionError('CUDA accessed')), \
                contextlib.redirect_stdout(io.StringIO()):
            summary = analyzer.run(self.output, output)
        self.assertEqual(summary['records'], 298)
        self.assertFalse(summary['full_dev_gate_passed'])
        self.assertFalse(summary['automatic_identity_verdict'])
        self.assertEqual(summary['source_bundle_sha256'], file_hash(self.output))
        self.assertLessEqual(summary['maximum_decode_error_pixels'], .002)
        with tarfile.open(output, 'r:gz') as archive:
            self.assertEqual(set(archive.getnames()), {'manifest.json', 'summary.json', 'per_frame.json', 'report.md'})
        with patch.object(analyzer, 'analyze_bundle', side_effect=AssertionError('input accessed')):
            with self.assertRaises(FileExistsError):
                analyzer.run(self.output, output)

    def test_gt_does_not_select_counterfactual_scores_or_boxes(self):
        record = self.records[0]
        trace = next(t for t in self.traces if t['sequence'] == record['sequence'])
        maps = {}
        for arm in CONFIGS:
            with np.load(io.BytesIO(self.payloads[record['arms'][arm]['maps_member']]), allow_pickle=False) as z:
                maps[arm] = dict(z)
        original = analyze_frame(record, trace, maps, self.shapes[record['image_member']])
        altered = deepcopy(trace)
        altered['gt_xywh'][record['frame_1based']-1] = [1., 2., .1, .1]
        changed = analyze_frame(record, altered, maps, self.shapes[record['image_member']])
        for name, control in original['one_step_controls'].items():
            self.assertEqual(control['peak_xy'], changed['one_step_controls'][name]['peak_xy'])
            self.assertEqual(control['bbox_xywh'], changed['one_step_controls'][name]['bbox_xywh'])
        altered['gt_xywh'][record['frame_1based']-1] = [1., 2., 0., 0.]
        invalid = analyze_frame(record, altered, maps, self.shapes[record['image_member']])
        self.assertIsNone(invalid['frozen_candidate_oracle']['max_iou'])
        self.assertEqual(invalid['candidate_status'], 'not_visible_or_invalid_gt')

    def test_checker_and_analysis_reject_rehashed_missing_frames_and_false_metrics(self):
        def missing(manifest, payloads):
            records = json.loads(payloads['records.json'])
            records.pop()
            payloads['records.json'] = entry.dump_json(records)

        def mask(manifest, payloads):
            record = json.loads(payloads['records.json'])[0]
            member = record['arms']['Rresp']['maps_member']
            with np.load(io.BytesIO(payloads[member]), allow_pickle=False) as z:
                arrays = dict(z)
            arrays['final_retained_mask'][:] = False
            payloads[member] = entry.npz_bytes(arrays)

        def wrong_iou(manifest, payloads):
            traces = json.loads(payloads['trajectories.json'])
            traces[0]['native']['Rresp'][1]['iou'] += .1
            payloads['trajectories.json'] = entry.dump_json(traces)

        for i, mutate in enumerate((missing, mask, wrong_iou)):
            target = self.root / f'tampered_{i}.tar.gz'
            rewrite_bundle(self.output, target, mutate)
            with self.assertRaises(ValueError):
                analyze_bundle(target)


if __name__ == '__main__':
    unittest.main()
