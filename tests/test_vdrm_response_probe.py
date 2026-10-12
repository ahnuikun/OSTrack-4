import contextlib
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

from test_vdrm_response import response_config, response_model
from test_vdrm_rscore import rscore_fixture
from test_vdrm_native_float import rewrite_bundle
from lib.test.analysis.response_probe import CONFIG, CONFIGS, check_bundle
from lib.test.evaluation import Sequence, Tracker
from lib.test.evaluation.running import _save_tracker_output
from lib.train.frozen_response import prepare_response_models
from lib.train.frozen_vdrm import file_hash
from tracking import diagnose_vdrm_response as entry
from tracking import check_vdrm_response as check_entry
from tracking import train, test_uav_suite, analyze_vdrm_module1_dev

PROJECT = Path(__file__).resolve().parents[1]


class ParsedCommand(Exception):
    pass


def response_fixture(root, factory=response_model):
    dataset = rscore_fixture(root, factory=factory)
    with patch('lib.models.ostrack.build_ostrack', side_effect=factory):
        model, baseline, _ = prepare_response_models(response_config(), root, CONFIG)
    with torch.no_grad():
        # Nonzero correction proves isolation beyond the zero-init identity.
        model.response_residual.correct[-1].weight.fill_(.003)
        model.response_residual.correct[-1].bias.fill_(.15)
    path = Path(root) / 'checkpoints/train/ostrack' / CONFIG / 'OSTrack_ep0300.pth.tar'
    path.parent.mkdir(parents=True)
    torch.save(dict(epoch=300, net_type='OSTrack', net=model.state_dict(),
                    frozen_response_contract=model.frozen_response_contract), path)
    return dataset




class ProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)
        cls.temp = tempfile.TemporaryDirectory(prefix='response_probe_')
        cls.root = Path(cls.temp.name)
        cls.dataset = response_fixture(cls.root)
        cls.output = cls.root / 'rresp.tar.gz'
        cls.paths = {a: cls.root / 'checkpoints/train/ostrack' / n / 'OSTrack_ep0300.pth.tar'
                     for a, n in CONFIGS.items()}
        cls.before = {a: file_hash(p) for a, p in cls.paths.items()}
        with patch('tracking.diagnose_vdrm_frozen_failures.build_ostrack', side_effect=response_model), \
                patch('lib.models.ostrack.build_ostrack', side_effect=response_model), \
                patch.object(entry, 'get_dataset', return_value=cls.dataset), \
                patch.object(torch.Tensor, 'cuda', lambda t, *a, **kw: t), contextlib.redirect_stdout(io.StringIO()):
            cls.manifest = entry.run(cls.root, cls.output, 'cpu')

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()
        torch.set_num_threads(cls.threads)

    def test_native_readonly_bundle_and_own_float_history(self):
        self.assertEqual(self.manifest, check_bundle(self.output))
        self.assertEqual(self.before, {a: file_hash(p) for a, p in self.paths.items()})
        self.assertEqual(self.manifest['native_tracking_steps'], 1996)
        self.assertEqual(self.manifest['records'], 179)
        self.assertFalse((self.root / 'test').exists())

    def test_standalone_checkpoint_check_runs_without_optimizer(self):
        args = ['check_vdrm_response.py', '--config', CONFIG, '--save_dir', str(self.root),
                '--checkpoint', str(self.paths['Rresp']), '--device', 'cpu']
        with patch('lib.models.ostrack.build_ostrack', side_effect=response_model), \
                patch.object(sys, 'argv', args), contextlib.redirect_stdout(io.StringIO()):
            check_entry.main()
        report = self.root / 'analysis/frozen_response' / CONFIG / 'checkpoint_audit.json'
        result = json.loads(report.read_text(encoding='utf-8'))
        self.assertEqual(result['optimizer_steps'], 0)
        self.assertTrue(result['bypass_exact_tclean'])

    def test_checker_rejects_history_and_regression_tampering(self):
        def history(manifest, payloads):
            traces = json.loads(payloads['trajectories.json'])
            traces[0]['native']['Rresp'][2]['anchor_xywh'][0] += .1
            payloads['trajectories.json'] = entry.dump_json(traces)

        def regression(manifest, payloads):
            records = json.loads(payloads['records.json'])
            member = records[0]['arms']['Rresp']['maps_member']
            with np.load(io.BytesIO(payloads[member]), allow_pickle=False) as maps:
                arrays = {key: maps[key] for key in maps.files}
            arrays['size_map'] += .01
            payloads[member] = entry.npz_bytes(arrays)

        for index, mutate in enumerate((history, regression)):
            path = self.root / f'bad{index}.tar.gz'
            rewrite_bundle(self.output, path, mutate)
            with self.assertRaises(ValueError):
                check_bundle(path)

    def test_no_overwrite_and_missing_prefix_fails_before_model_load(self):
        with patch.object(entry, 'get_dataset', side_effect=AssertionError('dataset accessed')):
            with self.assertRaises(FileExistsError):
                entry.run(self.root, self.output, 'cpu')
        path = Path(self.dataset[0].frames[24])
        moved = path.with_suffix('.hidden')
        path.rename(moved)
        try:
            with patch.object(entry, 'get_dataset', return_value=self.dataset), \
                    patch.object(entry, 'load_models', side_effect=AssertionError('models loaded')):
                with self.assertRaises(FileNotFoundError):
                    entry.run(self.root, self.root / 'missing.tar.gz', 'cpu')
        finally:
            moved.rename(path)

    def test_production_tracker_and_writer_match_native_float_trace(self):
        with tarfile.open(self.output, 'r:gz') as archive:
            trace = json.load(archive.extractfile('trajectories.json'))[0]
        source = next(s for s in self.dataset if s.name == trace['sequence'])
        length = len(trace['gt_xywh'])
        sequence = Sequence(source.name, source.frames[:length], 'got10k', source.ground_truth_rect[:length])
        from types import SimpleNamespace
        env = SimpleNamespace(prj_dir=str(PROJECT), save_dir=str(self.root),
                              results_path=str(self.root / 'production'))
        with patch('lib.test.evaluation.tracker.env_settings', return_value=env):
            tracker = Tracker('ostrack', CONFIG, 'got10k_vdrm_dev')
        with patch('lib.test.parameter.ostrack.env_settings', return_value=env), \
                patch('lib.test.tracker.ostrack.build_ostrack', side_effect=response_model), \
                patch.object(torch.nn.Module, 'cuda', lambda m, *a, **kw: m), \
                patch.object(torch.Tensor, 'cuda', lambda t, *a, **kw: t):
            output = tracker.run_sequence(sequence, debug=0)
        self.assertEqual(output['target_bbox'], [row['bbox_xywh'] for row in trace['native']['Rresp']])
        _save_tracker_output(sequence, tracker, output)
        self.assertTrue((Path(tracker.results_dir) / 'got10k' / (sequence.name + '.txt')).is_file())
        self.assertFalse((Path(tracker.results_dir) / (sequence.name + '.txt')).exists())


if __name__ == '__main__':
    unittest.main()
