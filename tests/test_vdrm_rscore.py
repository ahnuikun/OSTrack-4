"""Exact components, native closed loop, isolated paths, and copied commands."""

import contextlib
import io
import json
import pickle
from pathlib import Path
import re
import shlex
import shutil
import sys
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / 'tests'))

from test_vdrm_frozen import tiny_model
from test_vdrm_frozen_failure_diagnostics import synthetic_fixture
from test_vdrm_native_float import rewrite_bundle
from lib.test.analysis.rscore_probe import ARMS, FOCUS_PLAN, NATIVE_STEPS, SELECTED_FRAMES, check_bundle
from lib.test.evaluation.result_paths import result_bbox_path
from lib.test.evaluation import Sequence, Tracker
from lib.test.evaluation.running import _save_tracker_output
from lib.test.tracker.ostrack_rscore import PROBE_CONFIG, ScoreOnlyNetwork, OSTrackRscore
from lib.train.frozen_vdrm import file_hash
from tracking import analyze_vdrm_module1_dev as analyze
from tracking import diagnose_vdrm_rscore as entry
from tracking import test_uav_suite as suite
from tracking.diagnose_vdrm_frozen_failures import checkpoint_paths, config_for, load_models, state_hash


def rscore_fixture(root, factory=tiny_model):
    dataset = synthetic_fixture(root, factory=factory)
    for sequence in dataset:
        if sequence.name not in FOCUS_PLAN:
            continue
        stop = FOCUS_PLAN[sequence.name][1]
        originals = sequence.frames[:]
        folder = Path(originals[0]).parent
        for i in range(len(originals), stop):
            path = folder / f'{i + 1:08d}.jpg'
            shutil.copyfile(originals[i % len(originals)], path)
            sequence.frames.append(str(path))
        sequence.ground_truth_rect = np.tile([60.25, 60.375, 24.5, 24.75], (stop, 1))
        cover, absence = np.full(stop, 8), np.zeros(stop)
        cover[stop - 14:stop - 10] = 0
        np.savetxt(folder / 'cover.label', cover)
        np.savetxt(folder / 'absence.label', absence)
    return dataset


class CommandTests(unittest.TestCase):
    def test_documented_commands_parse_no_seed_and_correct_plugins(self):
        blocks = re.findall(r'```bash\n(.*?)```', (PROJECT / 'VDRM_RSCORE_PROBE.md').read_text(encoding='utf-8'), re.S)
        self.assertEqual(len(blocks), 4)
        for block in blocks:
            tokens = shlex.split(block.replace('\\\n', ''))
            args = tokens[tokens.index('python') + 1:]
            self.assertNotIn('--seed', args)
            with patch.object(sys, 'argv', args):
                if args[0] == 'tracking/diagnose_vdrm_rscore.py':
                    if '--check_bundle' in args:
                        with patch.object(entry, 'check_bundle', return_value=dict(records=179)) as checked, \
                                patch.object(entry, 'get_dataset', side_effect=AssertionError('dataset accessed')):
                            entry.main()
                        checked.assert_called_once_with('output/analysis/frozen_vdrm/rscore_probe.tar.gz')
                    else:
                        with patch.object(torch.cuda, 'is_available', return_value=True), patch.object(entry, 'run') as run:
                            entry.main()
                        self.assertEqual(run.call_args.args, ('./output', str(Path('output/analysis/frozen_vdrm/rscore_probe.tar.gz')), 'cuda'))
                elif args[0] == 'tracking/test_uav_suite.py':
                    parsed = suite.parse_args()
                    self.assertEqual((parsed.tracker_name, parsed.tracker_param), ('ostrack_rscore', PROBE_CONFIG))
                    self.assertEqual((parsed.dataset, parsed.threads, parsed.num_gpus), ('got10k_vdrm_dev', 4, 4))
                else:
                    self.assertEqual(args[0], 'tracking/analyze_vdrm_module1_dev.py')
                    with patch.object(analyze, 'get_dataset', side_effect=RuntimeError('parsed')):
                        with self.assertRaisesRegex(RuntimeError, 'parsed'):
                            analyze.main()

    def test_mismatched_plugin_list_and_duplicate_parameters_fail_before_dataset(self):
        for extra in (['--tracker_names', 'ostrack', 'ostrack'], ['--tracker_params', 'x', 'x']):
            args = ['analyze', '--tracker_params', 'x', '--reference', 'x', *extra]
            with patch.object(sys, 'argv', args), patch.object(analyze, 'get_dataset', side_effect=AssertionError('dataset accessed')), \
                    contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    analyze.main()


class RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)
        cls.temporary = tempfile.TemporaryDirectory(prefix='vdrm_rscore_')
        cls.root = Path(cls.temporary.name)
        cls.dataset = rscore_fixture(cls.root)
        cls.paths = checkpoint_paths(cls.root)
        cls.before = {a: file_hash(p) for a, p in cls.paths.items()}
        cls.output = cls.root / 'rscore.tar.gz'
        with patch('tracking.diagnose_vdrm_frozen_failures.build_ostrack', side_effect=tiny_model), \
                patch.object(entry, 'get_dataset', return_value=cls.dataset), \
                patch.object(torch.Tensor, 'cuda', lambda t, *a, **kw: t), contextlib.redirect_stdout(io.StringIO()):
            cls.manifest = entry.run(cls.root, cls.output, 'cpu')

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()
        torch.set_num_threads(cls.threads)

    def test_five_case_bundle_exact_readonly_native_history(self):
        self.assertEqual(self.manifest, check_bundle(self.output))
        self.assertEqual((NATIVE_STEPS, SELECTED_FRAMES), (1996, 179))
        self.assertEqual(self.before, {a: file_hash(p) for a, p in self.paths.items()})
        self.assertFalse((self.root / 'test').exists())
        with tarfile.open(self.output, 'r:gz') as archive:
            trajectories = json.load(archive.extractfile('trajectories.json'))
            summary = json.load(archive.extractfile('focus_summary.json'))
            records = json.load(archive.extractfile('records.json'))
        self.assertFalse(summary['full_dev_gate_passed'])
        self.assertFalse(summary['automatic_identity_verdict'])
        self.assertEqual(len(summary['cases']), 5)
        for trace in trajectories:
            self.assertEqual(set(trace['native']), set(ARMS))
            for arm in ARMS:
                self.assertEqual(trace['native'][arm][1]['anchor_xywh'], [60.25, 60.375, 24.5, 24.75])
        self.assertTrue(all(r['alpha0_exact'] for r in records))

    def test_bundle_refuses_overwrite_without_loading_dataset_or_weights(self):
        with patch.object(entry, 'get_dataset', side_effect=AssertionError('dataset accessed')):
            with self.assertRaises(FileExistsError):
                entry.run(self.root, self.output, 'cpu')

    def test_missing_prefix_image_aborts_before_gpu_model_loading(self):
        sequence = self.dataset[0]
        path = Path(sequence.frames[24])
        moved = path.with_suffix('.hidden')
        path.rename(moved)
        try:
            with patch.object(entry, 'get_dataset', return_value=self.dataset), \
                    patch.object(entry, 'load_models', side_effect=AssertionError('model loaded')):
                with self.assertRaises(FileNotFoundError):
                    entry.run(self.root, self.root / 'missing.tar.gz', 'cpu')
        finally:
            moved.rename(path)

    def test_checker_rejects_broken_history_and_components_even_with_new_checksums(self):
        def broken_history(manifest, payloads):
            traces = json.loads(payloads['trajectories.json'])
            traces[0]['native']['Rscore'][2]['anchor_xywh'][0] += .1
            payloads['trajectories.json'] = entry.dump_json(traces)

        def broken_components(manifest, payloads):
            records = json.loads(payloads['records.json'])
            member = records[0]['arms']['Rscore']['maps_member']
            with np.load(io.BytesIO(payloads[member]), allow_pickle=False) as maps:
                arrays = {k: maps[k] for k in maps.files}
            arrays['size_map'] = arrays['size_map'] + .01
            payloads[member] = entry.npz_bytes(arrays)

        for i, (mutate, message) in enumerate(((broken_history, 'closed-loop'), (broken_components, 'component'))):
            target = self.root / f'bad_{i}.tar.gz'
            rewrite_bundle(self.output, target, mutate)
            with self.assertRaisesRegex(ValueError, message):
                check_bundle(target)

    def test_checker_reads_each_archive_member_once_in_storage_order(self):
        original, calls = tarfile.TarFile.extractfile, []

        def observe(archive, member):
            calls.append(member if isinstance(member, str) else member.name)
            return original(archive, member)

        with patch.object(tarfile.TarFile, 'extractfile', observe):
            check_bundle(self.output)
        with tarfile.open(self.output, 'r:gz') as archive:
            self.assertEqual(calls, [m.name for m in archive.getmembers()])

    def test_plugin_has_separate_canonical_writer_path_and_native_ctor_binding(self):
        from lib.test.evaluation import Tracker
        from lib.test.parameter import ostrack_rscore as params_module
        env = SimpleNamespace(prj_dir=str(PROJECT), save_dir=str(self.root), results_path=str(self.root / 'test/tracking_results'))
        with patch('lib.test.evaluation.tracker.env_settings', return_value=env):
            tracker_info = Tracker('ostrack_rscore', PROBE_CONFIG, 'got10k_vdrm_dev')
        expected = self.root / 'test/tracking_results/ostrack_rscore' / PROBE_CONFIG / 'got10k' / (self.dataset[0].name + '.txt')
        self.assertEqual(Path(result_bbox_path(tracker_info.results_dir, self.dataset[0])), expected)
        with patch.object(params_module, 'env_settings', return_value=env), \
                patch('lib.test.parameter.ostrack.env_settings', return_value=env), contextlib.redirect_stdout(io.StringIO()):
            params = params_module.parameters(PROBE_CONFIG)
        params.debug = 0
        with patch('tracking.diagnose_vdrm_frozen_failures.build_ostrack', side_effect=tiny_model), \
                patch.object(torch.Tensor, 'cuda', lambda t, *a, **kw: t), \
                patch.object(torch.nn.Module, 'to', lambda m, *a, **kw: m), contextlib.redirect_stdout(io.StringIO()):
            tracker = OSTrackRscore(params, 'got10k_vdrm_dev')
        self.assertIsInstance(tracker.network, ScoreOnlyNetwork)
        self.assertEqual(set(tracker.provenance), {'Tclean', 'Rdisc'})
        for wrong in ('visdrone', 'got10k', 'lasot'):
            with self.assertRaisesRegex(ValueError, 'restricted'):
                OSTrackRscore(params, wrong)
        params.cfg.TEST.SEARCH_FACTOR += .1
        with self.assertRaisesRegex(ValueError, 'unmodified'):
            OSTrackRscore(params, 'got10k_vdrm_dev')

    def test_two_arm_loader_requires_no_rfreeze_but_rejects_wrong_arm_order(self):
        paths = {a: self.paths[a] for a in ('Tclean', 'Rdisc')}
        with patch('tracking.diagnose_vdrm_frozen_failures.build_ostrack', side_effect=tiny_model):
            models, _, _ = load_models(paths, torch.device('cpu'), arms=('Tclean', 'Rdisc'))
        self.assertEqual(set(models), set(paths))
        for arms in (('Rdisc', 'Tclean'), ('Tclean', 'Tclean'), ('Tclean', 'unknown')):
            with self.assertRaisesRegex(ValueError, 'unique registered'):
                load_models({}, torch.device('cpu'), arms=arms)

    def test_production_plugin_runs_native_sequence_and_writer_before_mixed_analysis(self):
        from lib.test.parameter import ostrack_rscore as params_module
        from lib.test.analysis import extract_results, plot_results
        folder = self.root / 'mixed_analysis'
        env = SimpleNamespace(prj_dir=str(PROJECT), save_dir=str(self.root),
            results_path=str(folder / 'tracking_results'), result_plot_path=str(folder / 'result_plots'))
        original = self.dataset[0]
        gt = original.ground_truth_rect[:14]
        source = Sequence('GOT-10k_Train_probe', original.frames[:14], 'got10k', gt)
        with patch('lib.test.evaluation.tracker.env_settings', return_value=env):
            probe_info = Tracker('ostrack_rscore', PROBE_CONFIG, 'got10k_vdrm_dev')
        with patch.object(params_module, 'env_settings', return_value=env), \
                patch('lib.test.parameter.ostrack.env_settings', return_value=env), \
                patch('tracking.diagnose_vdrm_frozen_failures.build_ostrack', side_effect=tiny_model), \
                patch.object(torch.Tensor, 'cuda', lambda t, *a, **kw: t), \
                patch.object(torch.nn.Module, 'to', lambda m, *a, **kw: m), contextlib.redirect_stdout(io.StringIO()):
            output = probe_info.run_sequence(source, debug=0)
        self.assertEqual(len(output['target_bbox']), 14)
        self.assertTrue(np.isfinite(output['target_bbox']).all())
        _save_tracker_output(source, probe_info, output)
        self.assertTrue(Path(result_bbox_path(probe_info.results_dir, source)).is_file())
        # Exercise the actual four-plugin reader, cache identity and stage
        # report on 152 synthetic sequences, not mocked scoring functions.
        names = list(entry.CONFIGS.values()) + [PROBE_CONFIG]
        plugins = ['ostrack'] * 3 + ['ostrack_rscore']
        dataset = [Sequence(f'GOT-10k_Train_fixture_{i:06d}', original.frames[:14], 'got10k', gt) for i in range(152)]
        cover, absence = np.full(len(original.frames), 8), np.zeros(len(original.frames))
        cover[3:6] = 0
        np.savetxt(Path(original.frames[0]).parent / 'cover.label', cover[:14])
        np.savetxt(Path(original.frames[0]).parent / 'absence.label', absence[:14])
        try:
            for plugin, name in zip(plugins, names):
                base = Path(env.results_path) / plugin / name / 'got10k'
                base.mkdir(parents=True, exist_ok=True)
                for seq in dataset:
                    np.savetxt(base / (seq.name + '.txt'), gt, delimiter='\t')
            args = ['analyze', '--tracker_params', *names, '--tracker_names', *plugins,
                    '--reference', names[0], '--report_name', 'mixed', '--visibility_stages', '--per_sequence']
            with patch.object(sys, 'argv', args), patch.object(analyze, 'get_dataset', return_value=dataset), \
                    patch.object(analyze, 'env_settings', return_value=env), \
                    patch('lib.test.evaluation.tracker.env_settings', return_value=env), \
                    patch.object(extract_results, 'env_settings', return_value=env), \
                    patch.object(plot_results, 'env_settings', return_value=env), contextlib.redirect_stdout(io.StringIO()):
                analyze.main()
            report = Path(env.result_plot_path) / 'mixed'
            with (report / 'eval_data.pkl').open('rb') as stream:
                cache = pickle.load(stream)
            self.assertEqual([(t['name'], t['param']) for t in cache['trackers']], list(zip(plugins, names)))
            self.assertEqual(sum(cache['valid_sequence']), 152)
            stages = json.loads((report / 'visibility_stages.json').read_text())
            self.assertEqual(stages['trackers'], names)
            self.assertEqual(stages['sequences'], 152)
        finally:
            cover[3:6] = 8
            stop = len(original.frames)
            cover[stop - 14:stop - 10] = 0
            np.savetxt(Path(original.frames[0]).parent / 'cover.label', cover)
            np.savetxt(Path(original.frames[0]).parent / 'absence.label', absence)

    def test_wrapper_exact_source_maps_raw_boxes_and_no_gradient_or_state_change(self):
        with patch('tracking.diagnose_vdrm_frozen_failures.build_ostrack', side_effect=tiny_model):
            models, _, _ = load_models(self.paths, torch.device('cpu'), arms=('Tclean', 'Rdisc'))
        probe = ScoreOnlyNetwork(models['Tclean'], models['Rdisc'])
        before = state_hash(probe)
        inputs = dict(template=torch.randn(1, 3, 128, 128), search=torch.randn(1, 3, 256, 256),
                      ce_template_mask=torch.ones(1, 64, dtype=torch.bool), template_bbox=torch.tensor([[.3, .3, .4, .4]]))
        out = probe.forward(**inputs)
        base, residual = probe.last_components
        self.assertTrue(torch.equal(out['score_map'], residual['score_map']))
        self.assertTrue(torch.equal(out['size_map'], base['size_map']))
        self.assertTrue(torch.equal(out['offset_map'], base['offset_map']))
        self.assertTrue(torch.equal(out['pred_boxes'], probe.box_head.cal_bbox(out['score_map'], base['size_map'], base['offset_map']).view(-1, 1, 4)))
        self.assertFalse(any(v.requires_grad for v in out.values() if isinstance(v, torch.Tensor)))
        self.assertEqual(before, state_hash(probe))
        with self.assertRaisesRegex(RuntimeError, 'inference-only'):
            probe.train()
        probe.residual.train()
        with self.assertRaisesRegex(RuntimeError, 'eval mode'):
            probe.forward(**inputs)

    def test_analysis_preflight_rejects_short_sequences_instead_of_padding(self):
        sequence = self.dataset[0]
        path = self.root / 'short' / 'got10k' / (sequence.name + '.txt')
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(path, sequence.ground_truth_rect[:-1], delimiter='\t')
        args = ['analyze', '--tracker_params', 'x', '--reference', 'x']
        tracker = SimpleNamespace(parameter_name='x', results_dir=str(self.root / 'short'))
        with patch.object(sys, 'argv', args), patch.object(analyze, 'get_dataset', return_value=self.dataset), \
                patch.object(analyze, 'trackerlist', return_value=[tracker]), \
                patch.object(analyze, 'print_results', side_effect=AssertionError('silently padded')):
            with self.assertRaisesRegex(ValueError, 'incomplete paired'):
                analyze.main()


if __name__ == '__main__':
    unittest.main()
