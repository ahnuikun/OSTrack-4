"""Guard isolation, early stopping, train/audit selection and complete bundle checks."""
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

import cv2
import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT)); sys.path.insert(0, str(PROJECT/'tests'))
from lib.train.identity_offline import ARMS, load_protocol, registered_ids
from lib.train.frozen_vdrm import file_hash
from lib.test.analysis.identity_guard import (PROTOCOL, partition, guard_select, grid_record, fit_policy,
    audit_policy, decide, policy_key, diagnose_sequence, diagnose_summary)
from lib.test.analysis.identity_guard_bundle import check_bundle
from lib.test.analysis.identity_native import load_screen_heads
from lib.test.analysis.candidate_cues import read_npz
from tracking import screen_vdrm_identity_guard as entry, analyze_vdrm_identity_guard as analyzer
from test_vdrm_identity_native import synthetic_outputs, native_fixture
from test_vdrm_native_float import rewrite_bundle
from test_vdrm_frozen import tiny_model


def guard_fixture(root, factory=tiny_model):
    screen, data, source = native_fixture(root, factory=factory)
    _, manifest = load_screen_heads(screen)
    parts = partition(manifest['train_ids'])
    names = [f'GOT-10k_Train_{i+1:06d}' for i in range(max(parts['fit']+parts['audit']+[9332])+1)]
    (data/'list.txt').write_text('\n'.join(names), encoding='utf-8')
    image = np.random.default_rng(42).integers(0, 256, (160, 160, 3), dtype=np.uint8)
    outputs = synthetic_outputs(absent_sentinel=True)
    for identity in parts['fit']+parts['audit']:
        folder = data/names[identity]; folder.mkdir()
        np.savetxt(folder/'groundtruth.txt', outputs['gt'], delimiter=',')
        np.savetxt(folder/'cover.label', outputs['cover']); np.savetxt(folder/'absence.label', outputs['absence'])
        for frame in range(1, 6): cv2.imwrite(str(folder/f'{frame:08d}.jpg'), image)
    return screen, data, source, parts


class GuardMathTests(unittest.TestCase):
    def test_fixed_partition_excludes_head_training_dev_and_each_other(self):
        train, dev = registered_ids(load_protocol()); parts = partition(train)
        self.assertEqual((len(parts['fit']), len(parts['audit'])), (128, 64))
        self.assertEqual(parts, partition(train))
        self.assertFalse(set(parts['fit']) & set(parts['audit']))
        self.assertFalse(set(parts['fit']+parts['audit']) & set(train+dev))

    def test_selection_takes_no_labels_and_disabled_is_exact_baseline(self):
        self.assertEqual(list(inspect.signature(guard_select).parameters),
            ['base_scores', 'identity_scores', 'boxes', 'common_cells', 'threshold'])
        x = synthetic_outputs(); base = x['base_scores'].argmax(1)
        for arm in ARMS:
            chosen = guard_select(x['base_scores'], x[arm+'_scores'], x['boxes'], x['common_cells'], 2.01)
            np.testing.assert_array_equal(chosen, base)

    def test_localization_lock_and_source_support_protection(self):
        x = synthetic_outputs()
        x['boxes'][0, 1] = x['boxes'][0, 0]+np.array([1., 0, 0, 0])
        x['mean_rank_scores'][0, 1] = 1.
        chosen = guard_select(x['base_scores'], x['mean_rank_scores'], x['boxes'], x['common_cells'], 0.)
        self.assertEqual(chosen[0], 0); self.assertEqual(chosen[1], 0)
        x['common_cells'][1, 1] = 0
        chosen = guard_select(x['base_scores'], x['mean_rank_scores'], x['boxes'], x['common_cells'], 0.)
        self.assertEqual(chosen[1], 1)

    def test_fit_cannot_access_audit_and_audit_runs_only_frozen_policy(self):
        x = synthetic_outputs()
        records = [grid_record(str(i), 'fit', x) for i in range(32)]
        choice, _ = fit_policy(records)
        self.assertEqual(choice['arm'], 'mean_rank'); self.assertEqual(choice['threshold'], .4)
        audit = grid_record('heldout', 'audit', x, selected=choice)
        self.assertEqual(set(audit['policies']), {policy_key('mean_rank', .4)})
        with self.assertRaisesRegex(ValueError, 'only fit'): fit_policy([audit])
        with self.assertRaisesRegex(ValueError, 'frozen'): grid_record('heldout', 'audit', x)

    def test_fail_audit_never_selects_runner_up(self):
        x = synthetic_outputs(); fit = [grid_record(str(i), 'fit', x) for i in range(32)]
        choice, _ = fit_policy(fit)
        # Reverse only the chosen head on audit; other heads are deliberately good.
        x['mean_rank_scores'] = -x['mean_rank_scores']
        audits = [grid_record(str(i+100), 'audit', x, selected=choice) for i in range(32)]
        report = decide(fit+audits)
        self.assertEqual(report['selected'], choice)
        self.assertFalse(report['heldout_screen_passed'])
        self.assertEqual(report['next_action'], 'stop_existing_identity_reranking_path')

    def test_no_gain_early_stop_and_insufficient_audit_do_not_pass(self):
        x = synthetic_outputs()
        for arm in ARMS: x[arm+'_scores'][:] = 0
        report = decide([grid_record(str(i), 'fit', x) for i in range(32)])
        self.assertEqual(report['selected']['threshold'], 2.01)
        self.assertIsNone(report['heldout']); self.assertFalse(report['heldout_screen_passed'])
        x = synthetic_outputs(); choice, _ = fit_policy([grid_record(str(i), 'fit', x) for i in range(32)])
        report = audit_policy([grid_record('one', 'audit', x, selected=choice)], choice)
        self.assertIn('insufficient_independent_switch_coverage', report['rejection_reasons'])

    def test_diagnostic_contributions_sum_to_total(self):
        x = synthetic_outputs(); record = diagnose_sequence('fixture', x); report = diagnose_summary([record])
        for arm in ARMS:
            groups = report['heads'][arm]['groups']
            self.assertAlmostEqual(sum(groups[k]['delta_pp'] for k in ('both_correct','damage','repair','both_wrong')),
                groups['changed_near']['delta_pp']+groups['changed_far']['delta_pp'])
        self.assertTrue(report['posthoc']); self.assertFalse(report['deployment_approved'])

    def test_existing_output_refused_before_source_access(self):
        with tempfile.TemporaryDirectory() as root:
            with patch.object(entry, 'load_screen_heads', side_effect=AssertionError('source read')):
                with self.assertRaises(FileExistsError): entry.run('missing','missing','missing',root)

    def test_documented_commands_parse_without_seed(self):
        document = (PROJECT/'VDRM_IDENTITY_GUARD_PLAN.md').read_text(encoding='utf-8')
        commands = re.findall(r'```(?:lua|css)\n(.*?)```', document, re.S)
        self.assertEqual(len(commands), 2)
        for command in commands:
            tokens = shlex.split(command.replace('\\\n',' ')); args = tokens[tokens.index('python')+1:]
            self.assertNotIn('--seed', args)
            with patch.object(sys, 'argv', args):
                if 'screen_' in args[0]:
                    with patch.object(torch.cuda, 'is_available', return_value=True), patch.object(entry, 'run') as run: entry.main()
                    self.assertEqual(run.call_args.args, ('output/analysis/identity_offline_v1/screen/identity_offline_results.tar.gz',
                        './data/got10k/train','./output','output/analysis/identity_guard_v1'))
                else:
                    with patch.object(analyzer, 'check_bundle', return_value=({'sequences': []}, {})), \
                         patch.object(analyzer, 'render_report', return_value='fixture'), \
                         patch.object(torch.cuda, 'is_available', side_effect=AssertionError('GPU access')), \
                         contextlib.redirect_stdout(io.StringIO()): analyzer.main()


class GuardRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads(); torch.set_num_threads(2)
        cls.temp = tempfile.TemporaryDirectory(prefix='identity_guard_test_'); cls.root = Path(cls.temp.name)
        with contextlib.redirect_stdout(io.StringIO()): cls.screen, cls.data, cls.source, cls.parts = guard_fixture(cls.root)
        cls.before = [file_hash(p) for p in (cls.screen, cls.source)]

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup(); torch.set_num_threads(cls.threads)

    def run_entry(self, name, collector=None):
        with patch('tracking.diagnose_vdrm_frozen_failures.build_ostrack', side_effect=tiny_model), \
             patch.object(torch.Tensor, 'cuda', lambda x,*a,**kw:x), \
             patch.object(torch.optim, 'AdamW', side_effect=AssertionError('optimizer')), \
             contextlib.redirect_stdout(io.StringIO()):
            manager = patch.object(entry, 'collect_sequence', side_effect=collector) if collector else contextlib.nullcontext()
            with manager: return entry.run(self.screen, self.data, self.root, self.root/name, 'cpu')

    def test_native_collection_and_standalone_recompute_read_only(self):
        result = self.run_entry('native')
        manifest, repeated = check_bundle(self.root/'native/identity_guard_results.tar.gz')
        self.assertEqual(result, repeated)
        self.assertIn(len(manifest['sequences']), (128,192))
        self.assertEqual(self.before, [file_hash(p) for p in (self.screen,self.source)])
        self.assertEqual(manifest['states_before'], manifest['states_after'])
        with tarfile.open(self.root/'native/identity_guard_results.tar.gz', 'r:gz') as archive:
            for spec in manifest['sequences']:
                arrays = read_npz(archive.extractfile(spec['member']).read())
                np.testing.assert_array_equal(arrays['gt'][3], [-1]*4)
                self.assertEqual(len(arrays['native_boxes']), 4)

    def test_active_policy_frozen_before_any_audit_inference_and_all192_checked(self):
        seen=[]
        def collector(plan,*args):
            seen.append(plan['split'])
            if plan['split']=='audit':
                frozen=json.loads((self.root/'active/frozen_choice.json').read_text())
                self.assertEqual(frozen['arm'],'mean_rank'); self.assertEqual(frozen['threshold'],.4)
            return synthetic_outputs(absent_sentinel=True),[2]
        result=self.run_entry('active',collector)
        manifest,_=check_bundle(self.root/'active/identity_guard_results.tar.gz')
        self.assertTrue(result['heldout_screen_passed'])
        self.assertEqual(seen,['fit']*128+['audit']*64)
        self.assertEqual(len(manifest['sequences']),192)
        path=self.root/'tampered.tar.gz'
        def mutate(m,p):m['frozen_choice']['arm']='spatial_id'
        rewrite_bundle(self.root/'active/identity_guard_results.tar.gz',path,mutate)
        with self.assertRaisesRegex(ValueError,'frozen|decision'):check_bundle(path)

    def test_disabled_fit_skips_all_audit_gpu_calls(self):
        seen=[]
        def collector(plan,*args):
            self.assertEqual(plan['split'],'fit');seen.append(plan['name'])
            x=synthetic_outputs(absent_sentinel=True)
            for arm in ARMS:x[arm+'_scores'][:]=0
            return x,[2]
        result=self.run_entry('disabled',collector)
        manifest,_=check_bundle(self.root/'disabled/identity_guard_results.tar.gz')
        self.assertEqual(len(seen),128);self.assertEqual(len(manifest['sequences']),128)
        self.assertIsNone(result['heldout']);self.assertFalse(result['heldout_screen_passed'])

    def test_inconsistent_sentinel_fails_before_gpu_and_output_creation(self):
        name = f'GOT-10k_Train_{self.parts["fit"][0]+1:06d}'
        path = self.data/name/'absence.label'
        original = path.read_bytes()
        try:
            labels = np.loadtxt(path); labels[3] = 0
            np.savetxt(path, labels)
            with patch.object(entry, 'load_models', side_effect=AssertionError('GPU access')):
                with self.assertRaisesRegex(ValueError, name+r'.*frames \(1-based\)=\[4\]'):
                    entry.run(self.screen, self.data, self.root, self.root/'bad_annotations', 'cpu')
            self.assertFalse((self.root/'bad_annotations').exists())
        finally:
            path.write_bytes(original)


if __name__=='__main__':unittest.main()
