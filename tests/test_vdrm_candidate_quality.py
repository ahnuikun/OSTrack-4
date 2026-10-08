"""Rqual wiring/geometry/loss/checkpoint tests; synthetic inputs, not performance."""
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
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from lib.models.layers.candidate_quality import (CandidateQuality, dense_center_boxes,
    quality_delta, apply_energy_correction)
from lib.models.layers.head import CenterPredictor
from lib.test.analysis.response_probe import QUALITY_CONFIG as CONFIG, QUALITY_CONFIGS as CONFIGS, check_bundle
from lib.test.analysis.response_failure_audit import analyze_bundle
from lib.train.actors.candidate_quality import candidate_iou, quality_losses, target_has_evidence
from lib.train.base_functions import validate_vdrm_experiment_contract, get_optimizer_scheduler
from lib.train.frozen_response import (prepare_response_models, audit_response_model,
    make_response_actor, assert_response_optimizer, baseline_hash, validate_response_checkpoint)
from lib.train.trainers.base_trainer import BaseTrainer
import test_vdrm_response as response_tests
from test_vdrm_response import response_model
from test_vdrm_frozen import fixed_batch
from test_vdrm_rscore import rscore_fixture
from test_vdrm_native_float import rewrite_bundle
from tracking import diagnose_vdrm_response as entry, check_vdrm_response as checker


def quality_config():
    return entry.config_for(CONFIG)


def quality_fixture(root, factory=response_model):
    dataset = rscore_fixture(root, factory=factory)
    with patch('lib.models.ostrack.build_ostrack', side_effect=factory):
        model, _, _ = prepare_response_models(quality_config(), root, CONFIG)
    with torch.no_grad():
        model.response_residual.correct[-1].weight.fill_(.04)
        model.response_residual.correct[-1].bias.fill_(.15)
    path = Path(root) / 'checkpoints/train/ostrack' / CONFIG / 'OSTrack_ep0020.pth.tar'
    path.parent.mkdir(parents=True)
    torch.save(dict(epoch=20, net_type='OSTrack', net=model.state_dict(),
                    frozen_response_contract=model.frozen_response_contract), path)
    return dataset


class QualityMathTests(unittest.TestCase):
    def test_all_candidates_exact_production_center_decode(self):
        torch.manual_seed(42)
        size, offset = torch.rand(1, 2, 16, 16), torch.rand(1, 2, 16, 16)
        expected = CenterPredictor.cal_bbox(SimpleNamespace(feat_sz=16),
            torch.eye(256).reshape(256, 1, 16, 16), size.repeat(256, 1, 1, 1), offset.repeat(256, 1, 1, 1))
        self.assertTrue(torch.equal(dense_center_boxes(size, offset)[0], expected))

    def test_common_shift_cancels_and_zero_is_bit_exact(self):
        logits = torch.tensor([[.25, -.5, .75, 1.]])
        valid = torch.tensor([[True, True, False, True]])
        delta = quality_delta(logits, valid, 2.)
        self.assertTrue(torch.allclose(delta, quality_delta(logits + 7, valid, 2.), atol=1e-6, rtol=0))
        self.assertEqual(delta[0, 2], 0)
        base = torch.tensor([[[[1e-4, .3], [.9, .9999]]]])
        self.assertTrue(torch.equal(base, apply_energy_correction(base, torch.zeros_like(base))))
        self.assertTrue(torch.equal(base.argmax(), apply_energy_correction(base, torch.full_like(base, 2.)).argmax()))
        corrected = apply_energy_correction(base, delta)
        self.assertLessEqual(corrected.max().item(), 1.)
        self.assertEqual(corrected.argmax(), (base.log() + delta.view_as(base)).argmax())

    def test_roi_support_can_survive_removed_center_and_graph_boundary(self):
        torch.manual_seed(42)
        head = CandidateQuality(32)
        with torch.no_grad():
            head.correct[-1].weight.fill_(.1)
        features = torch.randn(1, 80, 32, requires_grad=True)
        logits = torch.randn(1, 1, 4, 4, requires_grad=True)
        size = torch.full((1, 2, 4, 4), .8, requires_grad=True)
        offset = torch.full((1, 2, 4, 4), .5, requires_grad=True)
        kept = torch.ones(1, 16, dtype=torch.bool)
        kept[0, 5] = False
        out = head(features, logits, size, offset, torch.tensor([[.25, .25, .5, .5]]), kept)
        self.assertTrue(out['candidate_valid_mask'][0, 5])
        grads = torch.autograd.grad(out['candidate_quality_logits'].sum(),
            [features, logits, size, offset] + list(head.parameters()), allow_unused=True)
        self.assertTrue(all(g is None for g in grads[:4]))
        self.assertTrue(any(g is not None and g.abs().sum() > 0 for g in grads[4:]))
        out = head(features, logits, size, offset, torch.tensor([[.25, .25, .5, .5]]), torch.zeros_like(kept))
        self.assertFalse(out['candidate_valid_mask'].any())
        self.assertFalse(out['response_delta_energy'].any())

    def loss_fixture(self):
        logits = torch.zeros(1, 4, requires_grad=True)
        boxes = torch.tensor([[[.5, .5, .4, .4], [.1, .1, .1, .1], [.8, .8, .2, .2], [.3, .3, .1, .1]]])
        valid = torch.ones(1, 4, dtype=torch.bool)
        out = dict(candidate_quality_logits=logits, candidate_boxes=boxes, candidate_valid_mask=valid,
            response_delta_energy=quality_delta(logits, valid, 2.).reshape(1, 1, 2, 2),
            response_base_score_map=torch.tensor([[[[.9, .8], [.7, .6]]]]))
        return out, torch.tensor([[.3, .3, .4, .4]])

    def test_actual_iou_targets_and_ranking_gradient_protect_correct_box(self):
        out, gt = self.loss_fixture()
        self.assertTrue(torch.allclose(candidate_iou(out['candidate_boxes'], gt)[0], torch.tensor([1., 0., 0., .0149]), atol=.001))
        components, stats = quality_losses(out, gt, quality_config().TRAIN.RESPONSE_RESIDUAL)
        self.assertEqual(stats['ranking_samples'], 1)
        self.assertEqual(stats['preservation_samples'], 1)
        for name in ('candidate_rank', 'candidate_preserve'):
            grad = torch.autograd.grad(components[name], out['candidate_quality_logits'], retain_graph=True)[0]
            self.assertLess(grad[0, 0].item(), 0)
            self.assertTrue((grad[0, 1:] > 0).all())

    def test_no_evidence_or_good_candidate_has_no_fake_positive(self):
        config = quality_config().TRAIN.RESPONSE_RESIDUAL
        for kwargs in ({'targets': torch.tensor([[-1., .3, .1, .1]])},
                       {'padding': torch.ones(1, 256, 256, dtype=torch.bool)},
                       {'occlusion': torch.ones(1, 256, 256, dtype=torch.bool)}):
            out, gt = self.loss_fixture()
            targets = kwargs.pop('targets', gt)
            components, stats = quality_losses(out, targets, config, **kwargs)
            sum(components.values()).backward()
            self.assertEqual(stats['eligible_samples'], 0)
            self.assertFalse(out['candidate_quality_logits'].grad.any())
        out, gt = self.loss_fixture()
        out['candidate_boxes'][:, :, 2:] = .01
        components, stats = quality_losses(out, gt, config)
        self.assertEqual(stats['ranking_samples'], 0)
        self.assertEqual(stats['no_good_candidate_samples'], 1)
        self.assertEqual(components['candidate_rank'].item(), 0)
        self.assertTrue(target_has_evidence(torch.tensor([[.45, .45, .01, .01]])).item())


class QualityTrainingTests(unittest.TestCase):
    setUpClass = classmethod(response_tests.ResponseTests.setUpClass.__func__)
    tearDownClass = classmethod(response_tests.ResponseTests.tearDownClass.__func__)
    tearDown = response_tests.ResponseTests.tearDown

    def setUp(self):
        response_tests.ResponseTests.setUp(self)
        self.cfg = quality_config()

    def prepare(self, checkpoint=None):
        return prepare_response_models(self.cfg, self.root, CONFIG, checkpoint)

    test_loader_does_not_access_validation = response_tests.ResponseTests.test_training_loader_never_constructs_validation_dataset
    test_old_hashes_unchanged = response_tests.ResponseTests.test_old_config_hashes_are_unchanged

    def test_four_gpu_and_missing_source_fail_before_dataset_load(self):
        from lib.train import train_script
        settings = SimpleNamespace(script_name='ostrack', config_name=CONFIG, seed=42,
            local_rank=0, save_dir=str(self.root),
            cfg_file=str(Path(__file__).resolve().parents[1] / 'experiments/ostrack' / (CONFIG + '.yaml')))
        with patch('torch.distributed.get_world_size', return_value=2), \
                patch.object(train_script, 'prepare_response_models') as prepare, \
                patch.object(train_script, 'build_dataloaders') as loader:
            with self.assertRaisesRegex(ValueError, 'nproc_per_node 4'):
                train_script.run(settings)
        prepare.assert_not_called()
        loader.assert_not_called()
        with patch('torch.distributed.get_world_size', return_value=4), \
                patch.object(train_script, 'prepare_response_models', side_effect=FileNotFoundError('source missing')), \
                patch.object(train_script, 'build_dataloaders') as loader:
            with self.assertRaisesRegex(FileNotFoundError, 'source missing'):
                train_script.run(settings)
        loader.assert_not_called()

    def test_old_response_hash_unchanged(self):
        from lib.train.frozen_vdrm import configuration_hash
        self.assertEqual(configuration_hash(response_tests.response_config()),
            '88c2e3e7d0d20fb14a033e23083c3c33e227dc384d16571f62d2e255319b9c65')

    def test_strict_config_and_isolation_optimizer_and_epoch20_resume(self):
        validate_vdrm_experiment_contract(self.cfg, actual_seed=42)
        for field, value in (('EPOCH', 300), ('VDRM_REQUIRED_SEED', 43), ('AMP', True)):
            cfg = deepcopy(self.cfg)
            cfg.TRAIN[field] = value
            with self.assertRaises(ValueError):
                validate_vdrm_experiment_contract(cfg, actual_seed=42)
        model, baseline, _ = self.prepare()
        before = baseline_hash(model.state_dict(), False)
        result = audit_response_model(model, baseline, self.cfg, fixed_batch())
        self.assertEqual(before, baseline_hash(model.state_dict(), False))
        self.assertEqual(result['baseline_gradient_tensors'], 0)
        self.assertEqual(result['optimizer_steps'], 0)
        self.assertGreater(result['candidate_quality_head_gradient_l2'], 0)
        optimizer, scheduler = get_optimizer_scheduler(model, self.cfg)
        assert_response_optimizer(model, optimizer)
        loss, _ = make_response_actor(model, self.cfg)(fixed_batch())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        self.assertEqual(baseline_hash(model.state_dict()), baseline_hash(baseline.state_dict()))
        self.assertNotEqual(before, baseline_hash(model.state_dict(), False))
        audit_response_model(model, baseline, self.cfg, fixed_batch())
        trainer = object.__new__(BaseTrainer)
        trainer.actor = make_response_actor(model, self.cfg)
        trainer.optimizer, trainer.lr_scheduler = optimizer, scheduler
        trainer._checkpoint_dir = str(self.root / 'checkpoints')
        trainer.settings = SimpleNamespace(project_path='train/ostrack/' + CONFIG, config_name=CONFIG, local_rank=-1)
        trainer.epoch, trainer.stats, trainer.loaders = 20, {}, []
        trainer.save_checkpoint()
        path = self.root / 'checkpoints/train/ostrack' / CONFIG / 'OSTrack_ep0020.pth.tar'
        resumed, _, _ = self.prepare(path)
        self.assertEqual(baseline_hash(model.state_dict(), False), baseline_hash(resumed.state_dict(), False))
        trainer.load_checkpoint(str(path))
        checkpoint = torch.load(path, weights_only=False)
        validate_response_checkpoint(model, checkpoint, self.cfg, expected_epoch=20)
        checkpoint['epoch'] = 300
        with self.assertRaises(ValueError):
            validate_response_checkpoint(model, checkpoint, self.cfg)
        checkpoint['epoch'] = 20
        checkpoint['net']['backbone.vdrm.match_bias'] += .01
        with self.assertRaises(ValueError):
            validate_response_checkpoint(model, checkpoint, self.cfg)


class QualityProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)
        cls.temp = tempfile.TemporaryDirectory(prefix='quality_probe_')
        cls.root = Path(cls.temp.name)
        cls.dataset = quality_fixture(cls.root)
        cls.output = cls.root / 'quality.tar.gz'
        with patch('tracking.diagnose_vdrm_frozen_failures.build_ostrack', side_effect=response_model), \
                patch('lib.models.ostrack.build_ostrack', side_effect=response_model), \
                patch.object(entry, 'get_dataset', return_value=cls.dataset), \
                patch.object(torch.Tensor, 'cuda', lambda t, *a, **kw: t), contextlib.redirect_stdout(io.StringIO()):
            cls.manifest = entry.run(cls.root, cls.output, 'cpu', profile='early_failure', candidate_arm='Rqual')

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()
        torch.set_num_threads(cls.threads)

    def test_bundle_analysis_and_checkpoint_audit(self):
        self.assertEqual(check_bundle(self.output), self.manifest)
        self.assertEqual(self.manifest['records'], 298)
        summary, frames, report = analyze_bundle(self.output)
        self.assertEqual(summary['candidate_arm'], 'Rqual')
        self.assertEqual(len(frames), 298)
        self.assertIn('Rqual', report)
        self.assertNotIn('Rresp', report)
        args = ['check_vdrm_response.py', '--config', CONFIG, '--save_dir', str(self.root), '--device', 'cpu',
                '--checkpoint', str(self.root / 'checkpoints/train/ostrack' / CONFIG / 'OSTrack_ep0020.pth.tar')]
        with patch('lib.models.ostrack.build_ostrack', side_effect=response_model), patch.object(sys, 'argv', args):
            checker.main()

    def test_formula_tampering_is_rejected(self):
        def mutate(manifest, payloads):
            records = json.loads(payloads['records.json'])
            member = records[0]['arms']['Rqual']['maps_member']
            with np.load(io.BytesIO(payloads[member]), allow_pickle=False) as maps:
                arrays = {key: maps[key] for key in maps.files}
            arrays['response_delta_energy'] += .01
            payloads[member] = entry.npz_bytes(arrays)
        path = self.root / 'tampered.tar.gz'
        rewrite_bundle(self.output, path, mutate)
        with self.assertRaises(ValueError):
            check_bundle(path)

    def test_production_tracker_float_trace_and_nested_writer(self):
        from lib.test.evaluation import Sequence, Tracker
        from lib.test.evaluation.running import _save_tracker_output
        with tarfile.open(self.output, 'r:gz') as archive:
            traces = json.load(archive.extractfile('trajectories.json'))
        for trace in traces[:1]:
            source = next(s for s in self.dataset if s.name == trace['sequence'])
            length = len(trace['gt_xywh'])
            sequence = Sequence(source.name, source.frames[:length], 'got10k', source.ground_truth_rect[:length])
            env = SimpleNamespace(prj_dir=str(Path(__file__).resolve().parents[1]), save_dir=str(self.root),
                                  results_path=str(self.root / 'production'))
            with patch('lib.test.evaluation.tracker.env_settings', return_value=env):
                tracker = Tracker('ostrack', CONFIG, 'got10k_vdrm_dev')
            with patch('lib.test.parameter.ostrack.env_settings', return_value=env), \
                    patch('lib.test.tracker.ostrack.build_ostrack', side_effect=response_model), \
                    patch.object(torch.nn.Module, 'cuda', lambda m, *a, **kw: m), \
                    patch.object(torch.Tensor, 'cuda', lambda t, *a, **kw: t):
                output = tracker.run_sequence(sequence, debug=0)
            self.assertEqual(output['target_bbox'], [r['bbox_xywh'] for r in trace['native']['Rqual']])
            _save_tracker_output(sequence, tracker, output)
            self.assertTrue((Path(tracker.results_dir) / 'got10k' / (sequence.name + '.txt')).is_file())
            self.assertFalse((Path(tracker.results_dir) / (sequence.name + '.txt')).exists())


class QualityCommandTests(unittest.TestCase):
    def test_document_commands_and_default_seed_epoch_and_paths(self):
        from tracking import train, test_uav_suite, analyze_vdrm_module1_dev, analyze_vdrm_response
        project = Path(__file__).resolve().parents[1]
        document = (project / 'VDRM_RQUAL_PLAN.md').read_text(encoding='utf-8')
        class Parsed(Exception):
            pass
        count = 0
        for block in re.findall(r'```\w+\n(.*?)```', document, re.S):
            if 'python tracking/' not in block:
                continue
            tokens = shlex.split(block.replace('\\\n', ''))
            index = tokens.index('python')
            script, args = tokens[index+1], tokens[index+2:]
            self.assertNotIn('--seed', args)
            count += 1
            for option in ('--config', '--tracker_param', '--reference'):
                if option in args:
                    self.assertTrue((project / 'experiments/ostrack' / (args[args.index(option)+1]+'.yaml')).is_file())
            with patch.object(sys, 'argv', [script, *args]):
                if script == 'tracking/train.py':
                    with patch.object(train.os, 'system', return_value=0) as launch:
                        train.main()
                    self.assertIn('--seed 42', launch.call_args.args[0])
                    self.assertIn('--nproc_per_node 4', launch.call_args.args[0])
                elif script == 'tracking/check_vdrm_response.py':
                    self.assertEqual(args[args.index('--checkpoint')+1],
                        f'output/checkpoints/train/ostrack/{CONFIG}/OSTrack_ep0020.pth.tar')
                    with patch.object(checker, 'default_config', side_effect=Parsed):
                        with self.assertRaises(Parsed):
                            checker.main()
                elif script == 'tracking/diagnose_vdrm_response.py':
                    with patch.object(entry.torch.cuda, 'is_available', return_value=True), patch.object(entry, 'run') as call:
                        entry.main()
                    self.assertEqual(call.call_args.kwargs, dict(profile='early_failure', candidate_arm='Rqual'))
                    self.assertEqual(call.call_args.args[1], 'output/analysis/frozen_quality/rqual_early_probe.tar.gz')
                elif script == 'tracking/analyze_vdrm_response.py':
                    with patch.object(analyze_vdrm_response, 'run') as call:
                        analyze_vdrm_response.main()
                    self.assertEqual(call.call_args.args, ('output/analysis/frozen_quality/rqual_early_probe.tar.gz',
                                                          'output/analysis/frozen_quality/rqual_early_analysis.tar.gz'))
                elif script == 'tracking/test_uav_suite.py':
                    parsed = test_uav_suite.parse_args()
                    self.assertEqual(parsed.dataset, 'got10k_vdrm_dev')
                    self.assertEqual((parsed.num_gpus, parsed.threads), (4, 4))
                elif script == 'tracking/analyze_vdrm_module1_dev.py':
                    with patch.object(analyze_vdrm_module1_dev, 'get_dataset', side_effect=Parsed):
                        with self.assertRaises(Parsed):
                            analyze_vdrm_module1_dev.main()
                    self.assertEqual(args[args.index('--report_name')+1], 'vdrm_rqual_dev')
                else:
                    self.fail(script)
        self.assertEqual(count, 7)


if __name__ == '__main__':
    unittest.main()
