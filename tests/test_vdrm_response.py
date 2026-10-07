from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from lib.config.ostrack.config import default_config, update_config_from_file
from lib.train.base_functions import validate_vdrm_experiment_contract, get_optimizer_scheduler
from lib.train.frozen_vdrm import BASE_CONFIG, load_base_config, configuration_hash
from lib.train.frozen_response import (prepare_response_models, validate_response_checkpoint,
    audit_response_model, baseline_hash, make_response_actor, assert_response_optimizer)
from lib.train.actors.response_residual import response_losses
from lib.train import base_functions, train_script
from lib.train.trainers.base_trainer import BaseTrainer
from test_vdrm_frozen import tiny_model, fixed_batch

PROJECT = Path(__file__).resolve().parents[1]
CONFIG = 'vitb_256_mae_ce_vdrm_rresp_s42_32x4_ep300'


def response_config():
    cfg = default_config()
    update_config_from_file(PROJECT / 'experiments/ostrack' / (CONFIG + '.yaml'), cfg)
    return cfg


def response_model(cfg, training=False):
    model = tiny_model(cfg, training)
    if getattr(cfg.MODEL, 'RESPONSE_RESIDUAL', {}).get('ENABLED'):
        model.enable_response_residual(32, cfg.MODEL.RESPONSE_RESIDUAL)
    return model


class ResponseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        torch.manual_seed(42)
        self.cfg = response_config()
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = response_model(load_base_config()).eval()
        source_path = self.root / 'checkpoints/train/ostrack' / BASE_CONFIG / 'OSTrack_ep0300.pth.tar'
        source_path.parent.mkdir(parents=True)
        torch.save(dict(epoch=300, net_type='OSTrack', net=self.source.state_dict(),
                        settings=SimpleNamespace(config_name=BASE_CONFIG)), source_path)
        self.factory = patch('lib.models.ostrack.build_ostrack', side_effect=response_model)
        self.factory.start()

    def tearDown(self):
        self.factory.stop()
        self.temp.cleanup()

    def prepare(self, checkpoint=None):
        return prepare_response_models(self.cfg, self.root, CONFIG, checkpoint)

    def test_old_config_hashes_are_unchanged(self):
        hashes = {
            BASE_CONFIG: 'eeee473d8ef23fa2b01d225a64b3b64903e5c02ee560dbd6dc44d17da9c3b702',
            'vitb_256_mae_ce_vdrm_rfreeze_s42_32x4_ep300': '55638ed0d60e5df5a9a85af6e90d6ec28b88529b955e4e6ef9087b6ec8de1555',
            'vitb_256_mae_ce_vdrm_rdisc_s42_32x4_ep300': 'df47ce120236c7d3297518490c5437d8dc914d913cfce25d2a1aa423bc40a86e',
        }
        for name, expected in hashes.items():
            cfg = default_config()
            update_config_from_file(PROJECT / 'experiments/ostrack' / (name + '.yaml'), cfg)
            self.assertEqual(configuration_hash(cfg), expected)
            self.assertNotIn('RESPONSE_RESIDUAL', cfg.MODEL)

    def test_config_fail_closed_and_seed_default(self):
        validate_vdrm_experiment_contract(self.cfg, actual_seed=42)
        for section, field, value in (('TRAIN', 'VDRM_REQUIRED_SEED', 7),
                ('TRAIN', 'GIOU_WEIGHT', 2), ('TRAIN', 'VDRM_FROZEN_SKIP_VAL', False),
                ('TRAIN', 'VDRM_EXPERIMENT_ARM', 'ronly')):
            cfg = deepcopy(self.cfg)
            cfg[section][field] = value
            with self.assertRaises(ValueError):
                validate_vdrm_experiment_contract(cfg, actual_seed=42)
        cfg = deepcopy(self.cfg)
        cfg.MODEL.VDRM.TRAIN_ALPHA = True
        with self.assertRaises(ValueError):
            validate_vdrm_experiment_contract(cfg, actual_seed=42)
        for section in ('MODEL', 'TRAIN'):
            bad = self.root / 'bad.yaml'
            bad.write_text(section + ':\n  RESPONSE_RESIDUAL:\n    TYPO: 1\n', encoding='utf-8')
            with self.assertRaises(ValueError):
                update_config_from_file(bad, default_config())

    def test_audit_exact_outputs_isolated_gradients_and_no_updates(self):
        model, baseline, _ = self.prepare()
        before = baseline_hash(model.state_dict(), False)
        rng = torch.get_rng_state().clone()
        data = fixed_batch()
        rng = torch.get_rng_state().clone()
        result = audit_response_model(model, baseline, self.cfg, data)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(before, baseline_hash(model.state_dict(), False))
        self.assertEqual(result['baseline_gradient_tensors'], 0)
        self.assertGreater(result['response_hann_rank_head_gradient_l2'], 0)
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_optimizer_step_cannot_modify_baseline_and_resume_is_strict(self):
        model, baseline, _ = self.prepare()
        actor = make_response_actor(model, self.cfg)
        optimizer, _ = get_optimizer_scheduler(model, self.cfg)
        assert_response_optimizer(model, optimizer)
        loss, _ = actor(fixed_batch())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        self.assertEqual(baseline_hash(model.state_dict()), baseline_hash(baseline.state_dict()))
        self.assertEqual(model.backbone.vdrm.alpha.item(), 0)
        result = audit_response_model(model, baseline, self.cfg, fixed_batch())
        self.assertTrue(result['bypass_exact_tclean'])
        checkpoint = dict(epoch=300, net_type='OSTrack', net=model.state_dict(),
                          frozen_response_contract=model.frozen_response_contract)
        path = self.root / 'response.pth.tar'
        torch.save(checkpoint, path)
        resumed, _, _ = self.prepare(path)
        self.assertEqual(baseline_hash(resumed.state_dict(), False), baseline_hash(model.state_dict(), False))
        validate_response_checkpoint(resumed, checkpoint, cfg=self.cfg, expected_epoch=300)
        checkpoint['net']['backbone.vdrm.match_bias'] += .01
        with self.assertRaises(ValueError):
            validate_response_checkpoint(resumed, checkpoint, cfg=self.cfg)

    def test_ce_removed_cells_receive_zero_correction(self):
        model, baseline, _ = self.prepare()
        with torch.no_grad():
            model.response_residual.correct[-1].bias.fill_(.5)
        data = fixed_batch()
        model.eval()
        out = make_response_actor(model, self.cfg).forward_pass(data)
        self.assertFalse(out['size_map'].requires_grad)
        removed = torch.cat(out['removed_indexes_s'], 1).long()
        self.assertEqual(out['response_delta_logits'].flatten(1).gather(1, removed).abs().sum().item(), 0)
        self.assertLessEqual(out['response_delta_logits'].abs().max().item(), 2.)

    def test_missing_gt_ce_occlusion_or_padding_has_no_fabricated_label(self):
        cfg = self.cfg.TRAIN.RESPONSE_RESIDUAL
        boxes = torch.tensor([[.25, .25, .5, .5]])
        for kwargs in ({'boxes': torch.tensor([[-1., .25, .1, .1]])},
                       {'boxes': torch.tensor([[.45, .45, .01, .01]])},
                       {'removed_indexes': [torch.tensor([[5, 6, 9, 10]])]},
                       {'occlusion_mask': torch.ones(1, 16, 16)},
                       {'padding_mask': torch.ones(1, 16, 16)}):
            logits = torch.zeros(1, 1, 4, 4, requires_grad=True)
            args = dict(boxes=boxes, **kwargs) if 'boxes' not in kwargs else kwargs
            focal, rank, stats = response_losses(logits, logits.sigmoid(), config=cfg, **args)
            (focal + rank).backward()
            self.assertEqual(stats['eligible_samples'], 0)
            self.assertEqual(logits.grad.abs().sum().item(), 0)

    def test_hann_rank_matches_native_peak_and_gradient_direction(self):
        cfg = deepcopy(self.cfg.TRAIN.RESPONSE_RESIDUAL)
        cfg.NEGATIVE_GUARD_CELLS = 0
        logits = torch.zeros(1, 1, 4, 4, requires_grad=True)
        with torch.no_grad():
            logits[0, 0, 0, 0] = 4
        box = torch.tensor([[0., 0., .25, .25]])
        focal, rank, stats = response_losses(logits, logits.sigmoid(), box, cfg, return_masks=True)
        self.assertNotEqual(stats['final_response'].argmax().item(), logits.argmax().item())
        rank.backward()
        self.assertLess(logits.grad[0, 0, 0, 0].item(), 0)
        self.assertGreater(logits.grad[0, 0, 1, 1].item(), 0)

    def test_source_failure_and_four_gpu_guard_before_dataset_access(self):
        settings = SimpleNamespace(script_name='ostrack', config_name=CONFIG, seed=42,
            local_rank=0, save_dir=str(self.root),
            cfg_file=str(PROJECT / 'experiments/ostrack' / (CONFIG + '.yaml')))
        with patch('torch.distributed.get_world_size', return_value=4), \
                patch.object(train_script, 'prepare_response_models', side_effect=FileNotFoundError('source missing')), \
                patch.object(train_script, 'build_dataloaders') as loader:
            with self.assertRaisesRegex(FileNotFoundError, 'source missing'):
                train_script.run(settings)
        loader.assert_not_called()
        with patch('torch.distributed.get_world_size', return_value=2), \
                patch.object(train_script, 'prepare_response_models') as prepare, \
                patch.object(train_script, 'build_dataloaders') as loader:
            with self.assertRaisesRegex(ValueError, 'nproc_per_node 4'):
                train_script.run(settings)
        prepare.assert_not_called()
        loader.assert_not_called()

    def test_training_loader_never_constructs_validation_dataset(self):
        settings = SimpleNamespace(local_rank=-1)
        base_functions.update_settings(settings, self.cfg)
        with patch.object(base_functions, 'names2datasets', return_value=[object()]) as datasets, \
                patch.object(base_functions.sampler, 'TrackingSampler', return_value=object()), \
                patch.object(base_functions.processing, 'STARKProcessing', return_value=object()), \
                patch.object(base_functions, 'LTRLoader', return_value=object()):
            _, validation = base_functions.build_dataloaders(self.cfg, settings)
        self.assertIsNone(validation)
        self.assertEqual(datasets.call_count, 1)
        self.assertEqual(datasets.call_args.args[0], self.cfg.DATA.TRAIN.DATASETS_NAME)

    def test_official_checkpoint_save_resume_and_baseline_guard(self):
        model, _, _ = self.prepare()
        trainer = object.__new__(BaseTrainer)
        trainer.actor = make_response_actor(model, self.cfg)
        trainer.optimizer, trainer.lr_scheduler = get_optimizer_scheduler(model, self.cfg)
        trainer._checkpoint_dir = str(self.root / 'checkpoints')
        trainer.settings = SimpleNamespace(project_path='train/ostrack/' + CONFIG,
                                           config_name=CONFIG, local_rank=-1)
        trainer.epoch, trainer.stats, trainer.loaders = 300, {}, []
        trainer.save_checkpoint()
        path = self.root / 'checkpoints/train/ostrack' / CONFIG / 'OSTrack_ep0300.pth.tar'
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        self.assertEqual(checkpoint['frozen_response_contract'], model.frozen_response_contract)
        expected = model.response_residual.correct[-1].bias.detach().clone()
        with torch.no_grad():
            model.response_residual.correct[-1].bias.fill_(.75)
        trainer.load_checkpoint(str(path))
        self.assertTrue(torch.equal(model.response_residual.correct[-1].bias, expected))
        with torch.no_grad():
            model.backbone.vdrm.match_bias.add_(.01)
        with self.assertRaisesRegex(RuntimeError, 'modified Rresp baseline'):
            trainer.save_checkpoint()


if __name__ == '__main__':
    unittest.main()
