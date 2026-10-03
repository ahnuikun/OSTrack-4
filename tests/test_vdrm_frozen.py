import contextlib
from copy import deepcopy
import io
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import sys

import numpy as np
import torch

from lib.config.ostrack.config import default_config, update_config_from_file
from lib.models.layers.vdrm import VisibilityDrivenRepresentationModule
from lib.models.layers.discriminative_route import retained_coordinates
from lib.models.layers.head import build_box_head
from lib.models.ostrack.ostrack import OSTrack
from lib.models.ostrack.vit_ce import VisionTransformerCE
from lib.train.actors.discriminative_route_loss import compute_discriminative_route_loss
from lib.train import base_functions, train_script
from lib.train.base_functions import get_optimizer_scheduler, validate_vdrm_experiment_contract
from lib.train.frozen_vdrm import (BASE_CONFIG, PREFIX, audit_frozen_model, load_base_config,
    make_actor, prepare_frozen_models, validate_frozen_checkpoint, visual_hash)
from tracking import train as train_launcher

PROJECT = Path(__file__).resolve().parents[1]


def candidate_config(arm):
    name = f'vitb_256_mae_ce_vdrm_{arm}_s42_32x4_ep300'
    cfg = default_config()
    update_config_from_file(PROJECT / 'experiments' / 'ostrack' / (name + '.yaml'), cfg)
    return name, cfg


def tiny_model(cfg, training=False):
    # Real CE/backbone/prediction-head code, reduced feature width/depth only.
    backbone = VisionTransformerCE(embed_dim=32, depth=4, num_heads=4,
        ce_loc=[1, 2], ce_keep_ratio=[.7, .7], drop_path_rate=.1,
        vdrm_enabled=True, vdrm_insert_layer=2, vdrm_spatial_gate_mode='part_aligned',
        vdrm_alpha_max=1.5, vdrm_train_alpha=cfg.MODEL.VDRM.TRAIN_ALPHA)
    backbone.finetune_track(cfg)
    if cfg.MODEL.VDRM.DISCRIMINATIVE_ROUTE:
        backbone.vdrm.enable_discriminative_route(32)
    head_cfg = deepcopy(cfg)
    head_cfg.MODEL.HEAD.NUM_CHANNELS = 32
    return OSTrack(backbone, build_box_head(head_cfg, 32), head_type='CENTER', vdrm_enabled=True)


def fixed_batch(device='cpu'):
    return dict(template_images=torch.randn(1, 2, 3, 128, 128, device=device),
                search_images=torch.randn(1, 2, 3, 256, 256, device=device),
                template_anno=torch.tensor([[[.25, .25, .5, .5]] * 2], device=device),
                search_anno=torch.tensor([[[.25, .25, .5, .5]] * 2], device=device),
                search_att=torch.zeros(1, 2, 256, 256, device=device, dtype=torch.bool), epoch=1)


class RouteTests(unittest.TestCase):
    def test_launcher_default_seed_and_failure_exit_are_not_silent(self):
        argv = ['train.py', '--script', 'ostrack', '--config', 'test', '--save_dir', './output',
                '--mode', 'multiple', '--nproc_per_node', '4', '--use_lmdb', '0', '--use_wandb', '0']
        with patch.object(sys, 'argv', argv), patch.object(train_launcher.os, 'system', return_value=0) as launch:
            train_launcher.main()
            self.assertIn('--seed 42', launch.call_args.args[0])
            self.assertIn('--nproc_per_node 4', launch.call_args.args[0])
        with patch.object(sys, 'argv', argv), patch.object(train_launcher.os, 'system', return_value=1):
            with self.assertRaises(SystemExit) as error:
                train_launcher.main()
            self.assertNotEqual(error.exception.code, 0)

    def test_new_head_zero_correction_keeps_v8_identical_and_rng(self):
        torch.manual_seed(42)
        baseline = VisibilityDrivenRepresentationModule(spatial_gate_mode='part_aligned', alpha_max=1.5)
        with torch.no_grad():
            baseline.alpha.fill_(.2)
        candidate = deepcopy(baseline)
        state = torch.get_rng_state().clone()
        candidate.enable_discriminative_route(32)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        tokens = torch.randn(2, 64 + 21, 32)
        indices = torch.arange(21).repeat(2, 1)
        args = dict(template_length=64, template_bbox=torch.tensor([[.25, .25, .5, .5]] * 2),
                    search_global_index=indices, search_grid_size=16)
        expected, a = baseline(tokens, **args)
        actual, b = candidate(tokens, **args)
        self.assertTrue(torch.equal(expected, actual))
        self.assertTrue(torch.equal(a['part_route_logits'], b['part_route_logits']))
        self.assertEqual(set(baseline.state_dict()), set(candidate.state_dict()) -
                         {k for k in candidate.state_dict() if k.startswith('route_head.')})

    def test_route_aux_is_detached_from_visual_and_original_scalars(self):
        module = VisibilityDrivenRepresentationModule(spatial_gate_mode='part_aligned', alpha_max=1.5)
        module.enable_discriminative_route(32)
        tokens = torch.randn(2, 64 + 256, 32, requires_grad=True)
        _, out = module(tokens, 64, torch.tensor([[.25, .25, .5, .5]] * 2),
                        torch.arange(256).repeat(2, 1), 16)
        loss, stats = compute_discriminative_route_loss(out['discriminative_route_logits'],
            out['search_global_index'], torch.tensor([[.25, .25, .5, .5]] * 2), 16, 16,
            part_valid=out['part_valid'])
        grads = torch.autograd.grad(loss, [tokens] + list(module.parameters()), allow_unused=True)
        self.assertIsNone(grads[0])
        for (name, _), grad in zip(module.named_parameters(), grads[1:]):
            if not name.startswith('route_head.'):
                self.assertIsNone(grad, name)
        self.assertGreater(stats['disc_route_eligible_parts'].item(), 0)
        self.assertGreater(sum(g.abs().sum().item() for g in grads if g is not None), 0)

    def test_ce_mapping_and_no_fabricated_missing_part(self):
        logits = torch.zeros(1, 4, 3, requires_grad=True)
        loss, stats = compute_discriminative_route_loss(logits, torch.tensor([[15, 5, 0]]),
            torch.tensor([[.25, .25, .5, .5]]), 4, 4, negative_guard=0)
        loss.backward()
        self.assertEqual(stats['disc_route_eligible_parts'].item(), 1)
        self.assertLess(logits.grad[0, 0, 1].item(), 0)
        self.assertGreater(logits.grad[0, 0, 0].item(), 0)
        self.assertEqual(logits.grad[0, 1:].abs().sum().item(), 0)

    def test_absent_gt_or_missing_positive_yields_finite_zero_loss(self):
        for bbox, indices in (([-.5, .2, .1, .1], [0, 15]), ([.25, .25, .5, .5], [0, 15])):
            with self.subTest(box=bbox):
                logits = torch.randn(1, 4, 2, requires_grad=True)
                loss, stats = compute_discriminative_route_loss(logits, torch.tensor([indices]),
                    torch.tensor([bbox]), 4, 4)
                loss.backward()
                self.assertEqual(loss.item(), 0)
                self.assertEqual(logits.grad.abs().sum().item(), 0)
                self.assertEqual(stats['disc_route_eligible_parts'].item(), 0)

    def test_occlusion_padding_and_invalid_parts_are_ignored(self):
        indices = torch.arange(16).reshape(1, -1)
        box = torch.tensor([[.25, .25, .5, .5]])
        for kwargs in ({'occlusion_mask': torch.ones(1, 16, 16)},
                       {'padding_mask': torch.ones(1, 16, 16)},
                       {'part_valid': torch.zeros(1, 4)},
                       {'part_visibility': torch.zeros(1, 4)}):
            with self.subTest(kind=next(iter(kwargs))):
                loss, _ = compute_discriminative_route_loss(torch.zeros(1, 4, 16), indices, box, 4, 4, **kwargs)
                self.assertEqual(loss.item(), 0)

    def test_partially_clipped_box_does_not_shift_part_identity(self):
        loss, stats = compute_discriminative_route_loss(torch.zeros(1, 4, 16),
            torch.arange(16).reshape(1, -1), torch.tensor([[-.25, .25, .5, .5]]), 4, 4,
            negative_guard=0)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(stats['disc_route_eligible_parts'].item(), 2)

    def test_paste_removed_by_ce_has_no_false_nearest_paste_label(self):
        _, stats = compute_discriminative_route_loss(torch.zeros(1, 4, 16),
            torch.arange(16).reshape(1, -1), torch.tensor([[.25, .25, .5, .5]]), 4, 4,
            distractor_boxes=torch.tensor([[.8, .8, .05, .05]]), distractor_applied=torch.ones(1))
        self.assertEqual(stats['disc_route_pasted_parts'].item(), 0)

    def test_invalid_ce_indices_fail(self):
        for values in ([0, 0], [0, 16], [0, .5], [0, float('nan')]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                retained_coordinates(torch.tensor([values]), 1, 2, 4, 4, torch.float32, 'cpu')


class FrozenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        torch.manual_seed(42)
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source_path = self.root / 'checkpoints/train/ostrack' / BASE_CONFIG / 'OSTrack_ep0300.pth.tar'
        self.source_path.parent.mkdir(parents=True)
        self.source = tiny_model(load_base_config()).eval()
        torch.save(dict(epoch=300, net_type='OSTrack', net=self.source.state_dict(),
                        settings=SimpleNamespace(config_name=BASE_CONFIG)), self.source_path)
        self.build_patch = patch('lib.models.ostrack.build_ostrack', side_effect=tiny_model)
        self.build_patch.start()

    def tearDown(self):
        self.build_patch.stop()
        self.temp.cleanup()

    def prepare(self, arm):
        name, cfg = candidate_config(arm)
        net, base, _ = prepare_frozen_models(cfg, self.root, name)
        return name, cfg, net, base

    def test_configs_only_change_registered_route_factor(self):
        name_a, a = candidate_config('rfreeze')
        name_b, b = candidate_config('rdisc')
        validate_vdrm_experiment_contract(a, 42)
        validate_vdrm_experiment_contract(b, 42)
        expected = deepcopy(a)
        expected.TRAIN.VDRM_EXPERIMENT_ARM = 'rdisc'
        expected.TRAIN.VDRM_PART_ROUTE_WEIGHT = .5
        expected.MODEL.VDRM.DISCRIMINATIVE_ROUTE = True
        self.assertEqual(expected, b)
        for mutation in ('seed', 'pretrain', 'visual', 'override', 'aux', 'val', 'data', 'epochs'):
            invalid = deepcopy(b)
            if mutation == 'seed': invalid.TRAIN.VDRM_REQUIRED_SEED = 1
            elif mutation == 'pretrain': invalid.MODEL.PRETRAIN_FILE = 'mae.pth'
            elif mutation == 'visual': invalid.MODEL.BACKBONE.CE_KEEP_RATIO = [.8] * 3
            elif mutation == 'override': invalid.TEST.VDRM_ALPHA_OVERRIDE = 0.
            elif mutation == 'aux': invalid.TRAIN.VDRM_VISIBILITY_WEIGHT = .5
            elif mutation == 'val': invalid.TRAIN.VDRM_FROZEN_SKIP_VAL = False
            elif mutation == 'epochs': invalid.TEST.EPOCH = 299
            else: invalid.DATA.TRAIN.DATASETS_NAME = ['GOT10K_train_full']
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_vdrm_experiment_contract(invalid, 42)

    def test_missing_source_aborts_training_before_dataset_access(self):
        name, _ = candidate_config('rfreeze')
        settings = SimpleNamespace(script_name='ostrack', config_name=name, seed=42,
            local_rank=0, save_dir=str(self.root),
            cfg_file=str(PROJECT / 'experiments/ostrack' / (name + '.yaml')))
        with patch('torch.distributed.get_world_size', return_value=4), \
             patch.object(train_script, 'prepare_frozen_models', side_effect=FileNotFoundError('source missing')), \
             patch.object(train_script, 'build_dataloaders') as loader, \
             contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(FileNotFoundError, 'source missing'):
                train_script.run(settings)
        loader.assert_not_called()

    def test_frozen_loader_does_not_construct_validation_dataset(self):
        _, cfg = candidate_config('rdisc')
        settings = SimpleNamespace(local_rank=-1)
        base_functions.update_settings(settings, cfg)
        expected_loader = object()
        with patch.object(base_functions, 'names2datasets', return_value=[object()]) as datasets, \
             patch.object(base_functions.sampler, 'TrackingSampler', return_value=object()), \
             patch.object(base_functions.processing, 'STARKProcessing', return_value=object()), \
             patch.object(base_functions, 'LTRLoader', return_value=expected_loader) as loader, \
             contextlib.redirect_stdout(io.StringIO()):
            actual, validation = base_functions.build_dataloaders(cfg, settings)
        self.assertIs(actual, expected_loader)
        self.assertIsNone(validation)
        self.assertEqual(datasets.call_count, 1)
        self.assertEqual(datasets.call_args.args[0], cfg.DATA.TRAIN.DATASETS_NAME)
        self.assertEqual(loader.call_count, 1)

    def test_source_weights_modes_and_optimizer_are_exactly_frozen(self):
        for arm in ('rfreeze', 'rdisc'):
            with self.subTest(arm=arm):
                _, cfg, net, base = self.prepare(arm)
                self.assertEqual(visual_hash(net.state_dict()), visual_hash(base.state_dict()))
                net.train()
                self.assertTrue(net.training)
                self.assertFalse(net.backbone.training)
                self.assertFalse(net.box_head.training)
                self.assertTrue(net.backbone.vdrm.training)
                self.assertTrue(all(not m.training for n, m in net.named_modules()
                                    if isinstance(m, torch.nn.BatchNorm2d)))
                with contextlib.redirect_stdout(io.StringIO()):
                    optimizer, _ = get_optimizer_scheduler(net, cfg)
                parameters = {id(p) for g in optimizer.param_groups for p in g['params']}
                self.assertEqual(parameters, {id(p) for n, p in net.named_parameters() if n.startswith(PREFIX)})
                self.assertEqual({g['lr'] for g in optimizer.param_groups}, {cfg.TRAIN.LR})

    def test_real_network_audit_no_updates_rng_and_training_afterward(self):
        for arm in ('rfreeze', 'rdisc'):
            with self.subTest(arm=arm):
                _, cfg, net, base = self.prepare(arm)
                data = fixed_batch()
                before = {k: v.clone() for k, v in net.state_dict().items()}
                rng = torch.get_rng_state().clone()
                np_state, py_state = np.random.get_state(), random.getstate()
                result = audit_frozen_model(net, base, cfg, data)
                self.assertTrue(result['alpha0_exact'])
                self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                np.testing.assert_array_equal(np_state[1], np.random.get_state()[1])
                self.assertEqual(py_state, random.getstate())
                self.assertTrue(all(torch.equal(before[k], v) for k, v in net.state_dict().items()))
                self.assertTrue(all(p.grad is None for p in net.parameters()))
                with contextlib.redirect_stdout(io.StringIO()):
                    optimizer, _ = get_optimizer_scheduler(net, cfg)
                actor = make_actor(net, cfg)
                for _ in range(3):
                    loss, status = actor(data)
                    self.assertTrue(torch.isfinite(loss))
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    self.assertTrue(all(p.grad is None for n, p in net.named_parameters() if not n.startswith(PREFIX)))
                    optimizer.step()
                self.assertEqual(visual_hash(net.state_dict()), visual_hash(base.state_dict()))
                self.assertNotEqual(net.backbone.vdrm.alpha.item(), 0.)
                audit_frozen_model(net, base, cfg, data)

    def test_arms_initialize_with_same_visual_and_random_stream(self):
        result = []
        for arm in ('rfreeze', 'rdisc'):
            torch.manual_seed(42)
            _, _, net, _ = self.prepare(arm)
            result.append((net.state_dict(), torch.get_rng_state()))
        a, b = result
        self.assertTrue(torch.equal(a[1], b[1]))
        self.assertTrue(all(torch.equal(v, b[0][k]) for k, v in a[0].items()))

    def test_bad_source_identity_or_missing_keys_is_rejected(self):
        original = torch.load(self.source_path, weights_only=False)
        for kind in ('source', 'epoch', 'alpha', 'key'):
            modified = deepcopy(original)
            if kind == 'source': modified['settings'].config_name = 'ronly'
            elif kind == 'epoch': modified['epoch'] = 299
            elif kind == 'alpha': modified['net'][PREFIX + 'alpha'].fill_(.1)
            else: modified['net'].pop('backbone.pos_embed_x')
            torch.save(modified, self.source_path)
            with self.subTest(kind=kind), self.assertRaises((ValueError, RuntimeError)):
                self.prepare('rfreeze')

    def test_resume_requires_matching_provenance_and_frozen_buffers(self):
        name, cfg, net, base = self.prepare('rdisc')
        checkpoint = dict(net=net.state_dict(), net_type='OSTrack', epoch=40,
                          frozen_vdrm_contract=net.frozen_vdrm_contract)
        self.assertEqual(validate_frozen_checkpoint(net, checkpoint, cfg=cfg), net.frozen_vdrm_contract)
        for kind in ('metadata', 'arm', 'buffer', 'config', 'epoch', 'type'):
            bad = deepcopy(checkpoint)
            if kind == 'metadata': bad.pop('frozen_vdrm_contract')
            elif kind == 'arm': bad['frozen_vdrm_contract']['arm'] = 'rfreeze'
            elif kind == 'buffer':
                key = next(k for k in bad['net'] if k.endswith('running_mean'))
                bad['net'][key].add_(1.)
            elif kind == 'config': bad['frozen_vdrm_contract']['config_sha256'] = 'invalid'
            elif kind == 'epoch': bad['epoch'] = 301
            else: bad['net_type'] = 'wrong'
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                validate_frozen_checkpoint(net, bad, expected=net.frozen_vdrm_contract, cfg=cfg)
        with self.assertRaisesRegex(ValueError, 'expected 300'):
            validate_frozen_checkpoint(net, checkpoint, cfg=cfg, expected_epoch=300)
        dest = self.root / 'checkpoints/train/ostrack' / name / 'OSTrack_ep0040.pth.tar'
        dest.parent.mkdir(parents=True)
        with torch.no_grad():
            net.backbone.vdrm.alpha.fill_(.17)
        torch.save(checkpoint, dest)
        _, _, resumed, _ = self.prepare('rdisc')
        self.assertEqual(resumed.backbone.vdrm.alpha.item(), net.backbone.vdrm.alpha.item())


if __name__ == '__main__':
    unittest.main()
