import contextlib
import io
import json
import re
import shlex
from pathlib import Path
import sys
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / 'tests'))

from test_vdrm_frozen import tiny_model
from lib.models.ostrack import build_ostrack
from lib.train.frozen_vdrm import load_base_config, prepare_frozen_models, file_hash
from lib.test.analysis.frozen_failure_diagnostics import (
    CONFIGS, PLAN, check_bundle, checked_predictions, make_window_plan, overlapping_fraction, write_bundle)
from tracking.diagnose_vdrm_frozen_failures import (
    checkpoint_paths, config_for, crop_geometry, intervention, load_models, prepare_cases, run, state_hash)
from tracking import diagnose_vdrm_frozen_failures as entry


def synthetic_fixture(root, factory=tiny_model):
    """Temporary developer fixture: no user weights, no optimizer or training."""
    root = Path(root)
    torch.manual_seed(42)
    model = factory(load_base_config(), training=False)
    source = root / 'checkpoints/train/ostrack' / CONFIGS['Tclean'] / 'OSTrack_ep0300.pth.tar'
    source.parent.mkdir(parents=True)
    torch.save(dict(net=model.state_dict(), net_type='OSTrack', epoch=300,
                    settings=SimpleNamespace(config_name=CONFIGS['Tclean'])), source)
    del model
    with patch('lib.models.ostrack.build_ostrack', side_effect=factory):
        for arm in ('Rfreeze', 'Rdisc'):
            cfg = config_for(CONFIGS[arm])
            model, baseline, _ = prepare_frozen_models(cfg, root, CONFIGS[arm])
            with torch.no_grad():
                model.backbone.vdrm.alpha.fill_(.25 if arm == 'Rfreeze' else .35)
                if arm == 'Rdisc':
                    model.backbone.vdrm.route_head.classifier[-1].bias.fill_(.6)
            target = root / 'checkpoints/train/ostrack' / CONFIGS[arm] / 'OSTrack_ep0300.pth.tar'
            target.parent.mkdir(parents=True)
            torch.save(dict(net=model.state_dict(), net_type='OSTrack', epoch=300,
                            frozen_vdrm_contract=model.frozen_vdrm_contract), target)
            del model, baseline
    dataset = []
    for name in PLAN:
        folder = root / 'data/got10k/train' / name
        folder.mkdir(parents=True)
        gt = np.tile([60., 60., 24., 24.], (14, 1))
        cover, absence = np.full(14, 8), np.zeros(14)
        cover[3:6] = 0
        np.savetxt(folder / 'cover.label', cover)
        np.savetxt(folder / 'absence.label', absence)
        frames = []
        rng = np.random.default_rng(42)
        for i in range(14):
            path = folder / f'{i + 1:08d}.jpg'
            cv2.imwrite(str(path), rng.integers(0, 256, (160, 160, 3), dtype=np.uint8))
            frames.append(str(path))
        for arm, config in CONFIGS.items():
            pred = gt.copy()
            pred[1:, 0] += dict(Tclean=0, Rfreeze=4, Rdisc=7)[arm]
            target = root / 'results' / config / 'got10k' / (name + '.txt')
            target.parent.mkdir(parents=True, exist_ok=True)
            np.savetxt(target, pred, fmt='%d')
        dataset.append(SimpleNamespace(name=name, dataset='got10k', frames=frames, ground_truth_rect=gt))
    dataset += [SimpleNamespace(name='unselected_' + str(i)) for i in range(147)]
    return dataset


class PlanningTests(unittest.TestCase):

    def test_coverage_and_actual_border_support(self):
        self.assertEqual(overlapping_fraction([0, 0, 10, 10], [5, 0, 10, 10]), .5)
        self.assertIsNone(overlapping_fraction([0, 0, 0, 10], [5, 0, 10, 10]))
        self.assertEqual(crop_geometry([8, 8, 2, 2], 1, (10, 10, 3), [8, 8, 2, 2])
                         ['gt_fraction_in_visual_support'], .25)

    def test_native_integer_init_and_strict_result_lengths(self):
        gt = np.tile([4.2, 5.8, 20.9, 20.1], (6, 1))
        self.assertEqual(checked_predictions(np.trunc(gt), gt, 'fixture').shape, (6, 4))
        for prediction in (gt[:-1], np.full_like(gt, np.inf), gt * 0, gt + 4):
            with self.assertRaises(ValueError):
                checked_predictions(prediction, gt, 'fixture')

    def test_reappearance_selection_and_one_based_clipping(self):
        gt = np.tile([0., 0., 20., 20.], (20, 1))
        predictions = {a: gt.copy() for a in CONFIGS}
        predictions['Rdisc'][8:, 0] = 40
        cover = np.full(20, 8)
        cover[4:8] = 0
        plan = make_window_plan('GOT-10k_Train_008333', gt, cover, np.zeros(20), predictions)
        self.assertEqual(plan['selected_frames_1based'], list(range(2, 20)))
        self.assertEqual(plan['recovery_events'][0]['reappearance_1based'], 9)
        self.assertIsNone(plan['recovery_events'][0]['delay_frames']['Rdisc'])
        self.assertEqual(plan['recovery_events'][0]['delay_frames']['Tclean'], 0)

    def test_invalid_labels_rejected(self):
        gt = np.tile([0., 0., 20., 20.], (20, 1))
        for cover, absence in ((np.full(20, 9), np.zeros(20)), (np.full(20, 8), np.zeros(19))):
            with self.assertRaises(ValueError):
                make_window_plan('GOT-10k_Train_008333', gt, cover, absence, {a: gt.copy() for a in CONFIGS})


class RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        cls.temporary = tempfile.TemporaryDirectory(prefix='frozen_failure_test_')
        cls.root = Path(cls.temporary.name)
        cls.dataset = synthetic_fixture(cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()
        torch.set_num_threads(cls.old_threads)

    def test_preflight_nested_legacy_and_conflicting_results(self):
        cases = prepare_cases(self.root / 'results', self.dataset)
        self.assertEqual(len(cases), 5)
        sequence = self.dataset[0]
        parent = self.root / 'results' / CONFIGS['Tclean']
        nested = parent / 'got10k' / (sequence.name + '.txt')
        flat = parent / nested.name
        nested.rename(flat)
        try:
            self.assertEqual(len(prepare_cases(self.root / 'results', self.dataset)), 5)
            nested.write_bytes(flat.read_bytes() + b'1 2 3 4\n')
            with self.assertRaisesRegex(ValueError, 'Conflicting'):
                prepare_cases(self.root / 'results', self.dataset)
        finally:
            if nested.exists():
                nested.unlink()
            flat.rename(nested)

    def test_missing_images_abort_in_preflight(self):
        frame = Path(self.dataset[0].frames[1])
        moved = frame.with_suffix('.temporarily_hidden')
        frame.rename(moved)
        try:
            with self.assertRaises(FileNotFoundError):
                prepare_cases(self.root / 'results', self.dataset)
        finally:
            moved.rename(frame)

    def test_counterfactual_restoration_even_on_error(self):
        model = tiny_model(config_for(CONFIGS['Rdisc'])).eval()
        with torch.no_grad():
            model.backbone.vdrm.alpha.fill_(.37)
        before = state_hash(model)
        for mode in ('alpha0', 'route0'):
            with self.assertRaisesRegex(RuntimeError, 'intentional'):
                with intervention(model, mode):
                    raise RuntimeError('intentional')
            self.assertEqual(state_hash(model), before)
        self.assertFalse(model.backbone.vdrm.route_head._forward_hooks)

    def test_forged_source_checkpoint_rejected(self):
        paths = checkpoint_paths(self.root)
        path = paths['Rdisc']
        original = path.read_bytes()
        checkpoint = torch.load(path, weights_only=False, map_location='cpu')
        checkpoint['frozen_vdrm_contract']['source_sha256'] = 'invalid'
        torch.save(checkpoint, path)
        try:
            with patch('tracking.diagnose_vdrm_frozen_failures.build_ostrack', side_effect=tiny_model):
                with self.assertRaisesRegex(ValueError, 'exact Tclean'):
                    load_models(paths, torch.device('cpu'))
        finally:
            path.write_bytes(original)

    def test_complete_bundle_read_only_and_checksum_failure(self):
        paths = checkpoint_paths(self.root)
        before = {a: file_hash(p) for a, p in paths.items()}
        output = self.root / 'complete.tar.gz'
        with patch('tracking.diagnose_vdrm_frozen_failures.get_dataset', return_value=self.dataset), \
                patch('tracking.diagnose_vdrm_frozen_failures.build_ostrack', side_effect=tiny_model), \
                contextlib.redirect_stdout(io.StringIO()):
            manifest = run(self.root, self.root / 'results', output, 'cpu')
        self.assertEqual(manifest['records'], manifest['selected_frames'] * 3)
        self.assertEqual(before, {a: file_hash(p) for a, p in paths.items()})
        with tarfile.open(output, 'r:gz') as archive:
            records = json.load(archive.extractfile('records.json'))
            payloads = {m.name: archive.extractfile(m).read() for m in archive if m.name != 'manifest.json'}
        self.assertTrue(all(r['alpha0_exact'] for r in records))
        self.assertTrue(all(r['outputs']['Rdisc_alpha0']['scalars']['vdrm_alpha_raw'] == 0 for r in records))
        self.assertTrue(all(r['outputs']['Rdisc']['scalars']['vdrm_alpha_raw'] > 0 for r in records))
        self.assertTrue(all(r['outputs']['Rdisc_route0']['correction_abs_max'] == 0 for r in records))
        self.assertTrue(all(r['outputs']['Rdisc']['correction_abs_max'] > .5 for r in records))
        self.assertTrue(all(r['outputs']['Rdisc_alpha0']['actual_injected_residual']['residual_relative_norm_max'] == 0
                            for r in records))
        self.assertTrue(all(r['outputs']['Rdisc']['final_ce_retained_tokens'] < r['outputs']['Rdisc']['ce_retained_tokens']
                            for r in records))
        with self.assertRaises(FileExistsError):
            write_bundle(output, manifest, payloads)
        corrupt = self.root / 'corrupt.tar.gz'
        with tarfile.open(corrupt, 'w:gz') as archive:
            with tarfile.open(output, 'r:gz') as original:
                for member in original:
                    data = original.extractfile(member).read()
                    if member.name == 'records.json':
                        data += b' '
                    member.size = len(data)
                    archive.addfile(member, io.BytesIO(data))
        with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
            check_bundle(corrupt)


if __name__ == '__main__':
    unittest.main()
