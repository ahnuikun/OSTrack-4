"""Native tracker regression, exact float isolation and safe bundle tests."""

import contextlib
import io
import json
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
from lib.test.analysis.frozen_failure_diagnostics import CONFIGS, dump_json, write_bundle
from lib.test.analysis.native_float_audit import AUDIT_PLAN, check_bundle
from lib.train.frozen_vdrm import file_hash
from tracking import diagnose_vdrm_native_float as entry
from tracking.diagnose_vdrm_frozen_failures import checkpoint_paths, config_for, state_hash


def native_fixture(root, factory=tiny_model):
    dataset = synthetic_fixture(root, factory=factory)
    for sequence in dataset:
        if sequence.name not in AUDIT_PLAN:
            continue
        _, stop = AUDIT_PLAN[sequence.name]
        folder = Path(sequence.frames[0]).parent
        initial_frames = sequence.frames[:]
        for index in range(len(sequence.frames), stop):
            path = folder / f'{index + 1:08d}.jpg'
            shutil.copyfile(initial_frames[index % len(initial_frames)], path)
            sequence.frames.append(str(path))
        sequence.ground_truth_rect = np.tile([60.25, 60.375, 24.5, 24.75], (stop, 1))
        cover, absence = np.full(stop, 8), np.zeros(stop)
        cover[stop - 14:stop - 10] = 0
        np.savetxt(folder / 'cover.label', cover)
        np.savetxt(folder / 'absence.label', absence)
    return dataset


def rewrite_bundle(source, target, mutate):
    with tarfile.open(source, 'r:gz') as archive:
        manifest = json.load(archive.extractfile('manifest.json'))
        payloads = {m.name: archive.extractfile(m).read() for m in archive if m.name != 'manifest.json'}
    mutate(manifest, payloads)
    write_bundle(target, manifest, payloads)


class ProtocolTests(unittest.TestCase):

    def test_assert_exact_detects_changed_visual_route_and_ce_outputs(self):
        reference = {key: torch.zeros(1) for key in entry.REPLAY_KEYS}
        reference['removed_indexes_s'] = [torch.tensor([[3, 4]]), None]
        for key in entry.REPLAY_KEYS:
            replay = entry.clone_output(reference)
            replay[key].fill_(1)
            with self.assertRaisesRegex(RuntimeError, key):
                entry.assert_exact(reference, replay)
        replay = entry.clone_output(reference)
        replay['removed_indexes_s'][0][0, 0] = 2
        with self.assertRaisesRegex(RuntimeError, 'CE removal'):
            entry.assert_exact(reference, replay)
        # Alpha0 retains a DIFFERENT learned route head, yet visual outputs
        # must equal Tclean. Don't incorrectly demand route logits equal.
        replay = entry.clone_output(reference)
        replay['part_route_logits'].fill_(2)
        entry.assert_exact(reference, replay, keys=entry.EXACT_KEYS + ('search_global_index',))

    def test_snapshot_clones_raw_alpha_alias_before_restoration(self):
        parameter = torch.nn.Parameter(torch.tensor(0.))
        cloned = entry.clone_output(dict(vdrm_alpha_raw=parameter, removed_indexes_s=[None]))
        with torch.no_grad():
            parameter.fill_(.35)
        self.assertEqual(cloned['vdrm_alpha_raw'].item(), 0.)


class ConstructorTests(unittest.TestCase):
    def test_adapter_matches_unmodified_native_constructor_for_all_arms(self):
        old_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        try:
            with tempfile.TemporaryDirectory(prefix='vdrm_native_constructor_') as temporary:
                root = Path(temporary)
                dataset = synthetic_fixture(root)
                sequence = dataset[0]
                initial = entry.read_rgb(sequence.frames[0])
                init_bbox = [60.25, 60.375, 24.5, 24.75]
                paths = checkpoint_paths(root)
                with patch.object(entry.native, 'build_ostrack', side_effect=tiny_model), \
                        patch.object(torch.Tensor, 'cuda', lambda tensor, *a, **kw: tensor), \
                        patch.object(torch.nn.Module, 'cuda', lambda module, *a, **kw: module), torch.no_grad():
                    for arm in CONFIGS:
                        cfg = config_for(CONFIGS[arm])
                        params = SimpleNamespace(cfg=cfg, checkpoint=str(paths[arm]), debug=0, save_all_boxes=False,
                            template_factor=cfg.TEST.TEMPLATE_FACTOR, template_size=cfg.TEST.TEMPLATE_SIZE,
                            search_factor=cfg.TEST.SEARCH_FACTOR, search_size=cfg.TEST.SEARCH_SIZE)
                        original = entry.native.OSTrack(params, 'got10k')
                        adapter = entry.NativeTracker(original.network, cfg)
                        original.initialize(initial, dict(init_bbox=init_bbox.copy()))
                        adapter.initialize(initial, dict(init_bbox=init_bbox.copy()))
                        self.assertEqual(original.cfg, adapter.cfg)
                        self.assertTrue(torch.equal(original.output_window, adapter.output_window))
                        self.assertTrue(torch.equal(original.z_dict1.tensors, adapter.z_dict1.tensors))
                        self.assertTrue(torch.equal(original.vdrm_template_bbox, adapter.vdrm_template_bbox))
                        self.assertTrue(torch.equal(original.box_mask_z, adapter.box_mask_z))
                        for frame in sequence.frames[1:4]:
                            image = entry.read_rgb(frame)
                            a, b = original.track(image), adapter.track(image)
                            for key in a:
                                if isinstance(a[key], torch.Tensor):
                                    self.assertTrue(torch.equal(a[key], b[key]), (arm, key))
                                else:
                                    self.assertEqual(a[key], b[key], (arm, key))
                            self.assertEqual(original.state, adapter.state)
                            self.assertEqual(original.frame_id, adapter.frame_id)
        finally:
            torch.set_num_threads(old_threads)


class RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        cls.temporary = tempfile.TemporaryDirectory(prefix='vdrm_native_float_test_')
        cls.root = Path(cls.temporary.name)
        cls.dataset = native_fixture(cls.root)
        cls.output = cls.root / 'native.tar.gz'
        cls.before = {arm: file_hash(p) for arm, p in checkpoint_paths(cls.root).items()}
        # CPU-only tests execute the SAME native Preprocessor.process code;
        # only Tensor.cuda is an identity in this tightly scoped fixture.
        with patch('tracking.diagnose_vdrm_frozen_failures.build_ostrack', side_effect=tiny_model), \
                patch.object(entry, 'get_dataset', return_value=cls.dataset), \
                patch.object(torch.Tensor, 'cuda', lambda tensor, *a, **kw: tensor), \
                contextlib.redirect_stdout(io.StringIO()):
            cls.manifest = entry.run(cls.root, cls.output, 'cpu')

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()
        torch.set_num_threads(cls.old_threads)

    def test_complete_native_audit_read_only_and_no_integer_prediction_access(self):
        self.assertEqual(self.manifest, check_bundle(self.output))
        self.assertEqual((self.manifest['native_tracking_steps'], self.manifest['records'],
                          self.manifest['fork_records']), (516, 102, 102))
        self.assertEqual(self.before, {arm: file_hash(p) for arm, p in checkpoint_paths(self.root).items()})
        self.assertFalse((self.root / 'test').exists())
        with tarfile.open(self.output, 'r:gz') as archive:
            trajectories = json.load(archive.extractfile('trajectories.json'))
            records = json.load(archive.extractfile('records.json'))
            forks = json.load(archive.extractfile('forks.json'))
        for trajectory in trajectories:
            self.assertEqual(trajectory['native']['Rdisc'][1]['anchor_xywh'], [60.25, 60.375, 24.5, 24.75])
        self.assertTrue(all(r['native_replay_exact'] and r['alpha0_exact'] for r in records))
        self.assertTrue(all(r['outputs']['Rdisc_alpha0']['scalars']['vdrm_alpha_raw'] == 0. for r in records))
        self.assertTrue(all(r['outputs']['Rdisc_route0']['correction_abs_max'] == 0. for r in records))
        self.assertTrue(all(r['outputs']['scalars']['vdrm_alpha_raw'] == 0 for r in forks if r['mode'] == 'alpha0'))
        self.assertTrue(all(r['outputs']['correction_abs_max'] == 0 for r in forks if r['mode'] == 'route0'))
        self.assertTrue(all(r['control_exact'] for r in forks if r['mode'] == 'control'))
        for arm in CONFIGS:
            provenance = self.manifest['provenance'][arm]
            self.assertEqual(provenance['state_sha256_before'], provenance['state_sha256_after'])

    def test_refuses_overwrite_before_dataset_access(self):
        with patch.object(entry, 'get_dataset', side_effect=AssertionError('unexpected dataset access')):
            with self.assertRaises(FileExistsError):
                entry.run(self.root, self.output, 'cpu')

    def test_checker_reads_archive_members_once_in_storage_order(self):
        calls, original = [], tarfile.TarFile.extractfile

        def observe(archive, member):
            calls.append(member if isinstance(member, str) else member.name)
            return original(archive, member)

        with patch.object(tarfile.TarFile, 'extractfile', observe):
            check_bundle(self.output)
        with tarfile.open(self.output, 'r:gz') as archive:
            self.assertEqual(calls, [m.name for m in archive.getmembers()])
        self.assertEqual(len(calls), len(set(calls)))

    def test_whole_prefix_missing_frame_aborts_before_model_load(self):
        sequence = next(s for s in self.dataset if s.name.endswith('003240'))
        missing = Path(sequence.frames[24])
        moved = missing.with_suffix('.temporarily_hidden')
        missing.rename(moved)
        try:
            with patch.object(entry, 'get_dataset', return_value=self.dataset), \
                    patch.object(entry, 'load_models', side_effect=AssertionError('unexpected model access')):
                with self.assertRaises(FileNotFoundError):
                    entry.run(self.root, self.root / 'missing.tar.gz', 'cpu')
            self.assertFalse((self.root / 'missing.tar.gz').exists())
        finally:
            moved.rename(missing)

    def test_capture_restored_on_native_failure_and_existing_forward_override(self):
        model = tiny_model(config_for(CONFIGS['Rdisc'])).requires_grad_(False).eval()
        with patch.object(torch.Tensor, 'cuda', lambda tensor, *a, **kw: tensor):
            tracker = entry.NativeTracker(model, config_for(CONFIGS['Rdisc']))
        original_crop = entry.native.sample_target
        before = state_hash(model)
        for had_override in (False, True):
            if had_override:
                model.forward = lambda **kwargs: dict(fake=True)
            original_forward = model.forward
            with self.assertRaisesRegex(RuntimeError, 'intentional'):
                with entry.capture_native(tracker):
                    raise RuntimeError('intentional')
            self.assertEqual(model.forward, original_forward)
            self.assertEqual('forward' in model.__dict__, had_override)
            self.assertIs(entry.native.sample_target, original_crop)
            self.assertEqual(state_hash(model), before)

    def test_native_capture_rejects_current_gt_and_intervention_restores_on_error(self):
        model = tiny_model(config_for(CONFIGS['Rdisc'])).requires_grad_(False).eval()
        with patch.object(torch.Tensor, 'cuda', lambda tensor, *a, **kw: tensor):
            tracker = entry.NativeTracker(model, config_for(CONFIGS['Rdisc']))
        original_crop, before = entry.native.sample_target, state_hash(model)
        for mode in ('alpha0', 'route0'):
            with self.assertRaisesRegex(RuntimeError, 'no current GT'):
                with entry.intervention(model, mode), entry.capture_native(tracker):
                    model.forward(current_gt=torch.ones(4))
            self.assertEqual(state_hash(model), before)
            self.assertNotIn('forward', model.__dict__)
            self.assertFalse(model.backbone.vdrm.route_head._forward_hooks)
            self.assertIs(entry.native.sample_target, original_crop)

    def test_failed_self_replay_leaves_no_bundle_and_restores_hooks(self):
        models, cfg, provenance = None, None, None
        with patch('tracking.diagnose_vdrm_frozen_failures.build_ostrack', side_effect=tiny_model):
            models, cfg, provenance = entry.load_models(checkpoint_paths(self.root), torch.device('cpu'))
        target = self.root / 'failed_replay.tar.gz'
        with patch.object(entry, 'get_dataset', return_value=self.dataset), \
                patch.object(entry, 'load_models', return_value=(models, cfg, provenance)), \
                patch.object(torch.Tensor, 'cuda', lambda tensor, *a, **kw: tensor), \
                patch.object(entry, 'assert_exact', side_effect=RuntimeError('intentional replay failure')), \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, 'intentional replay'):
                entry.run(self.root, target, 'cpu')
        self.assertFalse(target.exists())
        for arm, model in models.items():
            self.assertFalse(model.backbone.vdrm._forward_hooks)
            self.assertNotIn('forward', model.__dict__)
            self.assertEqual(state_hash(model), provenance[arm]['state_sha256_before'])

    def test_bundle_checker_detects_hash_and_float_closure_corruption(self):
        def broken_anchor(manifest, payloads):
            trajectories = json.loads(payloads['trajectories.json'])
            trajectories[0]['native']['Tclean'][1]['anchor_xywh'][0] += .0000001
            payloads['trajectories.json'] = dump_json(trajectories)

        def broken_fork(manifest, payloads):
            forks = json.loads(payloads['forks.json'])
            forks[0]['anchor_xywh'][0] += .0000001
            payloads['forks.json'] = dump_json(forks)

        def broken_inputs(manifest, payloads):
            records = json.loads(payloads['records.json'])
            records[0]['input_sha256']['search'] = 'invalid'
            payloads['records.json'] = dump_json(records)

        for index, (mutate, message) in enumerate(((broken_anchor, 'closed-loop'), (broken_fork, 'fork history'),
                                                  (broken_inputs, 'same-crop'))):
            target = self.root / f'corrupt_{index}.tar.gz'
            rewrite_bundle(self.output, target, mutate)
            with self.assertRaisesRegex(ValueError, message):
                check_bundle(target)
        target = self.root / 'checksum_corrupt.tar.gz'
        with tarfile.open(self.output, 'r:gz') as source, tarfile.open(target, 'w:gz') as dest:
            for member in source:
                data = source.extractfile(member).read()
                if member.name == 'records.json':
                    data += b' '
                member.size = len(data)
                dest.addfile(member, io.BytesIO(data))
        with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
            check_bundle(target)


if __name__ == '__main__':
    unittest.main()
