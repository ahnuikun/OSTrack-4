"""Native identity: no-GT selection, no feedback, exact decode and complete coverage."""
import contextlib
import inspect
import io
import json
from pathlib import Path
import re
import shlex
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT)); sys.path.insert(0, str(PROJECT/'tests'))
from lib.train.identity_offline import ARMS, load_protocol
from lib.train.frozen_vdrm import BASE_CONFIG, file_hash, load_base_config
from lib.test.analysis.identity_native import (candidate_pools, choose_candidates, extract_cells,
    sequence_evidence, summarize, check_bundle, load_screen_heads)
from lib.test.analysis.candidate_cues import npz_bytes, read_npz
from lib.test.evaluation.got10kdataset import _vdrm_dev_sequence_ids
from tracking import probe_vdrm_identity_native as entry, analyze_vdrm_identity_native as analyzer
from tracking.cache_vdrm_identity import collect_pair
from tracking.train_vdrm_identity import run as train_fixture
from tracking.review_vdrm_identity_screen import review
from test_vdrm_identity_offline import synthetic_cache
from test_vdrm_native_float import rewrite_bundle
from test_vdrm_frozen import tiny_model


def synthetic_outputs(absent_sentinel=False):
    n = 4
    box = np.array([10., 20., 30., 40.])
    boxes = np.tile(box, (n, 256, 1))
    boxes[:, 1:, 0] += 100
    base = np.zeros((n, 256), dtype=np.float32)
    base[:, 0] = 1
    base[1, 1] = 2  # One recoverable native error.
    native = boxes[np.arange(n), base.argmax(1)]
    arrays = dict(boxes=boxes, base_scores=base, common_cells=np.full((n, 256), 9, dtype=np.int16),
        gt=np.tile(box, (n+1, 1)), cover=np.array([8, 8, 2, 0, 8.]), absence=np.zeros(n+1),
        native_boxes=native, anchors=np.concatenate((box[None], native[:-1])))
    for arm in ARMS:
        score = np.zeros_like(base); score[:, 0] = .9
        arrays[arm+'_scores'] = score
    if absent_sentinel:
        arrays['gt'][3] = -1
        arrays['absence'][3] = 1
    return arrays


def native_fixture(root, factory=tiny_model):
    """Real fixed IDs, synthetic images/labels; only fixture head training."""
    torch.manual_seed(42)
    model = factory(load_base_config(), training=False)
    with torch.no_grad(): model.backbone.vdrm.alpha.zero_()
    source = root/'checkpoints/train/ostrack'/BASE_CONFIG/'OSTrack_ep0300.pth.tar'
    source.parent.mkdir(parents=True)
    torch.save(dict(net=model.state_dict(), net_type='OSTrack', epoch=300,
                    settings={'config_name': BASE_CONFIG}), source)
    dimension = model.backbone.embed_dim
    del model
    index, protocol = synthetic_cache(root/'cache')
    if dimension != 32:
        with np.load(root/'cache/pairs.npz') as archive:
            arrays = {k: archive[k] for k in archive.files}
        for key in ('template', 'candidates'):
            arrays[key] = np.tile(arrays[key], (1,)*(arrays[key].ndim-1)+(dimension//32,))
        np.savez_compressed(root/'cache/pairs.npz', **arrays)
        index['feature_dim'] = dimension
        index['members_sha256']['pairs.npz'] = file_hash(root/'cache/pairs.npz')
    index['checkpoint_sha256'] = file_hash(source)
    (root/'cache/index.json').write_text(json.dumps(index), encoding='utf-8')
    train_fixture(root/'cache', root/'screen', 'cpu')
    data = root/'data/got10k/train'; data.mkdir(parents=True)
    ids = _vdrm_dev_sequence_ids()
    names = [f'GOT-10k_Train_{i+1:06d}' for i in range(max(ids)+1)]
    (data/'list.txt').write_text('\n'.join(names), encoding='utf-8')
    rng = np.random.default_rng(42)
    images = [rng.integers(0, 256, (160, 160, 3), dtype=np.uint8) for _ in range(3)]
    for identity in ids:
        folder = data/names[identity]; folder.mkdir()
        gt = np.tile([60.25, 60.375, 24.5, 24.75], (3, 1))
        gt[-1] = -1  # Real absent-target sentinel, retained through packaging.
        np.savetxt(folder/'groundtruth.txt', gt, delimiter=',')
        np.savetxt(folder/'cover.label', [8, 2, 0]); np.savetxt(folder/'absence.label', [0, 0, 1])
        for i, image in enumerate(images, 1): cv2.imwrite(str(folder/f'{i:08d}.jpg'), image)
    return root/'screen/identity_offline_results.tar.gz', data, source


class MathTests(unittest.TestCase):
    def test_no_gt_selection_signature(self):
        self.assertEqual(list(inspect.signature(candidate_pools).parameters), ['base', 'common'])
        self.assertEqual(list(inspect.signature(choose_candidates).parameters), ['base', 'scores', 'eligible'])

    def test_pool_does_not_inject_positive_or_resort_after_support_filter(self):
        base = np.arange(256, dtype=float)[None]
        common = np.full((1, 256), 9); common[0, 255] = 0
        pools = candidate_pools(base, common)
        self.assertEqual(np.flatnonzero(pools['top5'][0]).tolist(), [251, 252, 253, 254])
        self.assertFalse(pools['top5'][0, 0]); self.assertTrue(pools['all256'][0, 0])

    def test_ties_and_no_support_preserve_native_when_possible(self):
        base = np.array([[0., 2., 1.], [3., 2., 1.]])
        scores = np.ones_like(base); valid = np.ones_like(base, dtype=bool)
        chosen, fallback, tied = choose_candidates(base, scores, valid)
        np.testing.assert_array_equal(chosen, [1, 0]); self.assertTrue(tied.all())
        valid[:] = False
        chosen, fallback, tied = choose_candidates(base, scores, valid)
        np.testing.assert_array_equal(chosen, [1, 0]); self.assertTrue(fallback.all()); self.assertFalse(tied.any())

    def test_complete_counts_and_invisible_exclusion(self):
        record = sequence_evidence('fixture', synthetic_outputs())
        self.assertEqual(record['counts']['frames'], 4)
        self.assertEqual(record['counts']['visible'], 3)
        self.assertEqual(record['pools']['top5']['recoverable_hard'], 1)
        for arm in ARMS:
            self.assertEqual(record['pools']['top5']['heads'][arm]['repairs'], [3])
            self.assertEqual(record['pools']['top5']['heads'][arm]['damages'], [])
        report = summarize([record])
        self.assertFalse(report['pools']['top5']['hard_evidence_sufficient'])
        self.assertFalse(report['automatic_integration_approved'])

    def test_broken_float_history_and_decode_are_rejected(self):
        arrays = synthetic_outputs(); arrays['anchors'][2, 0] += 1
        with self.assertRaisesRegex(ValueError, 'trajectory'): sequence_evidence('fixture', arrays)
        arrays = synthetic_outputs(); arrays['boxes'][0, 0, 0] += 1
        with self.assertRaisesRegex(ValueError, 'decode'): sequence_evidence('fixture', arrays)

    def test_raw_roi_features_match_training_cache(self):
        torch.manual_seed(42)
        out = dict(backbone_feat=torch.randn(1, 320, 32), size_map=torch.full((1, 2, 16, 16), .25),
            offset_map=torch.full((1, 2, 16, 16), .5), score_map=torch.ones(1, 1, 16, 16), removed_indexes_s=[])
        zbox = torch.tensor([[.25, .25, .5, .5]]); target = torch.tensor([[.375, .375, .25, .25]])
        zp = torch.zeros(1, 128, 128, dtype=torch.bool); xp = torch.zeros(1, 256, 256, dtype=torch.bool)
        pair, _ = collect_pair(out, zbox, target, zp, xp, load_protocol())
        boxes, z, x, zm, xm = extract_cells(out, zbox, zp, xp)
        selected = pair[1]['candidate_indices']
        np.testing.assert_array_equal(z[0].numpy(), pair[0]['template'])
        np.testing.assert_array_equal(x[0, selected].numpy(), pair[0]['candidates'])
        np.testing.assert_array_equal(xm[0, selected].numpy(), pair[0]['candidate_mask'])

    def test_existing_output_and_invalid_dataset_fail_before_gpu(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(entry, 'load_models', side_effect=AssertionError('GPU access')):
                with self.assertRaises(FileExistsError): entry.run('missing', 'missing', 'missing', root)
                with self.assertRaises(FileNotFoundError): entry.run('missing', 'missing', 'missing', root/'new')
            (root/'list.txt').write_text('GOT-10k_Val_000001\n')
            with self.assertRaisesRegex(ValueError, 'ordered'): entry.plan_sequences(root)
            (root/'list.txt').write_text('GOT-10k_Train_000002\nGOT-10k_Train_000001\n')
            with self.assertRaisesRegex(ValueError, 'ordered'): entry.plan_sequences(root)

    def test_documented_commands_parse_and_analysis_never_loads_gpu(self):
        document = (PROJECT/'VDRM_IDENTITY_NATIVE_PLAN.md').read_text(encoding='utf-8')
        blocks = re.findall(r'```(?:lua|css)\n(.*?)```', document, re.S)
        self.assertEqual(len(blocks), 2)
        for block in blocks:
            tokens = shlex.split(block.replace('\\\n', ' ')); i = tokens.index('python')
            args = tokens[i+1:]; self.assertNotIn('--seed', args)
            with patch.object(sys, 'argv', args):
                if 'probe_' in args[0]:
                    with patch.object(torch.cuda, 'is_available', return_value=True), patch.object(entry, 'run') as run:
                        entry.main()
                    self.assertEqual(run.call_args.args, ('output/analysis/identity_offline_v1/screen/identity_offline_results.tar.gz',
                        './data/got10k/train', './output', 'output/analysis/identity_native_v1'))
                else:
                    with patch.object(analyzer, 'check_bundle', return_value=({}, {'sequences': 152})), \
                         patch.object(analyzer, 'render_report', return_value='fixture'), \
                         patch.object(torch.cuda, 'is_available', side_effect=AssertionError('GPU access')), \
                         contextlib.redirect_stdout(io.StringIO()): analyzer.main()


class NativeRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads(); torch.set_num_threads(2)
        cls.temporary = tempfile.TemporaryDirectory(prefix='identity_native_test_')
        cls.root = Path(cls.temporary.name)
        with contextlib.redirect_stdout(io.StringIO()):
            cls.screen, cls.data, cls.source = native_fixture(cls.root)
            cls.before = [file_hash(p) for p in (cls.screen, cls.source)]
            with patch('tracking.diagnose_vdrm_frozen_failures.build_ostrack', side_effect=tiny_model), \
                 patch.object(torch.Tensor, 'cuda', lambda v, *a, **kw: v), \
                 patch.object(torch.optim, 'AdamW', side_effect=AssertionError('optimizer in read-only probe')):
                cls.summary = entry.run(cls.screen, cls.data, cls.root, cls.root/'native', 'cpu')
        cls.bundle = cls.root/'native/identity_native_results.tar.gz'

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup(); torch.set_num_threads(cls.old_threads)

    def test_full152_native_path_readonly_and_cpu_recheck(self):
        manifest, summary = check_bundle(self.bundle)
        self.assertEqual(summary, self.summary)
        self.assertEqual(summary['sequences'], 152)
        self.assertEqual(summary['counts']['frames'], 304)
        self.assertEqual(summary['counts']['visible'], 152)
        self.assertEqual(manifest['states_before'], manifest['states_after'])
        self.assertEqual(self.before, [file_hash(p) for p in (self.screen, self.source)])
        self.assertFalse((self.root/'tracking_results').exists())

    def test_wrong_source_checkpoint_rejected_before_model_load(self):
        other = self.root/'other/checkpoints/train/ostrack'/BASE_CONFIG
        other.mkdir(parents=True); (other/'OSTrack_ep0300.pth.tar').write_bytes(b'wrong source')
        with patch.object(entry, 'load_models', side_effect=AssertionError('GPU loaded')):
            with self.assertRaisesRegex(ValueError, 'source'):
                entry.run(self.screen, self.data, self.root/'other', self.root/'bad')

    def test_new_labels_cannot_change_native_or_head_outputs(self):
        plan = entry.plan_sequences(self.data)[0]
        alternate = dict(plan, gt=plan['gt'].copy(), cover=np.full(3, 8), absence=np.zeros(3))
        alternate['gt'][1:] = plan['gt'][0] + [80, 80, 0, 0]
        with patch('tracking.diagnose_vdrm_frozen_failures.build_ostrack', side_effect=tiny_model), \
             patch.object(torch.Tensor, 'cuda', lambda v, *a, **kw: v):
            models, cfg, _ = entry.load_models({'Tclean': self.source}, torch.device('cpu'), arms=('Tclean',))
            heads, _ = load_screen_heads(self.screen)
            a, _ = entry.collect_sequence(plan, models['Tclean'], heads, cfg)
            b, _ = entry.collect_sequence(alternate, models['Tclean'], heads, cfg)
        for key in set(a)-{'gt', 'cover', 'absence'}: np.testing.assert_array_equal(a[key], b[key])

    def test_rehashed_state_and_trajectory_tampering_rejected(self):
        path = self.root/'bad_state.tar.gz'
        def mutate(manifest, payloads): manifest['states_after']['mean_id'] = 'changed'
        rewrite_bundle(self.bundle, path, mutate)
        with self.assertRaisesRegex(ValueError, 'provenance'): check_bundle(path)
        def mutate_trajectory(manifest, payloads):
            member = manifest['sequences'][0]['member']
            arrays = read_npz(payloads[member]); arrays['anchors'][1, 0] += .001
            payloads[member] = npz_bytes(arrays)
        path = self.root/'bad_trajectory.tar.gz'; rewrite_bundle(self.bundle, path, mutate_trajectory)
        with self.assertRaisesRegex(ValueError, 'trajectory'): check_bundle(path)

    def test_saved_weights_and_raw_cache_replay(self):
        # Fixture lacks per-sequence sampling metadata, so test weights separately.
        report = review(self.screen, screen_dir=self.root/'screen')
        self.assertTrue(report['saved_checkpoints_checked'])
        self.assertEqual(set(report['flips']), set(ARMS))


if __name__ == '__main__': unittest.main()
