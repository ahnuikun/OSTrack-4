"""No-GT crop selection, source closure, padding, native isolation and bundles."""
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

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT)); sys.path.insert(0, str(PROJECT/'tests'))
from lib.test.analysis.candidate_cues import read_npz, npz_bytes
from lib.test.analysis.search_coverage import (PROTOCOL, GROUPS, selected_ids, crop_specs,
    gt_crop_fraction, choose_group, sequence_evidence, summarize, check_bundle)
from lib.test.analysis.identity_guard import partition
from lib.train.frozen_vdrm import file_hash, BASE_CONFIG
from tracking import probe_vdrm_search_coverage as entry, analyze_vdrm_search_coverage as analyzer
from tracking import screen_vdrm_identity_guard as guard
from tracking.diagnose_vdrm_frozen_failures import load_models
from test_vdrm_identity_native import synthetic_outputs
from test_vdrm_identity_guard import guard_fixture
from test_vdrm_native_float import rewrite_bundle
from test_vdrm_frozen import tiny_model


def synthetic_coverage():
    source = synthetic_outputs(absent_sentinel=True)
    # A true no-good-candidate source error, rather than a ranking error.
    source['boxes'][0, :, 0] = 110
    source['native_boxes'] = source['boxes'][np.arange(4), source['base_scores'].argmax(1)]
    source['anchors'] = np.concatenate((source['gt'][:1], source['native_boxes'][:-1]))
    indices = np.arange(0, len(source['native_boxes']), PROTOCOL['sampling_stride'])
    n = len(indices)
    centers, squares = [], []
    for anchor in source['anchors'][indices]:
        c, _, s = crop_specs(anchor); centers.append(c); squares.append(s)
    values = dict(native_boxes=source['native_boxes'][indices].copy(), anchors=source['anchors'][indices].copy(),
        boxes=np.tile(source['boxes'][indices, None], (1, 6, 1, 1)),
        scores=np.tile(source['base_scores'][indices, None], (1, 6, 1)), supported=np.ones((n, 6, 256), dtype=bool),
        crop_anchors=np.stack(centers), crop_squares=np.stack(squares), image_hw=np.full((n, 2), 512),
        frame_indices=indices, gt=source['gt'].copy(), cover=source['cover'].copy(), absence=source['absence'].copy(),
        source_boxes=source['boxes'].copy(), source_scores=source['base_scores'].copy(),
        source_native_boxes=source['native_boxes'].copy(), source_anchors=source['anchors'].copy())
    values['boxes'][0, 1, 0] = source['gt'][1]
    values['scores'][0, 1, 0] = 3  # Below tests normalize to valid sigmoid/Hann range.
    values['scores'] /= 4
    values['source_scores'] /= 4
    return values


class CoverageMathTests(unittest.TestCase):
    def test_subset_uses_only_fixed_fit_and_does_not_sample_by_gt(self):
        train = list(range(4)); ids = selected_ids(train); parts = partition(train)
        self.assertEqual(len(ids), 128); self.assertEqual(ids, selected_ids(train))
        self.assertTrue(set(ids) <= set(parts['fit'])); self.assertFalse(set(ids) & set(parts['audit']))
        self.assertEqual(list(inspect.signature(crop_specs).parameters), ['anchor'])
        self.assertEqual(list(inspect.signature(choose_group).parameters), ['scores','supported','group'])

    def test_tiles_keep_scale_and_gt_fraction_accounts_for_image_padding(self):
        centers, factors, squares = crop_specs([20., 20., 10., 10.])
        self.assertEqual(factors.tolist(), [4, 6, 4, 4, 4, 4])
        self.assertEqual(squares[:, 2].tolist(), [40, 60, 40, 40, 40, 40])
        self.assertEqual(centers[2, 0], 0)
        fraction = gt_crop_fraction(np.array([[[0.,0.,100.,100.]]]), np.array([[50,50]]), np.array([[0.,0.,100.,100.]]))
        self.assertAlmostEqual(fraction[0,0], 49*49/10000)
        self.assertFalse(gt_crop_fraction(np.array([[[0.,0.,100.,100.]]]), np.array([[50,50]]), np.array([[-1.]*4])).any())

    def test_ties_keep_native_and_padding_or_removed_cells_cannot_take_over(self):
        x = synthetic_coverage(); x['scores'][0, 1] = x['scores'][0, 0]
        crop, cell = choose_group(x['scores'], x['supported'], GROUPS['wide6'])
        self.assertEqual(crop[0], 0)
        x = synthetic_coverage(); x['supported'][0, 1, 0] = False
        crop, cell = choose_group(x['scores'], x['supported'], GROUPS['wide6'])
        self.assertEqual(crop[0], 0); self.assertEqual(cell[0], 0)

    def test_opportunity_is_distinct_from_tracking_or_residual_approval(self):
        record = sequence_evidence('fixture', synthetic_coverage())
        self.assertEqual(record['counts']['frames'], 1); self.assertEqual(record['counts']['visible'], 1)
        self.assertEqual(record['counts']['original_history_steps'], 4)
        self.assertEqual(record['counts']['baseline_no_candidate'], 1)
        self.assertEqual(record['groups']['wide6']['added_candidate_frames_1based'], [2])
        self.assertEqual(record['groups']['wide6']['added_inside90_source'], 1)
        self.assertEqual(record['groups']['tiles4']['added_candidate_frames'], 0)
        report = summarize([record]*32)
        self.assertTrue(report['groups']['wide6']['candidate_opportunity_sufficient'])
        for flag in ('deployment_approved','closed_loop_tracking_passed','residual_effectiveness_passed','reliability_effectiveness_passed'):
            self.assertFalse(report[flag])

    def test_source_history_centers_scores_and_sentinels_remain_strict(self):
        for key, message in (('source_native_boxes','source native'), ('anchors','source native'),
                             ('source_boxes','center output'), ('crop_anchors','centers/scales')):
            x = synthetic_coverage(); x[key].flat[0] += 1
            with self.assertRaisesRegex(ValueError, message): sequence_evidence('fixture', x)
        x = synthetic_coverage(); x['absence'][3] = 0
        with self.assertRaisesRegex(ValueError, 'negative box'): sequence_evidence('fixture', x)

    def test_unsampled_history_and_registered_frame_positions_are_checked(self):
        x = synthetic_coverage(); x['source_anchors'][1, 0] += .1
        with self.assertRaisesRegex(ValueError, 'source float history'): sequence_evidence('fixture', x)
        x = synthetic_coverage(); x['frame_indices'][0] = 1
        with self.assertRaisesRegex(ValueError, 'sampling'): sequence_evidence('fixture', x)
        x = synthetic_coverage(); x['source_scores'][1, 0] = -1
        with self.assertRaisesRegex(ValueError, 'support/response'): sequence_evidence('fixture', x)

    def test_documented_commands_parse_without_seed_or_old_txt(self):
        document = (PROJECT/'VDRM_SEARCH_COVERAGE_PLAN.md').read_text(encoding='utf-8')
        commands = re.findall(r'```(?:lua|css)\n(.*?)```', document, re.S)
        self.assertEqual(len(commands), 2)
        for command in commands:
            tokens = shlex.split(command.replace('\\\n',' ')); args = tokens[tokens.index('python')+1:]
            self.assertNotIn('--seed', args)
            with patch.object(sys,'argv',args):
                if 'probe_' in args[0]:
                    with patch.object(torch.cuda,'is_available',return_value=True), patch.object(entry,'run') as run: entry.main()
                    self.assertEqual(run.call_args.args,('output/analysis/identity_guard_v1/identity_guard_results.tar.gz',
                        './data/got10k/train','./output','output/analysis/search_coverage_v1'))
                else:
                    with patch.object(analyzer,'check_bundle',return_value=({'sequences':[]},{})), \
                         patch.object(analyzer,'render_report',return_value='fixture'), \
                         patch.object(torch.cuda,'is_available',side_effect=AssertionError('GPU access')), \
                         contextlib.redirect_stdout(io.StringIO()): analyzer.main()


class CoverageRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads(); torch.set_num_threads(2)
        cls.temp = tempfile.TemporaryDirectory(prefix='coverage_test_'); cls.root = Path(cls.temp.name)
        with contextlib.redirect_stdout(io.StringIO()):
            cls.screen, cls.data, cls.source, cls.parts = guard_fixture(cls.root)
            with patch('tracking.diagnose_vdrm_frozen_failures.build_ostrack',side_effect=tiny_model), \
                 patch.object(torch.Tensor,'cuda',lambda x,*a,**kw:x):
                result = guard.run(cls.screen,cls.data,cls.root,cls.root/'guard','cpu')
                if result['next_action'] != 'stop_existing_identity_reranking_path':
                    raise RuntimeError('synthetic source must exercise a failed guard screen')
        cls.guard_bundle = cls.root/'guard/identity_guard_results.tar.gz'
        cls.before = [file_hash(p) for p in (cls.screen,cls.source,cls.guard_bundle)]
        with patch('tracking.diagnose_vdrm_frozen_failures.build_ostrack',side_effect=tiny_model), \
             patch.object(torch.Tensor,'cuda',lambda x,*a,**kw:x), \
             patch.object(torch.optim,'AdamW',side_effect=AssertionError('optimizer')), \
             contextlib.redirect_stdout(io.StringIO()):
            cls.summary = entry.run(cls.guard_bundle,cls.data,cls.root,cls.root/'coverage','cpu')
        cls.bundle = cls.root/'coverage/search_coverage_results.tar.gz'

    @classmethod
    def tearDownClass(cls): cls.temp.cleanup(); torch.set_num_threads(cls.threads)

    def test_complete128_uniform_sampling_full_history_readonly_and_cpu_recomputed(self):
        manifest, result = check_bundle(self.bundle)
        self.assertEqual(result,self.summary); self.assertEqual(len(manifest['sequences']),128)
        self.assertEqual(manifest['source_history_steps'],512); self.assertEqual(manifest['replayed_native_steps'],128)
        self.assertEqual(manifest['extra_crop_forwards'],640)
        self.assertEqual(manifest['state_before'],manifest['state_after'])
        self.assertEqual(self.before,[file_hash(p) for p in (self.screen,self.source,self.guard_bundle)])
        with tarfile.open(self.bundle,'r:gz') as archive:
            for spec in manifest['sequences']:
                arrays=read_npz(archive.extractfile(spec['member']).read())
                np.testing.assert_array_equal(arrays['gt'][3],[-1]*4)
        self.assertFalse((self.root/'tracking_results').exists())

    def test_current_labels_cannot_change_any_crop_prediction(self):
        with tarfile.open(self.bundle,'r:gz') as archive:
            manifest=json.load(archive.extractfile('manifest.json')); spec=manifest['sequences'][0]
        plans=entry.plan_sequences(self.data,dict(fit=[spec['sequence_id']],audit=[])); plan=plans[0]
        with tarfile.open(self.guard_bundle,'r:gz') as archive:
            source=json.load(archive.extractfile('manifest.json'))
            source_spec=next(s for s in source['sequences'] if s['name']==plan['name'])
            plan['expected']=read_npz(archive.extractfile(source_spec['member']).read())
        alternative=dict(plan,gt=plan['gt'].copy(),cover=np.full(5,8),absence=np.zeros(5))
        alternative['gt'][1:]=plan['gt'][0]+[80,80,0,0]
        with patch('tracking.diagnose_vdrm_frozen_failures.build_ostrack',side_effect=tiny_model), \
             patch.object(torch.Tensor,'cuda',lambda x,*a,**kw:x):
            models,cfg,_=load_models({'Tclean':self.source},torch.device('cpu'),arms=('Tclean',))
            a,_=entry.collect_sequence(plan,models['Tclean'],cfg)
            b,_=entry.collect_sequence(alternative,models['Tclean'],cfg)
        for key in set(a)-{'gt','cover','absence'}: np.testing.assert_array_equal(a[key],b[key])

    def test_existing_outputs_and_wrong_checkpoint_fail_before_gpu(self):
        with patch.object(entry,'load_models',side_effect=AssertionError('GPU access')):
            with self.assertRaises(FileExistsError): entry.run('missing','missing','missing',self.root/'coverage')
            other=self.root/'other/checkpoints/train/ostrack'/BASE_CONFIG; other.mkdir(parents=True)
            (other/'OSTrack_ep0300.pth.tar').write_bytes(b'wrong')
            with self.assertRaisesRegex(ValueError,'checkpoint/data list'):
                entry.run(self.guard_bundle,self.data,self.root/'other',self.root/'wrong_source','cpu')
        self.assertFalse((self.root/'wrong_source').exists())

    def test_rehashed_state_and_geometry_tampering_rejected(self):
        path=self.root/'bad_state.tar.gz'
        def mutate(m,p):m['state_after']='changed'
        rewrite_bundle(self.bundle,path,mutate)
        with self.assertRaisesRegex(ValueError,'state contract'):check_bundle(path)
        path=self.root/'bad_geometry.tar.gz'
        def geometry(m,p):
            member=m['sequences'][0]['member']; arrays=read_npz(p[member]); arrays['crop_squares'][0,1,0]+=1
            p[member]=npz_bytes(arrays)
        rewrite_bundle(self.bundle,path,geometry)
        with self.assertRaisesRegex(ValueError,'centers/scales'):check_bundle(path)


if __name__=='__main__':unittest.main()
