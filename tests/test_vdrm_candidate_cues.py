"""Real model code on synthetic data; controls are diagnostic only."""
import contextlib
import io
import json
from pathlib import Path
import re
import shlex
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from lib.test.analysis.candidate_cues import variants,check_bundle,check_native,read_npz,npz_bytes
from lib.train.frozen_vdrm import file_hash
from test_vdrm_candidate_quality import quality_fixture
from test_vdrm_response import response_model
from test_vdrm_native_float import rewrite_bundle
from tracking import diagnose_vdrm_response as source_entry
from tracking import probe_vdrm_candidate_cues as entry


class CueMathTests(unittest.TestCase):
    def test_native_decode_tolerates_cpu_gpu_subpixel_roundoff_not_wrong_boxes(self):
        record=dict(source_native_iou=.5,source_native_peak=0,source_native_bbox=[100,100,20,20])
        metric=dict(native_iou=.5,native_peak=0)
        boxes=np.tile(np.array([100.,100.,20.,20.]),(256,1))
        # Real downloaded frame 008333:65 differs by 0.0001220703125 px.
        boxes[0,0]+=0.0001220703125
        check_native(record,{'boxes_xywh':boxes},metric)
        boxes[0,0]+=.01
        with self.assertRaisesRegex(ValueError,'native candidate'):
            check_native(record,{'boxes_xywh':boxes},metric)

    def test_interventions_change_only_named_columns_and_do_not_mutate(self):
        torch.manual_seed(42)
        cues=torch.randn(1,256,71)
        candidates=torch.randn(1,256,32)
        before=cues.clone();donor=torch.randn(1,1,32)
        output=variants(cues,candidates,{'donor':donor})
        self.assertTrue(torch.equal(cues,before))
        for name,start,stop in [('identity_zero',0,65),('response_zero',65,66),
                                ('geometry_zero',66,70),('support_zero',70,71)]:
            self.assertFalse(output[name][...,start:stop].any())
            self.assertTrue(torch.equal(output[name][...,:start],before[...,:start]))
            self.assertTrue(torch.equal(output[name][...,stop:],before[...,stop:]))
        swapped=output['swap:donor']
        self.assertTrue(torch.equal(swapped[...,65:],before[...,65:]))
        self.assertTrue(torch.equal(swapped[...,:32],candidates*donor))
        self.assertTrue(torch.equal(swapped[...,32:64],(candidates-donor).abs()))

    def test_capture_hooks_restored_on_failure(self):
        from test_vdrm_candidate_quality import quality_config
        head=response_model(quality_config()).response_residual
        with self.assertRaisesRegex(RuntimeError,'deliberate'):
            with entry.capture_cues(head):raise RuntimeError('deliberate')
        self.assertFalse(head.project._forward_hooks)
        self.assertFalse(head.correct._forward_pre_hooks)

    def test_check_command_is_cpu_only(self):
        summary=dict(groups={},note='test')
        with patch.object(sys,'argv',['probe_vdrm_candidate_cues.py','--check_bundle','x.tar.gz']), \
                patch.object(entry,'check_bundle',return_value=({'records':298},summary,[])), \
                patch.object(entry.torch.cuda,'is_available',side_effect=AssertionError('CUDA queried')), \
                patch.object(entry,'prepare_response_models',side_effect=AssertionError('weights accessed')):
            entry.main()


    def test_modes_are_mutually_exclusive(self):
        with patch.object(sys,'argv',['probe_vdrm_candidate_cues.py','--bundle','a','--check_bundle','b']), \
                contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            entry.main()
        self.assertEqual(error.exception.code,2)


class CueRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads=torch.get_num_threads();torch.set_num_threads(2)
        cls.temp=tempfile.TemporaryDirectory(prefix='candidate_cues_');cls.root=Path(cls.temp.name)
        cls.dataset=quality_fixture(cls.root)
        cls.source=cls.root/'source.tar.gz';cls.target=cls.root/'cues.tar.gz'
        with patch('tracking.diagnose_vdrm_frozen_failures.build_ostrack',side_effect=response_model), \
                patch('lib.models.ostrack.build_ostrack',side_effect=response_model), \
                patch.object(source_entry,'get_dataset',return_value=cls.dataset), \
                patch.object(torch.Tensor,'cuda',lambda t,*a,**kw:t), contextlib.redirect_stdout(io.StringIO()):
            source_entry.run(cls.root,cls.source,'cpu',profile='early_failure',candidate_arm='Rqual')
        cls.source_hash=file_hash(cls.source)
        with patch('lib.models.ostrack.build_ostrack',side_effect=response_model), \
                patch.object(source_entry,'get_dataset',side_effect=AssertionError('dataset read')), \
                contextlib.redirect_stdout(io.StringIO()):
            cls.manifest=entry.run(cls.source,cls.root,cls.target,'cpu')

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup();torch.set_num_threads(cls.threads)

    def test_cpu_recomputation_every_control_and_readonly_source(self):
        manifest,summary,frames=check_bundle(self.target)
        self.assertEqual(manifest,self.manifest)
        self.assertEqual(summary['records'],298)
        self.assertEqual(summary['groups']['template_swap']['interventions'],1192)
        self.assertEqual(manifest['optimizer_steps'],0)
        self.assertFalse(summary['closed_loop'])
        self.assertEqual(file_hash(self.source),self.source_hash)
        self.assertEqual(manifest['state_sha256_before'],manifest['state_sha256_after'])

    def test_rehashed_tampering_of_head_features_score_and_report_rejected(self):
        def mutate_head(manifest,payloads):
            state=read_npz(payloads['head.npz']);state['4.bias']+=.1
            payloads['head.npz']=npz_bytes(state)
        def mutate_feature(manifest,payloads):
            row=json.loads(payloads['records.json'])[0]
            arrays=read_npz(payloads[row['member']]);arrays['cues'][...,0]+=.1
            payloads[row['member']]=npz_bytes(arrays)
        def mutate_score(manifest,payloads):
            row=json.loads(payloads['records.json'])[0]
            arrays=read_npz(payloads[row['member']]);arrays['scores'][1]+= .01
            payloads[row['member']]=npz_bytes(arrays)
        def mutate_report(manifest,payloads):
            summary=json.loads(payloads['summary.json']);summary['groups']['native']['changed_peaks']+=1
            payloads['summary.json']=json.dumps(summary).encode()
        def mutate_donor(manifest,payloads):
            donors=read_npz(payloads['donors.npz']);donors[next(iter(donors))]+=.1
            payloads['donors.npz']=npz_bytes(donors)
        def mutate_native(manifest,payloads):
            records=json.loads(payloads['records.json']);records[0]['source_native_bbox'][0]+=1
            payloads['records.json']=json.dumps(records).encode()
        def mutate_dataset(manifest,payloads):
            manifest['dataset']='visdrone'
        for i,mutate in enumerate((mutate_head,mutate_feature,mutate_score,mutate_report,
                                  mutate_donor,mutate_native,mutate_dataset)):
            target=self.root/f'bad{i}.tar.gz';rewrite_bundle(self.target,target,mutate)
            with self.assertRaises(ValueError):check_bundle(target)

    def test_no_overwrite_before_source_or_weights_access(self):
        with patch.object(entry,'check_source',side_effect=AssertionError('source read')):
            with self.assertRaises(FileExistsError):entry.run(self.source,self.root,self.target,'cpu')

    def test_wrong_checkpoint_is_rejected_before_model_construction(self):
        with patch.object(entry,'file_hash',return_value='wrong'), \
                patch.object(entry,'prepare_response_models',side_effect=AssertionError('model built')):
            with self.assertRaisesRegex(ValueError,'not the one used'):
                entry.run(self.source,self.root,self.root/'mismatch.tar.gz','cpu')


if __name__=='__main__':unittest.main()
