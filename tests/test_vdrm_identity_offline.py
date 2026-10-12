"""Offline-only identity protocol, geometry, gradients and full cached training."""
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
import cv2

from lib.train.identity_offline import (load_protocol, registered_ids, roi_cells, IdentityMatcher,
    losses, read_cache, protocol_hash, strict_top1, summarize_evaluation, ARMS)
from lib.test.analysis.identity_offline import check_bundle, donor_indices
from lib.train.frozen_vdrm import file_hash
from test_vdrm_native_float import rewrite_bundle
from tracking import cache_vdrm_identity as cache_entry, train_vdrm_identity as train_entry
from tracking import analyze_vdrm_identity as analyze_entry


def synthetic_cache(root):
    protocol = load_protocol()
    protocol.update(train_sequences=4,minimum_train_pairs=2,minimum_dev_pairs=2,
                    minimum_dev_sequences=2,minimum_hard_pairs=1,batch_size=8)
    train,dev = registered_ids(protocol)
    root.mkdir()
    (root/'protocol.json').write_text(json.dumps(protocol),encoding='utf-8')
    torch.manual_seed(7)
    arrays = dict(template=[],candidates=[],template_mask=[],candidate_mask=[])
    rows = []
    for split,ids in (('train',train),('dev',dev[:6])):
        for identity in ids:
            prototype = torch.randn(9,32)
            for frame in (2,3):
                candidates = torch.randn(5,9,32)
                candidates[0] = prototype+.03*torch.randn_like(prototype)
                arrays['template'].append(prototype.numpy())
                arrays['candidates'].append(candidates.numpy())
                arrays['template_mask'].append(np.ones(9,dtype=bool))
                arrays['candidate_mask'].append(np.ones((5,9),dtype=bool))
                rows.append(dict(split=split,sequence_id=identity,sequence=f'GOT-10k_Train_{identity+1:06d}',
                    frame_1based=frame,cover=8,member='pairs.npz',offset=len(rows),ious=[.9,0,0,0,0],
                    base_scores=[.3,.4,.2,.1,.1]))
    np.savez_compressed(root/'pairs.npz',**{k:np.stack(v) for k,v in arrays.items()})
    index = dict(kind='vdrm_identity_cache',version=1,protocol_sha256=protocol_hash(protocol),
        train_ids=train,dev_ids=dev,state_before='same',state_after='same',optimizer_steps=0,
        checkpoint_sha256='fixture',feature_dim=32,rows=rows,
        members_sha256={'pairs.npz':file_hash(root/'pairs.npz')})
    (root/'index.json').write_text(json.dumps(index),encoding='utf-8')
    return index,protocol


class IdentityMathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads=torch.get_num_threads();torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_fixed_partition_and_budget(self):
        protocol=load_protocol();train,dev=registered_ids(protocol)
        self.assertEqual(len(train),512);self.assertEqual(len(dev),152)
        self.assertFalse(set(train)&set(dev))
        self.assertEqual(registered_ids(protocol),(train,dev))
        self.assertEqual(protocol['epochs'],20);self.assertEqual(protocol['seed'],42)

    def test_roi_preserves_cells_and_never_backpropagates(self):
        feature=torch.arange(9.).reshape(1,1,3,3).requires_grad_()
        value,mask=roi_cells(feature,torch.tensor([[[.5,.5,1.,1.]]]),torch.ones(1,1,3,3))
        self.assertTrue(torch.allclose(value.flatten(),torch.arange(9.),atol=1e-5))
        self.assertTrue(mask.all());self.assertFalse(value.requires_grad)
        value,mask=roi_cells(feature,torch.tensor([[[2.,2.,1.,1.]]]),torch.ones(1,1,3,3))
        self.assertFalse(mask.any());self.assertFalse(value.any())

    def test_spatial_detects_permutation_mean_does_not(self):
        torch.manual_seed(42)
        template=torch.randn(2,9,32)
        candidates=torch.stack((template,template.flip(1)),dim=1)
        mask=torch.ones(2,9,dtype=torch.bool)
        mean=IdentityMatcher(32,'mean');spatial=IdentityMatcher(32,'spatial')
        spatial.load_state_dict(mean.state_dict())
        a=mean(template,candidates,mask,mask[:,None].expand(-1,2,-1))
        b=spatial(template,candidates,mask,mask[:,None].expand(-1,2,-1))
        self.assertTrue(torch.allclose(a[:,0],a[:,1],atol=1e-6))
        self.assertTrue((b[:,0]>b[:,1]+.1).all())
        self.assertEqual(sum(p.numel() for p in mean.parameters()),sum(p.numel() for p in spatial.parameters()))

    def test_isolation_and_same_sequence_not_negative(self):
        torch.manual_seed(42);protocol=load_protocol();model=IdentityMatcher(32,'spatial')
        batch=dict(template=torch.randn(4,9,32,requires_grad=True),candidates=torch.randn(4,5,9,32,requires_grad=True),
                   template_mask=torch.ones(4,9,dtype=torch.bool),candidate_mask=torch.ones(4,5,9,dtype=torch.bool),
                   sequence_id=torch.tensor([1,2,3,4]))
        audit=train_entry.audit_head(model,batch,protocol)
        self.assertEqual(audit['feature_gradient_tensors'],0);self.assertGreater(audit['head_gradient_l2'],0)
        value,status=losses(model,batch,protocol)
        rank_only,_=losses(model,batch,protocol,include_identity=False)
        self.assertAlmostEqual(float(value-rank_only),protocol['identity_weight']*status['identity'],places=5)
        value.backward()
        self.assertIsNone(batch['template'].grad);self.assertIsNone(batch['candidates'].grad)
        batch['sequence_id'].fill_(1)
        _,status=losses(model,batch,protocol)
        self.assertEqual(status['identity_queries'],0);self.assertEqual(status['identity'],0)

    def test_constant_scores_fail_and_donors_never_use_same_sequence(self):
        self.assertFalse(strict_top1(torch.ones(3,5)).any())
        rows=[dict(sequence_id=i) for i in range(6) for _ in range(2)]
        indices=donor_indices(rows)
        for i,donors in enumerate(indices):
            self.assertEqual(len(set(donors)),4)
            self.assertTrue(all(rows[j]['sequence_id']!=rows[i]['sequence_id'] for j in donors))

    def test_insufficient_sample_gate_never_passes(self):
        rows=[dict(sequence_id=i,cover=8,base_scores=[0,1,0,0,0]) for i in range(2)]
        outcomes={m:dict(candidate_correct=[True]*2,template_correct=[True]*2,template_eligible=[True]*2) for m in ARMS}
        report=summarize_evaluation(rows,outcomes,load_protocol(),1000)
        self.assertFalse(report['evidence_sufficient'])
        self.assertFalse(report['heads']['mean_id']['offline_gate_passed'])
        self.assertFalse(report['spatial_complexity_supported'])

    def test_jitter_is_fixed_and_does_not_mutate_anchor(self):
        box=np.array([10.,20.,30.,40.]);before=box.copy()
        a=cache_entry.jitter_anchor(box,4,20,42);b=cache_entry.jitter_anchor(box,4,20,42)
        np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(box,before)

    def test_candidate_selection_requires_real_supported_positive(self):
        torch.manual_seed(42)
        output=dict(backbone_feat=torch.randn(1,320,32),size_map=torch.full((1,2,16,16),.25),
            offset_map=torch.full((1,2,16,16),.5),score_map=torch.ones(1,1,16,16),removed_indexes_s=[])
        zbox=torch.tensor([[.25,.25,.5,.5]]);target=torch.tensor([[.375,.375,.25,.25]])
        zp=torch.zeros(1,128,128,dtype=torch.bool);xp=torch.zeros(1,256,256,dtype=torch.bool)
        pair,reason=cache_entry.collect_pair(output,zbox,target,zp,xp,load_protocol())
        self.assertIsNone(reason);self.assertEqual(pair[0]['candidates'].shape,(5,9,32))
        self.assertGreaterEqual(pair[1]['ious'][0],.5)
        self.assertTrue(all(v<=.1 for v in pair[1]['ious'][1:]))
        from lib.models.layers.candidate_quality import dense_center_boxes
        from lib.train.actors.candidate_quality import candidate_iou
        from lib.test.utils.hann import hann2d
        iou=candidate_iou(dense_center_boxes(output['size_map'],output['offset_map']),target)[0]
        base=(output['score_map']*hann2d(torch.tensor([16,16]),centered=True)).flatten()
        self.assertEqual(pair[1]['candidate_indices'][0],int(base.masked_fill(iou<.5,-torch.inf).argmax()))
        pair,reason=cache_entry.collect_pair(output,zbox,target,zp,torch.ones_like(xp),load_protocol())
        self.assertIsNone(pair);self.assertEqual(reason,'no_supported_positive')

    def test_existing_output_rejected_before_data_access(self):
        with tempfile.TemporaryDirectory() as root:
            with patch.object(cache_entry,'plan_sequences',side_effect=AssertionError('data read')):
                with self.assertRaises(FileExistsError):cache_entry.run('missing','missing',root)
            with patch.object(train_entry,'read_cache',side_effect=AssertionError('cache read')):
                with self.assertRaises(FileExistsError):train_entry.run('missing',root)

    def test_val_list_and_missing_weights_fail_before_inference(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);(root/'list.txt').write_text('GOT-10k_Val_000001\n')
            with self.assertRaisesRegex(ValueError,'training list'):
                cache_entry.plan_sequences(root,load_protocol())
            with patch.object(cache_entry,'plan_sequences',side_effect=AssertionError('data read')):
                with self.assertRaisesRegex(FileNotFoundError,'Tclean'):
                    cache_entry.run(root,root,root/'new')


    def test_full_cache_writer_real_split_ids_shapes_and_frozen_state(self):
        class FrozenFixture(torch.nn.Module):
            def __init__(self):
                super().__init__();self.anchor=torch.nn.Parameter(torch.tensor(1.),requires_grad=False)
            def forward(self,template,search,**kwargs):
                feature=torch.sin(torch.arange(320*32,dtype=torch.float32).reshape(1,320,32)*.07)
                return dict(backbone_feat=feature,size_map=torch.full((1,2,16,16),.25),
                    offset_map=torch.full((1,2,16,16),.5),score_map=torch.ones(1,1,16,16),removed_indexes_s=[])
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);data=root/'mounted_dataset';data.mkdir()
            protocol=load_protocol();protocol.update(train_sequences=2,frames_per_sequence=1)
            train,dev=registered_ids(protocol)
            names=[f'GOT-10k_Train_{i+1:06d}' for i in range(max(train+dev)+1)]
            (data/'list.txt').write_text('\n'.join(names),encoding='utf-8')
            for identity in train+dev:
                folder=data/names[identity];folder.mkdir()
                (folder/'groundtruth.txt').write_text('48,48,32,32\n48,48,32,32\n')
                (folder/'cover.label').write_text('8\n8\n');(folder/'absence.label').write_text('0\n0\n')
                for f in (1,2):cv2.imwrite(str(folder/f'{f:08d}.jpg'),np.full((128,128,3),120,dtype=np.uint8))
            config=root/'protocol.json';config.write_text(json.dumps(protocol))
            from lib.train.frozen_vdrm import BASE_CONFIG,load_base_config
            path=root/'checkpoints/train/ostrack'/BASE_CONFIG/'OSTrack_ep0300.pth.tar'
            path.parent.mkdir(parents=True);path.write_bytes(b'fixture')
            model=FrozenFixture().eval()
            loaded=({'Tclean':model},load_base_config(),{'Tclean':{'checkpoint_sha256':'fixture'}})
            with patch.object(cache_entry,'load_models',return_value=loaded),contextlib.redirect_stdout(io.StringIO()):
                result=cache_entry.run(data,root,root/'cache',config,'cpu')
            checked,_,tensors=read_cache(root/'cache')
            self.assertEqual(result,checked);self.assertEqual(len(result['sequences']),154)
            self.assertEqual(result['state_before'],result['state_after'])
            self.assertEqual(set(r['split'] for r in result['rows']),{'train','dev'})
            self.assertEqual(tensors['template'].shape[-2:],(9,32))
            self.assertFalse(model.training);self.assertIsNone(model.anchor.grad)


class IdentityPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads=torch.get_num_threads();torch.set_num_threads(2)
        cls.temp=tempfile.TemporaryDirectory(prefix='identity_screen_');cls.root=Path(cls.temp.name)
        cls.cache=cls.root/'cache';cls.index,cls.protocol=synthetic_cache(cls.cache)
        before=file_hash(cls.cache/'pairs.npz')
        cls.seen=set()
        original_loss=train_entry.losses
        def checked_loss(model,batch,protocol,**kwargs):
            cls.seen.update(batch['sequence_id'].tolist())
            return original_loss(model,batch,protocol,**kwargs)
        with contextlib.redirect_stdout(io.StringIO()),patch.object(train_entry,'losses',side_effect=checked_loss):
            cls.summary=train_entry.run(cls.cache,cls.root/'screen','cpu')
        cls.target=cls.root/'screen/identity_offline_results.tar.gz'
        if file_hash(cls.cache/'pairs.npz')!=before:raise AssertionError('cache changed')

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup();torch.set_num_threads(cls.threads)

    def test_train_checkpoints_and_independent_report(self):
        manifest,summary=check_bundle(self.target)
        self.assertEqual(summary,self.summary);self.assertEqual(manifest['epochs'],20)
        for mode in ARMS:
            self.assertTrue((self.root/'screen'/f'Identity_{mode}_ep0020.pth').is_file())
        self.assertFalse(manifest['backbone_loaded_during_training'])
        self.assertEqual(manifest['train_pairs'],8);self.assertEqual(manifest['dev_pairs'],12)
        self.assertEqual(self.seen,set(self.index['train_ids']))
        self.assertFalse(self.seen & set(self.index['dev_ids']))

    def test_second_run_is_deterministic(self):
        with contextlib.redirect_stdout(io.StringIO()):
            second=train_entry.run(self.cache,self.root/'repeat','cpu')
        self.assertEqual(second,self.summary)
        for mode in ARMS:
            a=torch.load(self.root/'screen'/f'Identity_{mode}_ep0020.pth',weights_only=False)
            b=torch.load(self.root/'repeat'/f'Identity_{mode}_ep0020.pth',weights_only=False)
            self.assertTrue(torch.equal(a['net']['project.weight'],b['net']['project.weight']))

    def test_rehashed_report_or_overlap_tampering_rejected(self):
        def bad_report(manifest,payloads):
            summary=json.loads(payloads['summary.json']);summary['evidence_sufficient']=not summary['evidence_sufficient']
            payloads['summary.json']=json.dumps(summary).encode()
        def bad_partition(manifest,payloads):
            manifest['train_ids'][0]=manifest['dev_ids'][0]
        for i,fn in enumerate((bad_report,bad_partition)):
            target=self.root/f'tampered{i}.tar.gz';rewrite_bundle(self.target,target,fn)
            with self.assertRaises(ValueError):check_bundle(target)

    def test_cache_sha_and_split_guards(self):
        index_path=self.cache/'index.json';original=index_path.read_text()
        index=json.loads(original);index['rows'][0]['split']='dev'
        try:
            index_path.write_text(json.dumps(index),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'boundary'):read_cache(self.cache)
        finally:index_path.write_text(original,encoding='utf-8')


if __name__=='__main__':unittest.main()
