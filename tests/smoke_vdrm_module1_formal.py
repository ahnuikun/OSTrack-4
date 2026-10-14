"""Full ViT-B integration: actual processing/entry/checkpoint/native tracker.

Synthetic images and a controlled regression head; this verifies code, not
tracking performance. It does not access or overwrite user checkpoints.
"""
from copy import deepcopy
from contextlib import ExitStack
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / 'tests'))
import numpy as np
import torch

from smoke_vdrm_module1_direction import fixture_model
from test_vdrm_frozen_failure_diagnostics import synthetic_fixture
from tracking import train_vdrm_module1_formal as entry
from lib.train.module1_formal import (CONFIG, config, original_rdisc, validate_checkpoint, backend)
from lib.train.module1_learning import adapter_insertion
from lib.train.module1_data import make_loader
from lib.utils.ce_utils import generate_mask_cond


class SyntheticDataset:
    def __init__(self, root, name):
        self.root, self.name = str(root), name
        self.sequence_list = ['GOT-10k_Train_%06d' % (i+1) for i in range(9335)] if name == 'got10k' else ['a', 'b']
        self.seq_per_class = self._build_seq_per_class()
        self.class_list = ['object']
        self.image = np.random.default_rng(42).integers(0, 256, (160, 160, 3), dtype=np.uint8)

    def _build_seq_per_class(self): return {'object': list(range(len(self.sequence_list)))}
    def get_name(self): return self.name
    def get_num_sequences(self): return len(self.sequence_list)
    def is_video_sequence(self): return self.name != 'coco'
    def has_class_info(self): return True
    def get_sequences_in_class(self, name): return self.seq_per_class[name]
    def get_sequence_info(self, index):
        return dict(bbox=torch.tensor([[60., 60., 24., 24.]] * 24),
                    visible=torch.ones(24, dtype=torch.bool), valid=torch.ones(24, dtype=torch.bool))
    def get_frames(self, seq_id, frame_ids, info):
        return ([self.image.copy() for _ in frame_ids], {'bbox': [info['bbox'][f].clone() for f in frame_ids]},
                {'object_class_name': 'object'})


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    torch.set_num_threads(2)
    backend()
    with tempfile.TemporaryDirectory(prefix='module1_formal_') as temporary:
        root = Path(temporary)
        fixture_sequences = synthetic_fixture(root, factory=fixture_model)
        (root / 'list.txt').write_text('\n'.join('GOT-10k_Train_%06d' % (i+1) for i in range(9335)))
        def datasets(names, settings, loader):
            assert names == ['LASOT', 'GOT10K_vottrain', 'COCO17', 'TRACKINGNET']
            return [SyntheticDataset(root, n) for n in ('lasot', 'got10k', 'coco', 'trackingnet')]
        # Only data sources/paths are synthetic. Actual sampler, crops, transforms,
        # augmentation, full backbone, loss, optimizer and file IO are exercised.
        with patch('lib.train.base_functions.names2datasets', side_effect=datasets):
            for mode in ('--precheck', '--smoke'):
                with patch.object(sys, 'argv', ['train_vdrm_module1_formal.py', '--save_dir', str(root),
                                               '--micro_batch', '2', '--workers', '0', mode]):
                    entry.main()
            with patch.object(sys, 'argv', ['train_vdrm_module1_formal.py', '--save_dir', str(root),
                                           '--micro_batch', '2', '--workers', '0', '--smoke', '--resume']):
                entry.main()
            cfg = config()
            saved = torch.load(root / 'analysis/module1_formal_smoke/latest.pth.tar', map_location='cpu', weights_only=False)
            validate_checkpoint(saved, cfg, allow_smoke=True)
            try:
                validate_checkpoint(saved, cfg)
            except ValueError:
                pass
            else:
                raise AssertionError('smoke incorrectly accepted as formal weights')
            model, _, _ = entry.initialize(cfg, root, 'cuda')
            loader, _ = make_loader(cfg, root, 2, 0, True)
            # Continuous-vs-resumed next epoch must agree in weights AND Adam state.
            endings = []
            for _ in range(2):
                model.load_state_dict(saved['net'], strict=True)
                optimizer, scheduler = entry.optimizer_for(model, cfg)
                optimizer.load_state_dict(deepcopy(saved['optimizer']))
                scheduler.load_state_dict(deepcopy(saved['scheduler']))
                updates, metrics = entry.train_epoch(model, cfg, loader, optimizer, 'cuda', 2, 128, saved['updates'])
                endings.append((deepcopy(model.backbone.module1_adapter.state_dict()),
                                deepcopy(optimizer.state_dict()), metrics))
            print('RESUME METRICS', endings[0][2], endings[1][2], flush=True)
            for key, value in endings[0][0].items():
                print('RESUME DELTA', key, float((value-endings[1][0][key]).abs().max()), flush=True)
                torch.testing.assert_close(value, endings[1][0][key], rtol=0, atol=0)
            for index, state in endings[0][1]['state'].items():
                for key, value in state.items():
                    torch.testing.assert_close(value, endings[1][1]['state'][index][key], rtol=0, atol=0)
            # Registered production implementation equals the historical diagnostic insertion.
            model.load_state_dict(saved['net'], strict=True)
            data = next(iter(loader))
            box = data['template_anno'][0][:1].cuda()
            inputs = dict(template=data['template_images'][0][:1].cuda(), search=data['search_images'][0][:1].cuda(),
                          template_bbox=box, ce_template_mask=generate_mask_cond(cfg, 1, 'cuda', box))
            with torch.no_grad():
                registered = model(**inputs)
                with original_rdisc(model), adapter_insertion(model, model.backbone.module1_adapter):
                    historical = model(**inputs)
            for key in ('score_map', 'size_map', 'offset_map'):
                torch.testing.assert_close(registered[key], historical[key], rtol=0, atol=0)
            # Produce an explicitly synthetic epoch-300 fixture to exercise the real
            # production constructor's fixed-epoch check (not a training claim).
            fixture = deepcopy(saved)
            fixture['epoch'] = 300
            fixture['run']['smoke'] = False
            checkpoint = root / 'checkpoints/train/ostrack' / CONFIG / 'OSTrack_ep0300.pth.tar'
            checkpoint.parent.mkdir(parents=True)
            torch.save(fixture, checkpoint)
            from lib.test.parameter.ostrack import parameters
            from lib.test.tracker.ostrack import OSTrack
            from tracking.diagnose_vdrm_native_float import NativeTracker, read_rgb
            with patch('lib.test.parameter.ostrack.env_settings', return_value=SimpleNamespace(prj_dir=str(PROJECT), save_dir=str(root))):
                params = parameters(CONFIG)
            params.debug = 0
            actual = OSTrack(params, 'got10k')
            bound = NativeTracker(model.eval(), cfg)
            seq = fixture_sequences[0]
            for tracker in (actual, bound):
                tracker.initialize(read_rgb(seq.frames[0]), dict(init_bbox=seq.ground_truth_rect[0].tolist()))
            for path in seq.frames[1:4]:
                image = read_rgb(path)
                assert actual.track(image)['target_bbox'] == bound.track(image)['target_bbox']
            corrupted = deepcopy(fixture)
            key = next(k for k in corrupted['net'] if k.startswith('backbone.vdrm.'))
            corrupted['net'][key].add_(.01)
            try:
                validate_checkpoint(corrupted, cfg, expected_epoch=300)
            except ValueError:
                pass
            else:
                raise AssertionError('changed frozen weights accepted')
            # Exercise the full acceptance CLI, all five dataset routes and the
            # real benchmark extractor. Only images/sequence lists are synthetic.
            from tracking import evaluate_vdrm_module1_formal as acceptance
            from lib.test.evaluation.data import Sequence, SequenceList
            evaluation_env = SimpleNamespace(prj_dir=str(PROJECT), save_dir=str(root),
                results_path=str(root / 'test/tracking_results'),
                segmentation_path=str(root / 'test/segmentation_results'),
                result_plot_path=str(root / 'test/result_plots'))
            evaluation_data = {name: SequenceList([Sequence(name + '_fixture', seq.frames[:3], name,
                seq.ground_truth_rect[:3], target_visible=np.ones(3, dtype=bool))]) for name in acceptance.DATASETS}
            evaluation_data['got10k_vdrm_dev'] = fixture_sequences
            with ExitStack() as stack:
                for module in ('tracking.evaluate_vdrm_module1_formal', 'lib.test.evaluation.tracker',
                               'lib.test.parameter.ostrack', 'lib.test.analysis.plot_results',
                               'lib.test.analysis.extract_results'):
                    stack.enter_context(patch(module + '.env_settings', return_value=evaluation_env))
                stack.enter_context(patch.object(acceptance, 'get_dataset', side_effect=lambda name: evaluation_data[name]))
                stack.enter_context(patch.object(acceptance, 'protocol_plan', return_value={s.name: (2, 3) for s in fixture_sequences[:5]}))
                stack.enter_context(patch.object(sys, 'argv', ['evaluate_vdrm_module1_formal.py', '--save_dir', str(root), '--run']))
                acceptance.main()
            assert (root / 'analysis/module1_formal_acceptance.tar.gz').is_file()
            for name in acceptance.DATASETS:
                result = json.loads((root / 'analysis/module1_formal_acceptance' / (name + '_evaluation.json')).read_text())
                assert len(result['trackers']) == 3 and result['valid_sequence'] == [1]
    report = dict(full_vit_b=True, synthetic_data=True, controlled_regression_head=True,
        real_four_dataset_pipeline=True, real_training_entry=True, saved_initial_parameters=True,
        epoch_boundary_resume_exact=True, registered_vs_diagnostic_exact=True,
        real_tracker_constructor_and_native_trajectory_exact=True, smoke_rejected_for_production=True,
        changed_frozen_checkpoint_rejected=True, real_five_dataset_acceptance_entry=True, performance_pass=False)
    output = PROJECT / 'output/analysis/module1_formal_cuda_smoke.json'
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('FORMAL FULL VIT-B CUDA INTEGRATION PASSED:', output)


if __name__ == '__main__':
    main()
