"""Actual MAE initialization, real full-network updates, production load/acceptance.

Images are synthetic. MAE weights, model, losses, optimizer, two data workers,
checkpoint and inference/evaluation code are real. No performance claim.
"""
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

from smoke_vdrm_module1_formal import SyntheticDataset
from smoke_vdrm_module1_direction import fixture_model
from test_vdrm_frozen_failure_diagnostics import synthetic_fixture
from tracking import train_vdrm_module1_formal as entry
from lib.train.module1_formal import SCRATCH_CONFIG, config, validate_checkpoint


def main():
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory(prefix='module1_scratch_') as temporary:
        root = Path(temporary)
        sequences = synthetic_fixture(root, factory=fixture_model)
        (root / 'list.txt').write_text('\n'.join('GOT-10k_Train_%06d' % (i+1) for i in range(9335)))
        def datasets(names, settings, loader):
            return [SyntheticDataset(root, n) for n in ('lasot', 'got10k', 'coco', 'trackingnet')]
        with patch('lib.train.base_functions.names2datasets', side_effect=datasets):
            for extra in (['--precheck'], ['--smoke'], ['--smoke', '--resume']):
                with patch.object(sys, 'argv', ['train_vdrm_module1_formal.py', '--stage', 'scratch',
                        '--save_dir', str(root), '--micro_batch', '2', '--workers', '2', *extra]):
                    entry.main()
        saved = torch.load(root / 'analysis/module1_scratch_smoke/latest.pth.tar', map_location='cpu', weights_only=False)
        cfg = config(SCRATCH_CONFIG)
        validate_checkpoint(saved, cfg, expected_epoch=3, allow_smoke=True)
        assert saved['module1_contract']['pretrained_backbone_tensors_verified'] >= 140
        assert saved['net']['backbone.vdrm.alpha'].abs() > 0
        for name in ('search.weight', 'template.weight', 'value.weight'):
            assert not torch.equal(saved['initial_adapter'][name], saved['net']['backbone.module1_adapter.'+name])
        # Synthetic fixed-epoch fixture, solely for exercising production checks.
        saved['epoch'] = 300
        saved['run']['smoke'] = False
        checkpoint = root / 'checkpoints/train/ostrack' / SCRATCH_CONFIG / 'OSTrack_ep0300.pth.tar'
        checkpoint.parent.mkdir(parents=True)
        torch.save(saved, checkpoint)
        del saved
        from tracking import evaluate_vdrm_module1_formal as acceptance
        from lib.test.evaluation.data import Sequence, SequenceList
        seq = sequences[0]
        evaluation_env = SimpleNamespace(prj_dir=str(PROJECT), save_dir=str(root),
            results_path=str(root / 'test/tracking_results'), segmentation_path=str(root / 'test/segmentation_results'),
            result_plot_path=str(root / 'test/result_plots'))
        evaluation_data = {name: SequenceList([Sequence(name + '_fixture', seq.frames[:3], name,
            seq.ground_truth_rect[:3], target_visible=np.ones(3, dtype=bool))]) for name in acceptance.DATASETS}
        evaluation_data['got10k_vdrm_dev'] = sequences
        with ExitStack() as stack:
            for module in ('tracking.evaluate_vdrm_module1_formal', 'lib.test.evaluation.tracker',
                           'lib.test.parameter.ostrack', 'lib.test.analysis.plot_results',
                           'lib.test.analysis.extract_results'):
                stack.enter_context(patch(module + '.env_settings', return_value=evaluation_env))
            stack.enter_context(patch.object(acceptance, 'get_dataset', side_effect=lambda name: evaluation_data[name]))
            stack.enter_context(patch.object(acceptance, 'protocol_plan', return_value={s.name: (2, 3) for s in sequences[:5]}))
            stack.enter_context(patch.object(sys, 'argv', ['evaluate_vdrm_module1_formal.py', '--stage', 'scratch',
                                                        '--save_dir', str(root), '--run']))
            acceptance.main()
        assert (root / 'analysis/module1_scratch_acceptance.tar.gz').is_file()
    output = PROJECT / 'output/analysis/module1_scratch_cuda_smoke.json'
    output.write_text(json.dumps(dict(actual_mae=True, synthetic_images=True, full_vit_b=True,
        multiprocess_data_workers=2, alpha_and_all_adapter_paths_train=True,
        full_network_optimizer=True, real_training_checkpoint_resume=True,
        real_production_tracker_and_five_benchmark_entry=True, performance_pass=False), indent=2))
    print('SCRATCH FULL VIT-B CUDA INTEGRATION PASSED:', output)


if __name__ == '__main__':
    main()
