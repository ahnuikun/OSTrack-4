"""Contracts which would otherwise waste a server training run."""
from copy import deepcopy
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pytest
import torch

from lib.models.layers.module1_adapter import ResidualAdapter
from lib.train.module1_formal import config, validate_config, excluded_got_sequences, overlap
from lib.train.data.sampler import TrackingSampler


def features(batch=3):
    torch.manual_seed(42)
    return (torch.randn(batch, 11, 24), torch.randn(batch, 4, 24),
            torch.ones(batch, 4, dtype=torch.bool), torch.randn(batch, 11, 24), torch.randn(batch, 11, 24))


def test_per_image_budget_and_batch_independence():
    torch.manual_seed(42)
    adapter = ResidualAdapter(24, 7).double()
    torch.nn.init.normal_(adapter.value.weight, std=3.)
    # Test independence at very different scales without conflating FP32 GEMM
    # cancellation near zero with coupling between images' norm budgets.
    values = tuple(v.double() if v.is_floating_point() else v for v in features())
    values[-1][0].mul_(.0001)
    values[-1][1].mul_(100)
    batched = adapter(*values)
    separate = torch.cat([adapter(*(v[i:i+1] for v in values)) for i in range(3)])
    torch.testing.assert_close(batched, separate, rtol=1e-10, atol=1e-10)
    assert (batched.flatten(1).norm(dim=1) <= values[-1].flatten(1).norm(dim=1) * 1.000001).all()


def test_zero_identity_zero_budget_and_gradient():
    adapter = ResidualAdapter(24, 7)
    values = features()
    assert torch.equal(adapter(*values), values[-1])
    adapter(*values).square().sum().backward()
    assert adapter.value.weight.grad.count_nonzero()
    values[-1].zero_()
    assert adapter(*values).count_nonzero() == 0


def test_scratch_zero_alpha_can_start_learning():
    adapter = ResidualAdapter(24, 7)
    values = list(features(1))
    alpha = torch.tensor(0., requires_grad=True)
    direction = torch.randn_like(values[-1])
    values[-1] = alpha * direction
    adapter(*values).sum().backward()
    torch.testing.assert_close(alpha.grad, direction.sum())
    assert alpha.grad != 0


def test_scratch_config_and_parameter_modes():
    from lib.train.module1_formal import SCRATCH_CONFIG
    cfg = config(SCRATCH_CONFIG)
    validate_config(cfg)
    assert cfg.MODEL.MODULE1_ADAPTER.MODE == 'joint'
    assert cfg.MODEL.PRETRAIN_FILE == 'mae_pretrain_vit_base.pth'
    assert cfg.TRAIN.VDRM_FROZEN_BASE_CONFIG == ''
    bad = deepcopy(cfg)
    bad.TRAIN.CE_START_EPOCH = 0
    with pytest.raises(ValueError):
        validate_config(bad)


def test_original_four_dataset_protocol_and_audit_exclusion():
    cfg = config()
    validate_config(cfg)
    assert cfg.DATA.TRAIN.DATASETS_NAME == ['LASOT', 'GOT10K_vottrain', 'COCO17', 'TRACKINGNET']
    assert all(i - 1 in excluded_got_sequences() for i in (3240, 8333, 1664, 7878, 8311))
    bad = deepcopy(cfg)
    bad.MODEL.MODULE1_ADAPTER.WIDTH = 32
    with pytest.raises(ValueError):
        validate_config(bad)
    bad = deepcopy(cfg)
    bad.DATA.TRAIN.DATASETS_NAME = ['GOT10K_vottrain']
    with pytest.raises(ValueError):
        validate_config(bad)


def test_geometry_labels_are_per_sample():
    boxes = torch.tensor([[[.5, .5, .2, .2]], [[.1, .1, .2, .2]]])
    gt = torch.tensor([[[.4, .4, .2, .2]], [[.4, .4, .2, .2]]])
    torch.testing.assert_close(overlap(boxes, gt), torch.tensor([[1.], [0.]]))


def test_bad_dataset_fails_instead_of_sampling_forever():
    class EmptyVisible:
        def get_num_sequences(self): return 1
        def get_name(self): return 'bad_fixture'
        def get_sequence_info(self, index): return {'visible': torch.zeros(20, dtype=torch.bool)}
    sampler = TrackingSampler([EmptyVisible()], p_datasets=[1], samples_per_epoch=1, max_gap=10,
                              num_search_frames=1, num_template_frames=1)
    sampler.max_sampling_attempts = 2
    with pytest.raises(RuntimeError, match='no eligible visible sequence'):
        sampler.sample_seq_from_dataset(sampler.datasets[0], True)


def test_evaluation_reuses_only_provenance_checked_results(tmp_path):
    from types import SimpleNamespace
    from tracking.evaluate_vdrm_module1_formal import stamp_results
    tracker = SimpleNamespace(results_dir=str(tmp_path / 'rdisc'), parameter_name='source')
    sequence = SimpleNamespace(name='sample', dataset='uav123')
    stamp_results(tracker, [sequence], 'uav123', 'a'*64, True)
    (tmp_path / 'rdisc/sample.txt').write_text('1 2 3 4\n')
    stamp_results(tracker, [sequence], 'uav123', 'a'*64, False)
    with pytest.raises(ValueError, match='different weights'):
        stamp_results(tracker, [sequence], 'uav123', 'b'*64, True)


def test_evaluation_rejects_unverified_legacy_results(tmp_path):
    from types import SimpleNamespace
    from tracking.evaluate_vdrm_module1_formal import stamp_results
    (tmp_path / 'sample.txt').write_text('1 2 3 4\n')
    tracker = SimpleNamespace(results_dir=str(tmp_path), parameter_name='source')
    sequence = SimpleNamespace(name='sample', dataset='uav123')
    with pytest.raises(ValueError, match='no module1 provenance'):
        stamp_results(tracker, [sequence], 'uav123', 'a'*64, True)
