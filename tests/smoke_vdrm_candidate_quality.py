"""Native ViT-B/CUDA Rqual test. One SYNTHETIC update, never user weights."""
import json
from pathlib import Path
import sys
import tarfile
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / 'tests'))

from lib.models.ostrack import build_ostrack
from lib.test.evaluation import Sequence, Tracker
from lib.test.evaluation.running import _save_tracker_output
from lib.test.analysis.response_probe import QUALITY_CONFIG as CONFIG
from lib.train.frozen_response import (prepare_response_models, audit_response_model, make_response_actor,
    baseline_hash, assert_response_optimizer, validate_response_checkpoint)
from lib.train.base_functions import get_optimizer_scheduler
from test_vdrm_candidate_quality import quality_fixture, quality_config
from test_vdrm_frozen import fixed_batch
from tracking.diagnose_vdrm_response import run
from tracking.analyze_vdrm_response import run as analyze


def batch32():
    # Per-GPU formal batch shape; all tensors remain synthetic.
    data = fixed_batch('cuda')
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            repeat = [1] * value.ndim
            repeat[1] = 16
            data[key] = value.repeat(*repeat)
    return data


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    torch.set_num_threads(2)
    torch.manual_seed(42)
    with tempfile.TemporaryDirectory(prefix='quality_native_') as temporary:
        root = Path(temporary)
        dataset = quality_fixture(root, factory=build_ostrack)
        cfg = quality_config()
        path = root / 'checkpoints/train/ostrack' / CONFIG / 'OSTrack_ep0020.pth.tar'
        model, baseline, _ = prepare_response_models(cfg, root, CONFIG, path)
        model, baseline = model.cuda(), baseline.cuda()
        torch.cuda.reset_peak_memory_stats()
        audit = audit_response_model(model, baseline, cfg, batch32())
        optimizer, _ = get_optimizer_scheduler(model, cfg)
        assert_response_optimizer(model, optimizer)
        before, full_before = baseline_hash(model.state_dict()), baseline_hash(model.state_dict(), False)
        loss, _ = make_response_actor(model, cfg)(batch32())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        peak_memory = torch.cuda.max_memory_allocated()
        if baseline_hash(model.state_dict()) != before or baseline_hash(model.state_dict(), False) == full_before:
            raise RuntimeError('synthetic update must change head only')
        checkpoint = dict(epoch=20, net_type='OSTrack', net=model.state_dict(),
                          frozen_response_contract=model.frozen_response_contract)
        validate_response_checkpoint(model, checkpoint, cfg, expected_epoch=20)
        torch.save(checkpoint, path)
        del model, baseline, optimizer, checkpoint, loss
        torch.cuda.empty_cache()
        probe = root / 'rqual_early_probe.tar.gz'
        with patch('tracking.diagnose_vdrm_response.get_dataset', return_value=dataset):
            manifest = run(root, probe, 'cuda', profile='early_failure', candidate_arm='Rqual')
        summary = analyze(probe, root / 'analysis.tar.gz')
        with tarfile.open(probe, 'r:gz') as archive:
            traces = json.load(archive.extractfile('trajectories.json'))
        env = SimpleNamespace(prj_dir=str(PROJECT), save_dir=str(root), results_path=str(root / 'production'))
        matched = []
        for trace in traces:
            if trace['sequence'][-6:] not in ('007878', '008311'):
                continue
            source = next(s for s in dataset if s.name == trace['sequence'])
            length = len(trace['gt_xywh'])
            sequence = Sequence(source.name, source.frames[:length], 'got10k', source.ground_truth_rect[:length])
            with patch('lib.test.evaluation.tracker.env_settings', return_value=env):
                tracker = Tracker('ostrack', CONFIG, 'got10k_vdrm_dev')
            with patch('lib.test.parameter.ostrack.env_settings', return_value=env):
                result = tracker.run_sequence(sequence, debug=0)
            if result['target_bbox'] != [row['bbox_xywh'] for row in trace['native']['Rqual']]:
                raise RuntimeError('quality diagnostic changed production trajectory: ' + trace['sequence'])
            _save_tracker_output(sequence, tracker, result)
            if not (Path(tracker.results_dir) / 'got10k' / (sequence.name + '.txt')).is_file():
                raise RuntimeError('wrong GOT result directory')
            matched.append(trace['sequence'])
        report = dict(device=torch.cuda.get_device_name(), synthetic_fixture=True,
            real_server_rqual_data_verified=False, synthetic_optimizer_steps=1, diagnostic_optimizer_steps=0,
            synthetic_batch_per_gpu=32, single_gpu_peak_allocated_bytes=peak_memory,
            full_baseline_unchanged=True, preflight_audit=audit,
            native_tracking_steps=manifest['native_tracking_steps'], records=manifest['records'],
            maximum_decode_error_pixels=summary['maximum_decode_error_pixels'],
            production_tracker_exact_sequences=matched, nested_result_writer_verified=True)
        target = PROJECT / 'output/analysis/vdrm_candidate_quality_gpu_smoke.json'
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2), encoding='utf-8')
        print('RQUAL NATIVE CUDA SMOKE PASSED:', target)


if __name__ == '__main__':
    main()
