"""Native ViT-B/CUDA synthetic developer fixture, NOT trained-model evidence."""

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
from lib.test.analysis.response_probe import CONFIG, check_bundle
from lib.train.frozen_response import (prepare_response_models, make_response_actor,
    audit_response_model, baseline_hash, assert_response_optimizer)
from lib.train.base_functions import get_optimizer_scheduler
from test_vdrm_frozen import fixed_batch
from test_vdrm_response import response_config
from test_vdrm_response_probe import response_fixture
from tracking.diagnose_vdrm_response import run


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory(prefix='response_native_gpu_') as temporary:
        root = Path(temporary)
        dataset = response_fixture(root, factory=build_ostrack)
        path = root / 'checkpoints/train/ostrack' / CONFIG / 'OSTrack_ep0300.pth.tar'
        cfg = response_config()
        model, baseline, _ = prepare_response_models(cfg, root, CONFIG, path)
        model, baseline = model.cuda(), baseline.cuda()
        audit = audit_response_model(model, baseline, cfg, fixed_batch('cuda'))
        # Exactly one synthetic optimizer step, never on user weights/data.
        optimizer, _ = get_optimizer_scheduler(model, cfg)
        assert_response_optimizer(model, optimizer)
        before = baseline_hash(model.state_dict())
        loss, _ = make_response_actor(model, cfg)(fixed_batch('cuda'))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if baseline_hash(model.state_dict()) != before:
            raise RuntimeError('native optimizer step changed baseline')
        torch.save(dict(epoch=300, net_type='OSTrack', net=model.state_dict(),
                        frozen_response_contract=model.frozen_response_contract), path)
        del model, baseline, optimizer
        torch.cuda.empty_cache()
        output = root / 'rresp.tar.gz'
        with patch('tracking.diagnose_vdrm_response.get_dataset', return_value=dataset):
            manifest = run(root, output, 'cuda')
        if manifest != check_bundle(output):
            raise RuntimeError('standalone native response recheck mismatch')
        with tarfile.open(output, 'r:gz') as archive:
            trace = json.load(archive.extractfile('trajectories.json'))[0]
        source = next(s for s in dataset if s.name == trace['sequence'])
        length = len(trace['gt_xywh'])
        sequence = Sequence(source.name, source.frames[:length], 'got10k', source.ground_truth_rect[:length])
        env = SimpleNamespace(prj_dir=str(PROJECT), save_dir=str(root),
                              results_path=str(root / 'test/tracking_results'))
        with patch('lib.test.evaluation.tracker.env_settings', return_value=env):
            tracker = Tracker('ostrack', CONFIG, 'got10k_vdrm_dev')
        with patch('lib.test.parameter.ostrack.env_settings', return_value=env):
            result = tracker.run_sequence(sequence, debug=0)
        if result['target_bbox'] != [r['bbox_xywh'] for r in trace['native']['Rresp']]:
            raise RuntimeError('production tracker differs from audited response trajectory')
        report = dict(device=torch.cuda.get_device_name(), synthetic_fixture=True,
            real_server_data_verified=False, synthetic_optimizer_steps=1,
            diagnostic_optimizer_steps=0, native_tracking_steps=manifest['native_tracking_steps'],
            records=manifest['records'], full_baseline_unchanged=True,
            production_tracker_matches_native_float=True, preflight_audit=audit)
        target = PROJECT / 'output/analysis/vdrm_response_gpu_smoke.json'
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2), encoding='utf-8')
        print('RRESP NATIVE CUDA SMOKE PASSED:', target)


if __name__ == '__main__':
    main()
