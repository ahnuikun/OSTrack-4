"""Short native ViT-B CUDA fixture; GT-label changes cannot change inference."""

import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

import cv2
import numpy as np
import torch

from lib.models.ostrack import build_ostrack
from lib.test.analysis.module1_evidence import ARMS, frame_metrics
from lib.test.analysis.frozen_failure_diagnostics import CONFIGS
from tracking.diagnose_vdrm_frozen_failures import config_for, state_hash
from tracking.diagnose_vdrm_module1_evidence import collect
from tracking.diagnose_vdrm_native_float import NativeTracker, read_rgb


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    torch.set_num_threads(2)
    torch.manual_seed(42)
    configs = {arm: config_for(CONFIGS[arm]) for arm in ARMS}
    models = {arm: build_ostrack(configs[arm], training=False) for arm in ARMS}
    for arm in ('Rfreeze', 'Rdisc'):
        models[arm].load_state_dict(models['Tclean'].state_dict(), strict=False)
        with torch.no_grad():
            models[arm].backbone.vdrm.alpha.fill_(-.056)
    with torch.no_grad():
        models['Rdisc'].backbone.vdrm.route_head.classifier[-1].weight.fill_(.003)
        models['Rdisc'].backbone.vdrm.route_head.classifier[-1].bias.fill_(.15)
    for model in models.values():
        model.requires_grad_(False).cuda().eval()
    before = {arm: state_hash(model) for arm, model in models.items()}
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        frames = []
        rng = np.random.default_rng(42)
        for frame in range(1, 4):
            path = root / f'{frame}.png'
            cv2.imwrite(str(path), rng.integers(0, 256, (160, 200, 3), dtype=np.uint8))
            frames.append(str(path))
        sequence = SimpleNamespace(name='synthetic_native', frames=frames)
        gt = np.tile([70.25, 60.375, 24.5, 24.75], (3, 1))
        cases = [(sequence, gt, np.full(3, 8.), np.zeros(3), 2, 3)]
        payloads = collect(cases, models, configs)
        traces = json.loads(payloads['trajectories.json'])
        records = json.loads(payloads['records.json'])
        decode_errors = []
        import io
        for record in records:
            for arm in ARMS:
                with np.load(io.BytesIO(payloads[record['arms'][arm]['arrays_member']]), allow_pickle=False) as raw:
                    result = frame_metrics({k: raw[k] for k in raw.files}, record,
                        traces[0]['native'][arm][record['frame_1based']-1], arm)
                decode_errors.append(result['decode_error_pixels'])
        changed = gt.copy()
        changed[1:] = [1., 1., 12., 12.]
        altered = collect([(sequence, changed, np.zeros(3), np.ones(3), 2, 3)], models, configs)
        altered_traces = json.loads(altered['trajectories.json'])
        for arm in ARMS:
            boxes = [row['bbox_xywh'] for row in traces[0]['native'][arm]]
            assert boxes == [row['bbox_xywh'] for row in altered_traces[0]['native'][arm]]
            tracker = NativeTracker(models[arm], configs[arm])
            with torch.no_grad():
                tracker.initialize(read_rgb(frames[0]), dict(init_bbox=gt[0].tolist()))
                production = [gt[0].tolist()] + [tracker.track(read_rgb(path))['target_bbox'] for path in frames[1:]]
            assert boxes == production
        assert before == {arm: state_hash(model) for arm, model in models.items()}
    report = dict(synthetic_fixture=True, real_server_data_verified=False, optimizer_steps=0,
        device=torch.cuda.get_device_name(), native_steps=12, saved_arm_records=12,
        labels_do_not_change_boxes=True, capture_matches_uninstrumented_native=True,
        state_unchanged=True, max_decode_error_pixels=max(decode_errors))
    path = PROJECT / 'output/analysis/module1_step01_cuda_smoke.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('STEP01 NATIVE CUDA SMOKE PASSED:', path)


if __name__ == '__main__':
    main()
