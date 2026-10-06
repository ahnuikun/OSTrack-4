"""Developer-only native ViT-B/CUDA fixture; no real data or training."""

import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / 'tests'))

from lib.models.ostrack import build_ostrack
from lib.test.analysis.native_float_audit import check_bundle
from test_vdrm_native_float import native_fixture
from tracking.diagnose_vdrm_native_float import run


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for native tracker smoke')
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory(prefix='vdrm_native_float_gpu_') as temporary:
        root = Path(temporary)
        dataset = native_fixture(root, factory=build_ostrack)
        output = root / 'native.tar.gz'
        with patch('tracking.diagnose_vdrm_native_float.get_dataset', return_value=dataset):
            manifest = run(root, output, 'cuda')
        if manifest != check_bundle(output):
            raise RuntimeError('standalone bundle recheck differs')
        report = dict(model='native ViT-B/CE/CENTER + native OSTrack tracker/Preprocessor',
            device=torch.cuda.get_device_name(), synthetic_fixture=True, real_server_data_verified=False,
            optimizer_steps=0, native_tracking_steps=manifest['native_tracking_steps'],
            same_crop_records=manifest['records'], fork_records=manifest['fork_records'],
            all_self_replays_exact=True, all_control_forks_exact=True, all_alpha0_visual_outputs_exact=True,
            all_model_states_unchanged=True)
        path = PROJECT / 'output/analysis/vdrm_native_float_gpu_smoke.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2), encoding='utf-8')
        print('NATIVE FLOAT CUDA SMOKE PASSED:', path)


if __name__ == '__main__':
    main()
