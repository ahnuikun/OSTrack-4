"""Developer-only real ViT-B read-only fixture; no training or real dataset."""

import argparse
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
from test_vdrm_frozen_failure_diagnostics import synthetic_fixture
from lib.test.analysis.frozen_failure_diagnostics import check_bundle
from tracking.diagnose_vdrm_frozen_failures import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    args = parser.parse_args()
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory(prefix='frozen_failure_native_smoke_') as temporary:
        root = Path(temporary)
        dataset = synthetic_fixture(root, factory=build_ostrack)
        output = root / 'native.tar.gz'
        with patch('tracking.diagnose_vdrm_frozen_failures.get_dataset', return_value=dataset):
            report = run(root, root / 'results', output, args.device)
        if report != check_bundle(output):
            raise RuntimeError('standalone bundle recheck differs')
        path = PROJECT / 'output/analysis/frozen_vdrm_failure_gpu_smoke.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(device=args.device, model='native ViT-B/CE/CENTER',
            synthetic_fixture=True, real_server_data_verified=False, optimizer_steps=0,
            selected_frames=report['selected_frames'], records=report['records'],
            all_states_unchanged=True, all_real_crop_alpha0_checks_exact=True), indent=2), encoding='utf-8')
        print('NATIVE FROZEN FAILURE DIAGNOSTICS SMOKE PASSED:', path)


if __name__ == '__main__':
    main()
