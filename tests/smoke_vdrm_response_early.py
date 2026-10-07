"""Native ViT-B/CUDA early audit; synthetic fixture, zero optimizer steps."""

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
from lib.test.analysis.response_probe import CONFIG
from test_vdrm_response_probe import response_fixture
from tracking.diagnose_vdrm_response import run
from tracking.analyze_vdrm_response import run as analyze


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    torch.set_num_threads(2)
    torch.manual_seed(42)
    with tempfile.TemporaryDirectory(prefix='response_early_native_') as temporary:
        root = Path(temporary)
        dataset = response_fixture(root, factory=build_ostrack)
        probe = root / 'rresp_early_probe.tar.gz'
        with patch('tracking.diagnose_vdrm_response.get_dataset', return_value=dataset):
            manifest = run(root, probe, 'cuda', profile='early_failure')
        summary = analyze(probe, root / 'analysis.tar.gz')
        with tarfile.open(probe, 'r:gz') as archive:
            traces = json.load(archive.extractfile('trajectories.json'))
        env = SimpleNamespace(prj_dir=str(PROJECT), save_dir=str(root),
                              results_path=str(root / 'production'))
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
            if result['target_bbox'] != [row['bbox_xywh'] for row in trace['native']['Rresp']]:
                raise RuntimeError('early diagnostic changed production trajectory: ' + trace['sequence'])
            matched.append(trace['sequence'])
        report = dict(device=torch.cuda.get_device_name(), synthetic_fixture=True,
            real_server_early_data_verified=False, optimizer_steps=0,
            native_tracking_steps=manifest['native_tracking_steps'], records=manifest['records'],
            maximum_decode_error_pixels=summary['maximum_decode_error_pixels'],
            production_tracker_exact_sequences=matched)
        target = PROJECT / 'output/analysis/vdrm_response_early_gpu_smoke.json'
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2), encoding='utf-8')
        print('RRESP EARLY NATIVE CUDA SMOKE PASSED:', target)


if __name__ == '__main__':
    main()
