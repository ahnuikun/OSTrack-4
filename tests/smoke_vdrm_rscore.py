"""Native ViT-B/CUDA developer fixture, not actual trained-weight evidence."""

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
from lib.test.parameter import ostrack_rscore as parameter_module
from lib.test.analysis.rscore_probe import check_bundle
from lib.test.tracker.ostrack_rscore import PROBE_CONFIG
from test_vdrm_rscore import rscore_fixture
from tracking.diagnose_vdrm_rscore import run


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory(prefix='rscore_native_gpu_') as temporary:
        root = Path(temporary)
        dataset = rscore_fixture(root, factory=build_ostrack)
        output = root / 'rscore.tar.gz'
        with patch('tracking.diagnose_vdrm_rscore.get_dataset', return_value=dataset):
            manifest = run(root, output, 'cuda')
        if manifest != check_bundle(output):
            raise RuntimeError('standalone recheck mismatch')
        with tarfile.open(output, 'r:gz') as archive:
            trace = json.load(archive.extractfile('trajectories.json'))[0]
        source = next(s for s in dataset if s.name == trace['sequence'])
        length = len(trace['gt_xywh'])
        sequence = Sequence(source.name, source.frames[:length], 'got10k', source.ground_truth_rect[:length])
        env = SimpleNamespace(prj_dir=str(PROJECT), save_dir=str(root), results_path=str(root / 'test/tracking_results'))
        with patch('lib.test.evaluation.tracker.env_settings', return_value=env):
            tracker = Tracker('ostrack_rscore', PROBE_CONFIG, 'got10k_vdrm_dev')
        with patch.object(parameter_module, 'env_settings', return_value=env), \
                patch('lib.test.parameter.ostrack.env_settings', return_value=env):
            result = tracker.run_sequence(sequence, debug=0)
        if result['target_bbox'] != [r['bbox_xywh'] for r in trace['native']['Rscore']]:
            raise RuntimeError('production plugin trajectory differs from audited Rscore adapter')
        report = dict(device=torch.cuda.get_device_name(), synthetic_fixture=True, real_server_data_verified=False,
            optimizer_steps=0, native_tracking_steps=manifest['native_tracking_steps'], records=manifest['records'],
            all_self_replays_exact=True, all_samepeak_bboxes_exact=True, all_alpha0_outputs_and_bboxes_exact=True,
            all_model_states_unchanged=True, production_plugin_matches_audited_trajectory=True)
        path = PROJECT / 'output/analysis/vdrm_rscore_gpu_smoke.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2), encoding='utf-8')
        print('RSCORE NATIVE CUDA SMOKE PASSED:', path)


if __name__ == '__main__':
    main()
