"""CUDA integration through real checkpoint loader, run/CLI and CPU checker.

Synthetic ViT-B weights/images in an isolated temporary directory. No server
data or performance claim. Run with: python tests/smoke_vdrm_module1_gates.py
"""

from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / 'tests'))

import numpy as np
import torch

from lib.models.ostrack import build_ostrack
from lib.test.analysis import module1_evidence as evidence
from lib.test.analysis.module1_gates import check_bundle, protocol_plan, read_source
from test_vdrm_frozen_failure_diagnostics import synthetic_fixture
from tracking import diagnose_vdrm_module1_evidence as step01
from tracking import diagnose_vdrm_module1_gates as step03
from tracking import analyze_vdrm_module1_gates as analysis
from tracking.diagnose_vdrm_native_float import NativeTracker, prepare_cases, read_rgb
from tracking.diagnose_vdrm_frozen_failures import config_for, load_models, state_hash
from lib.test.analysis.frozen_failure_diagnostics import CONFIGS


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    torch.set_num_threads(2)
    short_plan = protocol_plan(True)
    count = sum(stop-start+1 for start, stop in short_plan.values())
    with tempfile.TemporaryDirectory(prefix='module1_gates_') as temporary:
        root = Path(temporary)
        dataset = synthetic_fixture(root, factory=build_ostrack)
        source_path = root / 'source.tar.gz'
        output = root / 'gates.tar.gz'
        # Only external dataset and registered source duration are shortened.
        # Both stage loaders build and strictly load real full ViT-B checkpoints.
        with patch.object(evidence, 'PLAN', short_plan), patch.object(evidence, 'RECORDS', count), \
                patch.object(evidence, 'STEPS', count*3), patch.object(step01, 'PLAN', short_plan), \
                patch.object(step01, 'RECORDS', count), patch.object(step01, 'STEPS', count*3), \
                patch.object(step01, 'get_dataset', return_value=dataset), \
                patch.object(step03, 'get_dataset', return_value=dataset):
            step01.run(root, source_path)
            torch.cuda.empty_cache()
            with patch.object(sys, 'argv', ['diagnose_vdrm_module1_gates.py', '--save_dir', str(root),
                    '--source', str(source_path), '--output', str(output), '--smoke']):
                step03.main()
            summary = analysis.run(output, root / 'analysis.tar.gz')
            assert summary['native_steps'] == 50 and summary['forward_calls'] == 180
            source = read_source(source_path)
            reference = step03.make_reference(source, short_plan)
        paths = {a: root / 'checkpoints/train/ostrack' / CONFIGS[a] / 'OSTrack_ep0300.pth.tar' for a in step03.ARMS}
        models, _, _ = load_models(paths, 'cuda', arms=step03.ARMS)
        cfg = config_for(CONFIGS['Rdisc'])
        cases = prepare_cases(dataset, plan=short_plan)[:1]
        before = {a: state_hash(m) for a, m in models.items()}
        original = step03.collect(cases, models, cfg, reference)
        rows = json.loads(original['rows.json'])
        sequence, gt, cover, absence, start, stop = cases[0]
        changed = gt.copy()
        changed[1:] = [1., 1., 12., 12.]
        changed_reference = deepcopy(reference)
        for r in changed_reference['frames']:
            if r['sequence'] == sequence.name:
                r.update(gt_xywh=changed[r['frame_1based']-1].tolist(), cover=0., absence=1.)
        altered = step03.collect([(sequence, changed, np.zeros_like(cover), np.ones_like(absence), start, stop)],
                                 models, cfg, changed_reference)
        altered_rows = json.loads(altered['rows.json'])
        assert [r['bbox_xywh'] for r in rows] == [r['bbox_xywh'] for r in altered_rows]
        assert [r['input_sha256'] for r in rows] == [r['input_sha256'] for r in altered_rows]
        assert before == {a: state_hash(m) for a, m in models.items()}
        with torch.no_grad():
            tracker = NativeTracker(models['Rdisc'], cfg)
            tracker.initialize(read_rgb(sequence.frames[0]), dict(init_bbox=gt[0].tolist()))
            boxes = [tracker.track(read_rgb(path))['target_bbox'] for path in sequence.frames[1:stop]]
        assert boxes == [r['bbox_xywh'] for r in rows if r['scope'] == 'closed' and r['mode'] == 'control']
        checked, _ = check_bundle(output)
        report = dict(real_checkpoint_loader=True, full_vit_b=True, synthetic_fixture=True,
            server_data_verified=False, entry_smoke_and_analysis_passed=True, current_gt_does_not_change_predictions=True,
            instrumentation_matches_native=True, model_state_unchanged=True, optimizer_steps=0,
            gpu=torch.cuda.get_device_name(), max_decode_error_pixels=checked['max_decode_error_pixels'])
    result = PROJECT / 'output/analysis/module1_step03_cuda_smoke.json'
    result.parent.mkdir(parents=True, exist_ok=True)
    result.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('STEP03 FULL VIT-B CUDA INTEGRATION PASSED:', result)


if __name__ == '__main__':
    main()
