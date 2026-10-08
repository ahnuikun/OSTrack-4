"""Full ViT-B/CUDA cue export and independent CPU replay; synthetic weights only."""
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
from lib.test.analysis.candidate_cues import check_bundle
from lib.train.frozen_vdrm import file_hash
from test_vdrm_candidate_quality import quality_fixture
from tracking.diagnose_vdrm_response import run as make_source
from tracking.probe_vdrm_candidate_cues import run as export_cues


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    torch.set_num_threads(2)
    torch.manual_seed(42)
    with tempfile.TemporaryDirectory(prefix='cue_native_') as temporary:
        root = Path(temporary)
        dataset = quality_fixture(root, factory=build_ostrack)
        source, target = root / 'source.tar.gz', root / 'cues.tar.gz'
        with patch('tracking.diagnose_vdrm_response.get_dataset', return_value=dataset):
            source_manifest = make_source(root, source, 'cuda', profile='early_failure', candidate_arm='Rqual')
        original = file_hash(source)
        with patch('tracking.diagnose_vdrm_response.get_dataset', side_effect=AssertionError('dataset read')):
            manifest = export_cues(source, root, target, 'cuda')
        verified, summary, _ = check_bundle(target)
        if manifest != verified or file_hash(source) != original:
            raise RuntimeError('manifest/source mismatch')
        report = dict(device=torch.cuda.get_device_name(), synthetic_fixture=True,
            real_server_rqual_weights_tested=False, optimizer_steps=0,
            native_source_steps=source_manifest['native_tracking_steps'], records=summary['records'],
            template_swap_interventions=summary['groups']['template_swap']['interventions'],
            independent_cpu_recomputation=True, source_bundle_unchanged=True,
            model_state_unchanged=manifest['state_sha256_before'] == manifest['state_sha256_after'],
            maximum_saved_input_replay_error=manifest['maximum_saved_input_replay_error'],
            cue_bundle_bytes=target.stat().st_size)
        output = PROJECT / 'output/analysis/vdrm_candidate_cues_gpu_smoke.json'
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2), encoding='utf-8')
        print('RQUAL CUE NATIVE CUDA SMOKE PASSED:', output)


if __name__ == '__main__':
    main()
