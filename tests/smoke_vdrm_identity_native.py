"""Native ViT-B/CUDA, 152 synthetic sequences, no real-data training."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile

import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT)); sys.path.insert(0, str(PROJECT/'tests'))
from lib.models.ostrack import build_ostrack
from lib.test.analysis.identity_native import check_bundle
from tracking.probe_vdrm_identity_native import run
from test_vdrm_identity_native import native_fixture


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for native smoke')
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory(prefix='identity_native_cuda_') as temporary:
        root = Path(temporary)
        with contextlib.redirect_stdout(io.StringIO()):
            screen, data, _ = native_fixture(root, factory=build_ostrack)
        summary = run(screen, data, root, root/'native', 'cuda')
        manifest, checked = check_bundle(root/'native/identity_native_results.tar.gz')
        if checked != summary or manifest['states_before'] != manifest['states_after']:
            raise RuntimeError('native GPU probe recomputation differs')
        report = dict(synthetic_fixture=True, real_server_tclean_tested=False,
            model='native ViT-B/CE/CENTER and unmodified OSTrack.track', device=torch.cuda.get_device_name(),
            sequences=summary['sequences'], native_tracking_steps=manifest['native_tracking_steps'],
            optimizer_steps_during_probe=0, exact_native_decode=True, states_unchanged=True,
            native_replay_sequences=len(manifest['sequences']), cpu_metric_check=True)
        path = PROJECT/'output/analysis/identity_native_cuda_smoke.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2), encoding='utf-8')
        print('IDENTITY NATIVE CUDA SMOKE PASSED:', path)


if __name__ == '__main__':
    main()
