"""Native CUDA frozen-model guard collector smoke; synthetic fixture only."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import torch

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT));sys.path.insert(0,str(PROJECT/'tests'))
from lib.models.ostrack import build_ostrack
from lib.test.analysis.identity_guard_bundle import check_bundle
from test_vdrm_identity_guard import guard_fixture
from tracking.screen_vdrm_identity_guard import run


def main():
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory(prefix='identity_guard_cuda_') as tmp:
        root=Path(tmp)
        with contextlib.redirect_stdout(io.StringIO()):screen,data,_,_=guard_fixture(root,factory=build_ostrack)
        result=run(screen,data,root,root/'guard','cuda')
        manifest,repeated=check_bundle(root/'guard/identity_guard_results.tar.gz')
        if result!=repeated:raise RuntimeError('CUDA result differs from CPU verification')
        report=dict(synthetic_fixture=True,real_server_weights_tested=False,device=torch.cuda.get_device_name(),
            sequences=len(manifest['sequences']),native_steps=manifest['native_tracking_steps'],
            probe_optimizer_steps=0,states_unchanged=manifest['states_before']==manifest['states_after'],
            cpu_decision_recomputed=True)
        path=PROJECT/'output/analysis/identity_guard_cuda_smoke.json'
        path.write_text(json.dumps(report,indent=2),encoding='utf-8')
        print('IDENTITY GUARD CUDA SMOKE PASSED:',path)


if __name__=='__main__':main()
