"""Actual ViT-B CUDA paths and packages, synthetic weights/images only."""
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
from lib.test.analysis.search_coverage import check_bundle
from test_vdrm_identity_guard import guard_fixture
from tracking.screen_vdrm_identity_guard import run as guard_run
from tracking.probe_vdrm_search_coverage import run


def main():
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory(prefix='search_coverage_cuda_') as tmp:
        root=Path(tmp)
        with contextlib.redirect_stdout(io.StringIO()):
            screen,data,_,_=guard_fixture(root,factory=build_ostrack)
            guard_run(screen,data,root,root/'guard','cuda')
        result=run(root/'guard/identity_guard_results.tar.gz',data,root,root/'coverage','cuda')
        manifest,repeated=check_bundle(root/'coverage/search_coverage_results.tar.gz')
        if result!=repeated:raise RuntimeError('CUDA evidence differs from CPU recomputation')
        report=dict(synthetic_fixture=True,real_server_weights_tested=False,device=torch.cuda.get_device_name(),
            sequences=len(manifest['sequences']),source_history_steps=manifest['source_history_steps'],
            replayed_native_steps=manifest['replayed_native_steps'],
            extra_crop_forwards=manifest['extra_crop_forwards'],optimizer_steps=0,
            states_unchanged=manifest['state_before']==manifest['state_after'],
            source_center_exact=True,cpu_recomputed=True)
        path=PROJECT/'output/analysis/search_coverage_cuda_smoke.json'
        path.write_text(json.dumps(report,indent=2),encoding='utf-8')
        print('SEARCH COVERAGE CUDA SMOKE PASSED:',path)


if __name__=='__main__':main()
