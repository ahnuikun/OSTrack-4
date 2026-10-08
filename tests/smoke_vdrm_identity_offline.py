"""Native CUDA feature extraction + three small heads; wholly synthetic inputs."""
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT));sys.path.insert(0,str(PROJECT/'tests'))

from lib.models.ostrack import build_ostrack
from lib.models.layers.candidate_quality import dense_center_boxes
from lib.train.frozen_vdrm import load_base_config,file_hash
from lib.test.analysis.identity_offline import check_bundle
from tracking.diagnose_vdrm_frozen_failures import state_hash
from tracking.cache_vdrm_identity import collect_pair
from tracking.train_vdrm_identity import run
from test_vdrm_identity_offline import synthetic_cache
from lib.train.identity_offline import load_protocol


def main():
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    torch.set_num_threads(2);torch.manual_seed(42)
    model=build_ostrack(load_base_config(),training=False).requires_grad_(False).cuda().eval()
    with torch.no_grad():model.backbone.vdrm.alpha.zero_()
    before=state_hash(model)
    template=torch.randn(1,3,128,128,device='cuda');search=torch.randn(1,3,256,256,device='cuda')
    zbox=torch.tensor([[.25,.25,.5,.5]],device='cuda')
    from lib.utils.ce_utils import generate_mask_cond
    mask=generate_mask_cond(load_base_config(),1,torch.device('cuda'),zbox)
    with torch.no_grad():
        output=model(template=template,search=search,template_bbox=zbox,ce_template_mask=mask)
        boxes=dense_center_boxes(output['size_map'],output['offset_map'])
        pair=None
        for box in boxes[0]:
            # A synthetic label chosen to exercise native feature/box handling;
            # this is NOT ground truth or evidence of tracking performance.
            target=torch.cat((box[:2]-box[2:]/2,box[2:]))[None]
            pair,_=collect_pair(output,zbox,target,torch.zeros(1,128,128,device='cuda',dtype=torch.bool),
                torch.zeros(1,256,256,device='cuda',dtype=torch.bool),load_protocol())
            if pair is not None:break
        if pair is None:raise RuntimeError('synthetic native output lacks a supported pair')
    if state_hash(model)!=before or any(p.grad is not None for p in model.parameters()):
        raise RuntimeError('native frozen state changed')
    feature=pair[0];dim=feature['template'].shape[-1]
    del model,output;torch.cuda.empty_cache()
    with tempfile.TemporaryDirectory(prefix='identity_native_') as temporary:
        root=Path(temporary);index,protocol=synthetic_cache(root/'cache')
        n=len(index['rows']);rng=np.random.default_rng(42)
        # Repeat real CUDA-derived descriptor shapes with synthetic perturbations.
        arrays={k:np.repeat(v[None],n,axis=0) for k,v in feature.items()}
        for k in ('template','candidates'):
            arrays[k]+=rng.normal(0,.05,arrays[k].shape).astype(np.float32)
        np.savez_compressed(root/'cache/pairs.npz',**arrays)
        index['feature_dim']=dim;index['members_sha256']['pairs.npz']=file_hash(root/'cache/pairs.npz')
        (root/'cache/index.json').write_text(json.dumps(index),encoding='utf-8')
        run(root/'cache',root/'screen','cuda')
        _,summary=check_bundle(root/'screen/identity_offline_results.tar.gz')
        report=dict(synthetic_fixture=True,real_server_weights_tested=False,device=torch.cuda.get_device_name(),
            feature_dim=dim,parameters_per_head=32*dim,frozen_backbone_unchanged=True,cache_optimizer_steps=0,
            cached_head_epochs=20,trained_arms=['mean_rank','mean_id','spatial_id'],
            independent_cpu_metric_check=True,development_pairs=summary['development_pairs'])
        path=PROJECT/'output/analysis/vdrm_identity_offline_gpu_smoke.json'
        path.write_text(json.dumps(report,indent=2),encoding='utf-8')
        print('IDENTITY OFFLINE NATIVE CUDA SMOKE PASSED:',path)


if __name__=='__main__':main()
