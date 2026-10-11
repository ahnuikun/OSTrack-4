"""Full ViT-B, actual loader and CLI, nonzero insertion gradients on CUDA."""

import json
import math
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT));sys.path.insert(0,str(PROJECT/'tests'))

import torch
from lib.models.ostrack import build_ostrack
from lib.test.analysis import module1_evidence as evidence
from lib.test.analysis.module1_gates import protocol_plan
from lib.test.analysis.module1_direction import check_bundle
from test_vdrm_frozen_failure_diagnostics import synthetic_fixture
from tracking import diagnose_vdrm_module1_evidence as step01
from tracking import diagnose_vdrm_module1_direction as step06


def fixture_model(cfg,training=False):
    model=build_ostrack(cfg,training=training)
    # Give synthetic images meaningful good/low-IoU boxes. Keep the full
    # learned score branch nonzero so real downstream gradients are exercised.
    with torch.no_grad():
        model.box_head.conv5_size.weight.zero_();model.box_head.conv5_size.bias.fill_(math.log(1/3))
        model.box_head.conv5_offset.weight.zero_();model.box_head.conv5_offset.bias.fill_(.5)
    return model


def main():
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    torch.set_num_threads(2);plan=protocol_plan(True);count=sum(stop-start+1 for start,stop in plan.values())
    with tempfile.TemporaryDirectory(prefix='module1_direction_') as temporary:
        root=Path(temporary);dataset=synthetic_fixture(root,factory=fixture_model)
        source,output=root/'source.tar.gz',root/'direction.tar.gz'
        with patch.object(evidence,'PLAN',plan),patch.object(evidence,'RECORDS',count),patch.object(evidence,'STEPS',count*3), \
                patch.object(step01,'PLAN',plan),patch.object(step01,'RECORDS',count),patch.object(step01,'STEPS',count*3), \
                patch.object(step01,'get_dataset',return_value=dataset):
            step01.run(root,source);torch.cuda.empty_cache()
            with patch.object(sys,'argv',['diagnose_vdrm_module1_direction.py','--save_dir',str(root),'--source',str(source),'--output',str(output),'--smoke']):
                step06.main()
            summary,frames,gradients=check_bundle(output)
            assert summary['cases']==10 and summary['eligible_cases']>0 and summary['backward_calls']>0
            assert any(g['gradient_norm']>0 for g in gradients)
            resolved=[g for g in gradients if g['finite_difference_same_ce'] and g['finite_difference_resolvable']]
            assert resolved and all(g['finite_difference_relative_error']<.1 for g in resolved),gradients
            assert output.with_name('direction_analysis.tar.gz').is_file()
    report=dict(full_vit_b=True,real_checkpoint_loader=True,synthetic_fixture=True,
        fixture_regression_head_fixed=True,actual_nonzero_downstream_gradients=True,model_state_unchanged=True,
        optimizer_steps=0,cases=summary['cases'],eligible_cases=summary['eligible_cases'],forward_calls=summary['forward_calls'],
        backward_calls=summary['backward_calls'],finite_difference_checks=summary['finite_difference_checks'],
        max_decode_error_pixels=summary['max_decode_error_pixels'],gpu=torch.cuda.get_device_name())
    path=PROJECT/'output/analysis/module1_step06_cuda_smoke.json'
    path.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('STEP06 FULL VIT-B CUDA INTEGRATION PASSED:',path)


if __name__=='__main__':main()
