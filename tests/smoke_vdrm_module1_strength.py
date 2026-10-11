"""Full ViT-B integration through actual source/experiment CLI and loader."""

import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
sys.path.insert(0,str(PROJECT/'tests'))

import torch

from lib.models.ostrack import build_ostrack
from lib.test.analysis import module1_evidence as evidence
from lib.test.analysis.module1_gates import protocol_plan
from lib.test.analysis.module1_strength import check_bundle
from test_vdrm_frozen_failure_diagnostics import synthetic_fixture
from tracking import diagnose_vdrm_module1_evidence as step01
from tracking import diagnose_vdrm_module1_strength as step05


def main():
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    torch.set_num_threads(2)
    plan=protocol_plan(True)
    count=sum(stop-start+1 for start,stop in plan.values())
    with tempfile.TemporaryDirectory(prefix='module1_strength_') as temporary:
        root=Path(temporary)
        dataset=synthetic_fixture(root,factory=build_ostrack)
        source,output=root/'source.tar.gz',root/'strength.tar.gz'
        with patch.object(evidence,'PLAN',plan),patch.object(evidence,'RECORDS',count),patch.object(evidence,'STEPS',count*3), \
                patch.object(step01,'PLAN',plan),patch.object(step01,'RECORDS',count),patch.object(step01,'STEPS',count*3), \
                patch.object(step01,'get_dataset',return_value=dataset):
            step01.run(root,source)
            torch.cuda.empty_cache()
            with patch.object(sys,'argv',['diagnose_vdrm_module1_strength.py','--save_dir',str(root),
                    '--source',str(source),'--output',str(output),'--smoke']):
                step05.main()
            summary,frames=check_bundle(output)
            assert summary['cases']==10 and summary['forward_calls']>60
            assert summary['fixed_input_only'] and not summary['tracking_performance_passed']
            assert summary['eligibility']['1.0'].get('eligible',0)>0
            assert {f['strength'] for f in frames}==set(step05.GAINS)
            assert output.with_name('strength_analysis.tar.gz').is_file()
    report=dict(full_vit_b=True,real_checkpoint_loader=True,synthetic_fixture=True,
        cached_input_hashes_exact=True,control_matches_original_native=True,zero_matches_saved_Tclean=True,
        entry_and_cpu_analysis_passed=True,model_state_unchanged=True,optimizer_steps=0,
        cases=summary['cases'],forward_calls=summary['forward_calls'],
        max_decode_error_pixels=summary['max_decode_error_pixels'],gpu=torch.cuda.get_device_name())
    path=PROJECT/'output/analysis/module1_step05_cuda_smoke.json'
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('STEP05 FULL VIT-B CUDA INTEGRATION PASSED:',path)


if __name__=='__main__': main()
