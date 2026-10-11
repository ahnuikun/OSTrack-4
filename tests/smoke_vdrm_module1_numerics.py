"""Real full ViT-B/loader/CLI smoke; no fabricated derivative or suffix outputs."""

import json
import shutil
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT));sys.path.insert(0,str(PROJECT/'tests'))

import torch
from lib.test.analysis import module1_evidence as evidence
from lib.test.analysis.module1_gates import protocol_plan
from lib.test.analysis.module1_numerics import check_bundle
from test_vdrm_frozen_failure_diagnostics import synthetic_fixture
from smoke_vdrm_module1_direction import fixture_model
from tracking import diagnose_vdrm_module1_evidence as step01
from tracking import diagnose_vdrm_module1_direction as step06
from tracking import diagnose_vdrm_module1_numerics as step07


def main():
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    torch.set_num_threads(2);plan=protocol_plan(True);count=sum(stop-start+1 for start,stop in plan.values())
    with tempfile.TemporaryDirectory(prefix='module1_numerics_') as temporary:
        root=Path(temporary);dataset=synthetic_fixture(root,factory=fixture_model)
        source,direction,output=root/'source.tar.gz',root/'direction.tar.gz',root/'numerics.tar.gz'
        with patch.object(evidence,'PLAN',plan),patch.object(evidence,'RECORDS',count),patch.object(evidence,'STEPS',count*3), \
                patch.object(step01,'PLAN',plan),patch.object(step01,'RECORDS',count),patch.object(step01,'STEPS',count*3), \
                patch.object(step01,'get_dataset',return_value=dataset):
            step01.run(root,source);torch.cuda.empty_cache()
            step06.run(root,source,direction,True);torch.cuda.empty_cache()
            with patch.object(sys,'argv',['diagnose_vdrm_module1_numerics.py','--save_dir',str(root),'--source',str(source),
                    '--direction',str(direction),'--output',str(output),'--smoke']):
                step07.main()
            summary,frames,diagnostics,gradients,conv=check_bundle(output)
            saved=PROJECT/'output/analysis/module1_step07_cuda_evidence.tar.gz'
            saved.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(output,saved)
            (saved.parent/'module1_step07_cuda_diagnostics.json').write_text(json.dumps(diagnostics,indent=2),encoding='utf-8')
            assert summary['cases']==3 and summary['forward_calls']==429 and summary['backward_calls']==18
            assert summary['native_suffix_parity_passed']
            assert summary['float64_derivative_audit_passed'],[r for r in conv if not r['converged']]
            assert any(r['resolvable'] and r['same_ce'] and r['same_relu'] for r in diagnostics)
            assert output.with_name('numerics_analysis.tar.gz').is_file()
    report=dict(full_vit_b=True,real_checkpoint_loader=True,actual_entry=True,synthetic_fixture=True,
        fixture_regression_head_fixed=True,actual_nonzero_downstream_gradients=True,original_models_unchanged=True,
        backend_restored=True,**summary,gpu=torch.cuda.get_device_name())
    path=PROJECT/'output/analysis/module1_step07_cuda_smoke.json'
    path.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('STEP07 FULL VIT-B CUDA INTEGRATION PASSED:',path)


if __name__=='__main__':main()
