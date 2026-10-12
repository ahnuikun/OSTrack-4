"""Full ViT-B Step01 -> Step08 -> Step09, actual loaders and CLI entries."""

import json
from pathlib import Path
import shutil
import sys
from unittest.mock import patch

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT));sys.path.insert(0,str(PROJECT/'tests'))

import smoke_vdrm_module1_learning as learning_smoke
from tracking import diagnose_vdrm_module1_fit as entry
from lib.test.analysis.module1_fit_audit import check_bundle


def main():
    original=learning_smoke.step08.main
    reports=[]
    def learning_then_fit():
        original()
        argv=sys.argv
        def arg(flag):return argv[argv.index(flag)+1]
        root=Path(arg('--save_dir'));output=root/'fit.tar.gz'
        with patch.object(sys,'argv',['diagnose_vdrm_module1_fit.py','--save_dir',str(root),
                '--learning_dir',arg('--output_dir'),'--source',arg('--source'),'--output',str(output),'--smoke']):
            entry.main()
        summary,cases,rows=check_bundle(output)
        assert len(cases)==14 and summary['parameter_updates']==0
        assert sum(c['oracle_eligible'] for c in cases)>0
        assert any(r.get('trace') for r in rows)
        shutil.copyfile(output,PROJECT/'output/analysis/module1_step09_cuda_evidence.tar.gz')
        reports.append(dict(actual_full_vit_b=True,real_loader=True,actual_cli=True,synthetic_data=True,
            synthetic_regression_head_fixed=True,cases=len(cases),rows=len(rows),parameter_updates=0,
            learned_step08_head_parity=True,tracking_performance_passed=False))
    with patch.object(learning_smoke.step08,'main',side_effect=learning_then_fit):learning_smoke.main()
    path=PROJECT/'output/analysis/module1_step09_cuda_smoke.json'
    path.write_text(json.dumps(reports[0],indent=2),encoding='utf-8')
    print('STEP09 FULL VIT-B CUDA INTEGRATION PASSED:',path)


if __name__=='__main__':main()
