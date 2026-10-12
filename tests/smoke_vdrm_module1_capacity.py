"""Actual full ViT-B loader and width128 task backward through two frames."""
import json
from pathlib import Path
import shutil
import sys
from unittest.mock import patch
PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT));sys.path.insert(0,str(PROJECT/'tests'))
import smoke_vdrm_module1_learnability as prior
from tracking import verify_vdrm_module1_capacity as entry


def main():
    original=prior.entry.main;reports=[]
    def learn_then_fix():
        original();argv=sys.argv
        def arg(flag):return argv[argv.index(flag)+1]
        output=Path(arg('--save_dir'))/'capacity.tar.gz'
        with patch.object(sys,'argv',['verify_vdrm_module1_capacity.py','--save_dir',arg('--save_dir'),
                                     '--source',arg('--output'),'--output',str(output),'--smoke']):entry.main()
        summary=entry.check_bundle(output)
        assert summary['cases']==2 and summary['optimizer_updates']==2
        assert not summary['tracking_performance_passed']
        shutil.copyfile(output,PROJECT/'output/analysis/module1_capacity_cuda_evidence.tar.gz')
        reports.append(dict(actual_cli=True,real_loader=True,full_vit_b=True,synthetic_images=True,
                            synthetic_regression_head_fixed=True,old_width32_control_exact=True,**summary))
    with patch.object(prior.entry,'main',side_effect=learn_then_fix):prior.main()
    (PROJECT/'output/analysis/module1_capacity_cuda_smoke.json').write_text(json.dumps(reports[0],indent=2))
    print('CAPACITY FIX CUDA INTEGRATION PASSED')


if __name__=='__main__':main()
