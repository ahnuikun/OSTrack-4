"""Actual Step01/08/09/10 pipeline, real loaders, full ViT-B on synthetic data."""

import json
from pathlib import Path
import shutil
import sys
from unittest.mock import patch

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT));sys.path.insert(0,str(PROJECT/'tests'))
import smoke_vdrm_module1_fit as fit_smoke
from tracking import train_vdrm_module1_reweight as entry
from lib.test.analysis.module1_reweight import confirmation_ids,check_bundle


def main():
    original=fit_smoke.entry.main;reports=[]
    def fit_then_reweight():
        original();argv=sys.argv
        def arg(flag):return argv[argv.index(flag)+1]
        root=Path(arg('--save_dir'));dataset=root/'data/got10k/train'
        for identity in confirmation_ids(True):
            shutil.copytree(dataset/'GOT-10k_Train_003240',dataset/f'GOT-10k_Train_{identity+1:06d}')
        output=root/'reweight'
        with patch.object(sys,'argv',['train_vdrm_module1_reweight.py','--save_dir',str(root),'--data_root',str(dataset),
                '--learning_dir',arg('--learning_dir'),'--source',arg('--source'),'--fit',arg('--output'),
                '--output_dir',str(output),'--smoke']):entry.main()
        summary,rows=check_bundle(output/'module1_reweight.tar.gz')
        assert summary['training_updates']==4 and len(rows)==(7+7+5*2)*8
        assert summary['max_decode_error_pixels']<.002
        shutil.copyfile(output/'module1_reweight.tar.gz',PROJECT/'output/analysis/module1_step10_cuda_evidence.tar.gz')
        reports.append(dict(full_vit_b=True,real_loader=True,actual_cli=True,synthetic_data=True,synthetic_regression_head_fixed=True,
            training_updates=4,native_rows=len(rows),fresh_confirmation_sequences=1,source_control_exact=True,tracking_performance_passed=False))
    with patch.object(fit_smoke.entry,'main',side_effect=fit_then_reweight):fit_smoke.main()
    path=PROJECT/'output/analysis/module1_step10_cuda_smoke.json';path.write_text(json.dumps(reports[0],indent=2),encoding='utf-8')
    print('STEP10 FULL VIT-B CUDA INTEGRATION PASSED:',path)


if __name__=='__main__':main()
