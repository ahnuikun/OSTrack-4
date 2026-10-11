"""Actual full ViT-B checkpoint/cache/training/native-audit CLI integration."""

import json
from pathlib import Path
import shutil
import sys
import tempfile
from unittest.mock import patch

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT));sys.path.insert(0,str(PROJECT/'tests'))

import numpy as np
import torch

from lib.test.analysis import module1_evidence as evidence
from lib.test.analysis.module1_gates import protocol_plan
from lib.test.analysis.module1_learning import sequence_partition,check_bundle
from test_vdrm_frozen_failure_diagnostics import synthetic_fixture
from smoke_vdrm_module1_direction import fixture_model
from tracking import diagnose_vdrm_module1_evidence as step01
from tracking import diagnose_vdrm_module1_direction as step06
from tracking import diagnose_vdrm_module1_numerics as step07
from tracking import train_vdrm_module1_learning as step08


def main():
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    torch.set_num_threads(2);plan=protocol_plan(True);count=sum(b-a+1 for a,b in plan.values())
    with tempfile.TemporaryDirectory(prefix='module1_learning_') as tmp:
        root=Path(tmp);dataset=synthetic_fixture(root,factory=fixture_model)
        data=root/'data/got10k/train'
        for sequence in dataset[:5]:np.savetxt(data/sequence.name/'groundtruth.txt',sequence.ground_truth_rect,delimiter=',')
        names=[f'GOT-10k_Train_{i:06d}' for i in range(1,9336)]
        (data/'list.txt').write_text('\n'.join(names)+'\n',encoding='utf-8')
        partition,_=sequence_partition(True)
        for identity in partition['train']+partition['audit']:
            folder=data/names[identity]
            shutil.copytree(data/dataset[0].name,folder)
        source,direction,numerics=root/'source.tar.gz',root/'direction.tar.gz',root/'numerics.tar.gz'
        with patch.object(evidence,'PLAN',plan),patch.object(evidence,'RECORDS',count),patch.object(evidence,'STEPS',count*3), \
                patch.object(step01,'PLAN',plan),patch.object(step01,'RECORDS',count),patch.object(step01,'STEPS',count*3), \
                patch.object(step01,'get_dataset',return_value=dataset):
            step01.run(root,source);torch.cuda.empty_cache()
            step06.run(root,source,direction,True);torch.cuda.empty_cache()
            step07.run(root,source,direction,numerics,True);torch.cuda.empty_cache()
            output=root/'learning'
            with patch.object(sys,'argv',['train_vdrm_module1_learning.py','--save_dir',str(root),'--data_root',str(data),
                    '--source',str(source),'--numerics',str(numerics),'--output_dir',str(output),'--smoke']):
                step08.main()
            summary,rows,training=check_bundle(output/'module1_learning.tar.gz')
            assert summary['train_frames']==4 and summary['training_updates_per_arm']==4
            assert len(rows)==(7+5*2)*8
            assert all(t['initial_sha256']!=t['final_sha256'] for t in training.values())
            assert summary['max_decode_error_pixels']<.002
            saved=PROJECT/'output/analysis/module1_step08_cuda_evidence.tar.gz'
            shutil.copyfile(output/'module1_learning.tar.gz',saved)
    result=dict(actual_full_vit_b=True,real_loader=True,actual_cli=True,synthetic_data=True,synthetic_regression_head_fixed=True,
        learning_arms=list(training),train_frames=summary['train_frames'],updates_per_arm=summary['training_updates_per_arm'],
        native_and_shared_rows=len(rows),max_decode_error_pixels=summary['max_decode_error_pixels'],
        frozen_base_unchanged=True,paired_initialization_and_order=True,tracking_performance_passed=False)
    path=PROJECT/'output/analysis/module1_step08_cuda_smoke.json';path.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('STEP08 FULL VIT-B TRAINING INTEGRATION PASSED:',path)


if __name__=='__main__':main()
