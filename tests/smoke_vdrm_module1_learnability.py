"""Full ViT-B, actual checkpoint loader and Step11 CLI on synthetic images."""

import json
from pathlib import Path
import shutil
import sys
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT)); sys.path.insert(0, str(PROJECT / 'tests'))
import smoke_vdrm_module1_fit as fit_smoke
from tracking import diagnose_vdrm_module1_learnability as entry
from lib.test.analysis.module1_learnability import check_bundle


def main():
    original = fit_smoke.entry.main; reports = []

    def fit_then_learnability():
        original(); argv = sys.argv

        def arg(flag):
            return argv[argv.index(flag) + 1]

        output = Path(arg('--save_dir')) / 'learnability.tar.gz'
        with patch.object(sys, 'argv', ['diagnose_vdrm_module1_learnability.py', '--save_dir', arg('--save_dir'),
                                       '--learning_dir', arg('--learning_dir'), '--fit', arg('--output'),
                                       '--output', str(output), '--smoke']):
            entry.main()
        summary, cases, rows = check_bundle(output)
        assert len(cases) == 2 and len(rows) == len(cases) * 11
        assert summary['optimizer_updates'] == 2 * (2 * len(cases) + 2)
        assert not summary['tracking_performance_passed']
        shutil.copyfile(output, PROJECT / 'output/analysis/module1_step11_cuda_evidence.tar.gz')
        reports.append(dict(full_vit_b=True, actual_cli=True, real_loader=True, synthetic_data=True,
                            synthetic_regression_head_fixed=True, source_controls_exact=True,
                            final_weights_replayed=True, two_frame_shared_gradient_tested=True, **summary))

    # Give the synthetic source two oracle-eligible training frames so the real
    # Step11 shared arm traverses two actual suffix graphs in each update.
    def two_training_oracles(rows, smoke):
        assert smoke
        return [i for i, r in enumerate(rows) if r['good_cells']][:2]

    with patch.object(fit_smoke.entry, 'main', side_effect=fit_then_learnability), \
            patch.object(fit_smoke.entry, 'oracle_selection', side_effect=two_training_oracles):
        fit_smoke.main()
    path = PROJECT / 'output/analysis/module1_step11_cuda_smoke.json'
    path.write_text(json.dumps(reports[0], indent=2), encoding='utf-8')
    print('STEP11 FULL VIT-B CUDA INTEGRATION PASSED:', path)


if __name__ == '__main__':
    main()
