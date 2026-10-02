"""Exercise the complete seven-arm GOT-10k dev write/read/report contract."""

import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from lib.test.evaluation.running import _save_tracker_output
from tracking import analyze_vdrm_module1_dev as analysis


class VDRMModule1AnalysisTest(unittest.TestCase):
    def test_seven_arm_end_to_end_analysis(self):
        arms = ('b0', 'tclean', 'ronly', 'alpha0', 'qflat',
                'routeflat', 'clip20')
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            settings = SimpleNamespace(
                result_plot_path=str(root / 'plots')
            )
            boxes = np.array([[10, 10, 20, 20], [12, 12, 20, 20]])
            dataset = [
                SimpleNamespace(
                    dataset='got10k',
                    name=f'GOT-10k_Train_{index:06d}',
                    ground_truth_rect=boxes,
                    target_visible=None,
                )
                for index in range(152)
            ]
            trackers = {}
            for arm_index, arm in enumerate(arms):
                tracker = SimpleNamespace(
                    results_dir=str(root / 'results' / arm),
                    name='ostrack', parameter_name=arm,
                    run_id=None, display_name=arm,
                )
                trackers[arm] = tracker
                for sequence in dataset:
                    predicted = boxes.copy()
                    predicted[1, 0] += arm_index
                    _save_tracker_output(
                        sequence, tracker,
                        {'target_bbox': predicted.tolist(),
                         'time': [0.01, 0.01]},
                    )

            def trackerlist(*, parameter_name, **_):
                return [trackers[parameter_name]]

            argv = [
                'analyze_vdrm_module1_dev.py', '--tracker_params',
                *arms, '--reference', 'ronly', '--per_sequence',
            ]
            with patch.object(sys, 'argv', argv), \
                 patch.object(analysis, 'get_dataset', return_value=dataset), \
                 patch.object(analysis, 'trackerlist', side_effect=trackerlist), \
                 patch.object(analysis, 'env_settings', return_value=settings), \
                 patch('lib.test.analysis.plot_results.env_settings',
                       return_value=settings), \
                 patch('lib.test.analysis.extract_results.env_settings',
                       return_value=settings):
                analysis.main()

            cache_path = root / 'plots' / 'vdrm_module1_dev' / 'eval_data.pkl'
            self.assertTrue(cache_path.is_file())
            with cache_path.open('rb') as file:
                cache = pickle.load(file)
            self.assertEqual(len(cache['sequences']), 152)
            self.assertEqual(len(cache['trackers']), 7)
            self.assertTrue(all(cache['valid_sequence']))


if __name__ == '__main__':
    unittest.main()
