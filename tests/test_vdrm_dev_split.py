import unittest
import tempfile
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from lib.test.analysis.extract_results import extract_results, tracking_result_path
from lib.test.evaluation.running import _save_tracker_output
from lib.test.evaluation.got10kdataset import _vdrm_dev_sequence_ids


class VDRMDevSplitTest(unittest.TestCase):
    def test_extract_results_reads_got10k_runner_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            results_dir = root / 'results'
            boxes = np.array([[10, 10, 20, 20], [12, 12, 20, 20]])
            tracker = SimpleNamespace(
                results_dir=str(results_dir), name='ostrack',
                parameter_name='dev', run_id=None, display_name='dev',
            )
            sequence = SimpleNamespace(
                dataset='got10k', name='GOT-10k_Train_000035',
                ground_truth_rect=boxes, target_visible=None,
            )
            _save_tracker_output(
                sequence, tracker,
                {'target_bbox': boxes.tolist(), 'time': [0.01, 0.01]},
            )
            self.assertTrue(Path(tracking_result_path(tracker, sequence)).is_file())
            settings = SimpleNamespace(result_plot_path=str(root / 'plots'))
            with patch('lib.test.analysis.extract_results.env_settings',
                       return_value=settings):
                result = extract_results([tracker], [sequence], 'dev')
            self.assertEqual(result['valid_sequence'], [1])
            self.assertAlmostEqual(result['avg_overlap_all'][0][0], 1.0)

    def test_result_path_matches_runner_dataset_subdirectory(self):
        tracker = SimpleNamespace(results_dir=str(Path('/results') / 'ostrack'))
        got = SimpleNamespace(dataset='got10k', name='GOT-10k_Train_000035')
        uav = SimpleNamespace(dataset='uavdt', name='S0701')
        self.assertEqual(
            tracking_result_path(tracker, got),
            str(Path('/results') / 'ostrack' / 'got10k'
                / 'GOT-10k_Train_000035.txt'),
        )
        self.assertEqual(
            tracking_result_path(tracker, uav),
            str(Path('/results') / 'ostrack' / 'S0701.txt'),
        )

    def test_fixed_dev_ids_are_152_unique_and_sorted(self):
        ids = _vdrm_dev_sequence_ids()
        self.assertEqual(len(ids), 152)
        self.assertEqual(ids, sorted(set(ids)))


if __name__ == '__main__':
    unittest.main()
