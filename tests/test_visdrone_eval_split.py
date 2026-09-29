import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from lib.test.evaluation.datasets import load_dataset
from lib.test.evaluation.simple_sotdataset import VisDroneSOTDataset
from lib.test.evaluation.running import run_sequence
from tracking import test_uav_suite


class VisDroneEvaluationSplitTest(unittest.TestCase):
    def test_registry_and_default_load_only_test_sequences(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for split in ('train', 'test'):
                name = split + '_sequence'
                frames = root / split / 'sequences' / name
                annotations = root / split / 'annotations'
                frames.mkdir(parents=True)
                annotations.mkdir(parents=True)
                (frames / '0001.jpg').touch()
                (annotations / (name + '.txt')).write_text(
                    '1,2,3,4\n', encoding='utf-8'
                )

            settings = SimpleNamespace(visdrone_path=str(root))
            with patch(
                'lib.test.evaluation.data.env_settings',
                return_value=settings,
            ):
                registered = load_dataset('visdrone')
                default = VisDroneSOTDataset().get_sequence_list()

            for sequences in (registered, default):
                self.assertEqual(len(sequences), 1)
                self.assertEqual(sequences[0].name, 'test_sequence')
                self.assertEqual(
                    Path(sequences[0].frames[0]),
                    root / 'test' / 'sequences' / 'test_sequence' / '0001.jpg',
                )

    def test_force_reruns_existing_result(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            (Path(temp_dir) / 'test_sequence.txt').write_text(
                '1\t2\t3\t4\n', encoding='utf-8'
            )
            sequence = SimpleNamespace(
                dataset='visdrone', name='test_sequence', object_ids=None
            )
            tracker = SimpleNamespace(
                results_dir=temp_dir,
                name='ostrack',
                parameter_name='test_config',
                run_id=None,
            )
            tracker.run_sequence = Mock(return_value={'time': [1.0]})

            with patch('lib.test.evaluation.running._save_tracker_output'):
                run_sequence(sequence, tracker, force=False)
                tracker.run_sequence.assert_not_called()
                run_sequence(sequence, tracker, force=True)
                tracker.run_sequence.assert_called_once()

    def test_suite_forwards_force_to_runner(self):
        argv = [
            'tracking/test_uav_suite.py',
            '--tracker_param', 'test_config',
            '--dataset', 'visdrone',
            '--force',
        ]
        with patch.object(sys, 'argv', argv), patch(
            'tracking.test_uav_suite.run_tracker'
        ) as run_tracker:
            test_uav_suite.main()

        self.assertEqual(run_tracker.call_count, 1)
        self.assertTrue(run_tracker.call_args.kwargs['force'])


if __name__ == '__main__':
    unittest.main()
