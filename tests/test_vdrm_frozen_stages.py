import contextlib
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from lib.test.analysis.frozen_vdrm_stages import visibility_protocol, recovery_delay, summarize_stages


class VisibilityStageTest(unittest.TestCase):
    def test_stage_masks_and_recovery_use_gt_not_predictions(self):
        cover = np.array([8, 8, 0, 0, 0] + [8] * 10 + [2, 7])
        masks, events = visibility_protocol(cover, np.zeros_like(cover))
        self.assertFalse(masks['normal'][0])
        self.assertEqual(masks['reappearance'].sum(), 10)
        self.assertEqual(masks['partial'].sum(), 2)
        self.assertEqual(masks['severe_partial'].sum(), 1)
        self.assertEqual(len(events), 1)
        iou = np.zeros(len(cover))
        iou[7:12] = .7
        self.assertEqual(recovery_delay(iou, events[0]), 2)
        self.assertIsNone(recovery_delay(np.zeros(len(cover)), events[0]))

    def test_absence_counts_as_invisible_even_if_cover_positive(self):
        cover = np.full(12, 8)
        absence = np.array([0, 1, 1, 1] + [0] * 8)
        masks, events = visibility_protocol(cover, absence)
        self.assertFalse(masks['normal'][2])
        self.assertEqual(events[0]['start'], 4)

    def test_right_censored_reappearance_has_no_recovery_event(self):
        cover = np.array([8, 0, 0, 0, 8, 8, 8])
        _, events = visibility_protocol(cover, np.zeros_like(cover))
        self.assertEqual(events, [])

    def test_stage_report_reads_same_nested_tracker_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / 'raw'
            raw.mkdir()
            cover = np.array([8, 8, 0, 0, 0] + [8] * 8 + [2, 7])
            np.savetxt(raw / 'cover.label', cover)
            np.savetxt(raw / 'absence.label', np.zeros_like(cover))
            gt = np.tile([20., 20., 20., 20.], (len(cover), 1))
            sequence = SimpleNamespace(name='GOT-10k_Train_000035', dataset='got10k',
                frames=[str(raw / (str(i) + '.jpg')) for i in range(len(gt))], ground_truth_rect=gt)
            trackers = []
            for name in ('Tclean', 'Rdisc'):
                result_dir = root / name / 'got10k'
                result_dir.mkdir(parents=True)
                pred = gt.copy()
                pred[2:5, 0] += 100
                if name == 'Rdisc': pred[5:7, 0] += 100
                np.savetxt(result_dir / (sequence.name + '.txt'), pred)
                trackers.append(SimpleNamespace(parameter_name=name, results_dir=str(result_dir.parent)))
            with contextlib.redirect_stdout(io.StringIO()):
                result = summarize_stages(trackers, [sequence], 'Tclean', root / 'report')
            self.assertEqual(result['recovery']['Rdisc']['paired_delay_delta_frames'], 2.)
            self.assertEqual(result['recovery']['Rdisc']['recovered_events'], 1)
            self.assertEqual(result['stages']['normal']['eligible_sequences'], 1)
            self.assertTrue((root / 'report/visibility_stages.json').is_file())


if __name__ == '__main__':
    unittest.main()
