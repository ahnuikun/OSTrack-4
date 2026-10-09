"""Regression for the actual 007643 absent-GT failure, without real data writes."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT)); sys.path.insert(0, str(PROJECT/'tests'))
from lib.test.analysis.got10k_annotations import validate_native_annotations
from lib.test.analysis.identity_native import overlaps, sequence_evidence
from lib.test.analysis.identity_guard import grid_record, fit_policy
from tracking import screen_vdrm_identity_guard as guard, probe_vdrm_identity_native as native
from test_vdrm_identity_native import synthetic_outputs


def annotations():
    # Mirror the seven sentinel rows and 1-based frame numbers in real 007643.
    gt = np.tile([10., 20., 30., 40.], (32, 1))
    cover, absence = np.full(32, 8.), np.zeros(32)
    gt[23:30] = -1; cover[23:30] = 0; absence[23:30] = 1
    return gt, cover, absence


class AnnotationTests(unittest.TestCase):
    def test_real_failure_pattern_accepted_without_annotation_mutation(self):
        values = annotations(); before = [v.copy() for v in values]
        masks, _ = validate_native_annotations(*values, 'GOT-10k_Train_007643')
        for value, original in zip(values, before): np.testing.assert_array_equal(value, original)
        for mask in masks.values(): self.assertFalse(mask[23:30].any())
        self.assertTrue(masks['reappearance'][30:].all())

    def test_negative_geometry_is_not_broadly_accepted(self):
        for cover, absence, box in ((8, 0, [-1]*4), (0, 0, [-1]*4),
                                   (1, 1, [-1]*4), (0, 1, [10, 20, -1, -1]),
                                   (0, 1, [-2]*4), (0, 1, [10, 20, 30, -1])):
            with self.subTest(cover=cover, absence=absence, box=box):
                gt, c, a = annotations(); gt[23] = box; c[23] = cover; a[23] = absence
                with self.assertRaisesRegex(ValueError, r'007643.*frames \(1-based\)=\[24\]'):
                    validate_native_annotations(gt, c, a, 'GOT-10k_Train_007643')

    def test_initialization_finiteness_lengths_and_labels_remain_strict(self):
        for first in ([-1]*4, [10, 20, 0, 40]):
            gt, c, a = annotations(); gt[0] = first; c[0] = 0; a[0] = 1
            with self.assertRaisesRegex(ValueError, 'initial box.*=1'):
                validate_native_annotations(gt, c, a, 'fixture')
        gt, c, a = annotations(); gt[23, 0] = np.nan
        with self.assertRaisesRegex(ValueError, r'nonfinite GT.*\[24\]'):
            validate_native_annotations(gt, c, a, 'fixture')
        gt, c, a = annotations()
        with self.assertRaisesRegex(ValueError, 'lengths or shapes'):
            validate_native_annotations(gt, c[:-1], a, 'fixture')
        with self.assertRaisesRegex(ValueError, 'four values'):
            validate_native_annotations(gt[:, :3], c, a, 'fixture')
        c[23] = 9
        with self.assertRaisesRegex(ValueError, 'integer cover'):
            validate_native_annotations(gt, c, a, 'fixture')

    def test_both_planners_keep_every_frame_and_raw_sentinel(self):
        with tempfile.TemporaryDirectory(prefix='identity_annotation_test_') as temporary:
            root = Path(temporary); identity = 7642
            names = [f'GOT-10k_Train_{i+1:06d}' for i in range(identity+1)]
            (root/'list.txt').write_text('\n'.join(names), encoding='utf-8')
            folder = root/names[identity]; folder.mkdir()
            gt, cover, absence = annotations()
            np.savetxt(folder/'groundtruth.txt', gt, delimiter=',')
            np.savetxt(folder/'cover.label', cover); np.savetxt(folder/'absence.label', absence)
            for i in range(len(gt)): (folder/f'{i+1:08d}.jpg').touch()
            source_files = list(folder.iterdir())+[root/'list.txt']
            before = {p: p.read_bytes() for p in source_files}
            with patch.object(native, '_vdrm_dev_sequence_ids', return_value=[identity]):
                planners = (lambda: guard.plan_sequences(root, dict(fit=[identity], audit=[])),
                            lambda: native.plan_sequences(root))
                for planner in planners:
                    plan, = planner()
                    self.assertEqual(len(plan['frames']), 32)
                    for key, expected in (('gt', gt), ('cover', cover), ('absence', absence)):
                        np.testing.assert_array_equal(plan[key], expected)
                self.assertEqual(before, {p: p.read_bytes() for p in source_files})
                # Accepting absent GT must not bypass complete image coverage.
                (folder/'00000024.jpg').rename(folder/'wrong_frame.jpg')
                for planner in planners:
                    with self.assertRaisesRegex(ValueError, 'missing/extra'): planner()

    def test_native_and_guard_metrics_ignore_sentinel_but_keep_trajectory(self):
        reference = synthetic_outputs()
        reference['absence'][3] = 1
        sentinel = synthetic_outputs(absent_sentinel=True)
        before = {k: v.copy() for k, v in sentinel.items()}
        a = sequence_evidence('fixture', reference); b = sequence_evidence('fixture', sentinel)
        self.assertEqual(a, b)
        self.assertEqual(b['counts']['frames'], 4)
        self.assertEqual(b['counts']['visible'], 3)
        self.assertEqual(b['counts']['invisible_or_invalid'], 1)
        fit = grid_record('fixture', 'fit', reference)
        self.assertEqual(fit, grid_record('fixture', 'fit', sentinel))
        choice, _ = fit_policy([fit]*32)
        self.assertEqual(grid_record('fixture', 'audit', reference, selected=choice),
                         grid_record('fixture', 'audit', sentinel, selected=choice))
        for key in sentinel: np.testing.assert_array_equal(sentinel[key], before[key])

    def test_invalid_candidate_geometry_and_float_history_still_rejected(self):
        x = synthetic_outputs(absent_sentinel=True); x['boxes'][2, 0, 2] = -1
        with self.assertRaisesRegex(ValueError, 'box geometry'): sequence_evidence('fixture', x)
        x = synthetic_outputs(absent_sentinel=True); x['anchors'][3] = -1
        with self.assertRaisesRegex(ValueError, 'trajectory'): sequence_evidence('fixture', x)

    def test_invalid_gt_iou_zero_and_valid_iou_bitwise_unchanged(self):
        x = synthetic_outputs(absent_sentinel=True); gt = x['gt'][1:]; boxes = x['boxes']
        end = np.minimum(boxes[..., :2]+boxes[..., 2:], gt[:, None, :2]+gt[:, None, 2:])
        start = np.maximum(boxes[..., :2], gt[:, None, :2])
        area = np.maximum(end-start, 0).prod(-1)
        union = boxes[..., 2:].prod(-1)+gt[:, None, 2:].prod(-1)-area
        old = np.divide(area, union, out=np.zeros_like(area), where=union > 0)
        result = overlaps(boxes, gt); valid = (gt[:, 2:] > 0).all(1)
        np.testing.assert_array_equal(result[valid], old[valid])
        self.assertFalse(result[~valid].any()); self.assertTrue(np.isfinite(result).all())


if __name__ == '__main__': unittest.main()
