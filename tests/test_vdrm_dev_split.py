import unittest

from lib.test.evaluation.got10kdataset import _vdrm_dev_sequence_ids


class VDRMDevSplitTest(unittest.TestCase):
    def test_fixed_dev_ids_are_152_unique_and_sorted(self):
        ids = _vdrm_dev_sequence_ids()
        self.assertEqual(len(ids), 152)
        self.assertEqual(ids, sorted(set(ids)))


if __name__ == '__main__':
    unittest.main()
