import unittest
from copy import deepcopy
from pathlib import Path

from lib.config.ostrack.config import cfg, update_config_from_file
from lib.train.base_functions import validate_vdrm_experiment_contract


CONFIG_DIR = Path(__file__).resolve().parents[1] / 'experiments' / 'ostrack'
RON = 'vitb_256_mae_ce_vdrm_v8_ronly_s42_32x4_ep300'


def load_config(name):
    value = deepcopy(cfg)
    update_config_from_file(CONFIG_DIR / f'{name}.yaml', value)
    return value


class VDRMModule1AblationConfigTest(unittest.TestCase):
    def test_probe_configs_match_ronly_inference_contract(self):
        baseline = load_config(RON)
        probes = {
            'qflat': ('part_mean', 0.0),
            'routeflat': ('route_spatial_mean', 0.0),
            'clip20': (None, 0.2),
        }
        for suffix, (ablation, clip_ratio) in probes.items():
            with self.subTest(suffix=suffix):
                name = f'vitb_256_mae_ce_vdrm_v8_ronly_{suffix}_s42_32x4_ep300'
                candidate = load_config(name)
                expected_model = deepcopy(baseline.MODEL)
                expected_model.VDRM.RESIDUAL_MAX_RATIO = clip_ratio
                self.assertEqual(candidate.MODEL, expected_model)
                self.assertEqual(candidate.TRAIN.DROP_PATH_RATE,
                                 baseline.TRAIN.DROP_PATH_RATE)
                self.assertEqual(candidate.DATA.MEAN, baseline.DATA.MEAN)
                self.assertEqual(candidate.DATA.STD, baseline.DATA.STD)
                for field in ('EPOCH', 'SEARCH_FACTOR', 'SEARCH_SIZE',
                              'TEMPLATE_FACTOR', 'TEMPLATE_SIZE'):
                    self.assertEqual(candidate.TEST[field], baseline.TEST[field])
                self.assertEqual(candidate.TEST.CHECKPOINT_CONFIG, RON)
                self.assertEqual(candidate.TEST.VDRM_INFERENCE_ABLATION,
                                 ablation)
                self.assertEqual(candidate.TRAIN.VDRM_EXPERIMENT_ARM,
                                 'inference_only')
                with self.assertRaisesRegex(ValueError, 'VDRM_EXPERIMENT_ARM'):
                    validate_vdrm_experiment_contract(candidate, actual_seed=42)


if __name__ == '__main__':
    unittest.main()
