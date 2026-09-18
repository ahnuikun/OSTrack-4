import copy
import inspect
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from lib.config.ostrack.config import cfg, update_config_from_file
from lib.test.evaluation.simple_sotdataset import VisDroneSOTDataset
from tracking import train as training_launcher


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "experiments" / "ostrack"

ABLATION_CONFIGS = {
    "a1": "vitb_256_mae_ce_vdrm_v8_a1_structure_32x4_ep300.yaml",
    "a2": "vitb_256_mae_ce_vdrm_v8_a2_rank_32x4_ep300.yaml",
    "a3": "vitb_256_mae_ce_vdrm_v8_a3_route_32x4_ep300.yaml",
    "a4": "vitb_256_mae_ce_vdrm_v8_a4_rank_route_32x4_ep300.yaml",
    "a5": "vitb_256_mae_ce_vdrm_v8_a5_occlusion_32x4_ep300.yaml",
    "a6": "vitb_256_mae_ce_vdrm_v8_a6_route_visibility_32x4_ep300.yaml",
    "a7": "vitb_256_mae_ce_vdrm_v8_a7_visibility_loss_32x4_ep300.yaml",
    "a8": "vitb_256_mae_ce_vdrm_v8_a8_full_32x4_ep300.yaml",
}

EXPECTED_ABLATIONS = {
    "a1": (0.0, 0.0, 0.0, 0.0, 0.0, False),
    "a2": (0.0, 0.0, 0.5, 0.0, 0.0, False),
    "a3": (0.0, 0.0, 0.0, 0.1, 0.0, False),
    "a4": (0.0, 0.0, 0.5, 0.1, 0.0, False),
    "a5": (0.5, 0.0, 0.5, 0.1, 0.0, False),
    "a6": (0.5, 0.0, 0.5, 0.1, 0.0, True),
    "a7": (0.5, 0.0, 0.5, 0.1, 0.5, True),
    "a8": (0.5, 0.3, 0.5, 0.1, 0.5, True),
}


def load_config(filename):
    loaded = copy.deepcopy(cfg)
    update_config_from_file(str(CONFIG_DIR / filename), base_cfg=loaded)
    return loaded


def to_plain_dict(value):
    if isinstance(value, dict):
        return {key: to_plain_dict(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_plain_dict(item) for item in value]
    return value


class VDRMAblationConfigTest(unittest.TestCase):
    def test_visdrone_defaults_to_test_split(self):
        default = inspect.signature(
            VisDroneSOTDataset.__init__
        ).parameters["split"].default
        self.assertEqual(default, "test")

    def test_a1_to_a8_match_the_registered_matrix(self):
        for name, filename in ABLATION_CONFIGS.items():
            with self.subTest(ablation=name):
                loaded = load_config(filename)
                expected = EXPECTED_ABLATIONS[name]

                self.assertTrue(loaded.MODEL.VDRM.ENABLED)
                self.assertEqual(
                    loaded.MODEL.VDRM.SPATIAL_GATE_MODE, "part_aligned"
                )
                self.assertEqual(loaded.MODEL.VDRM.ALPHA_MAX, 1.5)
                self.assertEqual(loaded.MODEL.VDRM.NUM_PARTS, 4)
                self.assertEqual(loaded.MODEL.VDRM.TOPK, 4)
                self.assertEqual(
                    loaded.DATA.SEARCH.VDRM_OCCLUSION_PROBABILITY,
                    expected[0],
                )
                self.assertEqual(
                    loaded.DATA.SEARCH.VDRM_DISTRACTOR_PROBABILITY,
                    expected[1],
                )
                self.assertEqual(
                    loaded.TRAIN.VDRM_RANK_WEIGHT, expected[2]
                )
                self.assertEqual(
                    loaded.TRAIN.VDRM_PART_ROUTE_WEIGHT, expected[3]
                )
                self.assertEqual(
                    loaded.TRAIN.VDRM_VISIBILITY_WEIGHT, expected[4]
                )
                self.assertEqual(
                    loaded.TRAIN.VDRM_ROUTE_VISIBILITY_WEIGHTED,
                    expected[5],
                )
                self.assertEqual(loaded.TRAIN.VDRM_CANDIDATE_WEIGHT, 0.0)
                self.assertEqual(loaded.TRAIN.EPOCH, 300)
                self.assertEqual(loaded.TRAIN.LR_DROP_EPOCH, 240)
                self.assertEqual(loaded.TRAIN.BATCH_SIZE, 32)
                self.assertEqual(loaded.DATA.TRAIN.SAMPLE_PER_EPOCH, 60000)
                self.assertEqual(
                    loaded.DATA.TRAIN.DATASETS_NAME,
                    ["LASOT", "GOT10K_vottrain", "COCO17", "TRACKINGNET"],
                )
                self.assertEqual(
                    loaded.DATA.TRAIN.DATASETS_RATIO, [1, 1, 1, 1]
                )
                self.assertEqual(loaded.TEST.EPOCH, 300)

    def test_a8_is_behaviorally_equal_to_released_v8(self):
        released_v8 = load_config(
            "vitb_256_mae_ce_vdrm_v8_par_hncp_32x4_ep300.yaml"
        )
        a8 = load_config(ABLATION_CONFIGS["a8"])
        self.assertEqual(to_plain_dict(a8), to_plain_dict(released_v8))

    def test_training_launcher_forwards_the_registered_seed(self):
        argv = [
            "tracking/train.py",
            "--script", "ostrack",
            "--config", ABLATION_CONFIGS["a1"].removesuffix(".yaml"),
            "--save_dir", "./output",
            "--mode", "multiple",
            "--nproc_per_node", "4",
            "--seed", "123",
            "--use_lmdb", "0",
            "--use_wandb", "0",
        ]
        with patch.object(sys, "argv", argv), patch.object(
            training_launcher.random, "randint", return_value=29500
        ), patch.object(training_launcher.os, "system") as run_command:
            training_launcher.main()

        command = run_command.call_args.args[0]
        self.assertIn("torchrun --standalone --nproc_per_node 4", command)
        self.assertIn("--master_port 29500", command)
        self.assertIn("--seed 123", command)
        self.assertIn(
            "--config vitb_256_mae_ce_vdrm_v8_a1_structure_32x4_ep300",
            command,
        )


if __name__ == "__main__":
    unittest.main()
