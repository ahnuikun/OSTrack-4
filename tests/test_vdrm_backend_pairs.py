import math
import unittest

import torch

from tracking.vdrm_diagnostics.backend_pair_metrics import (
    bbox_iou_xywh,
    binary_auc,
    mix_box,
    pair_status,
    residual_spatial_statistics,
    response_statistics,
    summarize_pair_rows,
)
from tracking.vdrm_diagnostics.diagnose_vdrm_backend_pairs import (
    assert_backend_compatibility,
    load_config,
    map_box_back_from_anchor,
)


class VDRMBackendPairMetricsTest(unittest.TestCase):
    def test_registered_configs_are_isolated_and_pair_compatible(self):
        _, _, baseline = load_config(
            "vitb_256_mae_ce_32x4_ep300_fulltn"
        )
        _, _, vdrm = load_config(
            "vitb_256_mae_ce_vdrm_v3_hncp_32x4_ep300"
        )

        assert_backend_compatibility(baseline, vdrm)
        self.assertFalse(baseline.MODEL.VDRM.ENABLED)
        self.assertTrue(vdrm.MODEL.VDRM.ENABLED)
        self.assertIsNot(baseline, vdrm)
        self.assertIsNot(baseline.MODEL, vdrm.MODEL)

    def test_v8_part_aligned_config_is_registered(self):
        _, _, v8 = load_config(
            "vitb_256_mae_ce_vdrm_v8_par_hncp_32x4_ep300"
        )

        self.assertTrue(v8.MODEL.VDRM.ENABLED)
        self.assertEqual(
            v8.MODEL.VDRM.SPATIAL_GATE_MODE, "part_aligned"
        )
        self.assertEqual(v8.MODEL.VDRM.ALPHA_MAX, 1.5)
        self.assertEqual(v8.TRAIN.VDRM_CANDIDATE_WEIGHT, 0.0)
        self.assertEqual(v8.TRAIN.VDRM_PART_ROUTE_WEIGHT, 0.1)

    def test_v9_candidate_consistent_part_routing_config_is_registered(self):
        _, _, v9 = load_config(
            "vitb_256_mae_ce_vdrm_v9_pacr_hncp_32x4_ep300"
        )

        self.assertTrue(v9.MODEL.VDRM.ENABLED)
        self.assertEqual(
            v9.MODEL.VDRM.SPATIAL_GATE_MODE,
            "part_aligned_consensus",
        )
        self.assertEqual(v9.MODEL.VDRM.CANDIDATE_LOCAL_RADIUS, 1)
        self.assertEqual(v9.MODEL.VDRM.CANDIDATE_CONSENSUS_PARTS, 3)
        self.assertEqual(v9.MODEL.VDRM.ALPHA_MAX, 1.5)
        self.assertEqual(v9.TRAIN.VDRM_CANDIDATE_WEIGHT, 0.1)
        self.assertEqual(v9.TRAIN.VDRM_PART_ROUTE_WEIGHT, 0.1)

    def test_v10_residual_preserving_guidance_config_is_registered(self):
        _, _, v10 = load_config(
            "vitb_256_mae_ce_vdrm_v10_rpcg_hncp_32x4_ep300"
        )

        self.assertTrue(v10.MODEL.VDRM.ENABLED)
        self.assertEqual(
            v10.MODEL.VDRM.SPATIAL_GATE_MODE,
            "part_aligned_guidance",
        )
        self.assertEqual(v10.MODEL.VDRM.CANDIDATE_LOCAL_RADIUS, 1)
        self.assertEqual(v10.MODEL.VDRM.CANDIDATE_CONSENSUS_PARTS, 3)
        self.assertEqual(v10.MODEL.VDRM.CANDIDATE_MODULATION_MAX, 0.5)
        self.assertEqual(v10.MODEL.VDRM.ALPHA_MAX, 1.5)
        self.assertEqual(v10.TRAIN.VDRM_CANDIDATE_WEIGHT, 0.1)
        self.assertEqual(v10.TRAIN.VDRM_PART_ROUTE_WEIGHT, 0.1)

    def test_v11_background_suppressed_route_config_is_registered(self):
        _, _, v11 = load_config(
            "vitb_256_mae_ce_vdrm_v11_bspar_hncp_32x4_ep300"
        )

        self.assertTrue(v11.MODEL.VDRM.ENABLED)
        self.assertEqual(
            v11.MODEL.VDRM.SPATIAL_GATE_MODE,
            "part_aligned_sharpened",
        )
        self.assertEqual(
            v11.MODEL.VDRM.PART_ROUTE_RESIDUAL_FLOOR, 0.25
        )
        self.assertEqual(v11.MODEL.VDRM.ALPHA_MAX, 1.5)
        self.assertEqual(v11.TRAIN.VDRM_CANDIDATE_WEIGHT, 0.0)
        self.assertEqual(v11.TRAIN.VDRM_PART_ROUTE_WEIGHT, 0.1)

    def test_v12_positive_preserved_route_config_is_registered(self):
        _, _, v12 = load_config(
            "vitb_256_mae_ce_vdrm_v12_pepr_hncp_32x4_ep300"
        )

        self.assertTrue(v12.MODEL.VDRM.ENABLED)
        self.assertEqual(
            v12.MODEL.VDRM.SPATIAL_GATE_MODE,
            "part_aligned_positive_preserved",
        )
        self.assertEqual(
            v12.MODEL.VDRM.PART_ROUTE_RESIDUAL_FLOOR, 0.25
        )
        self.assertEqual(v12.MODEL.VDRM.ALPHA_MAX, 1.5)
        self.assertEqual(v12.MODEL.PRETRAIN_FILE, "mae_pretrain_vit_base.pth")
        self.assertEqual(v12.TRAIN.VDRM_CANDIDATE_WEIGHT, 0.0)
        self.assertEqual(v12.TRAIN.VDRM_PART_ROUTE_WEIGHT, 0.1)

    def test_v13_reliability_safe_route_config_is_registered(self):
        _, _, v13 = load_config(
            "vitb_256_mae_ce_vdrm_v13_rmps_hncp_32x4_ep300"
        )

        self.assertTrue(v13.MODEL.VDRM.ENABLED)
        self.assertEqual(
            v13.MODEL.VDRM.SPATIAL_GATE_MODE,
            "part_aligned_reliability_safe",
        )
        self.assertEqual(
            v13.MODEL.VDRM.PART_ROUTE_RESIDUAL_FLOOR, 0.25
        )
        self.assertEqual(v13.MODEL.VDRM.ALPHA_MAX, 1.5)
        self.assertEqual(
            v13.MODEL.PRETRAIN_FILE, "mae_pretrain_vit_base.pth"
        )
        self.assertEqual(v13.TRAIN.VDRM_CANDIDATE_WEIGHT, 0.0)
        self.assertEqual(v13.TRAIN.VDRM_PART_ROUTE_WEIGHT, 0.1)

    def test_v15_tail_bounded_v8_config_is_registered(self):
        _, _, v15 = load_config(
            "vitb_256_mae_ce_vdrm_v15_tbr_hncp_32x4_ep300"
        )

        self.assertTrue(v15.MODEL.VDRM.ENABLED)
        self.assertEqual(v15.MODEL.VDRM.SPATIAL_GATE_MODE, "part_aligned")
        self.assertEqual(v15.MODEL.VDRM.RESIDUAL_MAX_RATIO, 0.35)
        self.assertEqual(v15.MODEL.VDRM.ALPHA_MAX, 1.5)
        self.assertEqual(v15.MODEL.PRETRAIN_FILE, "mae_pretrain_vit_base.pth")
        self.assertEqual(v15.TRAIN.VDRM_CANDIDATE_WEIGHT, 0.0)
        self.assertEqual(v15.TRAIN.VDRM_PART_ROUTE_WEIGHT, 0.1)

    def test_v16_mass_conserving_bidirectional_config_is_registered(self):
        _, _, v16 = load_config(
            "vitb_256_mae_ce_vdrm_v16_mcba_hncp_32x4_ep300"
        )

        self.assertTrue(v16.MODEL.VDRM.ENABLED)
        self.assertEqual(
            v16.MODEL.VDRM.SPATIAL_GATE_MODE,
            "part_aligned_bidirectional",
        )
        self.assertEqual(v16.MODEL.VDRM.RESIDUAL_MAX_RATIO, 0.0)
        self.assertEqual(v16.MODEL.VDRM.ALPHA_MAX, 1.5)
        self.assertEqual(v16.MODEL.PRETRAIN_FILE, "mae_pretrain_vit_base.pth")
        self.assertEqual(v16.TRAIN.VDRM_CANDIDATE_WEIGHT, 0.0)
        self.assertEqual(v16.TRAIN.VDRM_PART_ROUTE_WEIGHT, 0.1)

    def test_map_box_back_uses_the_shared_anchor(self):
        mapped = map_box_back_from_anchor(
            predicted_box=[50.0, 50.0, 20.0, 10.0],
            anchor_bbox=[40.0, 30.0, 20.0, 20.0],
            search_size=100.0,
            resize_factor=1.0,
        )

        self.assertEqual(mapped, [40.0, 35.0, 20.0, 10.0])

    def test_mixed_boxes_separate_center_and_size(self):
        baseline = [0.0, 0.0, 4.0, 4.0]
        vdrm = [1.0, 1.0, 2.0, 2.0]

        vdrm_center_baseline_size = mix_box(vdrm, baseline)
        baseline_center_vdrm_size = mix_box(baseline, vdrm)

        self.assertEqual(vdrm_center_baseline_size, [0.0, 0.0, 4.0, 4.0])
        self.assertEqual(baseline_center_vdrm_size, [1.0, 1.0, 2.0, 2.0])
        self.assertEqual(bbox_iou_xywh(baseline, baseline), 1.0)

    def test_response_statistics_suppress_first_peak_neighborhood(self):
        response = torch.zeros(3, 3)
        response[0, 0] = 0.9
        response[0, 1] = 0.8
        response[2, 2] = 0.45

        metrics = response_statistics(response, nms_radius=1)

        self.assertAlmostEqual(metrics["p1"], 0.9, places=6)
        self.assertAlmostEqual(metrics["p2"], 0.45, places=6)
        self.assertEqual((metrics["p1_x"], metrics["p1_y"]), (0, 0))
        self.assertEqual((metrics["p2_x"], metrics["p2_y"]), (2, 2))
        self.assertGreaterEqual(metrics["entropy_normalized"], 0.0)
        self.assertLessEqual(metrics["entropy_normalized"], 1.0)

    def test_residual_statistics_measure_spatial_concentration(self):
        input_tokens = torch.ones(1, 6, 4)
        output_tokens = input_tokens.clone()
        output_tokens[:, 2, :] += 1.0

        metrics = residual_spatial_statistics(
            input_tokens, output_tokens, template_length=2
        )

        self.assertAlmostEqual(
            metrics["residual_active_token_fraction"], 0.25
        )
        self.assertAlmostEqual(
            metrics["residual_top10_energy_fraction"], 1.0
        )
        self.assertAlmostEqual(
            metrics["residual_spatial_entropy_normalized"], 0.0
        )
        self.assertGreater(metrics["residual_relative_norm_max"], 0.0)

    def test_pair_status_uses_registered_delta_threshold(self):
        self.assertEqual(pair_status(0.06), "vdrm_better")
        self.assertEqual(pair_status(-0.06), "vdrm_worse")
        self.assertEqual(pair_status(0.01), "similar")
        self.assertEqual(pair_status(math.nan), "invalid_gt")

    def test_binary_auc_rewards_correct_reliability_order(self):
        self.assertEqual(binary_auc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]), 1.0)
        self.assertEqual(binary_auc([0.1, 0.2, 0.8, 0.9], [1, 1, 0, 0]), 0.0)

    def test_summary_compares_raw_response_and_combined_reliability(self):
        rows = []
        for index, (delta, iou, reliability) in enumerate(
            [
                (0.20, 0.80, 0.90),
                (0.10, 0.70, 0.80),
                (-0.10, 0.05, 0.20),
                (-0.20, 0.02, 0.10),
            ]
        ):
            rows.append(
                {
                    "frame_index": index,
                    "baseline_iou": iou - delta,
                    "vdrm_iou": iou,
                    "delta_iou": delta,
                    "vdrm_status": "correct" if iou >= 0.5 else "failed",
                    "pair_status": pair_status(delta),
                    "center_only_delta_iou": 0.5 * delta,
                    "size_only_delta_iou": 0.25 * delta,
                    "backend_center_shift_normalized": abs(delta),
                    "log_width_ratio": delta,
                    "log_height_ratio": delta,
                    "response_peak_shift_normalized": abs(delta),
                    "residual_active_token_fraction": 1.0,
                    "residual_spatial_entropy_normalized": 0.9,
                    "part_assignment_total_variation": 0.15,
                    "part_assignment_entropy": 0.8,
                    "part_assignment_mass_error_max": 1e-7,
                    "candidate_modulation": 0.2,
                    "candidate_modulation_factor_min": 0.9,
                    "candidate_modulation_factor_max": 1.1,
                    "part_route_residual_retention_mean": 0.6,
                    "part_route_residual_retention_min": 0.3,
                    "part_route_residual_retention_max": 0.9,
                    "part_route_positive_preservation_scale_mean": 1.2,
                    "part_route_positive_preservation_scale_min": 1.0,
                    "part_route_positive_preservation_scale_max": 1.4,
                    "part_route_positive_mass_ratio": 1.0,
                    "part_route_positive_part_fraction": 0.75,
                    "part_reliability_safety_factor_mean": 0.8,
                    "part_reliability_safety_factor_min": 0.4,
                    "part_reliability_safety_factor_max": 1.0,
                    "part_reliability_suppressed_fraction": 0.5,
                    "part_reliability_suppression_mean": 0.2,
                    "visual_reliability": reliability,
                    "vdrm_response_reliability": reliability,
                    "combined_reliability": reliability,
                }
            )

        summary = summarize_pair_rows(rows)

        self.assertEqual(summary["vdrm_better_frames"], 2)
        self.assertEqual(summary["vdrm_worse_frames"], 2)
        for metric in (
            "visual_reliability",
            "vdrm_response_reliability",
            "combined_reliability",
        ):
            self.assertEqual(
                summary["reliability"][metric]["correct_vs_failed_auc"],
                1.0,
            )
            self.assertEqual(
                summary["reliability"][metric][
                    "vdrm_better_vs_worse_auc"
                ],
                1.0,
            )

        residual = summary["residual"][
            "residual_active_token_fraction"
        ]
        self.assertEqual(residual["mean"], 1.0)
        self.assertEqual(residual["correct_mean"], 1.0)
        self.assertEqual(residual["failed_mean"], 1.0)
        self.assertIsNone(residual["spearman_with_vdrm_iou"])
        self.assertEqual(
            summary["mean_residual_active_token_fraction"], 1.0
        )
        self.assertEqual(
            summary["mean_part_assignment_total_variation"], 0.15
        )
        self.assertEqual(summary["mean_part_assignment_entropy"], 0.8)
        self.assertEqual(
            summary["mean_part_assignment_mass_error_max"], 1e-7
        )
        self.assertEqual(
            summary["candidate_guidance"]["candidate_modulation"]["mean"],
            0.2,
        )
        self.assertEqual(
            summary["mean_candidate_modulation_factor_min"], 0.9
        )
        self.assertEqual(
            summary["mean_candidate_modulation_factor_max"], 1.1
        )
        self.assertEqual(
            summary["route_safeguard"][
                "part_route_residual_retention_mean"
            ]["mean"],
            0.6,
        )
        self.assertEqual(
            summary["mean_part_route_residual_retention_min"], 0.3
        )
        self.assertEqual(
            summary["mean_part_route_residual_retention_max"], 0.9
        )
        self.assertEqual(
            summary[
                "mean_part_route_positive_preservation_scale_mean"
            ],
            1.2,
        )
        self.assertEqual(
            summary[
                "mean_part_route_positive_preservation_scale_min"
            ],
            1.0,
        )
        self.assertEqual(
            summary[
                "mean_part_route_positive_preservation_scale_max"
            ],
            1.4,
        )
        self.assertEqual(
            summary["mean_part_route_positive_mass_ratio"], 1.0
        )
        self.assertEqual(
            summary["mean_part_route_positive_part_fraction"], 0.75
        )
        self.assertEqual(
            summary["mean_part_reliability_safety_factor_mean"], 0.8
        )
        self.assertEqual(
            summary["mean_part_reliability_safety_factor_min"], 0.4
        )
        self.assertEqual(
            summary["mean_part_reliability_safety_factor_max"], 1.0
        )
        self.assertEqual(
            summary["mean_part_reliability_suppressed_fraction"], 0.5
        )
        self.assertEqual(
            summary["mean_part_reliability_suppression_mean"], 0.2
        )


if __name__ == "__main__":
    unittest.main()
