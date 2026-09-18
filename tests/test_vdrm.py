import unittest

import numpy as np
import torch

from lib.models.layers.vdrm import VisibilityDrivenRepresentationModule
from lib.train.actors.ostrack import (
    build_vdrm_part_route_targets,
    compute_vdrm_candidate_focal_loss,
    compute_vdrm_part_route_loss,
    compute_vdrm_part_rank_loss,
    compute_vdrm_response_rank_loss,
    select_vdrm_part_route_weight,
)
from lib.train.data.sampler import TrackingSampler
from lib.train.data.vdrm_augmentation import (
    apply_same_class_distractor_copy_paste,
    apply_structured_target_occlusion,
)
from lib.train.data.vdrm_paired_diagnostics import (
    compute_condition_metrics,
    create_paired_copy_pastes,
    normalized_center_distance,
)


class VDRMTest(unittest.TestCase):
    class _FakeClassDataset:
        def has_class_info(self):
            return True

        def get_sequences_in_class(self, class_name):
            return [0, 1] if class_name == "car" else []

        def is_video_sequence(self):
            return True

        def get_sequence_info(self, seq_id):
            bbox = torch.tensor([[8.0, 8.0, 16.0, 16.0]]).repeat(20, 1)
            visible = torch.ones(20, dtype=torch.uint8)
            return {"bbox": bbox, "visible": visible, "valid": visible}

        def get_frames(self, seq_id, frame_ids, anno=None):
            frames = [
                np.full((32, 32, 3), seq_id, dtype=np.uint8)
                for _ in frame_ids
            ]
            boxes = [anno["bbox"][frame_id].clone() for frame_id in frame_ids]
            return frames, {"bbox": boxes}, {"object_class_name": "car"}

    def test_zero_initialized_residual_preserves_tokens(self):
        torch.manual_seed(0)
        module = VisibilityDrivenRepresentationModule(
            num_parts=4,
            topk=4,
        )
        tokens = torch.randn(2, 64 + 37, 32)
        template_bbox = torch.tensor(
            [
                [0.25, 0.25, 0.50, 0.50],
                [0.20, 0.20, 0.60, 0.60],
            ]
        )

        output, diagnostics = module(
            tokens,
            template_length=64,
            template_bbox=template_bbox,
        )

        self.assertTrue(torch.equal(output, tokens))
        self.assertEqual(diagnostics["part_reliability"].shape, (2, 4))
        self.assertEqual(diagnostics["part_valid"].shape, (2, 4))
        self.assertEqual(diagnostics["visual_reliability"].shape, (2,))
        self.assertTrue(diagnostics["part_valid"].all())
        self.assertTrue(
            ((diagnostics["visual_reliability"] >= 0.0)
             & (diagnostics["visual_reliability"] <= 1.0)).all()
        )

    def test_gradients_reach_residual_and_reliability_parameters(self):
        torch.manual_seed(1)
        module = VisibilityDrivenRepresentationModule(
            num_parts=4,
            topk=4,
        )
        tokens = torch.randn(2, 64 + 37, 32, requires_grad=True)
        template_bbox = torch.tensor(
            [[0.25, 0.25, 0.50, 0.50]] * 2
        )

        output, diagnostics = module(
            tokens,
            template_length=64,
            template_bbox=template_bbox,
        )
        loss = output.square().mean() + diagnostics[
            "visual_reliability"
        ].mean()
        loss.backward()

        self.assertIsNotNone(module.alpha.grad)
        self.assertIsNotNone(module.log_match_scale.grad)
        self.assertIsNotNone(module.match_bias.grad)
        self.assertTrue(torch.isfinite(module.alpha.grad))
        self.assertTrue(torch.isfinite(module.log_match_scale.grad))
        self.assertTrue(torch.isfinite(module.match_bias.grad))
        self.assertIsNotNone(tokens.grad)
        self.assertTrue(torch.isfinite(tokens.grad).all())

    def test_relative_norm_bound_caps_the_complete_residual_update(self):
        torch.manual_seed(11)
        max_ratio = 0.25
        module = VisibilityDrivenRepresentationModule(
            num_parts=4,
            topk=4,
            residual_max_ratio=max_ratio,
        )
        module.alpha.data.fill_(-20.0)
        tokens = torch.randn(2, 64 + 37, 32)
        template_bbox = torch.tensor(
            [[0.25, 0.25, 0.50, 0.50]] * 2
        )

        output, diagnostics = module(
            tokens,
            template_length=64,
            template_bbox=template_bbox,
        )

        search_input = tokens[:, 64:]
        search_delta = output[:, 64:] - search_input
        relative_norm = (
            torch.linalg.vector_norm(search_delta, dim=-1)
            / torch.linalg.vector_norm(search_input, dim=-1).clamp_min(1e-6)
        )
        self.assertLessEqual(relative_norm.max().item(), max_ratio + 1e-5)
        self.assertGreater(
            diagnostics["vdrm_residual_clip_rate"].item(), 0.0
        )
        self.assertGreater(
            diagnostics["vdrm_raw_delta_relative_norm"].item(),
            diagnostics["vdrm_delta_relative_norm"].item(),
        )

    def test_disabled_relative_norm_bound_preserves_v1_forward(self):
        torch.manual_seed(13)
        default_module = VisibilityDrivenRepresentationModule(
            num_parts=4,
            topk=4,
        )
        explicit_v1_module = VisibilityDrivenRepresentationModule(
            num_parts=4,
            topk=4,
            residual_max_ratio=0.0,
        )
        explicit_v1_module.load_state_dict(default_module.state_dict())
        default_module.alpha.data.fill_(-1.046)
        explicit_v1_module.alpha.data.copy_(default_module.alpha.data)
        tokens = torch.randn(2, 64 + 37, 32)
        template_bbox = torch.tensor(
            [[0.25, 0.25, 0.50, 0.50]] * 2
        )

        default_output, _ = default_module(
            tokens,
            template_length=64,
            template_bbox=template_bbox,
        )
        explicit_output, _ = explicit_v1_module(
            tokens,
            template_length=64,
            template_bbox=template_bbox,
        )

        self.assertTrue(torch.equal(default_output, explicit_output))

    def test_default_mode_preserves_pre_v7_state_dict_schema(self):
        module = VisibilityDrivenRepresentationModule(num_parts=4, topk=4)

        self.assertEqual(
            set(module.state_dict()),
            {"log_match_scale", "match_bias", "alpha"},
        )

    def test_part_aligned_mode_preserves_zero_initialized_forward(self):
        torch.manual_seed(23)
        module = VisibilityDrivenRepresentationModule(
            num_parts=4,
            spatial_gate_mode="part_aligned",
            alpha_max=1.5,
        )
        tokens = torch.randn(2, 64 + 37, 32, requires_grad=True)
        global_index = torch.arange(37).unsqueeze(0).repeat(2, 1)

        output, diagnostics = module(
            tokens,
            template_length=64,
            template_bbox=torch.tensor(
                [[0.25, 0.25, 0.50, 0.50]] * 2
            ),
            search_global_index=global_index,
            search_grid_size=16,
        )

        self.assertTrue(torch.equal(output, tokens))
        self.assertEqual(diagnostics["part_route_logits"].shape, (2, 4, 37))
        self.assertEqual(diagnostics["part_route_gate"].shape, (2, 4, 37))
        self.assertEqual(diagnostics["vdrm_alpha"].item(), 0.0)
        self.assertEqual(
            set(module.state_dict()),
            {
                "log_match_scale",
                "match_bias",
                "part_route_log_match_scale",
                "part_route_match_bias",
                "alpha",
            },
        )

    def test_part_aligned_layerscale_is_bounded(self):
        torch.manual_seed(29)
        module = VisibilityDrivenRepresentationModule(
            num_parts=4,
            spatial_gate_mode="part_aligned",
            alpha_max=1.5,
        )
        module.alpha.data.fill_(-100.0)
        tokens = torch.randn(1, 64 + 37, 16)

        output, diagnostics = module(
            tokens,
            template_length=64,
            search_global_index=torch.arange(37).unsqueeze(0),
            search_grid_size=16,
        )

        self.assertFalse(torch.equal(output, tokens))
        self.assertGreaterEqual(diagnostics["vdrm_alpha"].item(), -1.5)
        self.assertLessEqual(diagnostics["vdrm_alpha"].item(), 1.5)
        self.assertEqual(diagnostics["vdrm_alpha_raw"].item(), -100.0)

    def test_part_route_targets_follow_the_four_target_parts(self):
        targets = build_vdrm_part_route_targets(
            torch.tensor([[0.25, 0.25, 0.50, 0.50]]),
            height=8,
            width=8,
            num_parts=4,
            dilation=0.0,
        )

        self.assertEqual(targets.shape, (1, 4, 8, 8))
        expected_core_tokens = [(2, 2), (2, 4), (4, 2), (4, 4)]
        for part_index, (row, col) in enumerate(expected_core_tokens):
            self.assertEqual(targets[0, part_index, row, col].item(), 1.0)
        self.assertEqual(targets[0, 0, 2, 4].item(), 0.0)
        self.assertEqual(targets[0, 3, 2, 2].item(), 0.0)

    def test_part_route_loss_rewards_corresponding_part_regions(self):
        bbox = torch.tensor([[0.25, 0.25, 0.50, 0.50]])
        targets = build_vdrm_part_route_targets(
            bbox,
            height=8,
            width=8,
            num_parts=4,
            dilation=0.0,
        ).flatten(2)
        good_logits = torch.where(
            targets > 0.0,
            torch.full_like(targets, 3.0),
            torch.full_like(targets, -3.0),
        ).requires_grad_()
        bad_logits = torch.where(
            targets.flip(1) > 0.0,
            torch.full_like(targets, 3.0),
            torch.full_like(targets, -3.0),
        )
        global_index = torch.arange(64).unsqueeze(0)

        good_loss, diagnostics = compute_vdrm_part_route_loss(
            good_logits,
            global_index,
            bbox,
            grid_height=8,
            grid_width=8,
            dilation=0.0,
        )
        bad_loss, _ = compute_vdrm_part_route_loss(
            bad_logits,
            global_index,
            bbox,
            grid_height=8,
            grid_width=8,
            dilation=0.0,
        )

        self.assertLess(good_loss.item(), bad_loss.item())
        self.assertGreater(
            diagnostics["part_route_positive_probability"].item(),
            diagnostics["part_route_background_probability"].item(),
        )
        good_loss.backward()
        self.assertIsNotNone(good_logits.grad)
        self.assertTrue(torch.isfinite(good_logits.grad).all())

    def test_part_route_visibility_weight_can_be_ablated(self):
        visibility_target = torch.tensor([[1.0, 0.8, 0.4, 0.0]])
        pred_dict = {'vdrm_visibility_target': visibility_target}

        self.assertIs(
            select_vdrm_part_route_weight(
                pred_dict, visibility_weighted=True
            ),
            visibility_target,
        )
        self.assertIsNone(
            select_vdrm_part_route_weight(
                pred_dict, visibility_weighted=False
            )
        )
        self.assertIsNone(
            select_vdrm_part_route_weight({}, visibility_weighted=True)
        )

    def test_candidate_consensus_prefers_colocated_multi_part_evidence(self):
        module = VisibilityDrivenRepresentationModule(
            num_parts=4,
            spatial_gate_mode="candidate_consensus",
            candidate_local_radius=1,
            candidate_consensus_parts=2,
        )
        similarity = torch.zeros(1, 4, 25)
        # Two different parts agree around candidate (2, 2).
        similarity[0, 0, 6] = 0.9
        similarity[0, 1, 7] = 0.8
        # A stronger but isolated part appears at the opposite corner.
        similarity[0, 2, 24] = 0.95
        part_reliability = torch.ones(1, 4)
        part_valid = torch.ones(1, 4, dtype=torch.bool)
        global_index = torch.arange(25).unsqueeze(0)

        logits, gate, candidate_map, valid = (
            module._candidate_consensus_statistics(
                similarity,
                part_reliability,
                part_valid,
                global_index,
                search_grid_size=5,
            )
        )

        self.assertEqual(logits.shape, (1, 25))
        self.assertEqual(gate.shape, (1, 25))
        self.assertEqual(candidate_map.shape, (1, 1, 5, 5))
        self.assertTrue(valid.item())
        self.assertGreater(gate[0, 12].item(), gate[0, 24].item() + 0.2)

    def test_candidate_mode_preserves_zero_initialized_forward(self):
        torch.manual_seed(17)
        module = VisibilityDrivenRepresentationModule(
            num_parts=4,
            spatial_gate_mode="candidate_consensus",
        )
        tokens = torch.randn(2, 64 + 37, 32, requires_grad=True)
        global_index = torch.arange(37).unsqueeze(0).repeat(2, 1)

        output, diagnostics = module(
            tokens,
            template_length=64,
            template_bbox=torch.tensor(
                [[0.25, 0.25, 0.50, 0.50]] * 2
            ),
            search_global_index=global_index,
            search_grid_size=16,
        )

        self.assertTrue(torch.equal(output, tokens))
        self.assertEqual(
            diagnostics["candidate_identity_logits"].shape, (2, 37)
        )
        self.assertEqual(
            diagnostics["candidate_reliability_map"].shape,
            (2, 1, 16, 16),
        )
        gaussian_map = torch.zeros(2, 16, 16)
        gaussian_map[:, 1, 1] = 1.0
        candidate_loss = compute_vdrm_candidate_focal_loss(
            diagnostics["candidate_identity_logits"],
            diagnostics["search_global_index"],
            gaussian_map,
            sample_valid=diagnostics["candidate_consensus_valid"],
        )
        candidate_loss.backward()
        self.assertIsNotNone(module.candidate_log_match_scale.grad)
        self.assertIsNotNone(module.candidate_match_bias.grad)
        self.assertTrue(torch.isfinite(tokens.grad).all())

    def test_candidate_focal_loss_rewards_the_target_candidate(self):
        global_index = torch.arange(16).unsqueeze(0)
        gaussian_map = torch.zeros(1, 4, 4)
        gaussian_map[0, 1, 1] = 1.0
        good_logits = torch.full((1, 16), -3.0, requires_grad=True)
        bad_logits = torch.full((1, 16), -3.0)
        with torch.no_grad():
            good_logits[0, 5] = 3.0
            bad_logits[0, 15] = 3.0

        good_loss = compute_vdrm_candidate_focal_loss(
            good_logits, global_index, gaussian_map
        )
        bad_loss = compute_vdrm_candidate_focal_loss(
            bad_logits, global_index, gaussian_map
        )

        self.assertLess(good_loss.item(), bad_loss.item())
        good_loss.backward()
        self.assertIsNotNone(good_logits.grad)
        self.assertTrue(torch.isfinite(good_logits.grad).all())

    def test_margin_reliability_suppresses_the_first_peak_neighborhood(self):
        module = VisibilityDrivenRepresentationModule(
            num_parts=4,
            reliability_mode="margin",
            nms_radius=1,
        )
        similarity = torch.tensor(
            [[[0.90, 0.85, 0.70, 0.80, 0.60]]]
        )
        global_index = torch.tensor([[0, 1, 2, 5, 15]])

        peak, hard_negative, margin = module._margin_statistics(
            similarity,
            global_index,
            search_grid_size=4,
        )

        self.assertTrue(torch.allclose(peak, torch.tensor([[0.90]])))
        # Locations 1 and 5 belong to the 3x3 neighborhood around location 0.
        self.assertTrue(
            torch.allclose(hard_negative, torch.tensor([[0.70]]))
        )
        self.assertTrue(torch.allclose(margin, torch.tensor([[0.20]])))

    def test_margin_forward_preserves_zero_initialized_residual(self):
        torch.manual_seed(3)
        module = VisibilityDrivenRepresentationModule(
            num_parts=4,
            reliability_mode="margin",
            nms_radius=1,
            initial_match_bias=0.0,
        )
        tokens = torch.randn(2, 64 + 37, 32)
        global_index = torch.arange(37).unsqueeze(0).repeat(2, 1)

        output, diagnostics = module(
            tokens,
            template_length=64,
            template_bbox=torch.tensor(
                [[0.25, 0.25, 0.50, 0.50]] * 2
            ),
            search_global_index=global_index,
            search_grid_size=16,
        )

        self.assertTrue(torch.equal(output, tokens))
        self.assertEqual(diagnostics["part_similarity"].shape, (2, 4, 37))
        self.assertEqual(diagnostics["part_match_margin"].shape, (2, 4))
        self.assertTrue((diagnostics["part_match_margin"] >= 0.0).all())

    def test_part_rank_loss_rewards_target_over_background(self):
        good_similarity = torch.tensor(
            [[[0.90, 0.80, 0.10, 0.20],
              [0.70, 0.60, 0.30, 0.10]]],
            requires_grad=True,
        )
        bad_similarity = torch.tensor(
            [[[0.10, 0.20, 0.90, 0.80],
              [0.20, 0.10, 0.70, 0.60]]]
        )
        global_index = torch.tensor([[0, 1, 4, 15]])
        gaussian_map = torch.zeros(1, 4, 4)
        gaussian_map[0, 0, :2] = 1.0
        part_valid = torch.ones(1, 2, dtype=torch.bool)

        good_loss = compute_vdrm_part_rank_loss(
            good_similarity,
            global_index,
            gaussian_map,
            part_valid=part_valid,
            margin=0.1,
        )
        bad_loss = compute_vdrm_part_rank_loss(
            bad_similarity,
            global_index,
            gaussian_map,
            part_valid=part_valid,
            margin=0.1,
        )

        self.assertLess(good_loss.item(), bad_loss.item())
        good_loss.backward()
        self.assertIsNotNone(good_similarity.grad)
        self.assertTrue(torch.isfinite(good_similarity.grad).all())

    def test_response_rank_uses_pasted_distractor_for_applied_samples(self):
        score_logits = torch.zeros(2, 4, 4, requires_grad=True)
        with torch.no_grad():
            score_logits[:, 1, 1] = 2.0
            score_logits[0, 0, 0] = 1.0
            score_logits[:, 3, 3] = 5.0
        gaussian_map = torch.zeros_like(score_logits)
        gaussian_map[:, 1, 1] = 1.0
        distractor_boxes = torch.tensor(
            [
                [0.00, 0.00, 0.25, 0.25],
                [0.00, 0.00, 0.00, 0.00],
            ]
        )
        distractor_applied = torch.tensor([1.0, 0.0])

        loss, diagnostics = compute_vdrm_response_rank_loss(
            score_logits,
            gaussian_map,
            distractor_boxes=distractor_boxes,
            distractor_applied=distractor_applied,
        )
        expected = (
            torch.nn.functional.softplus(torch.tensor(1.0 - 2.0))
            + torch.nn.functional.softplus(torch.tensor(5.0 - 2.0))
        ) / 2.0

        self.assertTrue(torch.allclose(loss, expected))
        self.assertEqual(diagnostics["alignment_success_rate"].item(), 1.0)
        self.assertEqual(diagnostics["distractor_rank_margin"].item(), 1.0)
        loss.backward()
        self.assertNotEqual(score_logits.grad[0, 0, 0].item(), 0.0)
        self.assertEqual(score_logits.grad[0, 3, 3].item(), 0.0)

    def test_response_rank_without_alignment_preserves_global_negative(self):
        score_logits = torch.zeros(1, 4, 4)
        score_logits[0, 1, 1] = 2.0
        score_logits[0, 0, 0] = 1.0
        score_logits[0, 3, 3] = 5.0
        gaussian_map = torch.zeros_like(score_logits)
        gaussian_map[0, 1, 1] = 1.0

        loss, diagnostics = compute_vdrm_response_rank_loss(
            score_logits,
            gaussian_map,
        )

        expected = torch.nn.functional.softplus(torch.tensor(5.0 - 2.0))
        self.assertTrue(torch.allclose(loss, expected))
        self.assertEqual(diagnostics["alignment_success_rate"].item(), 0.0)

    def test_distractor_diagnostics_do_not_replace_global_negative(self):
        score_logits = torch.zeros(1, 4, 4)
        score_logits[0, 1, 1] = 2.0
        score_logits[0, 0, 0] = 1.0
        score_logits[0, 3, 3] = 5.0
        gaussian_map = torch.zeros_like(score_logits)
        gaussian_map[0, 1, 1] = 1.0

        loss, diagnostics = compute_vdrm_response_rank_loss(
            score_logits,
            gaussian_map,
            distractor_boxes=torch.tensor(
                [[0.00, 0.00, 0.25, 0.25]]
            ),
            distractor_applied=torch.tensor([1.0]),
            align_distractor=False,
        )

        expected = torch.nn.functional.softplus(torch.tensor(5.0 - 2.0))
        self.assertTrue(torch.allclose(loss, expected))
        self.assertEqual(
            diagnostics["distractor_hard_hit_rate"].item(), 0.0
        )
        self.assertEqual(diagnostics["distractor_global_gap"].item(), 4.0)
        self.assertEqual(diagnostics["distractor_rank_margin"].item(), 1.0)

    def test_response_rank_maps_tiny_distractor_to_nearest_cell(self):
        score_logits = torch.zeros(1, 4, 4)
        score_logits[0, 2, 2] = 2.0
        score_logits[0, 0, 0] = 1.0
        score_logits[0, 3, 3] = 5.0
        gaussian_map = torch.zeros_like(score_logits)
        gaussian_map[0, 2, 2] = 1.0

        loss, diagnostics = compute_vdrm_response_rank_loss(
            score_logits,
            gaussian_map,
            distractor_boxes=torch.tensor([[0.01, 0.01, 0.01, 0.01]]),
            distractor_applied=torch.tensor([1.0]),
        )

        expected = torch.nn.functional.softplus(torch.tensor(1.0 - 2.0))
        self.assertTrue(torch.allclose(loss, expected))
        self.assertEqual(diagnostics["alignment_success_rate"].item(), 1.0)

    def test_structured_occlusion_returns_soft_part_labels(self):
        torch.manual_seed(2)
        images = torch.ones(2, 3, 32, 32)
        boxes = torch.tensor(
            [
                [0.25, 0.25, 0.50, 0.50],
                [0.20, 0.20, 0.60, 0.60],
            ]
        )

        occluded, visibility, applied = apply_structured_target_occlusion(
            images,
            boxes,
            probability=1.0,
            min_area_ratio=0.3,
            max_area_ratio=0.3,
            part_grid=2,
        )

        self.assertTrue(applied.all())
        self.assertEqual(visibility.shape, (2, 4))
        self.assertTrue((visibility >= 0.0).all())
        self.assertTrue((visibility <= 1.0).all())
        self.assertTrue((visibility < 1.0).any(dim=1).all())
        self.assertGreater((occluded == 0.0).sum().item(), 0)

        unchanged, clean_visibility, clean_applied = (
            apply_structured_target_occlusion(
                images,
                boxes,
                probability=0.0,
                part_grid=2,
            )
        )
        self.assertTrue(torch.equal(unchanged, images))
        self.assertTrue((clean_visibility == 1.0).all())
        self.assertFalse(clean_applied.any())

    def test_same_class_sampler_uses_a_different_instance(self):
        sampler = TrackingSampler(
            datasets=[],
            p_datasets=[],
            samples_per_epoch=1,
            max_gap=10,
            num_search_frames=1,
            same_class_distractor_probability=1.0,
        )
        distractor = sampler._sample_same_class_distractor(
            self._FakeClassDataset(),
            source_seq_id=0,
            class_name="car",
        )

        self.assertIsNotNone(distractor)
        self.assertTrue((distractor["vdrm_distractor_images"][0] == 1).all())
        self.assertIsNone(
            sampler._sample_same_class_distractor(
                self._FakeClassDataset(),
                source_seq_id=0,
                class_name="Unknown",
            )
        )

    def test_same_class_copy_paste_preserves_target_pixels(self):
        torch.manual_seed(7)
        image = torch.zeros(3, 64, 64)
        distractor = torch.ones(3, 64, 64)
        target_box = torch.tensor([0.375, 0.375, 0.25, 0.25])
        distractor_box = torch.tensor([0.25, 0.25, 0.50, 0.50])

        augmented, applied, pasted_box = apply_same_class_distractor_copy_paste(
            image,
            target_box,
            distractor,
            distractor_box,
            min_scale=1.0,
            max_scale=1.0,
            invalid_mask=torch.zeros(64, 64, dtype=torch.bool),
        )

        self.assertTrue(applied)
        self.assertGreater(augmented.count_nonzero().item(), 0)
        self.assertTrue(torch.equal(augmented[:, 24:40, 24:40], image[:, 24:40, 24:40]))
        self.assertTrue((pasted_box >= 0.0).all())
        self.assertTrue((pasted_box <= 1.0).all())
        self.assertGreater(pasted_box[2].item(), 0.0)
        self.assertGreater(pasted_box[3].item(), 0.0)

        paste_x0 = int(round(pasted_box[0].item() * 64))
        paste_y0 = int(round(pasted_box[1].item() * 64))
        paste_x1 = int(round((pasted_box[0] + pasted_box[2]).item() * 64))
        paste_y1 = int(round((pasted_box[1] + pasted_box[3]).item() * 64))
        overlaps_target = not (
            paste_x1 <= 24
            or paste_x0 >= 40
            or paste_y1 <= 24
            or paste_y0 >= 40
        )
        self.assertFalse(overlaps_target)

    def test_nearest_copy_paste_is_no_farther_than_random_placement(self):
        image = torch.zeros(3, 64, 64)
        distractor = torch.ones(3, 64, 64)
        target_box = torch.tensor([0.375, 0.375, 0.25, 0.25])
        distractor_box = torch.tensor([0.25, 0.25, 0.50, 0.50])
        invalid_mask = torch.zeros(64, 64, dtype=torch.bool)

        def normalized_center_distance(pasted_box):
            target_center = target_box[:2] + 0.5 * target_box[2:]
            pasted_center = pasted_box[:2] + 0.5 * pasted_box[2:]
            scale = target_box[2:] + pasted_box[2:]
            return (((pasted_center - target_center) / scale) ** 2).sum()

        found_strict_improvement = False
        for seed in range(8):
            torch.manual_seed(seed)
            _, random_applied, random_box = (
                apply_same_class_distractor_copy_paste(
                    image,
                    target_box,
                    distractor,
                    distractor_box,
                    min_scale=1.0,
                    max_scale=1.0,
                    invalid_mask=invalid_mask,
                    placement_mode="random",
                )
            )
            torch.manual_seed(seed)
            _, nearest_applied, nearest_box = (
                apply_same_class_distractor_copy_paste(
                    image,
                    target_box,
                    distractor,
                    distractor_box,
                    min_scale=1.0,
                    max_scale=1.0,
                    invalid_mask=invalid_mask,
                    placement_mode="nearest",
                )
            )

            self.assertTrue(random_applied and nearest_applied)
            random_distance = normalized_center_distance(random_box)
            nearest_distance = normalized_center_distance(nearest_box)
            self.assertLessEqual(
                nearest_distance.item(), random_distance.item() + 1e-7
            )
            found_strict_improvement |= (
                nearest_distance.item() + 1e-7 < random_distance.item()
            )

        self.assertTrue(found_strict_improvement)

    def test_paired_copy_paste_reuses_scale_and_candidate_sequence(self):
        image = torch.zeros(3, 64, 64)
        distractor = torch.ones(3, 64, 64)
        target_box = torch.tensor([0.375, 0.375, 0.25, 0.25])
        source_box = torch.tensor([0.25, 0.25, 0.50, 0.50])
        invalid_mask = torch.zeros(64, 64, dtype=torch.bool)

        torch.manual_seed(19)
        (
            random_image,
            near_image,
            random_applied,
            near_applied,
            random_box,
            near_box,
        ) = create_paired_copy_pastes(
            image,
            target_box,
            distractor,
            source_box,
            invalid_mask,
            min_scale=0.7,
            max_scale=1.3,
        )

        torch.manual_seed(19)
        expected_random = apply_same_class_distractor_copy_paste(
            image,
            target_box,
            distractor,
            source_box,
            min_scale=0.7,
            max_scale=1.3,
            invalid_mask=invalid_mask,
            placement_mode="random",
        )
        torch.manual_seed(19)
        expected_near = apply_same_class_distractor_copy_paste(
            image,
            target_box,
            distractor,
            source_box,
            min_scale=0.7,
            max_scale=1.3,
            invalid_mask=invalid_mask,
            placement_mode="nearest",
        )

        self.assertTrue(random_applied and near_applied)
        self.assertTrue(expected_random[1] and expected_near[1])
        self.assertTrue(torch.equal(random_image, expected_random[0]))
        self.assertTrue(torch.equal(near_image, expected_near[0]))
        self.assertTrue(torch.equal(random_box, expected_random[2]))
        self.assertTrue(torch.equal(near_box, expected_near[2]))
        self.assertTrue(torch.equal(random_box[2:], near_box[2:]))
        self.assertLessEqual(
            normalized_center_distance(target_box, near_box).item(),
            normalized_center_distance(target_box, random_box).item(),
        )

    def test_paired_response_metrics_report_paste_hard_hit(self):
        logits = torch.zeros(1, 1, 4, 4)
        logits[0, 0, 2, 2] = 4.0
        logits[0, 0, 0, 0] = 5.0
        score_map = logits.sigmoid()
        output = {
            "score_logits": logits,
            "score_map": score_map,
            "pred_boxes": torch.tensor(
                [[[0.375, 0.375, 0.25, 0.25]]]
            ),
            "visual_reliability": torch.tensor([0.8]),
        }
        target_box = torch.tensor([[0.25, 0.25, 0.25, 0.25]])
        paste_box = torch.tensor([[0.0, 0.0, 0.25, 0.25]])

        metrics = compute_condition_metrics(
            output,
            target_box,
            search_size=64,
            stride=16,
            distractor_boxes=paste_box,
        )

        self.assertEqual(metrics["target_logit"].item(), 4.0)
        self.assertEqual(metrics["global_negative_logit"].item(), 5.0)
        self.assertEqual(metrics["paste_logit"].item(), 5.0)
        self.assertEqual(metrics["paste_global_gap"].item(), 0.0)
        self.assertEqual(metrics["paste_hard_hit"].item(), 1.0)
        self.assertAlmostEqual(metrics["pred_iou"].item(), 1.0)


if __name__ == "__main__":
    unittest.main()
