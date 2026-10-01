import unittest
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import torch

from lib.models.layers.vdrm import VisibilityDrivenRepresentationModule
from lib.train.actors.ostrack import (
    OSTrackActor,
    build_vdrm_part_route_targets,
    compute_vdrm_candidate_focal_loss,
    compute_vdrm_part_route_loss,
    compute_vdrm_part_rank_loss,
    compute_vdrm_response_rank_loss,
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
from lib.train.base_functions import validate_vdrm_experiment_contract


class VDRMTest(unittest.TestCase):
    def test_v8_inference_ablations_preserve_checkpoint_and_raw_diagnostics(self):
        torch.manual_seed(420)
        module = VisibilityDrivenRepresentationModule(
            num_parts=4, spatial_gate_mode="part_aligned", alpha_max=1.5,
        ).eval()
        module.alpha.data.fill_(-0.6)
        tokens = torch.randn(2, 64 + 37, 32)
        bbox = torch.tensor([[0.2, 0.2, 0.6, 0.6]] * 2)
        original, original_diag = module(
            tokens, template_length=64, template_bbox=bbox,
        )
        state_keys = set(module.state_dict())
        part_masks, part_valid = module._build_part_masks(bbox, 64)
        part_count = part_masks.sum(dim=-1, keepdim=True).clamp_min(1.0)
        prototypes = torch.einsum(
            "bkl,blc->bkc", part_masks, tokens[:, :64]
        ) / part_count
        prototypes = prototypes * part_valid.unsqueeze(-1)
        valid_count = part_valid.sum(dim=-1, keepdim=True).clamp_min(1)

        for ablation in ("part_mean", "route_spatial_mean"):
            probed = deepcopy(module)
            probed.set_inference_ablation(ablation)
            output, diagnostics = probed(
                tokens, template_length=64, template_bbox=bbox,
            )
            self.assertEqual(set(probed.state_dict()), state_keys)
            self.assertFalse(torch.equal(output, original))
            self.assertTrue(torch.equal(
                diagnostics["part_reliability"],
                original_diag["part_reliability"],
            ))
            self.assertTrue(torch.equal(
                diagnostics["part_route_gate"],
                original_diag["part_route_gate"],
            ))
            self.assertTrue(torch.equal(
                diagnostics["visual_reliability"],
                original_diag["visual_reliability"],
            ))
            route = original_diag["part_route_gate"]
            reliability = original_diag["part_reliability"]
            if ablation == "part_mean":
                reliability = (
                    reliability.sum(dim=-1, keepdim=True)
                    / valid_count
                ) * part_valid
            else:
                route = route.mean(dim=-1, keepdim=True).expand_as(route)
            expected_residual = torch.einsum(
                "bkl,bkc->blc", route * reliability.unsqueeze(-1),
                prototypes,
            ) / valid_count.unsqueeze(-1)
            expected_search = tokens[:, 64:] + (
                diagnostics["vdrm_alpha"] * expected_residual
            )
            torch.testing.assert_close(output[:, 64:], expected_search)

        with self.assertRaisesRegex(ValueError, "inference ablation"):
            module.set_inference_ablation("unknown")

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

    def test_frozen_zero_alpha_is_an_exact_training_identity(self):
        torch.manual_seed(5)
        module = VisibilityDrivenRepresentationModule(
            num_parts=4,
            topk=4,
            spatial_gate_mode="part_aligned",
            alpha_max=1.5,
            train_alpha=False,
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
        output.square().mean().backward()

        self.assertTrue(torch.equal(output, tokens))
        self.assertEqual(diagnostics["vdrm_alpha"].item(), 0.0)
        self.assertFalse(module.alpha.requires_grad)
        self.assertIsNone(module.alpha.grad)

    def test_clean_arm_contract_accepts_only_seed_42_clean_configs(self):
        train = SimpleNamespace(
            VDRM_EXPERIMENT_ARM="tclean",
            VDRM_REQUIRED_SEED=42,
            VDRM_VISIBILITY_WEIGHT=0.0,
            VDRM_RANK_WEIGHT=0.0,
            VDRM_CANDIDATE_WEIGHT=0.0,
            VDRM_PART_ROUTE_WEIGHT=0.0,
        )
        vdrm = SimpleNamespace(ENABLED=True, TRAIN_ALPHA=False)
        cfg = SimpleNamespace(
            TRAIN=train,
            MODEL=SimpleNamespace(VDRM=vdrm),
        )

        validate_vdrm_experiment_contract(cfg, actual_seed=42)

        train.VDRM_EXPERIMENT_ARM = "ronly"
        vdrm.TRAIN_ALPHA = True
        validate_vdrm_experiment_contract(cfg, actual_seed=42)

        with self.assertRaisesRegex(ValueError, "requires seed=42"):
            validate_vdrm_experiment_contract(cfg, actual_seed=7)

        train.VDRM_VISIBILITY_WEIGHT = 0.5
        with self.assertRaisesRegex(ValueError, "auxiliary weight"):
            validate_vdrm_experiment_contract(cfg, actual_seed=42)

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

    def test_v15_tail_bound_preserves_v8_below_bound_and_caps_tail(self):
        torch.manual_seed(59)
        v8 = VisibilityDrivenRepresentationModule(
            num_parts=4,
            spatial_gate_mode="part_aligned",
            alpha_max=1.5,
        )
        v15 = VisibilityDrivenRepresentationModule(
            num_parts=4,
            spatial_gate_mode="part_aligned",
            residual_max_ratio=0.35,
            alpha_max=1.5,
        )
        v15.load_state_dict(v8.state_dict())
        v8.alpha.data.fill_(-20.0)
        v15.alpha.data.copy_(v8.alpha.data)
        template_tokens = torch.randn(2, 64, 32)
        search_tokens = torch.randn(2, 36, 32)
        search_tokens[:, :18] *= 20.0
        search_tokens[:, 18:] *= 0.05
        tokens = torch.cat((template_tokens, search_tokens), dim=1)
        template_bbox = torch.tensor(
            [[0.25, 0.25, 0.50, 0.50]] * 2
        )
        global_index = torch.arange(36).unsqueeze(0).repeat(2, 1)
        kwargs = {
            "template_length": 64,
            "template_bbox": template_bbox,
            "search_global_index": global_index,
            "search_grid_size": 6,
        }

        v8_output, _ = v8(tokens, **kwargs)
        v15_output, diagnostics = v15(tokens, **kwargs)
        input_search = tokens[:, 64:]
        raw_update = v8_output[:, 64:] - input_search
        bounded_update = v15_output[:, 64:] - input_search
        input_norm = torch.linalg.vector_norm(input_search, dim=-1)
        raw_ratio = (
            torch.linalg.vector_norm(raw_update, dim=-1)
            / input_norm.clamp_min(1e-6)
        )
        bounded_ratio = (
            torch.linalg.vector_norm(bounded_update, dim=-1)
            / input_norm.clamp_min(1e-6)
        )
        below_bound = raw_ratio <= 0.35
        above_bound = raw_ratio > 0.35

        self.assertTrue(below_bound.any().item())
        self.assertTrue(above_bound.any().item())
        self.assertTrue(
            torch.equal(
                v15_output[:, 64:][below_bound],
                v8_output[:, 64:][below_bound],
            )
        )
        self.assertLessEqual(bounded_ratio.max().item(), 0.35 + 1e-5)
        self.assertGreater(diagnostics["vdrm_residual_clip_rate"].item(), 0.0)
        self.assertLess(
            diagnostics["vdrm_residual_clip_scale_min"].item(), 1.0
        )
        self.assertLess(
            diagnostics["vdrm_residual_clip_scale_mean"].item(), 1.0
        )
        self.assertEqual(set(v15.state_dict()), set(v8.state_dict()))

    def test_v15_tail_bound_has_finite_gradients_and_zero_alpha_identity(self):
        torch.manual_seed(61)
        module = VisibilityDrivenRepresentationModule(
            num_parts=4,
            spatial_gate_mode="part_aligned",
            residual_max_ratio=0.35,
            alpha_max=1.5,
        )
        tokens = torch.randn(2, 64 + 36, 24, requires_grad=True)
        kwargs = {
            "template_length": 64,
            "template_bbox": torch.tensor(
                [[0.25, 0.25, 0.50, 0.50]] * 2
            ),
            "search_global_index": torch.arange(36).unsqueeze(0).repeat(2, 1),
            "search_grid_size": 6,
        }

        identity_output, identity_diagnostics = module(tokens, **kwargs)
        self.assertTrue(torch.equal(identity_output, tokens))
        self.assertEqual(
            identity_diagnostics["vdrm_residual_clip_scale_mean"].item(),
            1.0,
        )

        module.alpha.data.fill_(-20.0)
        output, diagnostics = module(tokens, **kwargs)
        loss = output[:, 64:].square().mean()
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(tokens.grad).all())
        for parameter in (
            module.alpha,
            module.log_match_scale,
            module.match_bias,
            module.part_route_log_match_scale,
            module.part_route_match_bias,
        ):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertGreaterEqual(
            diagnostics["vdrm_residual_clip_scale_min"].item(), 0.0
        )

    def test_v16_bidirectional_assignment_conserves_v8_route_mass(self):
        module = VisibilityDrivenRepresentationModule(
            num_parts=4,
            spatial_gate_mode="part_aligned_bidirectional",
            alpha_max=1.5,
        )
        route_logits = torch.tensor(
            [
                [
                    [3.0, 1.0, -1.0, -2.0, 0.0],
                    [-1.0, 2.5, 0.5, -2.0, 0.0],
                    [-2.0, -1.0, 3.0, 0.5, 0.0],
                    [0.5, -2.0, -1.0, 3.0, 0.0],
                ],
                [
                    [1.0, 0.0, -1.0, 2.0, -2.0],
                    [0.0, 2.0, -1.0, 1.0, -2.0],
                    [2.0, -1.0, 0.0, 1.0, -2.0],
                    [-1.0, 1.0, 2.0, 0.0, -2.0],
                ],
            ],
            requires_grad=True,
        )
        part_reliability = torch.tensor(
            [[0.9, 0.7, 0.5, 0.3], [0.8, 0.6, 0.4, 0.0]]
        )
        part_valid = torch.tensor(
            [[True, True, True, True], [True, True, True, False]]
        )
        v8_route_weight = (
            route_logits.sigmoid()
            * part_reliability.unsqueeze(-1)
            * part_valid.unsqueeze(-1)
        )

        route_weight, diagnostics = (
            module._mass_conserving_bidirectional_assignment(
                route_logits,
                v8_route_weight,
                part_valid,
            )
        )

        self.assertTrue(
            torch.allclose(
                route_weight.sum(dim=1),
                v8_route_weight.sum(dim=1),
                atol=1e-6,
                rtol=1e-6,
            )
        )
        self.assertTrue(torch.equal(route_weight[1, 3], torch.zeros(5)))
        self.assertLessEqual(
            diagnostics["part_assignment_mass_error_max"].max().item(),
            1e-6,
        )
        self.assertTrue(
            (
                diagnostics["part_assignment_total_variation"] > 0.0
            ).all().item()
        )
        loss = route_weight.square().mean()
        loss.backward()
        self.assertIsNotNone(route_logits.grad)
        self.assertTrue(torch.isfinite(route_logits.grad).all())

        empty_weight, empty_diagnostics = (
            module._mass_conserving_bidirectional_assignment(
                torch.zeros(1, 4, 3),
                torch.zeros(1, 4, 3),
                torch.zeros(1, 4, dtype=torch.bool),
            )
        )
        self.assertTrue(
            torch.equal(empty_weight, torch.zeros_like(empty_weight))
        )
        for value in empty_diagnostics.values():
            self.assertTrue(torch.isfinite(value).all())

    def test_v16_uniform_spatial_evidence_reduces_to_v8_mixture(self):
        module = VisibilityDrivenRepresentationModule(
            num_parts=4,
            spatial_gate_mode="part_aligned_bidirectional",
            alpha_max=1.5,
        )
        route_logits = torch.zeros(2, 4, 7)
        reliability = torch.tensor(
            [[0.2, 0.4, 0.6, 0.8], [0.9, 0.3, 0.7, 0.5]]
        )
        part_valid = torch.ones(2, 4, dtype=torch.bool)
        v8_route_weight = route_logits.sigmoid() * reliability.unsqueeze(-1)

        route_weight, diagnostics = (
            module._mass_conserving_bidirectional_assignment(
                route_logits,
                v8_route_weight,
                part_valid,
            )
        )

        self.assertTrue(
            torch.allclose(
                route_weight, v8_route_weight, atol=1e-6, rtol=1e-6
            )
        )
        self.assertLessEqual(
            diagnostics["part_assignment_total_variation"].max().item(),
            1e-6,
        )

    def test_v16_forward_keeps_schema_identity_and_finite_gradients(self):
        torch.manual_seed(67)
        v8 = VisibilityDrivenRepresentationModule(
            num_parts=4,
            spatial_gate_mode="part_aligned",
            alpha_max=1.5,
        )
        v16 = VisibilityDrivenRepresentationModule(
            num_parts=4,
            spatial_gate_mode="part_aligned_bidirectional",
            alpha_max=1.5,
        )
        v16.load_state_dict(v8.state_dict(), strict=True)
        self.assertEqual(set(v16.state_dict()), set(v8.state_dict()))

        tokens = torch.randn(2, 64 + 31, 24, requires_grad=True)
        kwargs = {
            "template_length": 64,
            "template_bbox": torch.tensor(
                [[0.25, 0.25, 0.50, 0.50]] * 2
            ),
            "search_global_index": torch.arange(31).unsqueeze(0).repeat(2, 1),
            "search_grid_size": 6,
        }

        identity_output, identity_diagnostics = v16(tokens, **kwargs)
        self.assertTrue(torch.equal(identity_output, tokens))
        self.assertLessEqual(
            identity_diagnostics[
                "part_assignment_mass_error_max"
            ].max().item(),
            1e-6,
        )

        v16.alpha.data.fill_(-1.0)
        output, diagnostics = v16(tokens, **kwargs)
        loss = output[:, 64:].square().mean()
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(tokens.grad).all())
        for parameter in v16.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertTrue(
            torch.isfinite(diagnostics["part_assignment_entropy"]).all()
        )
        self.assertLessEqual(
            diagnostics[
                "part_assignment_mass_error_max"
            ].max().item(),
            1e-6,
        )

    def test_v18_coherent_weights_are_positive_normalized_and_detached(self):
        module = VisibilityDrivenRepresentationModule(
            num_parts=1,
            candidate_consensus_parts=1,
            spatial_gate_mode="part_aligned_coherent_prototype",
            alpha_max=1.5,
        )
        template_tokens = torch.tensor(
            [[[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]]],
            requires_grad=True,
        )
        part_masks = torch.ones(1, 1, 3)
        part_valid = torch.ones(1, 1, dtype=torch.bool)

        prototypes, weights, diagnostics = (
            module._identity_coherent_part_prototypes(
                template_tokens, part_masks, part_valid
            )
        )

        self.assertFalse(weights.requires_grad)
        for value in diagnostics.values():
            self.assertFalse(value.requires_grad)
        self.assertTrue((weights > 0.0).all().item())
        self.assertTrue(
            torch.equal(weights.sum(dim=-1), torch.ones(1, 1))
        )
        self.assertLess(weights[0, 0, 2].item(), weights[0, 0, 0].item())
        self.assertEqual(
            weights[0, 0, 0].item(), weights[0, 0, 1].item()
        )

        prototypes.sum().backward()
        expected_gradient = weights.squeeze(1).unsqueeze(-1).expand_as(
            template_tokens
        )
        self.assertTrue(
            torch.equal(template_tokens.grad, expected_gradient)
        )
        for value in diagnostics.values():
            self.assertTrue(torch.isfinite(value).all())

    def test_v18_uniform_and_single_token_parts_are_exact_v8_means(self):
        module = VisibilityDrivenRepresentationModule(
            num_parts=1,
            candidate_consensus_parts=1,
            spatial_gate_mode="part_aligned_coherent_prototype",
            alpha_max=1.5,
        )
        repeated = torch.tensor(
            [[[2.0, -1.0, 0.5]] * 4]
        )
        repeated_mask = torch.ones(1, 1, 4)
        valid = torch.ones(1, 1, dtype=torch.bool)
        repeated_prototype, repeated_weight, diagnostics = (
            module._identity_coherent_part_prototypes(
                repeated, repeated_mask, valid
            )
        )
        v8_repeated = repeated.mean(dim=1, keepdim=True)

        self.assertTrue(torch.equal(repeated_prototype, v8_repeated))
        self.assertTrue(
            torch.equal(
                repeated_weight,
                torch.full_like(repeated_weight, 0.25),
            )
        )
        self.assertEqual(
            diagnostics["prototype_weight_entropy"].item(), 1.0
        )
        self.assertEqual(
            diagnostics["prototype_effective_token_fraction"].item(),
            1.0,
        )

        singleton = torch.tensor([[[0.5, -2.0, 3.0]]])
        singleton_prototype, singleton_weight, _ = (
            module._identity_coherent_part_prototypes(
                singleton,
                torch.ones(1, 1, 1),
                valid,
            )
        )
        self.assertTrue(torch.equal(singleton_prototype, singleton))
        self.assertTrue(
            torch.equal(singleton_weight, torch.ones_like(singleton_weight))
        )

    def test_v18_invalid_part_is_zero_and_finite(self):
        module = VisibilityDrivenRepresentationModule(
            num_parts=1,
            candidate_consensus_parts=1,
            spatial_gate_mode="part_aligned_coherent_prototype",
            alpha_max=1.5,
        )
        prototype, weight, diagnostics = (
            module._identity_coherent_part_prototypes(
                torch.randn(1, 3, 4),
                torch.zeros(1, 1, 3),
                torch.zeros(1, 1, dtype=torch.bool),
            )
        )

        self.assertTrue(torch.equal(prototype, torch.zeros_like(prototype)))
        self.assertTrue(torch.equal(weight, torch.zeros_like(weight)))
        for value in diagnostics.values():
            self.assertTrue(torch.equal(value, torch.zeros_like(value)))

    def test_v18_uniform_forward_is_bit_identical_to_v8(self):
        torch.manual_seed(83)
        v8 = VisibilityDrivenRepresentationModule(
            num_parts=4,
            spatial_gate_mode="part_aligned",
            alpha_max=1.5,
        )
        v18 = VisibilityDrivenRepresentationModule(
            num_parts=4,
            spatial_gate_mode="part_aligned_coherent_prototype",
            alpha_max=1.5,
        )
        v18.load_state_dict(v8.state_dict(), strict=True)
        self.assertEqual(tuple(v18.state_dict()), tuple(v8.state_dict()))
        v8.alpha.data.fill_(-0.75)
        v18.alpha.data.copy_(v8.alpha.data)

        template = torch.empty(1, 8, 8, 8)
        part_values = torch.tensor(
            [
                [1.0, 0.0, 0.5, -0.5, 0.2, 0.3, -0.2, 0.7],
                [0.0, 1.0, -0.5, 0.5, 0.4, -0.3, 0.6, 0.1],
                [0.5, -0.5, 1.0, 0.0, -0.2, 0.7, 0.3, 0.4],
                [-0.5, 0.5, 0.0, 1.0, 0.6, 0.2, 0.1, -0.4],
            ]
        )
        template[:, :4, :4] = part_values[0]
        template[:, :4, 4:] = part_values[1]
        template[:, 4:, :4] = part_values[2]
        template[:, 4:, 4:] = part_values[3]
        tokens = torch.cat(
            (template.reshape(1, 64, 8), torch.randn(1, 36, 8)),
            dim=1,
        )
        kwargs = {
            "template_length": 64,
            "template_bbox": torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
            "search_global_index": torch.arange(36).unsqueeze(0),
            "search_grid_size": 6,
        }

        v8_output, _ = v8(tokens, **kwargs)
        v18_output, diagnostics = v18(tokens, **kwargs)

        self.assertTrue(torch.equal(v18_output, v8_output))
        self.assertTrue(
            torch.equal(
                diagnostics["prototype_weight_entropy"],
                torch.ones(1),
            )
        )
        self.assertTrue(
            torch.equal(
                diagnostics["prototype_effective_token_fraction"],
                torch.ones(1),
            )
        )
        self.assertEqual(diagnostics["prototype_max_weight"].item(), 1 / 16)

    def test_v18_forward_has_finite_gradients_and_zero_alpha_identity(self):
        torch.manual_seed(89)
        module = VisibilityDrivenRepresentationModule(
            num_parts=4,
            spatial_gate_mode="part_aligned_coherent_prototype",
            alpha_max=1.5,
        )
        tokens = torch.randn(2, 64 + 31, 24, requires_grad=True)
        kwargs = {
            "template_length": 64,
            "template_bbox": torch.tensor(
                [[0.25, 0.25, 0.50, 0.50]] * 2
            ),
            "search_global_index": torch.arange(31).unsqueeze(0).repeat(2, 1),
            "search_grid_size": 6,
        }

        identity_output, identity_diagnostics = module(tokens, **kwargs)
        self.assertTrue(torch.equal(identity_output, tokens))
        for name in (
            "prototype_weight_entropy",
            "prototype_effective_token_fraction",
            "prototype_cosine_to_uniform",
            "prototype_max_weight",
        ):
            self.assertTrue(torch.isfinite(identity_diagnostics[name]).all())

        module.alpha.data.fill_(-1.0)
        output, diagnostics = module(tokens, **kwargs)
        output[:, 64:].square().mean().backward()

        self.assertTrue(torch.isfinite(tokens.grad).all())
        for parameter in module.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertTrue(
            (
                diagnostics["prototype_effective_token_fraction"] > 0.0
            ).all().item()
        )
        self.assertTrue(
            (diagnostics["prototype_max_weight"] < 1.0).all().item()
        )

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

    def test_background_suppressed_route_sharpens_v8_without_new_parameters(self):
        torch.manual_seed(30)
        sharpened = VisibilityDrivenRepresentationModule(
            num_parts=4,
            spatial_gate_mode="part_aligned_sharpened",
            part_route_residual_floor=0.25,
            alpha_max=1.5,
        )
        reference = VisibilityDrivenRepresentationModule(
            num_parts=4,
            spatial_gate_mode="part_aligned",
            alpha_max=1.5,
        )
        for name in reference.state_dict():
            reference.state_dict()[name].copy_(
                sharpened.state_dict()[name]
            )
        sharpened.alpha.data.fill_(-0.5)
        reference.alpha.data.fill_(-0.5)
        tokens = torch.randn(2, 64 + 25, 16)
        global_index = torch.arange(25).unsqueeze(0).repeat(2, 1)
        template_bbox = torch.tensor(
            [[0.25, 0.25, 0.50, 0.50]] * 2
        )

        output, diagnostics = sharpened(
            tokens,
            template_length=64,
            template_bbox=template_bbox,
            search_global_index=global_index,
            search_grid_size=5,
        )
        reference_output, reference_diagnostics = reference(
            tokens,
            template_length=64,
            template_bbox=template_bbox,
            search_global_index=global_index,
            search_grid_size=5,
        )

        torch.testing.assert_close(
            diagnostics["part_route_gate"],
            reference_diagnostics["part_route_gate"],
        )
        self.assertFalse(torch.equal(output, reference_output))
        self.assertGreaterEqual(
            diagnostics[
                "part_route_residual_retention_min"
            ].min().item(),
            0.25,
        )
        self.assertLessEqual(
            diagnostics[
                "part_route_residual_retention_max"
            ].max().item(),
            1.0,
        )
        self.assertEqual(
            set(sharpened.state_dict()), set(reference.state_dict())
        )

    def test_background_suppressed_route_floor_one_is_exact_v8(self):
        torch.manual_seed(32)
        sharpened = VisibilityDrivenRepresentationModule(
            num_parts=4,
            spatial_gate_mode="part_aligned_sharpened",
            part_route_residual_floor=1.0,
            alpha_max=1.5,
        )
        reference = VisibilityDrivenRepresentationModule(
            num_parts=4,
            spatial_gate_mode="part_aligned",
            alpha_max=1.5,
        )
        for name in reference.state_dict():
            reference.state_dict()[name].copy_(
                sharpened.state_dict()[name]
            )
        sharpened.alpha.data.fill_(-0.5)
        reference.alpha.data.fill_(-0.5)
        tokens = torch.randn(1, 64 + 25, 16)
        kwargs = {
            "template_length": 64,
            "template_bbox": torch.tensor(
                [[0.25, 0.25, 0.50, 0.50]]
            ),
            "search_global_index": torch.arange(25).unsqueeze(0),
            "search_grid_size": 5,
        }

        sharpened_output, diagnostics = sharpened(tokens, **kwargs)
        reference_output, _ = reference(tokens, **kwargs)

        torch.testing.assert_close(sharpened_output, reference_output)
        torch.testing.assert_close(
            diagnostics["part_route_residual_retention_mean"],
            torch.ones(1),
        )

        with self.assertRaisesRegex(ValueError, r"in \(0, 1\]"):
            VisibilityDrivenRepresentationModule(
                num_parts=4,
                spatial_gate_mode="part_aligned_sharpened",
                part_route_residual_floor=0.0,
            )

    def test_reliability_safe_route_is_exact_v11_for_confident_parts(self):
        v11 = VisibilityDrivenRepresentationModule(
            num_parts=4,
            spatial_gate_mode="part_aligned_sharpened",
            part_route_residual_floor=0.25,
            alpha_max=1.5,
        )
        v13 = VisibilityDrivenRepresentationModule(
            num_parts=4,
            spatial_gate_mode="part_aligned_reliability_safe",
            part_route_residual_floor=0.25,
            alpha_max=1.5,
        )
        v13.load_state_dict(v11.state_dict())
        similarity = torch.randn(2, 4, 7)
        prototypes = torch.randn(2, 4, 8)
        part_reliability = torch.tensor(
            [[0.50, 0.60, 0.80, 1.00], [0.55, 0.70, 0.90, 0.95]]
        )
        part_valid = torch.ones(2, 4, dtype=torch.bool)

        _, v11_gate, v11_residual, v11_diagnostics = (
            v11._part_aligned_statistics(
                similarity, prototypes, part_reliability, part_valid
            )
        )
        _, v13_gate, v13_residual, v13_diagnostics = (
            v13._part_aligned_statistics(
                similarity, prototypes, part_reliability, part_valid
            )
        )

        torch.testing.assert_close(v13_gate, v11_gate)
        torch.testing.assert_close(v13_residual, v11_residual)
        torch.testing.assert_close(
            v13_diagnostics["part_route_residual_retention"],
            v11_diagnostics["part_route_residual_retention"],
        )
        torch.testing.assert_close(
            v13_diagnostics["part_reliability_safety_factor"],
            torch.ones_like(part_reliability),
        )
        self.assertEqual(set(v13.state_dict()), set(v11.state_dict()))

    def test_reliability_safe_route_is_monotone_and_never_amplifies(self):
        module = VisibilityDrivenRepresentationModule(
            num_parts=4,
            spatial_gate_mode="part_aligned_reliability_safe",
            part_route_residual_floor=0.25,
            alpha_max=1.5,
        )
        similarity = torch.randn(1, 4, 6)
        prototypes = torch.randn(1, 4, 8, requires_grad=True)
        part_reliability = torch.tensor(
            [[0.00, 0.25, 0.49, 0.50]], requires_grad=True
        )
        part_valid = torch.ones(1, 4, dtype=torch.bool)

        _, route_gate, residual, diagnostics = (
            module._part_aligned_statistics(
                similarity, prototypes, part_reliability, part_valid
            )
        )
        expected_factor = torch.tensor([[0.00, 0.50, 0.98, 1.00]])
        torch.testing.assert_close(
            diagnostics["part_reliability_safety_factor"],
            expected_factor,
        )
        effective_reliability = part_reliability.detach() * expected_factor
        self.assertTrue(
            torch.all(effective_reliability[:, 1:]
                      >= effective_reliability[:, :-1])
        )
        self.assertTrue(
            torch.all(effective_reliability <= part_reliability.detach())
        )
        expected_weight = (
            route_gate
            * diagnostics["part_route_residual_retention"]
            * part_reliability.unsqueeze(-1)
            * expected_factor.unsqueeze(-1)
        )
        expected_residual = torch.einsum(
            "bkl,bkc->blc", expected_weight, prototypes
        ) / 4.0
        torch.testing.assert_close(residual, expected_residual)

        residual.square().mean().backward()
        self.assertFalse(
            diagnostics["part_reliability_safety_factor"].requires_grad
        )
        self.assertTrue(torch.isfinite(part_reliability.grad).all())
        self.assertTrue(torch.isfinite(prototypes.grad).all())

    def test_reliability_safe_route_zero_alpha_preserves_forward(self):
        torch.manual_seed(36)
        module = VisibilityDrivenRepresentationModule(
            num_parts=4,
            spatial_gate_mode="part_aligned_reliability_safe",
            part_route_residual_floor=0.25,
            alpha_max=1.5,
        )
        tokens = torch.randn(2, 64 + 25, 16)

        output, diagnostics = module(
            tokens,
            template_length=64,
            template_bbox=torch.tensor(
                [[0.25, 0.25, 0.50, 0.50]] * 2
            ),
            search_global_index=torch.arange(25).unsqueeze(0).repeat(2, 1),
            search_grid_size=5,
        )

        self.assertTrue(torch.equal(output, tokens))
        for name in (
            "part_reliability_safety_factor_mean",
            "part_reliability_safety_factor_min",
            "part_reliability_safety_factor_max",
            "part_reliability_suppressed_fraction",
            "part_reliability_suppression_mean",
        ):
            self.assertTrue(torch.isfinite(diagnostics[name]).all())
            self.assertFalse(diagnostics[name].requires_grad)
        self.assertEqual(diagnostics["vdrm_alpha"].item(), 0.0)

    def test_reliability_safe_route_reports_neutral_empty_parts(self):
        torch.manual_seed(37)
        module = VisibilityDrivenRepresentationModule(
            num_parts=4,
            spatial_gate_mode="part_aligned_reliability_safe",
            part_route_residual_floor=0.25,
            alpha_max=1.5,
        )
        module.alpha.data.fill_(-0.5)
        tokens = torch.randn(1, 64 + 25, 16)

        output, diagnostics = module(
            tokens,
            template_length=64,
            template_bbox=torch.zeros(1, 4),
            search_global_index=torch.arange(25).unsqueeze(0),
            search_grid_size=5,
        )

        self.assertTrue(torch.equal(output, tokens))
        self.assertFalse(diagnostics["part_valid"].any().item())
        for name in (
            "part_reliability_safety_factor_mean",
            "part_reliability_safety_factor_min",
            "part_reliability_safety_factor_max",
        ):
            torch.testing.assert_close(diagnostics[name], torch.ones(1))
        torch.testing.assert_close(
            diagnostics["part_reliability_suppressed_fraction"],
            torch.zeros(1),
        )
        torch.testing.assert_close(
            diagnostics["part_reliability_suppression_mean"],
            torch.zeros(1),
        )

    def test_positive_preserved_route_restores_v8_positive_mass(self):
        v8 = VisibilityDrivenRepresentationModule(
            num_parts=4,
            spatial_gate_mode="part_aligned",
            alpha_max=1.5,
        )
        v11 = VisibilityDrivenRepresentationModule(
            num_parts=4,
            spatial_gate_mode="part_aligned_sharpened",
            part_route_residual_floor=0.25,
            alpha_max=1.5,
        )
        v12 = VisibilityDrivenRepresentationModule(
            num_parts=4,
            spatial_gate_mode="part_aligned_positive_preserved",
            part_route_residual_floor=0.25,
            alpha_max=1.5,
        )
        calibrated_scale = torch.log(torch.expm1(torch.tensor(1.0)))
        for module in (v8, v11, v12):
            module.part_route_log_match_scale.data.copy_(calibrated_scale)
            module.part_route_match_bias.data.zero_()

        similarity = torch.tensor(
            [[
                [0.9, 0.5, 0.1, -0.2, -0.6],
                [0.7, 0.3, 0.0, -0.4, -0.8],
                [0.8, 0.2, -0.1, -0.3, -0.7],
                [0.6, 0.4, 0.0, -0.5, -0.9],
            ]]
        )
        prototypes = torch.randn(1, 4, 8)
        part_reliability = torch.ones(1, 4)
        part_valid = torch.ones(1, 4, dtype=torch.bool)

        _, v8_gate, _, v8_diagnostics = v8._part_aligned_statistics(
            similarity, prototypes, part_reliability, part_valid
        )
        _, v11_gate, _, v11_diagnostics = v11._part_aligned_statistics(
            similarity, prototypes, part_reliability, part_valid
        )
        _, v12_gate, _, v12_diagnostics = v12._part_aligned_statistics(
            similarity, prototypes, part_reliability, part_valid
        )

        torch.testing.assert_close(v11_gate, v8_gate)
        torch.testing.assert_close(v12_gate, v8_gate)
        positive_mask = v8_gate >= 0.5
        background_mask = ~positive_mask
        v8_positive_mass = (v8_gate * positive_mask).sum(dim=-1)
        v12_effective_gate = (
            v12_gate
            * v12_diagnostics["part_route_residual_retention"]
        )
        v12_positive_mass = (
            v12_effective_gate * positive_mask
        ).sum(dim=-1)
        torch.testing.assert_close(
            v12_positive_mass, v8_positive_mass, rtol=1e-6, atol=1e-6
        )

        v11_effective_gate = (
            v11_gate
            * v11_diagnostics["part_route_residual_retention"]
        )
        self.assertTrue(
            torch.all(v11_effective_gate[positive_mask]
                      < v8_gate[positive_mask])
        )
        self.assertTrue(
            torch.all(v12_effective_gate[background_mask]
                      <= v8_gate[background_mask] + 1e-7)
        )
        self.assertTrue(
            torch.any(v12_effective_gate[background_mask]
                      < v8_gate[background_mask])
        )
        preservation_scale = v12_diagnostics[
            "part_route_positive_preservation_scale"
        ]
        self.assertGreaterEqual(preservation_scale.min().item(), 1.0)
        self.assertLessEqual(preservation_scale.max().item(), 1.6 + 1e-6)
        torch.testing.assert_close(
            v12_diagnostics["part_route_positive_mass_ratio"],
            torch.ones(1, 4, 1),
            rtol=1e-6,
            atol=1e-6,
        )
        self.assertEqual(set(v12.state_dict()), set(v8.state_dict()))
        self.assertEqual(set(v11.state_dict()), set(v8.state_dict()))
        self.assertTrue(
            torch.equal(
                v8_diagnostics["part_route_residual_retention"],
                torch.ones_like(v8_gate),
            )
        )

    def test_positive_preserved_route_without_positive_evidence_is_v11(self):
        v11 = VisibilityDrivenRepresentationModule(
            num_parts=4,
            spatial_gate_mode="part_aligned_sharpened",
            part_route_residual_floor=0.25,
            alpha_max=1.5,
        )
        v12 = VisibilityDrivenRepresentationModule(
            num_parts=4,
            spatial_gate_mode="part_aligned_positive_preserved",
            part_route_residual_floor=0.25,
            alpha_max=1.5,
        )
        v12.load_state_dict(v11.state_dict())
        calibrated_scale = torch.log(torch.expm1(torch.tensor(1.0)))
        for module in (v11, v12):
            module.part_route_log_match_scale.data.copy_(calibrated_scale)
            module.part_route_match_bias.data.zero_()

        similarity = -torch.ones(2, 4, 6)
        prototypes = torch.randn(2, 4, 8)
        part_reliability = torch.ones(2, 4)
        part_valid = torch.ones(2, 4, dtype=torch.bool)
        _, _, v11_residual, v11_diagnostics = (
            v11._part_aligned_statistics(
                similarity, prototypes, part_reliability, part_valid
            )
        )
        _, _, v12_residual, v12_diagnostics = (
            v12._part_aligned_statistics(
                similarity, prototypes, part_reliability, part_valid
            )
        )

        torch.testing.assert_close(v12_residual, v11_residual)
        torch.testing.assert_close(
            v12_diagnostics["part_route_residual_retention"],
            v11_diagnostics["part_route_residual_retention"],
        )
        torch.testing.assert_close(
            v12_diagnostics["part_route_positive_preservation_scale"],
            torch.ones(2, 4, 1),
        )
        self.assertFalse(
            v12_diagnostics["part_route_positive_present"].any().item()
        )

    def test_positive_preserved_route_has_finite_route_gradients(self):
        torch.manual_seed(33)
        module = VisibilityDrivenRepresentationModule(
            num_parts=4,
            spatial_gate_mode="part_aligned_positive_preserved",
            part_route_residual_floor=0.25,
            alpha_max=1.5,
        )
        module.alpha.data.fill_(-0.5)
        tokens = torch.randn(2, 64 + 25, 16, requires_grad=True)
        output, diagnostics = module(
            tokens,
            template_length=64,
            template_bbox=torch.tensor(
                [[0.25, 0.25, 0.50, 0.50]] * 2
            ),
            search_global_index=torch.arange(25).unsqueeze(0).repeat(2, 1),
            search_grid_size=5,
        )

        output.square().mean().backward()

        self.assertIsNotNone(module.part_route_log_match_scale.grad)
        self.assertIsNotNone(module.part_route_match_bias.grad)
        self.assertTrue(
            torch.isfinite(module.part_route_log_match_scale.grad).item()
        )
        self.assertTrue(
            torch.isfinite(module.part_route_match_bias.grad).item()
        )
        self.assertFalse(
            diagnostics[
                "part_route_positive_preservation_scale_mean"
            ].requires_grad
        )
        self.assertTrue(torch.isfinite(tokens.grad).all())

    def test_positive_preserved_route_zero_alpha_preserves_forward(self):
        torch.manual_seed(34)
        module = VisibilityDrivenRepresentationModule(
            num_parts=4,
            spatial_gate_mode="part_aligned_positive_preserved",
            part_route_residual_floor=0.25,
            alpha_max=1.5,
        )
        tokens = torch.randn(2, 64 + 25, 16)

        output, diagnostics = module(
            tokens,
            template_length=64,
            template_bbox=torch.tensor(
                [[0.25, 0.25, 0.50, 0.50]] * 2
            ),
            search_global_index=torch.arange(25).unsqueeze(0).repeat(2, 1),
            search_grid_size=5,
        )

        self.assertTrue(torch.equal(output, tokens))
        self.assertTrue(
            torch.isfinite(
                diagnostics[
                    "part_route_positive_preservation_scale_mean"
                ]
            ).all()
        )
        self.assertEqual(diagnostics["vdrm_alpha"].item(), 0.0)

    def test_positive_preserved_route_floor_one_is_exact_v8(self):
        torch.manual_seed(35)
        v8 = VisibilityDrivenRepresentationModule(
            num_parts=4,
            spatial_gate_mode="part_aligned",
            alpha_max=1.5,
        )
        v12 = VisibilityDrivenRepresentationModule(
            num_parts=4,
            spatial_gate_mode="part_aligned_positive_preserved",
            part_route_residual_floor=1.0,
            alpha_max=1.5,
        )
        v12.load_state_dict(v8.state_dict())
        v8.alpha.data.fill_(-0.5)
        v12.alpha.data.copy_(v8.alpha.data)
        tokens = torch.randn(1, 64 + 25, 16)
        kwargs = {
            "template_length": 64,
            "template_bbox": torch.tensor(
                [[0.25, 0.25, 0.50, 0.50]]
            ),
            "search_global_index": torch.arange(25).unsqueeze(0),
            "search_grid_size": 5,
        }

        v8_output, _ = v8(tokens, **kwargs)
        v12_output, diagnostics = v12(tokens, **kwargs)

        torch.testing.assert_close(v12_output, v8_output)
        torch.testing.assert_close(
            diagnostics["part_route_residual_retention_mean"],
            torch.ones(1),
        )
        torch.testing.assert_close(
            diagnostics[
                "part_route_positive_preservation_scale_mean"
            ],
            torch.ones(1),
        )

    def test_part_aligned_consensus_exposes_both_supervised_gates(self):
        torch.manual_seed(31)
        module = VisibilityDrivenRepresentationModule(
            num_parts=4,
            spatial_gate_mode="part_aligned_consensus",
            candidate_local_radius=1,
            candidate_consensus_parts=3,
            alpha_max=1.5,
        )
        tokens = torch.randn(2, 64 + 25, 32, requires_grad=True)
        global_index = torch.arange(25).unsqueeze(0).repeat(2, 1)

        output, diagnostics = module(
            tokens,
            template_length=64,
            template_bbox=torch.tensor(
                [[0.25, 0.25, 0.50, 0.50]] * 2
            ),
            search_global_index=global_index,
            search_grid_size=5,
        )

        self.assertTrue(torch.equal(output, tokens))
        self.assertEqual(diagnostics["part_route_logits"].shape, (2, 4, 25))
        self.assertEqual(
            diagnostics["candidate_identity_logits"].shape, (2, 25)
        )
        self.assertEqual(
            diagnostics["candidate_reliability_map"].shape,
            (2, 1, 5, 5),
        )
        self.assertEqual(
            set(module.state_dict()),
            {
                "log_match_scale",
                "match_bias",
                "candidate_log_match_scale",
                "candidate_match_bias",
                "part_route_log_match_scale",
                "part_route_match_bias",
                "alpha",
            },
        )

    def test_part_aligned_consensus_decouples_tracking_and_gate_gradients(self):
        torch.manual_seed(37)
        module = VisibilityDrivenRepresentationModule(
            num_parts=4,
            spatial_gate_mode="part_aligned_consensus",
            candidate_local_radius=1,
            candidate_consensus_parts=3,
            alpha_max=1.5,
        )
        module.alpha.data.fill_(-0.5)
        global_index = torch.arange(25).unsqueeze(0)
        template_bbox = torch.tensor([[0.25, 0.25, 0.50, 0.50]])
        tokens = torch.randn(1, 64 + 25, 16, requires_grad=True)

        output, _ = module(
            tokens,
            template_length=64,
            template_bbox=template_bbox,
            search_global_index=global_index,
            search_grid_size=5,
        )
        output[:, 64:].square().mean().backward()

        self.assertIsNone(module.candidate_log_match_scale.grad)
        self.assertIsNone(module.candidate_match_bias.grad)
        self.assertIsNotNone(module.part_route_log_match_scale.grad)
        self.assertIsNotNone(module.part_route_match_bias.grad)

        module.zero_grad(set_to_none=True)
        _, diagnostics = module(
            torch.randn(1, 64 + 25, 16),
            template_length=64,
            template_bbox=template_bbox,
            search_global_index=global_index,
            search_grid_size=5,
        )
        gaussian_map = torch.zeros(1, 5, 5)
        gaussian_map[:, 2, 2] = 1.0
        candidate_loss = compute_vdrm_candidate_focal_loss(
            diagnostics["candidate_identity_logits"],
            diagnostics["search_global_index"],
            gaussian_map,
            sample_valid=diagnostics["candidate_consensus_valid"],
        )
        candidate_loss.backward()

        self.assertIsNotNone(module.candidate_log_match_scale.grad)
        self.assertIsNotNone(module.candidate_match_bias.grad)
        self.assertIsNotNone(module.part_route_log_match_scale.grad)
        self.assertTrue(
            torch.isfinite(module.candidate_log_match_scale.grad).all()
        )

    def test_part_aligned_guidance_starts_as_exact_v8_and_is_bounded(self):
        torch.manual_seed(41)
        guided = VisibilityDrivenRepresentationModule(
            num_parts=4,
            spatial_gate_mode="part_aligned_guidance",
            candidate_local_radius=1,
            candidate_consensus_parts=3,
            candidate_modulation_max=0.5,
            alpha_max=1.5,
        )
        reference = VisibilityDrivenRepresentationModule(
            num_parts=4,
            spatial_gate_mode="part_aligned",
            alpha_max=1.5,
        )
        for name in (
            "log_match_scale",
            "match_bias",
            "part_route_log_match_scale",
            "part_route_match_bias",
            "alpha",
        ):
            getattr(reference, name).data.copy_(getattr(guided, name).data)
        guided.alpha.data.fill_(-0.5)
        reference.alpha.data.fill_(-0.5)
        tokens = torch.randn(2, 64 + 25, 16)
        template_bbox = torch.tensor(
            [[0.25, 0.25, 0.50, 0.50]] * 2
        )
        global_index = torch.arange(25).unsqueeze(0).repeat(2, 1)

        guided_output, diagnostics = guided(
            tokens,
            template_length=64,
            template_bbox=template_bbox,
            search_global_index=global_index,
            search_grid_size=5,
        )
        reference_output, _ = reference(
            tokens,
            template_length=64,
            template_bbox=template_bbox,
            search_global_index=global_index,
            search_grid_size=5,
        )

        torch.testing.assert_close(guided_output, reference_output)
        self.assertEqual(diagnostics["candidate_modulation"].item(), 0.0)
        torch.testing.assert_close(
            diagnostics["candidate_modulation_factor_min"],
            torch.ones(2),
        )
        torch.testing.assert_close(
            diagnostics["candidate_modulation_factor_max"],
            torch.ones(2),
        )
        self.assertEqual(
            set(guided.state_dict()),
            {
                "log_match_scale",
                "match_bias",
                "candidate_log_match_scale",
                "candidate_match_bias",
                "part_route_log_match_scale",
                "part_route_match_bias",
                "candidate_modulation",
                "alpha",
            },
        )

        for raw_value in (-100.0, 100.0):
            guided.candidate_modulation.data.fill_(raw_value)
            _, bounded = guided(
                tokens,
                template_length=64,
                template_bbox=template_bbox,
                search_global_index=global_index,
                search_grid_size=5,
            )
            self.assertGreaterEqual(
                bounded["candidate_modulation_factor_min"].min().item(),
                0.5,
            )
            self.assertLessEqual(
                bounded["candidate_modulation_factor_max"].max().item(),
                1.5,
            )

        with self.assertRaisesRegex(ValueError, r"in \(0, 0\.5\]"):
            VisibilityDrivenRepresentationModule(
                num_parts=4,
                spatial_gate_mode="part_aligned_guidance",
                candidate_modulation_max=0.75,
            )

    def test_part_aligned_guidance_isolates_candidate_and_route_gradients(self):
        torch.manual_seed(43)
        module = VisibilityDrivenRepresentationModule(
            num_parts=4,
            spatial_gate_mode="part_aligned_guidance",
            candidate_local_radius=1,
            candidate_consensus_parts=3,
            candidate_modulation_max=0.5,
            alpha_max=1.5,
        )
        module.alpha.data.fill_(-0.5)
        module.candidate_modulation.data.fill_(0.1)
        global_index = torch.arange(25).unsqueeze(0)
        template_bbox = torch.tensor([[0.25, 0.25, 0.50, 0.50]])
        tokens = torch.randn(1, 64 + 25, 16, requires_grad=True)

        output, _ = module(
            tokens,
            template_length=64,
            template_bbox=template_bbox,
            search_global_index=global_index,
            search_grid_size=5,
        )
        output[:, 64:].square().mean().backward()

        self.assertIsNone(module.candidate_log_match_scale.grad)
        self.assertIsNone(module.candidate_match_bias.grad)
        self.assertIsNotNone(module.candidate_modulation.grad)
        self.assertIsNotNone(module.part_route_log_match_scale.grad)
        self.assertIsNotNone(module.part_route_match_bias.grad)

        module.zero_grad(set_to_none=True)
        _, diagnostics = module(
            torch.randn(1, 64 + 25, 16),
            template_length=64,
            template_bbox=template_bbox,
            search_global_index=global_index,
            search_grid_size=5,
        )
        gaussian_map = torch.zeros(1, 5, 5)
        gaussian_map[:, 2, 2] = 1.0
        candidate_loss = compute_vdrm_candidate_focal_loss(
            diagnostics["candidate_identity_logits"],
            diagnostics["search_global_index"],
            gaussian_map,
            sample_valid=diagnostics["candidate_consensus_valid"],
        )
        candidate_loss.backward()

        self.assertIsNotNone(module.candidate_log_match_scale.grad)
        self.assertIsNotNone(module.candidate_match_bias.grad)
        self.assertIsNone(module.candidate_modulation.grad)
        self.assertIsNone(module.part_route_log_match_scale.grad)
        self.assertIsNone(module.part_route_match_bias.grad)
        self.assertIsNone(module.log_match_scale.grad)
        self.assertIsNone(module.match_bias.grad)

    def test_v14_identity_aux_is_exact_v8_for_fixed_common_parameters(self):
        torch.manual_seed(47)
        v8 = VisibilityDrivenRepresentationModule(
            num_parts=4,
            spatial_gate_mode="part_aligned",
            alpha_max=1.5,
        )
        v14 = VisibilityDrivenRepresentationModule(
            num_parts=4,
            spatial_gate_mode="part_aligned_identity_aux",
            candidate_local_radius=1,
            candidate_consensus_parts=3,
            alpha_max=1.5,
        )
        for name, value in v8.state_dict().items():
            getattr(v14, name).data.copy_(value)
        v8.alpha.data.fill_(-0.65)
        v14.alpha.data.copy_(v8.alpha.data)
        tokens = torch.randn(2, 64 + 25, 16)
        kwargs = {
            "template_length": 64,
            "template_bbox": torch.tensor(
                [[0.25, 0.25, 0.50, 0.50]] * 2
            ),
            "search_global_index": torch.arange(25).unsqueeze(0).repeat(2, 1),
            "search_grid_size": 5,
        }

        v8_output, v8_diagnostics = v8(tokens, **kwargs)
        v14_output, v14_diagnostics = v14(tokens, **kwargs)

        self.assertTrue(torch.equal(v14_output, v8_output))
        torch.testing.assert_close(
            v14_diagnostics["part_route_gate"],
            v8_diagnostics["part_route_gate"],
            rtol=0.0,
            atol=0.0,
        )
        self.assertEqual(
            v14_diagnostics["candidate_identity_logits"].shape,
            (2, 25),
        )
        self.assertEqual(
            set(v14.state_dict()).difference(v8.state_dict()),
            {"candidate_log_match_scale", "candidate_match_bias"},
        )

    def test_v14_identity_aux_isolates_tracking_and_candidate_gradients(self):
        torch.manual_seed(53)
        module = VisibilityDrivenRepresentationModule(
            num_parts=4,
            spatial_gate_mode="part_aligned_identity_aux",
            candidate_local_radius=1,
            candidate_consensus_parts=3,
            alpha_max=1.5,
        )
        module.alpha.data.fill_(-0.5)
        global_index = torch.arange(25).unsqueeze(0)
        template_bbox = torch.tensor([[0.25, 0.25, 0.50, 0.50]])
        tokens = torch.randn(1, 64 + 25, 16, requires_grad=True)

        output, _ = module(
            tokens,
            template_length=64,
            template_bbox=template_bbox,
            search_global_index=global_index,
            search_grid_size=5,
        )
        output[:, 64:].square().mean().backward()

        self.assertIsNone(module.candidate_log_match_scale.grad)
        self.assertIsNone(module.candidate_match_bias.grad)
        self.assertIsNotNone(module.part_route_log_match_scale.grad)
        self.assertIsNotNone(module.part_route_match_bias.grad)

        module.zero_grad(set_to_none=True)
        candidate_tokens = torch.randn(
            1, 64 + 25, 16, requires_grad=True
        )
        _, diagnostics = module(
            candidate_tokens,
            template_length=64,
            template_bbox=template_bbox,
            search_global_index=global_index,
            search_grid_size=5,
        )
        gaussian_map = torch.zeros(1, 5, 5)
        gaussian_map[:, 2, 2] = 1.0
        candidate_loss = compute_vdrm_candidate_focal_loss(
            diagnostics["candidate_identity_logits"],
            diagnostics["search_global_index"],
            gaussian_map,
            sample_valid=diagnostics["candidate_consensus_valid"],
        )
        candidate_loss.backward()

        self.assertIsNotNone(module.candidate_log_match_scale.grad)
        self.assertIsNotNone(module.candidate_match_bias.grad)
        self.assertTrue(
            torch.isfinite(module.candidate_log_match_scale.grad).all()
        )
        self.assertTrue(torch.isfinite(candidate_tokens.grad).all())
        self.assertIsNone(module.part_route_log_match_scale.grad)
        self.assertIsNone(module.part_route_match_bias.grad)
        self.assertIsNone(module.log_match_scale.grad)
        self.assertIsNone(module.match_bias.grad)
        self.assertIsNone(module.alpha.grad)

    def test_v9_actor_trains_candidate_and_part_route_objectives(self):
        vdrm_cfg = SimpleNamespace(
            ENABLED=True,
            RELIABILITY_MODE="topk",
            SPATIAL_GATE_MODE="part_aligned_consensus",
        )
        cfg = SimpleNamespace(
            DATA=SimpleNamespace(SEARCH=SimpleNamespace(SIZE=64)),
            MODEL=SimpleNamespace(
                VDRM=vdrm_cfg,
                BACKBONE=SimpleNamespace(STRIDE=16),
            ),
            TRAIN=SimpleNamespace(
                VDRM_AUX_WARMUP_EPOCHS=1,
                VDRM_VISIBILITY_WEIGHT=0.5,
                VDRM_RANK_WEIGHT=0.5,
                VDRM_CANDIDATE_WEIGHT=0.1,
                VDRM_PART_ROUTE_WEIGHT=0.1,
                VDRM_PART_TARGET_DILATION=1.0,
            ),
        )

        def giou_objective(prediction, target):
            return (
                (prediction - target).square().mean(),
                prediction.new_ones(prediction.shape[0]),
            )

        actor = OSTrackActor(
            net=None,
            objective={
                "giou": giou_objective,
                "l1": lambda prediction, target: (
                    prediction - target
                ).abs().mean(),
                "focal": lambda prediction, target: (
                    prediction - target
                ).square().mean(),
            },
            loss_weight={"giou": 2.0, "l1": 5.0, "focal": 1.0},
            settings=SimpleNamespace(batchsize=1),
            cfg=cfg,
        )
        candidate_logits = torch.zeros(1, 16, requires_grad=True)
        part_route_logits = torch.zeros(1, 4, 16, requires_grad=True)
        pred_dict = {
            "pred_boxes": torch.tensor(
                [[[0.5, 0.5, 0.5, 0.5]]], requires_grad=True
            ),
            "score_map": torch.zeros(1, 1, 4, 4, requires_grad=True),
            "candidate_identity_logits": candidate_logits,
            "candidate_consensus_valid": torch.ones(1, dtype=torch.bool),
            "part_route_logits": part_route_logits,
            "part_valid": torch.ones(1, 4, dtype=torch.bool),
            "search_global_index": torch.arange(16).unsqueeze(0),
            "visual_reliability": torch.ones(1),
            "vdrm_alpha": torch.zeros(()),
        }
        gt_dict = {
            "search_anno": torch.tensor([[[0.25, 0.25, 0.5, 0.5]]]),
            "epoch": 1,
        }

        loss, status = actor.compute_losses(pred_dict, gt_dict)

        self.assertGreater(status["Loss/vdrm_candidate"], 0.0)
        self.assertGreater(status["Loss/vdrm_part_route"], 0.0)
        loss.backward()
        self.assertIsNotNone(candidate_logits.grad)
        self.assertIsNotNone(part_route_logits.grad)
        self.assertTrue(torch.isfinite(candidate_logits.grad).all())
        self.assertTrue(torch.isfinite(part_route_logits.grad).all())

    def test_v14_actor_trains_identity_aux_and_v8_part_route_objectives(self):
        cfg = SimpleNamespace(
            DATA=SimpleNamespace(SEARCH=SimpleNamespace(SIZE=64)),
            MODEL=SimpleNamespace(
                VDRM=SimpleNamespace(
                    ENABLED=True,
                    RELIABILITY_MODE="topk",
                    SPATIAL_GATE_MODE="part_aligned_identity_aux",
                ),
                BACKBONE=SimpleNamespace(STRIDE=16),
            ),
            TRAIN=SimpleNamespace(
                VDRM_AUX_WARMUP_EPOCHS=1,
                VDRM_VISIBILITY_WEIGHT=0.5,
                VDRM_RANK_WEIGHT=0.5,
                VDRM_CANDIDATE_WEIGHT=0.02,
                VDRM_PART_ROUTE_WEIGHT=0.1,
                VDRM_PART_TARGET_DILATION=1.0,
            ),
        )

        def giou_objective(prediction, target):
            return (
                (prediction - target).square().mean(),
                prediction.new_ones(prediction.shape[0]),
            )

        actor = OSTrackActor(
            net=None,
            objective={
                "giou": giou_objective,
                "l1": lambda prediction, target: (
                    prediction - target
                ).abs().mean(),
                "focal": lambda prediction, target: (
                    prediction - target
                ).square().mean(),
            },
            loss_weight={"giou": 2.0, "l1": 5.0, "focal": 1.0},
            settings=SimpleNamespace(batchsize=1),
            cfg=cfg,
        )
        candidate_logits = torch.zeros(1, 16, requires_grad=True)
        part_route_logits = torch.zeros(1, 4, 16, requires_grad=True)
        pred_dict = {
            "pred_boxes": torch.tensor(
                [[[0.5, 0.5, 0.5, 0.5]]], requires_grad=True
            ),
            "score_map": torch.zeros(1, 1, 4, 4, requires_grad=True),
            "candidate_identity_logits": candidate_logits,
            "candidate_consensus_valid": torch.ones(1, dtype=torch.bool),
            "part_route_logits": part_route_logits,
            "part_valid": torch.ones(1, 4, dtype=torch.bool),
            "search_global_index": torch.arange(16).unsqueeze(0),
            "visual_reliability": torch.ones(1),
            "vdrm_alpha": torch.zeros(()),
        }
        gt_dict = {
            "search_anno": torch.tensor([[[0.25, 0.25, 0.5, 0.5]]]),
            "epoch": 1,
        }

        loss, status = actor.compute_losses(pred_dict, gt_dict)

        self.assertGreater(status["Loss/vdrm_candidate"], 0.0)
        self.assertGreater(status["Loss/vdrm_part_route"], 0.0)
        loss.backward()
        self.assertIsNotNone(candidate_logits.grad)
        self.assertIsNotNone(part_route_logits.grad)
        self.assertTrue(torch.isfinite(candidate_logits.grad).all())
        self.assertTrue(torch.isfinite(part_route_logits.grad).all())

    def test_v12_actor_trains_route_and_logs_preservation(self):
        cfg = SimpleNamespace(
            DATA=SimpleNamespace(SEARCH=SimpleNamespace(SIZE=64)),
            MODEL=SimpleNamespace(
                VDRM=SimpleNamespace(
                    ENABLED=True,
                    RELIABILITY_MODE="topk",
                    SPATIAL_GATE_MODE=(
                        "part_aligned_positive_preserved"
                    ),
                ),
                BACKBONE=SimpleNamespace(STRIDE=16),
            ),
            TRAIN=SimpleNamespace(
                VDRM_AUX_WARMUP_EPOCHS=1,
                VDRM_VISIBILITY_WEIGHT=0.5,
                VDRM_RANK_WEIGHT=0.5,
                VDRM_CANDIDATE_WEIGHT=0.0,
                VDRM_PART_ROUTE_WEIGHT=0.1,
                VDRM_PART_TARGET_DILATION=1.0,
            ),
        )

        def giou_objective(prediction, target):
            return (
                (prediction - target).square().mean(),
                prediction.new_ones(prediction.shape[0]),
            )

        actor = OSTrackActor(
            net=None,
            objective={
                "giou": giou_objective,
                "l1": lambda prediction, target: (
                    prediction - target
                ).abs().mean(),
                "focal": lambda prediction, target: (
                    prediction - target
                ).square().mean(),
            },
            loss_weight={"giou": 2.0, "l1": 5.0, "focal": 1.0},
            settings=SimpleNamespace(batchsize=1),
            cfg=cfg,
        )
        part_route_logits = torch.zeros(1, 4, 16, requires_grad=True)
        pred_dict = {
            "pred_boxes": torch.tensor(
                [[[0.5, 0.5, 0.5, 0.5]]], requires_grad=True
            ),
            "score_map": torch.zeros(1, 1, 4, 4, requires_grad=True),
            "part_route_logits": part_route_logits,
            "part_valid": torch.ones(1, 4, dtype=torch.bool),
            "search_global_index": torch.arange(16).unsqueeze(0),
            "visual_reliability": torch.ones(1),
            "vdrm_alpha": torch.zeros(()),
            "part_route_positive_preservation_scale_mean": (
                torch.tensor([1.2])
            ),
            "part_route_positive_preservation_scale_min": (
                torch.tensor([1.0])
            ),
            "part_route_positive_preservation_scale_max": (
                torch.tensor([1.4])
            ),
            "part_route_positive_mass_ratio": torch.tensor([1.0]),
            "part_route_positive_part_fraction": torch.tensor([0.75]),
        }
        gt_dict = {
            "search_anno": torch.tensor([[[0.25, 0.25, 0.5, 0.5]]]),
            "epoch": 1,
        }

        loss, status = actor.compute_losses(pred_dict, gt_dict)

        self.assertGreater(status["Loss/vdrm_part_route"], 0.0)
        self.assertEqual(status["Loss/vdrm_candidate"], 0.0)
        self.assertAlmostEqual(
            status[
                "VDRM/part_route_positive_preservation_scale_mean"
            ],
            1.2,
        )
        self.assertEqual(
            status["VDRM/part_route_positive_mass_ratio"], 1.0
        )
        loss.backward()
        self.assertIsNotNone(part_route_logits.grad)
        self.assertTrue(torch.isfinite(part_route_logits.grad).all())

    def test_v13_actor_trains_route_and_logs_reliability_safety(self):
        cfg = SimpleNamespace(
            DATA=SimpleNamespace(SEARCH=SimpleNamespace(SIZE=64)),
            MODEL=SimpleNamespace(
                VDRM=SimpleNamespace(
                    ENABLED=True,
                    RELIABILITY_MODE="topk",
                    SPATIAL_GATE_MODE=(
                        "part_aligned_reliability_safe"
                    ),
                ),
                BACKBONE=SimpleNamespace(STRIDE=16),
            ),
            TRAIN=SimpleNamespace(
                VDRM_AUX_WARMUP_EPOCHS=1,
                VDRM_VISIBILITY_WEIGHT=0.5,
                VDRM_RANK_WEIGHT=0.5,
                VDRM_CANDIDATE_WEIGHT=0.0,
                VDRM_PART_ROUTE_WEIGHT=0.1,
                VDRM_PART_TARGET_DILATION=1.0,
            ),
        )

        def giou_objective(prediction, target):
            return (
                (prediction - target).square().mean(),
                prediction.new_ones(prediction.shape[0]),
            )

        actor = OSTrackActor(
            net=None,
            objective={
                "giou": giou_objective,
                "l1": lambda prediction, target: (
                    prediction - target
                ).abs().mean(),
                "focal": lambda prediction, target: (
                    prediction - target
                ).square().mean(),
            },
            loss_weight={"giou": 2.0, "l1": 5.0, "focal": 1.0},
            settings=SimpleNamespace(batchsize=1),
            cfg=cfg,
        )
        part_route_logits = torch.zeros(1, 4, 16, requires_grad=True)
        pred_dict = {
            "pred_boxes": torch.tensor(
                [[[0.5, 0.5, 0.5, 0.5]]], requires_grad=True
            ),
            "score_map": torch.zeros(1, 1, 4, 4, requires_grad=True),
            "part_route_logits": part_route_logits,
            "part_valid": torch.ones(1, 4, dtype=torch.bool),
            "search_global_index": torch.arange(16).unsqueeze(0),
            "visual_reliability": torch.ones(1),
            "vdrm_alpha": torch.zeros(()),
            "part_reliability_safety_factor_mean": torch.tensor([0.8]),
            "part_reliability_safety_factor_min": torch.tensor([0.4]),
            "part_reliability_safety_factor_max": torch.tensor([1.0]),
            "part_reliability_suppressed_fraction": torch.tensor([0.5]),
            "part_reliability_suppression_mean": torch.tensor([0.2]),
        }
        gt_dict = {
            "search_anno": torch.tensor([[[0.25, 0.25, 0.5, 0.5]]]),
            "epoch": 1,
        }

        loss, status = actor.compute_losses(pred_dict, gt_dict)

        self.assertGreater(status["Loss/vdrm_part_route"], 0.0)
        self.assertEqual(status["Loss/vdrm_candidate"], 0.0)
        self.assertAlmostEqual(
            status["VDRM/part_reliability_safety_factor_mean"], 0.8
        )
        self.assertAlmostEqual(
            status["VDRM/part_reliability_suppressed_fraction"], 0.5
        )
        loss.backward()
        self.assertIsNotNone(part_route_logits.grad)
        self.assertTrue(torch.isfinite(part_route_logits.grad).all())

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

    def test_v17_no_hncp_is_bit_identical_to_v8_loss_and_gradient(self):
        torch.manual_seed(73)
        v8_logits = torch.randn(2, 4, 64, requires_grad=True)
        v17_logits = v8_logits.detach().clone().requires_grad_()
        global_index = torch.arange(64).unsqueeze(0).repeat(2, 1)
        bbox = torch.tensor(
            [[0.25, 0.25, 0.50, 0.50], [0.20, 0.30, 0.40, 0.35]]
        )
        part_valid = torch.tensor(
            [[True, True, True, True], [True, False, True, True]]
        )
        part_weight = torch.tensor(
            [[1.0, 0.8, 0.6, 0.4], [0.9, 0.0, 0.7, 0.5]]
        )

        v8_loss, _ = compute_vdrm_part_route_loss(
            v8_logits,
            global_index,
            bbox,
            grid_height=8,
            grid_width=8,
            part_valid=part_valid,
            part_weight=part_weight,
            dilation=1.0,
        )
        v17_loss, diagnostics = compute_vdrm_part_route_loss(
            v17_logits,
            global_index,
            bbox,
            grid_height=8,
            grid_width=8,
            part_valid=part_valid,
            part_weight=part_weight,
            dilation=1.0,
            distractor_boxes=torch.tensor(
                [[0.0, 0.0, 0.2, 0.2], [0.8, 0.8, 0.1, 0.1]]
            ),
            distractor_applied=torch.zeros(2),
            group_balance_distractor=True,
        )

        self.assertTrue(torch.equal(v17_loss, v8_loss))
        v8_loss.backward()
        v17_loss.backward()
        self.assertTrue(torch.equal(v17_logits.grad, v8_logits.grad))
        self.assertEqual(
            diagnostics["part_route_distractor_alignment_rate"].item(),
            0.0,
        )

    def test_v17_group_balance_penalizes_distractor_and_keeps_background(self):
        bbox = torch.tensor([[0.25, 0.25, 0.50, 0.50]])
        global_index = torch.arange(64).unsqueeze(0)
        low_logits = torch.zeros(1, 4, 64, requires_grad=True)
        high_logits = low_logits.detach().clone()
        high_logits[:, :, 0] = 3.0
        high_logits.requires_grad_()
        kwargs = {
            "search_global_index": global_index,
            "search_bbox": bbox,
            "grid_height": 8,
            "grid_width": 8,
            "dilation": 0.0,
            "distractor_boxes": torch.tensor(
                [[0.0, 0.0, 0.125, 0.125]]
            ),
            "distractor_applied": torch.ones(1),
            "group_balance_distractor": True,
        }

        low_loss, low_diagnostics = compute_vdrm_part_route_loss(
            low_logits, **kwargs
        )
        high_loss, high_diagnostics = compute_vdrm_part_route_loss(
            high_logits, **kwargs
        )

        self.assertGreater(high_loss.item(), low_loss.item())
        self.assertGreater(
            high_diagnostics[
                "part_route_distractor_probability"
            ].item(),
            low_diagnostics[
                "part_route_distractor_probability"
            ].item(),
        )
        self.assertEqual(
            high_diagnostics[
                "part_route_distractor_alignment_rate"
            ].item(),
            1.0,
        )
        high_loss.backward()
        self.assertGreater(high_logits.grad[0, 0, 0].item(), 0.0)
        self.assertNotEqual(high_logits.grad[0, 0, 1].item(), 0.0)
        self.assertTrue(torch.isfinite(high_logits.grad).all())

    def test_v17_tiny_distractor_falls_back_to_nearest_legal_negative(self):
        logits = torch.zeros(1, 4, 64, requires_grad=True)
        loss, diagnostics = compute_vdrm_part_route_loss(
            logits,
            torch.arange(64).unsqueeze(0),
            torch.tensor([[0.25, 0.25, 0.50, 0.50]]),
            grid_height=8,
            grid_width=8,
            dilation=0.0,
            distractor_boxes=torch.tensor(
                [[0.001, 0.001, 0.01, 0.01]]
            ),
            distractor_applied=torch.ones(1),
            group_balance_distractor=True,
        )

        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(
            diagnostics[
                "part_route_distractor_alignment_rate"
            ].item(),
            1.0,
        )
        self.assertGreater(
            diagnostics[
                "part_route_distractor_mass_fraction"
            ].item(),
            0.0,
        )

    def test_v17_missing_ordinary_group_falls_back_exactly_to_v8(self):
        torch.manual_seed(79)
        v8_logits = torch.randn(1, 4, 16, requires_grad=True)
        v17_logits = v8_logits.detach().clone().requires_grad_()
        args = (
            torch.arange(16).unsqueeze(0),
            torch.tensor([[0.25, 0.25, 0.50, 0.50]]),
        )
        v8_loss, _ = compute_vdrm_part_route_loss(
            v8_logits,
            *args,
            grid_height=4,
            grid_width=4,
            dilation=0.0,
        )
        v17_loss, diagnostics = compute_vdrm_part_route_loss(
            v17_logits,
            *args,
            grid_height=4,
            grid_width=4,
            dilation=0.0,
            distractor_boxes=torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
            distractor_applied=torch.ones(1),
            group_balance_distractor=True,
        )

        self.assertTrue(torch.equal(v17_loss, v8_loss))
        v8_loss.backward()
        v17_loss.backward()
        self.assertTrue(torch.equal(v17_logits.grad, v8_logits.grad))
        self.assertEqual(
            diagnostics[
                "part_route_distractor_alignment_rate"
            ].item(),
            0.0,
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
