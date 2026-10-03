"""Visibility-driven representation module for OSTrack.

The first implementation intentionally keeps the design small:

* the template target is split into a fixed 2 x 2 grid;
* each grid cell is represented by masked mean pooling;
* cosine matching estimates whether each part is visible in the search;
* matched, reliable prototypes are added to search tokens through one
  zero-initialized residual scalar;
* VDRM-v4 can deterministically bound the complete residual update relative
  to each input search-token norm.
* VDRM-v7 can require multiple template parts to agree inside one local
  search candidate before allowing a spatial residual update.
* VDRM-v8 routes every template part to its own search locations and builds
  the residual from the same calibrated part evidence. Its LayerScale is
  bounded so a vanishing spatial gate cannot be offset by an unbounded
  residual scalar.
* VDRM-v9 keeps the supervised V8 part routes, but only injects their
  residual where several routed parts support the same local candidate. The
  candidate gate is detached from the tracking loss, so only its explicit
  target supervision can calibrate it.
* VDRM-v10 preserves the complete V8 residual and treats candidate consensus
  as isolated guidance. Candidate supervision cannot alter part routing, and
  a zero-initialized bounded scalar can only apply a mean-centered correction
  that retains a hard residual floor at every token.
* VDRM-v11 keeps V8's supervised part routes and checkpoint schema, but
  suppresses the accumulated background leakage before residual injection.
  A deterministic confidence factor preserves most positive-route evidence,
  retains a configurable fraction of every V8 contribution, and introduces
  no candidate gate or additional learned scalar.
* VDRM-v12 preserves V11's background suppression and restores the aggregate
  contribution of routes on the calibrated positive side of the existing
  part-route classifier. The deterministic per-part compensation is detached,
  bounded by the V11 floor, and keeps the exact V8 parameter schema.
* VDRM-v14 returns to V8's unmodified part-aligned residual and adds a
  candidate-level identity auxiliary readout. The readout is supervised from
  raw part similarity, cannot gate the residual, and receives detached global
  reliability weights so it cannot recalibrate V8's route or reliability
  branches through its auxiliary loss.
* VDRM-v15 keeps V8's route, losses, and parameter schema, but independently
  bounds each complete post-LayerScale token update to the high-tail trust
  region selected before training. This reuses V4's parameter-free projection
  at a deliberately non-restrictive V8-tail boundary.
* VDRM-v16 keeps V8's effective route mass at every search token, but
  redistributes that mass among template parts using bidirectional part-token
  evidence. It changes identity assignment without a new gate, parameter,
  loss, threshold, or residual-magnitude control.
* VDRM-v18 returns to V8's complete route and residual path, but replaces the
  uniform template-part mean with a detached identity-coherence weighting.
  Every in-part token remains active, while tokens that agree with the V8
  uniform prototype contribute more strongly to the prototype used for both
  matching and residual injection.

The original ``topk`` reliability is retained for VDRM-v1 checkpoint
compatibility. VDRM-v2 uses the margin between a part's best match and its
strongest spatially distinct match.

No temporal state or motion information is used here.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from .discriminative_route import DiscriminativePartRoute


class VisibilityDrivenRepresentationModule(nn.Module):
    """Lightweight fixed-part prototype matching and residual refinement."""

    def __init__(
        self,
        num_parts: int = 4,
        topk: int = 4,
        reliability_mode: str = "topk",
        nms_radius: int = 1,
        initial_match_scale: float = 5.0,
        initial_match_bias: float = -2.5,
        residual_max_ratio: float = 0.0,
        spatial_gate_mode: str = "token_match",
        candidate_local_radius: int = 1,
        candidate_consensus_parts: int = 2,
        candidate_initial_match_scale: float = 5.0,
        candidate_initial_match_bias: float = -2.5,
        part_route_initial_match_scale: float = 5.0,
        part_route_initial_match_bias: float = -2.5,
        part_route_residual_floor: float = 1.0,
        candidate_modulation_max: float = 0.5,
        alpha_max: float = 0.0,
        train_alpha: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        part_grid = int(math.sqrt(num_parts))
        if part_grid * part_grid != num_parts:
            raise ValueError(f"num_parts must be a square number, got {num_parts}")
        if topk < 1:
            raise ValueError(f"topk must be positive, got {topk}")
        if reliability_mode not in ("topk", "margin"):
            raise ValueError(
                "reliability_mode must be 'topk' or 'margin', "
                f"got {reliability_mode!r}"
            )
        if nms_radius < 0:
            raise ValueError(
                f"nms_radius must be non-negative, got {nms_radius}"
            )
        if residual_max_ratio < 0.0:
            raise ValueError(
                "residual_max_ratio must be non-negative, got "
                f"{residual_max_ratio}"
            )
        if spatial_gate_mode not in (
            "token_match",
            "candidate_consensus",
            "part_aligned",
            "part_aligned_consensus",
            "part_aligned_guidance",
            "part_aligned_sharpened",
            "part_aligned_positive_preserved",
            "part_aligned_reliability_safe",
            "part_aligned_identity_aux",
            "part_aligned_bidirectional",
            "part_aligned_coherent_prototype",
        ):
            raise ValueError(
                "spatial_gate_mode must be 'token_match', "
                "'candidate_consensus', 'part_aligned', or "
                "'part_aligned_consensus', 'part_aligned_guidance', or "
                "'part_aligned_sharpened', "
                "'part_aligned_positive_preserved', or "
                "'part_aligned_reliability_safe', or "
                "'part_aligned_identity_aux', "
                "'part_aligned_bidirectional', "
                "'part_aligned_coherent_prototype', "
                f"got {spatial_gate_mode!r}"
            )
        if candidate_local_radius < 0:
            raise ValueError(
                "candidate_local_radius must be non-negative, got "
                f"{candidate_local_radius}"
            )
        if not 1 <= candidate_consensus_parts <= num_parts:
            raise ValueError(
                "candidate_consensus_parts must be in [1, num_parts], got "
                f"{candidate_consensus_parts} for {num_parts} parts"
            )
        if alpha_max < 0.0:
            raise ValueError(
                f"alpha_max must be non-negative, got {alpha_max}"
            )
        if not 0.0 < part_route_residual_floor <= 1.0:
            raise ValueError(
                "part_route_residual_floor must be in (0, 1], got "
                f"{part_route_residual_floor}"
            )
        if candidate_modulation_max < 0.0:
            raise ValueError(
                "candidate_modulation_max must be non-negative, got "
                f"{candidate_modulation_max}"
            )
        if (
            spatial_gate_mode == "part_aligned_guidance"
            and not 0.0 < candidate_modulation_max <= 0.5
        ):
            raise ValueError(
                "part_aligned_guidance requires "
                "candidate_modulation_max in (0, 0.5]"
            )

        self.num_parts = num_parts
        self.part_grid = part_grid
        self.topk = topk
        self.reliability_mode = reliability_mode
        self.nms_radius = nms_radius
        self.residual_max_ratio = float(residual_max_ratio)
        self.spatial_gate_mode = spatial_gate_mode
        self.candidate_local_radius = int(candidate_local_radius)
        self.candidate_consensus_parts = int(candidate_consensus_parts)
        self.part_route_residual_floor = float(part_route_residual_floor)
        self.candidate_modulation_max = float(candidate_modulation_max)
        self.alpha_max = float(alpha_max)
        self.eps = eps

        # A shared monotonic calibration is enough for the first version.
        # softplus(log_match_scale) keeps higher similarity mapped to higher
        # reliability without adding an MLP or another gating branch.
        initial_scale_tensor = torch.tensor(float(initial_match_scale))
        self.log_match_scale = nn.Parameter(
            torch.log(torch.expm1(initial_scale_tensor))
        )
        self.match_bias = nn.Parameter(torch.tensor(float(initial_match_bias)))

        # Candidate calibration is used by V7, V9, V10, and V14. Keeping these
        # parameters absent in the other modes preserves every previous
        # checkpoint contract.
        if self.spatial_gate_mode in (
            "candidate_consensus",
            "part_aligned_consensus",
            "part_aligned_guidance",
            "part_aligned_identity_aux",
        ):
            initial_candidate_scale = torch.tensor(
                float(candidate_initial_match_scale)
            )
            self.candidate_log_match_scale = nn.Parameter(
                torch.log(torch.expm1(initial_candidate_scale))
            )
            self.candidate_match_bias = nn.Parameter(
                torch.tensor(float(candidate_initial_match_bias))
            )
        else:
            self.register_parameter("candidate_log_match_scale", None)
            self.register_parameter("candidate_match_bias", None)

        # V8-V18 calibrate each part-to-token similarity independently. These
        # parameters remain absent from every earlier forward path.
        if self.spatial_gate_mode in (
            "part_aligned",
            "part_aligned_consensus",
            "part_aligned_guidance",
            "part_aligned_sharpened",
            "part_aligned_positive_preserved",
            "part_aligned_reliability_safe",
            "part_aligned_identity_aux",
            "part_aligned_bidirectional",
            "part_aligned_coherent_prototype",
        ):
            initial_part_route_scale = torch.tensor(
                float(part_route_initial_match_scale)
            )
            self.part_route_log_match_scale = nn.Parameter(
                torch.log(torch.expm1(initial_part_route_scale))
            )
            self.part_route_match_bias = nn.Parameter(
                torch.tensor(float(part_route_initial_match_bias))
            )
        else:
            self.register_parameter("part_route_log_match_scale", None)
            self.register_parameter("part_route_match_bias", None)

        # V10 starts as the exact V8 residual path. Tracking loss may learn a
        # small candidate correction, but the bounded scalar and centered gate
        # cannot erase the base residual or change its training gradients.
        if self.spatial_gate_mode == "part_aligned_guidance":
            self.candidate_modulation = nn.Parameter(torch.zeros(()))
        else:
            self.register_parameter("candidate_modulation", None)

        # Zero initialization preserves the original OSTrack forward path.
        self.alpha = nn.Parameter(torch.zeros(()))
        self.alpha.requires_grad_(bool(train_alpha))
        # Runtime-only probes retain the exact trained checkpoint schema.
        self.inference_ablation = None
        self.route_head = None

    def enable_discriminative_route(self, embed_dim, projection_dim=32, hidden_dim=64):
        if self.spatial_gate_mode != 'part_aligned' or self.route_head is not None:
            raise ValueError('Rdisc requires an unmodified V8 part_aligned path')
        # Extra head initialization must not shift the training loader's RNG
        # stream relative to Rfreeze, or the two arms see different samples.
        with torch.random.fork_rng(devices=[]):
            self.route_head = DiscriminativePartRoute(embed_dim, projection_dim, hidden_dim)

    def set_inference_ablation(self, ablation: str) -> None:
        """Select one V8 residual component to neutralize at inference."""
        if self.spatial_gate_mode != "part_aligned":
            raise ValueError(
                "VDRM inference ablations require V8 part_aligned mode"
            )
        if ablation not in ("part_mean", "route_spatial_mean"):
            raise ValueError(
                "VDRM inference ablation must be 'part_mean' or "
                f"'route_spatial_mean', got {ablation!r}"
            )
        self.inference_ablation = ablation

    def _effective_alpha(self) -> torch.Tensor:
        """Return the residual LayerScale, optionally bounded for V8."""
        if self.alpha_max <= 0.0:
            return self.alpha
        return self.alpha_max * torch.tanh(self.alpha / self.alpha_max)

    def _effective_candidate_modulation(self) -> torch.Tensor:
        """Return V10's bounded residual-preserving candidate correction."""
        if self.candidate_modulation is None:
            raise RuntimeError(
                "candidate modulation is only defined for "
                "part_aligned_guidance"
            )
        return self.candidate_modulation_max * torch.tanh(
            self.candidate_modulation / self.candidate_modulation_max
        )

    def _mass_conserving_bidirectional_assignment(
        self,
        route_logits: torch.Tensor,
        v8_route_weight: torch.Tensor,
        part_valid: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Redistribute each token's V8 route mass by mutual correspondence.

        ``v8_route_weight`` already contains V8's calibrated part route and
        continuous part reliability. Token-to-part identity probability is
        therefore computed from that exact effective evidence. Part-to-token
        spatial probability is computed from the calibrated route logits.
        Combining both directions in log space avoids probability underflow.

        The resulting assignment is normalized only across parts at each
        token and multiplied by the original per-token V8 mass. Consequently,
        this operation changes prototype identity mixture without adding a
        gate or changing the scalar amount of route evidence at that token.
        """
        if route_logits.shape != v8_route_weight.shape:
            raise ValueError(
                "route_logits and v8_route_weight must have equal shape, "
                f"got {tuple(route_logits.shape)} and "
                f"{tuple(v8_route_weight.shape)}"
            )
        if part_valid.shape != route_logits.shape[:2]:
            raise ValueError(
                "part_valid must have shape [B, K], got "
                f"{tuple(part_valid.shape)} for route logits "
                f"{tuple(route_logits.shape)}"
            )

        valid_mask = part_valid.unsqueeze(-1)
        valid_float = valid_mask.to(route_logits.dtype)
        very_negative = torch.finfo(route_logits.dtype).min

        identity_logits = torch.log(
            v8_route_weight.clamp_min(self.eps)
        ).masked_fill(~valid_mask, very_negative)
        identity_log_probability = F.log_softmax(identity_logits, dim=1)

        spatial_logits = route_logits.masked_fill(
            ~valid_mask, very_negative
        )
        spatial_log_probability = F.log_softmax(spatial_logits, dim=-1)

        joint_logits = (
            identity_log_probability + spatial_log_probability
        ).masked_fill(~valid_mask, very_negative)
        assignment = F.softmax(joint_logits, dim=1) * valid_float
        assignment_sum = assignment.sum(dim=1, keepdim=True)
        assignment = torch.where(
            assignment_sum > 0.0,
            assignment / assignment_sum.clamp_min(self.eps),
            torch.zeros_like(assignment),
        )

        v8_route_mass = v8_route_weight.sum(dim=1, keepdim=True)
        route_weight = assignment * v8_route_mass

        active_token = (v8_route_mass.squeeze(1) > self.eps).to(
            route_logits.dtype
        )
        active_count = active_token.sum(dim=1).clamp_min(1.0)
        v8_assignment = (
            v8_route_weight / v8_route_mass.clamp_min(self.eps)
        )
        total_variation = 0.5 * (
            assignment - v8_assignment
        ).abs().sum(dim=1)
        assignment_entropy = -(
            assignment * torch.log(assignment.clamp_min(self.eps))
        ).sum(dim=1)
        valid_count = part_valid.sum(dim=1).to(route_logits.dtype)
        entropy_normalizer = torch.log(valid_count.clamp_min(2.0))
        normalized_entropy = torch.where(
            valid_count[:, None] > 1.0,
            assignment_entropy / entropy_normalizer[:, None],
            torch.zeros_like(assignment_entropy),
        )
        mass_error = (
            route_weight.sum(dim=1, keepdim=True) - v8_route_mass
        ).abs()
        diagnostics = {
            "part_assignment_total_variation": (
                (total_variation * active_token).sum(dim=1) / active_count
            ),
            "part_assignment_entropy": (
                (normalized_entropy * active_token).sum(dim=1)
                / active_count
            ),
            "part_assignment_mass_error_max": mass_error.amax(dim=(1, 2)),
        }
        return route_weight, diagnostics

    def _part_aligned_statistics(
        self,
        similarity: torch.Tensor,
        prototypes: torch.Tensor,
        part_reliability: torch.Tensor,
        part_valid: torch.Tensor,
        route_logits_override=None,
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]
    ]:
        """Route each part and build its residual from the same evidence.

        Unlike V7, this path has no center-candidate gate. A calibrated map is
        produced for every template part at the exact search locations where
        that part's prototype is injected. Division by the fixed number of
        valid parts keeps the residual scale identifiable without normalizing
        away background suppression.
        """
        route_scale = F.softplus(self.part_route_log_match_scale)
        route_logits = (
            route_scale * similarity + self.part_route_match_bias
        )
        if route_logits_override is not None:
            if route_logits_override.shape != route_logits.shape:
                raise ValueError('discriminative route logits shape mismatch')
            route_logits = route_logits_override
        route_gate = torch.sigmoid(route_logits)
        route_gate = route_gate * part_valid.unsqueeze(-1).to(
            route_gate.dtype
        )
        # V11-V13 sharpen only the route used by the residual. The raw sigmoid
        # remains unchanged for the balanced part-route objective and its
        # diagnostics. At a learned route probability ``p``, V11 retains
        # ``floor + (1 - floor) * p`` of V8's contribution. Consequently,
        # confident target routes remain nearly unchanged while the small
        # probability accumulated over many background tokens is strongly
        # reduced. Detaching the retention factor prevents the tracking loss
        # from exploiting the sharpening nonlinearity to saturate route
        # logits; V8's original route gradient is merely scaled in
        # ``[floor, 1]``.
        route_retention = torch.ones_like(route_gate)
        residual_route_gate = route_gate
        residual_route_diagnostics = {}
        if self.inference_ablation == "route_spatial_mean":
            # Preserve each part's mean route mass while removing its spatial
            # selectivity. Diagnostics continue to report the original map.
            residual_route_gate = route_gate.mean(
                dim=-1, keepdim=True
            ).expand_as(route_gate)
        if self.spatial_gate_mode in (
            "part_aligned_sharpened",
            "part_aligned_positive_preserved",
            "part_aligned_reliability_safe",
        ):
            route_retention = self.part_route_residual_floor + (
                (1.0 - self.part_route_residual_floor) * route_gate.detach()
            )
            residual_route_gate = route_gate * route_retention

        if self.spatial_gate_mode == "part_aligned_positive_preserved":
            # V12 uses the calibrated classifier boundary, not another tuned
            # threshold, to identify positive route evidence. It restores the
            # aggregate V8 route mass on that side while retaining V11's
            # relative sharpening. Both the mask and the compensation are
            # detached: the tracking loss sees only a bounded rescaling of
            # V8's original route gradient and receives no gradient through
            # the positive-set selection or its normalization factor.
            positive_mask = (
                route_gate.detach() >= 0.5
            ) & part_valid.unsqueeze(-1)
            positive_mask_float = positive_mask.to(route_gate.dtype)
            positive_mass_v8 = (
                route_gate.detach() * positive_mask_float
            ).sum(dim=-1, keepdim=True)
            positive_mass_sharpened = (
                residual_route_gate.detach() * positive_mask_float
            ).sum(dim=-1, keepdim=True)
            positive_present = positive_mask.any(dim=-1, keepdim=True)
            preservation_scale = torch.where(
                positive_present,
                positive_mass_v8
                / positive_mass_sharpened.clamp_min(self.eps),
                torch.ones_like(positive_mass_v8),
            )
            residual_route_gate = residual_route_gate * preservation_scale
            route_retention = route_retention * preservation_scale

            positive_mass_preserved = (
                residual_route_gate.detach() * positive_mask_float
            ).sum(dim=-1, keepdim=True)
            positive_mass_ratio = torch.where(
                positive_present,
                positive_mass_preserved
                / positive_mass_v8.clamp_min(self.eps),
                torch.ones_like(positive_mass_v8),
            )
            residual_route_diagnostics.update({
                "part_route_positive_preservation_scale": (
                    preservation_scale
                ),
                "part_route_positive_mass_ratio": positive_mass_ratio,
                "part_route_positive_present": positive_present,
            })

        residual_route_diagnostics[
            "part_route_residual_retention"
        ] = route_retention

        # V13 leaves confident parts exactly on the V11 path and applies only
        # a detached, monotone safety attenuation to uncertain parts. The
        # sigmoid decision boundary supplies the fixed 0.5 pivot: s(r)=1 for
        # r>=0.5 and s(r)=2r otherwise. Consequently r*s(r) is continuous,
        # non-decreasing, and never exceeds V11's original reliability weight.
        # Applying this factor per part avoids suppressing reliable visible
        # parts because another part is occluded. Detaching it preserves the
        # existing reliability objective instead of giving tracking loss a
        # shortcut through the safety path.
        part_reliability_safety_factor = torch.ones_like(part_reliability)
        if self.spatial_gate_mode == "part_aligned_reliability_safe":
            part_reliability_safety_factor = (
                2.0 * part_reliability.detach()
            ).clamp(max=1.0)
            part_reliability_safety_factor = (
                part_reliability_safety_factor
                * part_valid.to(part_reliability_safety_factor.dtype)
            )
            residual_route_diagnostics[
                "part_reliability_safety_factor"
            ] = part_reliability_safety_factor

        route_weight = (
            residual_route_gate
            * part_reliability.unsqueeze(-1)
            * part_reliability_safety_factor.unsqueeze(-1)
        )
        if self.spatial_gate_mode == "part_aligned_bidirectional":
            route_weight, assignment_diagnostics = (
                self._mass_conserving_bidirectional_assignment(
                    route_logits,
                    route_weight,
                    part_valid,
                )
            )
            residual_route_diagnostics.update(assignment_diagnostics)
        residual = torch.einsum(
            "bkl,bkc->blc", route_weight, prototypes
        )
        valid_count = part_valid.sum(dim=1, keepdim=True).clamp_min(1)
        residual = residual / valid_count.unsqueeze(-1).to(residual.dtype)
        return route_logits, route_gate, residual, residual_route_diagnostics

    def _candidate_consensus_statistics(
        self,
        part_evidence: torch.Tensor,
        part_reliability: torch.Tensor,
        part_valid: torch.Tensor,
        search_global_index: torch.Tensor,
        search_grid_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build a candidate-level reliability gate from local part agreement.

        Each retained token is treated as a candidate location. For every
        template part, the strongest positive evidence in a fixed local
        neighbourhood is collected on the original search grid. V7 and V14
        supply raw similarities, while V9/V10 supply supervised part-route
        probabilities.
        The gate is calibrated from the strongest
        ``candidate_consensus_parts`` values, so one isolated part or
        spatially scattered evidence cannot enable the residual by itself.
        """
        if search_grid_size < 1:
            raise ValueError(
                f"search_grid_size must be positive, got {search_grid_size}"
            )
        batch_size, num_parts, search_length = part_evidence.shape
        if search_global_index.shape != (batch_size, search_length):
            raise ValueError(
                "search_global_index must have shape [B, L_s], got "
                f"{tuple(search_global_index.shape)} for part evidence "
                f"{tuple(part_evidence.shape)}"
            )

        global_index = search_global_index.to(
            device=part_evidence.device, dtype=torch.long
        )
        full_length = search_grid_size ** 2
        if (
            global_index.numel() == 0
            or global_index.min() < 0
            or global_index.max() >= full_length
        ):
            raise ValueError(
                "search_global_index contains values outside the original "
                f"{search_grid_size}x{search_grid_size} search grid"
            )

        weighted_evidence = (
            part_evidence.clamp_min(0.0) * part_reliability.unsqueeze(-1)
        )
        weighted_evidence = weighted_evidence * part_valid.unsqueeze(
            -1
        ).to(weighted_evidence.dtype)
        full_evidence = part_evidence.new_zeros(
            batch_size, num_parts, full_length
        )
        full_evidence.scatter_(
            2,
            global_index[:, None, :].expand(-1, num_parts, -1),
            weighted_evidence,
        )
        full_evidence = full_evidence.view(
            batch_size, num_parts, search_grid_size, search_grid_size
        )

        radius = self.candidate_local_radius
        if radius > 0:
            local_evidence = F.max_pool2d(
                full_evidence,
                kernel_size=2 * radius + 1,
                stride=1,
                padding=radius,
            )
        else:
            local_evidence = full_evidence
        local_evidence = local_evidence.flatten(2).gather(
            2, global_index[:, None, :].expand(-1, num_parts, -1)
        )

        consensus_values = local_evidence.topk(
            self.candidate_consensus_parts, dim=1
        ).values
        candidate_evidence = consensus_values.mean(dim=1)
        consensus_valid = (
            part_valid.sum(dim=1) >= self.candidate_consensus_parts
        )
        candidate_scale = F.softplus(self.candidate_log_match_scale)
        candidate_logits = (
            candidate_scale * candidate_evidence + self.candidate_match_bias
        )
        candidate_logits = torch.where(
            consensus_valid[:, None],
            candidate_logits,
            candidate_logits.new_full(candidate_logits.shape, -20.0),
        )
        candidate_gate = torch.sigmoid(candidate_logits)

        candidate_map = part_evidence.new_zeros(batch_size, full_length)
        candidate_map.scatter_(1, global_index, candidate_gate)
        candidate_map = candidate_map.view(
            batch_size, 1, search_grid_size, search_grid_size
        )
        return (
            candidate_logits,
            candidate_gate,
            candidate_map,
            consensus_valid,
        )

    def _margin_statistics(
        self,
        similarity: torch.Tensor,
        search_global_index: torch.Tensor,
        search_grid_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the first peak, spatially distinct hard peak, and margin."""
        if search_grid_size < 1:
            raise ValueError(
                f"search_grid_size must be positive, got {search_grid_size}"
            )
        if search_global_index.shape != (
            similarity.shape[0], similarity.shape[2]
        ):
            raise ValueError(
                "search_global_index must have shape [B, L_s], got "
                f"{tuple(search_global_index.shape)} for similarity "
                f"{tuple(similarity.shape)}"
            )

        global_index = search_global_index.to(
            device=similarity.device, dtype=torch.long
        )
        if (
            global_index.numel() == 0
            or global_index.min() < 0
            or global_index.max() >= search_grid_size ** 2
        ):
            raise ValueError(
                "search_global_index contains values outside the original "
                f"{search_grid_size}x{search_grid_size} search grid"
            )

        peak_similarity, peak_local_index = similarity.max(dim=-1)
        expanded_global_index = global_index[:, None, :].expand(
            -1, similarity.shape[1], -1
        )
        peak_global_index = expanded_global_index.gather(
            dim=2, index=peak_local_index.unsqueeze(-1)
        ).squeeze(-1)

        token_x = global_index.remainder(search_grid_size)[:, None, :]
        token_y = global_index.div(
            search_grid_size, rounding_mode="floor"
        )[:, None, :]
        peak_x = peak_global_index.remainder(search_grid_size).unsqueeze(-1)
        peak_y = peak_global_index.div(
            search_grid_size, rounding_mode="floor"
        ).unsqueeze(-1)
        same_peak_neighborhood = (
            (token_x - peak_x).abs() <= self.nms_radius
        ) & ((token_y - peak_y).abs() <= self.nms_radius)

        hard_negative_similarity = similarity.masked_fill(
            same_peak_neighborhood, -torch.inf
        ).amax(dim=-1)
        has_hard_negative = (~same_peak_neighborhood).any(dim=-1)
        hard_negative_similarity = torch.where(
            has_hard_negative,
            hard_negative_similarity,
            peak_similarity,
        )
        match_margin = (peak_similarity - hard_negative_similarity).clamp_min(
            0.0
        )
        return peak_similarity, hard_negative_similarity, match_margin

    def _default_template_bbox(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Return a conservative centered template box for API fallback."""
        return torch.tensor(
            [0.25, 0.25, 0.5, 0.5], device=device, dtype=dtype
        ).expand(batch_size, -1)

    def _identity_coherent_part_prototypes(
        self,
        template_tokens: torch.Tensor,
        part_masks: torch.Tensor,
        part_valid: torch.Tensor,
    ) -> Tuple[
        torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]
    ]:
        """Refine V8 part means using detached within-part coherence.

        The V8 uniform prototype is used only as a read-only identity anchor.
        Cosine scores and their softmax weights are detached, preventing the
        backbone from learning to manipulate the selector. Gradients still
        reach every template token through its strictly positive weighted
        contribution to the refined prototype.
        """
        if template_tokens.ndim != 3:
            raise ValueError(
                "template_tokens must have shape [B, L, C], got "
                f"{tuple(template_tokens.shape)}"
            )
        expected_mask_shape = (
            template_tokens.shape[0],
            self.num_parts,
            template_tokens.shape[1],
        )
        if part_masks.shape != expected_mask_shape:
            raise ValueError(
                "part_masks must have shape [B, K, L], got "
                f"{tuple(part_masks.shape)} instead of "
                f"{expected_mask_shape}"
            )
        if part_valid.shape != expected_mask_shape[:2]:
            raise ValueError(
                "part_valid must have shape [B, K], got "
                f"{tuple(part_valid.shape)}"
            )

        mask_float = part_masks.to(
            device=template_tokens.device, dtype=template_tokens.dtype
        )
        mask_bool = mask_float.gt(0.0)
        valid = part_valid.to(
            device=template_tokens.device, dtype=torch.bool
        )
        part_count = mask_float.sum(dim=-1, keepdim=True).clamp_min(1.0)
        uniform_prototypes = torch.einsum(
            "bkl,blc->bkc", mask_float, template_tokens.detach()
        ) / part_count
        uniform_prototypes = uniform_prototypes * valid.unsqueeze(-1).to(
            uniform_prototypes.dtype
        )

        detached_tokens = F.normalize(
            template_tokens.detach(), dim=-1, eps=self.eps
        )
        detached_uniform = F.normalize(
            uniform_prototypes, dim=-1, eps=self.eps
        )
        coherence = torch.einsum(
            "bkc,blc->bkl", detached_uniform, detached_tokens
        )
        very_negative = torch.finfo(coherence.dtype).min
        coherence = coherence.masked_fill(~mask_bool, very_negative)
        weights = F.softmax(coherence, dim=-1) * mask_float
        weights = torch.where(
            valid.unsqueeze(-1),
            weights / weights.sum(dim=-1, keepdim=True).clamp_min(self.eps),
            torch.zeros_like(weights),
        ).detach()

        prototypes = torch.einsum(
            "bkl,blc->bkc", weights, template_tokens
        )
        prototypes = prototypes * valid.unsqueeze(-1).to(prototypes.dtype)

        token_count = mask_float.sum(dim=-1)
        weight_entropy = -(
            weights * weights.clamp_min(self.eps).log()
        ).sum(dim=-1)
        normalized_entropy = torch.where(
            token_count > 1.0,
            weight_entropy / token_count.clamp_min(2.0).log(),
            torch.ones_like(weight_entropy),
        )
        effective_token_fraction = 1.0 / (
            token_count.clamp_min(1.0)
            * weights.square().sum(dim=-1).clamp_min(self.eps)
        )
        effective_token_fraction = torch.where(
            valid,
            effective_token_fraction,
            torch.zeros_like(effective_token_fraction),
        )
        cosine_to_uniform = F.cosine_similarity(
            prototypes.detach(), uniform_prototypes, dim=-1, eps=self.eps
        )
        max_weight = weights.amax(dim=-1)

        valid_float = valid.to(template_tokens.dtype)
        valid_count = valid_float.sum(dim=-1).clamp_min(1.0)

        def valid_mean(value: torch.Tensor) -> torch.Tensor:
            return (value * valid_float).sum(dim=-1) / valid_count

        has_valid = valid.any(dim=-1)
        diagnostics = {
            "prototype_weight_entropy": torch.where(
                has_valid,
                valid_mean(normalized_entropy),
                torch.zeros_like(valid_count),
            ),
            "prototype_effective_token_fraction": torch.where(
                has_valid,
                valid_mean(effective_token_fraction),
                torch.zeros_like(valid_count),
            ),
            "prototype_cosine_to_uniform": torch.where(
                has_valid,
                valid_mean(cosine_to_uniform),
                torch.zeros_like(valid_count),
            ),
            "prototype_max_weight": torch.where(
                has_valid,
                max_weight.masked_fill(~valid, 0.0).amax(dim=-1),
                torch.zeros_like(valid_count),
            ),
        }
        return prototypes, weights, diagnostics

    def _build_part_masks(
        self,
        template_bbox: torch.Tensor,
        template_length: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create fixed target-local grid masks on the template token grid.

        Args:
            template_bbox: normalized ``xywh`` boxes with shape ``[B, 4]``.
            template_length: number of template tokens.

        Returns:
            part_masks: float masks with shape ``[B, K, L_t]``.
            part_valid: whether each part owns at least one token, ``[B, K]``.
        """
        template_size = int(math.sqrt(template_length))
        if template_size * template_size != template_length:
            raise ValueError(
                "VDRM requires a square template token grid, "
                f"got {template_length} tokens"
            )

        bbox = template_bbox.to(dtype=torch.float32).clamp(0.0, 1.0)
        x0, y0, width, height = bbox.unbind(dim=-1)
        x1 = (x0 + width).clamp(0.0, 1.0)
        y1 = (y0 + height).clamp(0.0, 1.0)

        coord = (
            torch.arange(template_size, device=bbox.device, dtype=bbox.dtype)
            + 0.5
        ) / template_size
        grid_y, grid_x = torch.meshgrid(coord, coord, indexing="ij")
        grid_x = grid_x.flatten().view(1, 1, -1)
        grid_y = grid_y.flatten().view(1, 1, -1)

        part_masks = []
        for row in range(self.part_grid):
            part_y0 = y0 + (y1 - y0) * (row / self.part_grid)
            part_y1 = y0 + (y1 - y0) * ((row + 1) / self.part_grid)
            for col in range(self.part_grid):
                part_x0 = x0 + (x1 - x0) * (col / self.part_grid)
                part_x1 = x0 + (x1 - x0) * ((col + 1) / self.part_grid)
                mask = (
                    (grid_x >= part_x0[:, None, None])
                    & (grid_x < part_x1[:, None, None])
                    & (grid_y >= part_y0[:, None, None])
                    & (grid_y < part_y1[:, None, None])
                )
                part_masks.append(mask.squeeze(1))

        part_masks = torch.stack(part_masks, dim=1)
        part_valid = part_masks.any(dim=-1)
        return part_masks.to(dtype=template_bbox.dtype), part_valid

    def forward(
        self,
        tokens: torch.Tensor,
        template_length: int,
        template_bbox: Optional[torch.Tensor] = None,
        search_global_index: Optional[torch.Tensor] = None,
        search_grid_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Refine search tokens and return reliability diagnostics.

        ``tokens`` must be ordered as template tokens followed by the currently
        retained search tokens, which is the default direct concatenation used
        by OSTrack-CE.
        """
        if tokens.ndim != 3:
            raise ValueError(f"tokens must have shape [B, L, C], got {tokens.shape}")
        if not 0 < template_length < tokens.shape[1]:
            raise ValueError(
                f"invalid template length {template_length} for {tokens.shape[1]} tokens"
            )

        batch_size = tokens.shape[0]
        if template_bbox is None:
            template_bbox = self._default_template_bbox(
                batch_size, tokens.device, tokens.dtype
            )
        else:
            template_bbox = template_bbox.reshape(batch_size, 4).to(
                device=tokens.device, dtype=tokens.dtype
            )

        template_tokens = tokens[:, :template_length]
        search_tokens = tokens[:, template_length:]
        part_masks, part_valid = self._build_part_masks(
            template_bbox, template_length
        )
        part_masks = part_masks.to(dtype=template_tokens.dtype)

        part_count = part_masks.sum(dim=-1, keepdim=True).clamp_min(1.0)
        prototypes = torch.einsum(
            "bkl,blc->bkc", part_masks, template_tokens
        ) / part_count
        prototypes = prototypes * part_valid.unsqueeze(-1).to(prototypes.dtype)
        prototype_diagnostics = {}
        if self.spatial_gate_mode == "part_aligned_coherent_prototype":
            prototypes, _, prototype_diagnostics = (
                self._identity_coherent_part_prototypes(
                    template_tokens,
                    part_masks,
                    part_valid,
                )
            )

        normalized_prototypes = F.normalize(prototypes, dim=-1, eps=self.eps)
        normalized_search = F.normalize(search_tokens, dim=-1, eps=self.eps)
        similarity = torch.einsum(
            "bkc,blc->bkl", normalized_prototypes, normalized_search
        )

        margin_diagnostics = {}
        if self.reliability_mode == "topk":
            match_topk = min(self.topk, search_tokens.shape[1])
            reliability_evidence = similarity.topk(
                match_topk, dim=-1
            ).values.mean(dim=-1)
        else:
            if search_global_index is None or search_grid_size is None:
                raise ValueError(
                    "margin reliability requires search_global_index and "
                    "search_grid_size"
                )
            peak_similarity, hard_negative_similarity, match_margin = (
                self._margin_statistics(
                    similarity,
                    search_global_index,
                    int(search_grid_size),
                )
            )
            reliability_evidence = match_margin
            margin_diagnostics = {
                "part_peak_similarity": peak_similarity,
                "part_hard_negative_similarity": hard_negative_similarity,
                "part_match_margin": match_margin,
                "part_similarity": similarity,
                "search_global_index": search_global_index,
            }
        match_scale = F.softplus(self.log_match_scale)
        part_reliability = torch.sigmoid(
            match_scale * reliability_evidence + self.match_bias
        )
        part_reliability = (
            part_reliability * part_valid.to(part_reliability.dtype)
        )
        residual_part_reliability = part_reliability
        if self.inference_ablation == "part_mean":
            # Preserve per-frame mean gate strength, but remove the relative
            # preference between template parts. Raw q remains diagnostic.
            valid_count = part_valid.sum(dim=-1, keepdim=True).clamp_min(1)
            mean_reliability = (
                part_reliability.sum(dim=-1, keepdim=True) / valid_count
            )
            residual_part_reliability = (
                mean_reliability * part_valid.to(part_reliability.dtype)
            )

        route_diagnostics = {}
        part_route_gate = None
        route_logits_override = None
        if self.route_head is not None:
            correction = self.route_head(
                prototypes, search_tokens, search_global_index, search_grid_size,
            )
            prior = F.softplus(self.part_route_log_match_scale) * similarity + self.part_route_match_bias
            route_logits_override = prior + correction
            # Auxiliary supervision is intentionally independent of every
            # original scalar, alpha, and the visual feature graph.
            discriminative_route_logits = prior.detach() + correction
        if self.spatial_gate_mode in (
            "part_aligned",
            "part_aligned_consensus",
            "part_aligned_guidance",
            "part_aligned_sharpened",
            "part_aligned_positive_preserved",
            "part_aligned_reliability_safe",
            "part_aligned_identity_aux",
            "part_aligned_bidirectional",
            "part_aligned_coherent_prototype",
        ):
            (
                part_route_logits,
                part_route_gate,
                residual,
                residual_route_diagnostics,
            ) = (
                self._part_aligned_statistics(
                    similarity,
                    prototypes,
                    residual_part_reliability,
                    part_valid,
                    route_logits_override=route_logits_override,
                )
            )
            route_diagnostics = {
                "part_route_logits": part_route_logits,
                "part_route_gate": part_route_gate,
                "part_similarity": similarity,
                "search_global_index": search_global_index,
            }
            if self.route_head is not None:
                route_diagnostics['discriminative_route_logits'] = discriminative_route_logits
            if self.spatial_gate_mode in (
                "part_aligned_sharpened",
                "part_aligned_positive_preserved",
                "part_aligned_reliability_safe",
            ):
                part_route_residual_retention = (
                    residual_route_diagnostics[
                        "part_route_residual_retention"
                    ]
                )
                route_diagnostics.update({
                    "part_route_residual_retention_mean": (
                        part_route_residual_retention.mean(dim=(1, 2))
                    ),
                    "part_route_residual_retention_min": (
                        part_route_residual_retention.amin(dim=(1, 2))
                    ),
                    "part_route_residual_retention_max": (
                        part_route_residual_retention.amax(dim=(1, 2))
                    ),
                })
            if self.spatial_gate_mode == "part_aligned_reliability_safe":
                safety_factor = residual_route_diagnostics[
                    "part_reliability_safety_factor"
                ]
                valid_float = part_valid.to(safety_factor.dtype)
                valid_count = valid_float.sum(dim=1).clamp_min(1.0)
                safety_min = safety_factor.masked_fill(~part_valid, 1.0)
                safety_max = safety_factor.masked_fill(~part_valid, 0.0)
                has_valid_part = part_valid.any(dim=1)
                safety_mean = (
                    (safety_factor * valid_float).sum(dim=1)
                    / valid_count
                )
                route_diagnostics.update({
                    "part_reliability_safety_factor_mean": (
                        torch.where(
                            has_valid_part,
                            safety_mean,
                            torch.ones_like(valid_count),
                        )
                    ),
                    "part_reliability_safety_factor_min": (
                        torch.where(
                            has_valid_part,
                            safety_min.amin(dim=1),
                            torch.ones_like(valid_count),
                        )
                    ),
                    "part_reliability_safety_factor_max": (
                        torch.where(
                            has_valid_part,
                            safety_max.amax(dim=1),
                            torch.ones_like(valid_count),
                        )
                    ),
                    "part_reliability_suppressed_fraction": (
                        (
                            (safety_factor < 1.0) & part_valid
                        ).to(safety_factor.dtype).sum(dim=1)
                        / valid_count
                    ),
                    "part_reliability_suppression_mean": (
                        ((1.0 - safety_factor) * valid_float).sum(dim=1)
                        / valid_count
                    ),
                })
            if self.spatial_gate_mode == "part_aligned_positive_preserved":
                preservation_scale = residual_route_diagnostics[
                    "part_route_positive_preservation_scale"
                ]
                positive_mass_ratio = residual_route_diagnostics[
                    "part_route_positive_mass_ratio"
                ]
                positive_present = residual_route_diagnostics[
                    "part_route_positive_present"
                ]
                route_diagnostics.update({
                    "part_route_positive_preservation_scale_mean": (
                        preservation_scale.mean(dim=(1, 2))
                    ),
                    "part_route_positive_preservation_scale_min": (
                        preservation_scale.amin(dim=(1, 2))
                    ),
                    "part_route_positive_preservation_scale_max": (
                        preservation_scale.amax(dim=(1, 2))
                    ),
                    "part_route_positive_mass_ratio": (
                        positive_mass_ratio.mean(dim=(1, 2))
                    ),
                    "part_route_positive_part_fraction": (
                        positive_present.to(
                            part_route_gate.dtype
                        ).mean(dim=(1, 2))
                    ),
                })
            if self.spatial_gate_mode == "part_aligned_bidirectional":
                route_diagnostics.update({
                    "part_assignment_total_variation": (
                        residual_route_diagnostics[
                            "part_assignment_total_variation"
                        ]
                    ),
                    "part_assignment_entropy": (
                        residual_route_diagnostics[
                            "part_assignment_entropy"
                        ]
                    ),
                    "part_assignment_mass_error_max": (
                        residual_route_diagnostics[
                            "part_assignment_mass_error_max"
                        ]
                    ),
                })
        else:
            part_attention = torch.softmax(similarity, dim=1)
            weighted_prototypes = (
                prototypes * part_reliability.unsqueeze(-1)
            )
            reconstructed = torch.einsum(
                "bkl,bkc->blc", part_attention, weighted_prototypes
            )
            token_match = similarity.max(dim=1).values.clamp_min(0.0)
            residual = reconstructed * token_match.unsqueeze(-1)
        candidate_diagnostics = {}
        if self.spatial_gate_mode in (
            "candidate_consensus",
            "part_aligned_consensus",
            "part_aligned_guidance",
            "part_aligned_identity_aux",
        ):
            if search_global_index is None or search_grid_size is None:
                raise ValueError(
                    "candidate-consensus readout requires search_global_index "
                    "and search_grid_size"
                )
            candidate_evidence = similarity
            candidate_reliability = part_reliability
            if self.spatial_gate_mode in (
                "part_aligned_consensus",
                "part_aligned_guidance",
            ):
                if part_route_gate is None:
                    raise RuntimeError(
                        "part-aligned candidate guidance requires "
                        "part-route evidence"
                    )
                candidate_evidence = part_route_gate
            if self.spatial_gate_mode == "part_aligned_guidance":
                # Candidate focal supervision calibrates only its own branch.
                # It must not repeat V9's reduction of positive part-route
                # probabilities or change backbone features.
                candidate_evidence = candidate_evidence.detach()
                candidate_reliability = candidate_reliability.detach()
            if self.spatial_gate_mode == "part_aligned_identity_aux":
                # V14 is an auxiliary identity objective on raw cosine part
                # evidence. Its candidate map is deliberately absent from the
                # residual forward path. Detaching only reliability prevents
                # this loss from moving V8's global match calibration while
                # preserving identity gradients to template/search features.
                # Raw similarity also bypasses the part-route calibration, so
                # candidate focal loss has no direct V9-style shortcut through
                # the route scale and bias.
                candidate_reliability = candidate_reliability.detach()
            (
                candidate_logits,
                candidate_gate,
                candidate_map,
                consensus_valid,
            ) = self._candidate_consensus_statistics(
                candidate_evidence,
                candidate_reliability,
                part_valid,
                search_global_index,
                int(search_grid_size),
            )
            if self.spatial_gate_mode == "part_aligned_consensus":
                # The tracking objective must not learn to close the spatial
                # gate and compensate with LayerScale. Candidate focal loss
                # still trains the gate (and its part-route evidence) through
                # candidate_logits, while the bounded alpha learns only the
                # magnitude of an accepted residual.
                residual = residual * candidate_gate.detach().unsqueeze(-1)
            candidate_diagnostics = {
                "candidate_identity_logits": candidate_logits,
                "candidate_reliability_map": candidate_map,
                "candidate_consensus_valid": consensus_valid,
                "candidate_reliability_peak": candidate_gate.max(dim=1).values,
                "candidate_reliability_mean": candidate_gate.mean(dim=1),
                "search_global_index": search_global_index,
            }
            if self.spatial_gate_mode == "part_aligned_guidance":
                # Centering prevents a sparse gate from globally shrinking the
                # residual. Detaching both inputs leaves the V8 base route's
                # tracking gradient unchanged; only this scalar learns from
                # the correction. With max=0.5 every token retains a forward
                # factor in [0.5, 1.5], independent of gate calibration.
                centered_candidate_gate = candidate_gate.detach()
                centered_candidate_gate = centered_candidate_gate - (
                    centered_candidate_gate.mean(dim=1, keepdim=True)
                )
                candidate_modulation = (
                    self._effective_candidate_modulation()
                )
                modulation_factor = 1.0 + (
                    candidate_modulation * centered_candidate_gate
                )
                candidate_correction = (
                    residual.detach()
                    * centered_candidate_gate.unsqueeze(-1)
                )
                residual = residual + (
                    candidate_modulation * candidate_correction
                )
                candidate_diagnostics.update({
                    "candidate_modulation": candidate_modulation,
                    "candidate_modulation_raw": self.candidate_modulation,
                    "candidate_modulation_factor_min": (
                        modulation_factor.min(dim=1).values
                    ),
                    "candidate_modulation_factor_max": (
                        modulation_factor.max(dim=1).values
                    ),
                })
        effective_alpha = self._effective_alpha()
        raw_delta = effective_alpha * residual

        # V4/V15 bound the *complete* update after alpha, so the learned scalar
        # cannot compensate for the bound by growing in magnitude. The
        # reference norm is detached to prevent the backbone from increasing
        # token norms merely to relax the constraint. A ratio of zero retains
        # the exact unbounded forward path and checkpoint behavior.
        reference_norm = torch.linalg.vector_norm(
            search_tokens.detach(), dim=-1, keepdim=True
        )
        raw_delta_norm = torch.linalg.vector_norm(
            raw_delta, dim=-1, keepdim=True
        )
        if self.residual_max_ratio > 0.0:
            max_delta_norm = self.residual_max_ratio * reference_norm
            clip_scale = (
                max_delta_norm / raw_delta_norm.clamp_min(self.eps)
            ).clamp(max=1.0)
            delta = raw_delta * clip_scale
            residual_clip_rate = (
                raw_delta_norm > max_delta_norm
            ).to(raw_delta.dtype).mean()
        else:
            clip_scale = torch.ones_like(raw_delta_norm)
            delta = raw_delta
            residual_clip_rate = raw_delta.new_zeros(())
        residual_clip_scale_mean = clip_scale.mean()
        residual_clip_scale_min = clip_scale.amin()

        safe_reference_norm = reference_norm.clamp_min(self.eps)
        raw_delta_relative_norm = (
            raw_delta_norm / safe_reference_norm
        ).mean()
        delta_relative_norm = (
            torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
            / safe_reference_norm
        ).mean()

        residual_concentration_diagnostics = {}
        if self.spatial_gate_mode in (
            "part_aligned",
            "part_aligned_consensus",
            "part_aligned_guidance",
            "part_aligned_sharpened",
            "part_aligned_positive_preserved",
            "part_aligned_reliability_safe",
            "part_aligned_bidirectional",
            "part_aligned_coherent_prototype",
        ):
            # Measure before LayerScale so zero initialization cannot hide a
            # collapsed routing map during early V8-V18 training.
            residual_energy = residual.square().sum(dim=-1)
            residual_energy_sum = residual_energy.sum(dim=1).clamp_min(
                self.eps
            )
            top_count = max(
                1, int(math.ceil(0.1 * residual_energy.shape[1]))
            )
            residual_top10_energy_fraction = (
                residual_energy.topk(top_count, dim=1).values.sum(dim=1)
                / residual_energy_sum
            )
            residual_probability = (
                residual_energy / residual_energy_sum[:, None]
            )
            if residual_energy.shape[1] > 1:
                residual_spatial_entropy = -(
                    residual_probability
                    * residual_probability.clamp_min(self.eps).log()
                ).sum(dim=1) / math.log(residual_energy.shape[1])
            else:
                residual_spatial_entropy = residual_energy.new_zeros(
                    batch_size
                )
            residual_concentration_diagnostics = {
                "vdrm_residual_top10_energy_fraction": (
                    residual_top10_energy_fraction
                ),
                "vdrm_residual_spatial_entropy": residual_spatial_entropy,
            }

        template_tokens_out = template_tokens
        search_tokens_out = search_tokens + delta
        output_tokens = torch.cat(
            [template_tokens_out, search_tokens_out], dim=1
        )

        valid_count = part_valid.sum(dim=-1).clamp_min(1)
        visual_reliability = part_reliability.sum(dim=-1) / valid_count
        visual_reliability = torch.where(
            part_valid.any(dim=-1),
            visual_reliability,
            torch.zeros_like(visual_reliability),
        )

        diagnostics = {
            "part_reliability": part_reliability,
            "part_valid": part_valid,
            "visual_reliability": visual_reliability,
            "vdrm_alpha": effective_alpha,
            "vdrm_alpha_raw": self.alpha,
            "vdrm_residual_clip_rate": residual_clip_rate,
            "vdrm_residual_clip_scale_mean": residual_clip_scale_mean,
            "vdrm_residual_clip_scale_min": residual_clip_scale_min,
            "vdrm_raw_delta_relative_norm": raw_delta_relative_norm,
            "vdrm_delta_relative_norm": delta_relative_norm,
        }
        diagnostics.update(margin_diagnostics)
        diagnostics.update(candidate_diagnostics)
        diagnostics.update(route_diagnostics)
        diagnostics.update(residual_concentration_diagnostics)
        diagnostics.update(prototype_diagnostics)
        return output_tokens, diagnostics
