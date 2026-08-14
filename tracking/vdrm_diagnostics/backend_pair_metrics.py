"""Pure metric helpers for paired OSTrack/VDRM backend diagnostics."""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import torch


EPS = 1e-12
CORRECT_IOU_THRESHOLD = 0.5
FAILURE_IOU_THRESHOLD = 0.1
PAIR_DELTA_THRESHOLD = 0.05
RESIDUAL_METRICS = (
    "residual_active_token_fraction",
    "residual_relative_norm_mean",
    "residual_relative_norm_p50",
    "residual_relative_norm_p90",
    "residual_relative_norm_p99",
    "residual_relative_norm_max",
    "residual_top10_energy_fraction",
    "residual_spatial_entropy_normalized",
    "part_assignment_total_variation",
    "part_assignment_entropy",
    "part_assignment_mass_error_max",
    "vdrm_residual_clip_rate",
    "vdrm_residual_clip_scale_mean",
    "vdrm_residual_clip_scale_min",
    "vdrm_raw_delta_relative_norm",
    "vdrm_delta_relative_norm",
)
GUIDANCE_METRICS = (
    "candidate_modulation",
    "candidate_modulation_factor_min",
    "candidate_modulation_factor_max",
)
ROUTE_SAFEGUARD_METRICS = (
    "part_route_residual_retention_mean",
    "part_route_residual_retention_min",
    "part_route_residual_retention_max",
    "part_route_positive_preservation_scale_mean",
    "part_route_positive_preservation_scale_min",
    "part_route_positive_preservation_scale_max",
    "part_route_positive_mass_ratio",
    "part_route_positive_part_fraction",
    "part_reliability_safety_factor_mean",
    "part_reliability_safety_factor_min",
    "part_reliability_safety_factor_max",
    "part_reliability_suppressed_fraction",
    "part_reliability_suppression_mean",
)
PROTOTYPE_METRICS = (
    "prototype_weight_entropy",
    "prototype_effective_token_fraction",
    "prototype_cosine_to_uniform",
    "prototype_max_weight",
)


def valid_box(box: Sequence[float]) -> bool:
    values = np.asarray(box, dtype=np.float64).reshape(-1)
    return bool(
        values.size >= 4
        and np.isfinite(values[:4]).all()
        and values[2] > 0.0
        and values[3] > 0.0
    )


def bbox_iou_xywh(first: Sequence[float], second: Sequence[float]) -> float:
    first = np.asarray(first, dtype=np.float64).reshape(-1)[:4]
    second = np.asarray(second, dtype=np.float64).reshape(-1)[:4]
    if not valid_box(first) or not valid_box(second):
        return math.nan

    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[0] + first[2], second[0] + second[2])
    bottom = min(first[1] + first[3], second[1] + second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    union = first[2] * first[3] + second[2] * second[3] - intersection
    return float(intersection / union) if union > 0.0 else math.nan


def normalized_center_error(
    predicted_box: Sequence[float], ground_truth: Sequence[float]
) -> float:
    predicted = np.asarray(predicted_box, dtype=np.float64).reshape(-1)[:4]
    target = np.asarray(ground_truth, dtype=np.float64).reshape(-1)[:4]
    if not valid_box(predicted) or not valid_box(target):
        return math.nan
    predicted_center = predicted[:2] + 0.5 * predicted[2:]
    target_center = target[:2] + 0.5 * target[2:]
    scale = max(math.sqrt(float(target[2] * target[3])), EPS)
    return float(np.linalg.norm(predicted_center - target_center) / scale)


def normalized_box_center_shift(
    first: Sequence[float], second: Sequence[float], reference: Sequence[float]
) -> float:
    first = np.asarray(first, dtype=np.float64).reshape(-1)[:4]
    second = np.asarray(second, dtype=np.float64).reshape(-1)[:4]
    reference = np.asarray(reference, dtype=np.float64).reshape(-1)[:4]
    if not valid_box(first) or not valid_box(second) or not valid_box(reference):
        return math.nan
    first_center = first[:2] + 0.5 * first[2:]
    second_center = second[:2] + 0.5 * second[2:]
    scale = max(math.sqrt(float(reference[2] * reference[3])), EPS)
    return float(np.linalg.norm(first_center - second_center) / scale)


def mix_box(
    center_box: Sequence[float], size_box: Sequence[float]
) -> List[float]:
    center_box = np.asarray(center_box, dtype=np.float64).reshape(-1)[:4]
    size_box = np.asarray(size_box, dtype=np.float64).reshape(-1)[:4]
    if not valid_box(center_box) or not valid_box(size_box):
        return [math.nan] * 4
    center = center_box[:2] + 0.5 * center_box[2:]
    top_left = center - 0.5 * size_box[2:]
    return [
        float(top_left[0]),
        float(top_left[1]),
        float(size_box[2]),
        float(size_box[3]),
    ]


def response_statistics(response_map, nms_radius: int = 1) -> Dict[str, float]:
    if nms_radius < 0:
        raise ValueError(f"nms_radius must be non-negative, got {nms_radius}")

    response = torch.as_tensor(response_map).detach().float().squeeze()
    if response.ndim != 2 or response.numel() == 0:
        raise ValueError(
            "response_map must reduce to a non-empty 2-D map, "
            f"got shape {tuple(response.shape)}"
        )

    height, width = response.shape
    flat = response.flatten()
    peak1_index = int(flat.argmax().item())
    peak1_y, peak1_x = divmod(peak1_index, width)
    peak1 = float(flat[peak1_index].item())

    suppressed = response.clone()
    y0 = max(0, peak1_y - nms_radius)
    y1 = min(height, peak1_y + nms_radius + 1)
    x0 = max(0, peak1_x - nms_radius)
    x1 = min(width, peak1_x + nms_radius + 1)
    suppressed[y0:y1, x0:x1] = -torch.inf
    peak2_index = int(suppressed.flatten().argmax().item())
    peak2_y, peak2_x = divmod(peak2_index, width)
    peak2 = float(suppressed.flatten()[peak2_index].item())
    if not math.isfinite(peak2):
        peak2 = 0.0
        peak2_x = -1
        peak2_y = -1

    probability = response.clamp_min(0.0).flatten()
    probability_sum = float(probability.sum().item())
    if probability_sum <= EPS:
        entropy_normalized = 1.0
    elif probability.numel() <= 1:
        entropy_normalized = 0.0
    else:
        probability = probability / probability.sum()
        entropy = -(probability * probability.clamp_min(EPS).log()).sum()
        entropy_normalized = float(
            (entropy / math.log(probability.numel())).clamp(0.0, 1.0).item()
        )

    peak_ratio = peak2 / (peak1 + EPS)
    peak_separation = max(0.0, min(1.0, 1.0 - peak_ratio))
    response_reliability = (
        max(0.0, min(1.0, peak1))
        * peak_separation
        * max(0.0, 1.0 - entropy_normalized)
    )
    return {
        "p1": peak1,
        "p2": peak2,
        "peak_ratio": peak_ratio,
        "peak_margin": peak1 - peak2,
        "entropy_normalized": entropy_normalized,
        "response_reliability": response_reliability,
        "p1_x": peak1_x,
        "p1_y": peak1_y,
        "p2_x": peak2_x,
        "p2_y": peak2_y,
        "height": height,
        "width": width,
    }

def residual_spatial_statistics(
    input_tokens: torch.Tensor,
    output_tokens: torch.Tensor,
    template_length: int,
) -> Dict[str, float]:
    if input_tokens.shape != output_tokens.shape:
        raise ValueError(
            "input and output token shapes must match, got "
            f"{tuple(input_tokens.shape)} and {tuple(output_tokens.shape)}"
        )
    if input_tokens.ndim != 3 or not 0 < template_length < input_tokens.shape[1]:
        raise ValueError(
            f"invalid token shape/template length: {input_tokens.shape}, "
            f"{template_length}"
        )

    input_search = input_tokens[:, template_length:].detach().float()
    delta = (
        output_tokens[:, template_length:].detach().float() - input_search
    )
    reference_norm = torch.linalg.vector_norm(input_search, dim=-1)
    delta_norm = torch.linalg.vector_norm(delta, dim=-1)
    relative_norm = delta_norm / reference_norm.clamp_min(1e-6)
    flat_relative = relative_norm.flatten()
    flat_energy = delta_norm.square().flatten()
    active_fraction = float((flat_relative > 1e-8).float().mean().item())

    total_energy = flat_energy.sum()
    if float(total_energy.item()) <= EPS:
        top10_energy_fraction = 0.0
        spatial_entropy_normalized = 0.0
    else:
        top_count = max(1, math.ceil(0.1 * flat_energy.numel()))
        top10_energy_fraction = float(
            (flat_energy.topk(top_count).values.sum() / total_energy).item()
        )
        probability = flat_energy / total_energy
        if probability.numel() <= 1:
            spatial_entropy_normalized = 0.0
        else:
            entropy = -(
                probability * probability.clamp_min(EPS).log()
            ).sum()
            spatial_entropy_normalized = float(
                (entropy / math.log(probability.numel()))
                .clamp(0.0, 1.0)
                .item()
            )

    quantiles = torch.quantile(
        flat_relative,
        torch.tensor(
            [0.5, 0.9, 0.99], device=flat_relative.device
        ),
    )
    return {
        "residual_active_token_fraction": active_fraction,
        "residual_relative_norm_mean": float(flat_relative.mean().item()),
        "residual_relative_norm_p50": float(quantiles[0].item()),
        "residual_relative_norm_p90": float(quantiles[1].item()),
        "residual_relative_norm_p99": float(quantiles[2].item()),
        "residual_relative_norm_max": float(flat_relative.max().item()),
        "residual_top10_energy_fraction": top10_energy_fraction,
        "residual_spatial_entropy_normalized": spatial_entropy_normalized,
    }


def tracking_status(iou: float) -> str:
    if not math.isfinite(iou):
        return "invalid_gt"
    if iou >= CORRECT_IOU_THRESHOLD:
        return "correct"
    if iou < FAILURE_IOU_THRESHOLD:
        return "failed"
    return "ambiguous"


def pair_status(delta_iou: float) -> str:
    if not math.isfinite(delta_iou):
        return "invalid_gt"
    if delta_iou >= PAIR_DELTA_THRESHOLD:
        return "vdrm_better"
    if delta_iou <= -PAIR_DELTA_THRESHOLD:
        return "vdrm_worse"
    return "similar"


def _finite(values: Iterable[float]) -> List[float]:
    return [float(value) for value in values if math.isfinite(float(value))]


def _mean(values: Iterable[float]) -> Optional[float]:
    values = _finite(values)
    return float(np.mean(values)) if values else None


def _median(values: Iterable[float]) -> Optional[float]:
    values = _finite(values)
    return float(np.median(values)) if values else None


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def spearman(values: Sequence[float], targets: Sequence[float]) -> Optional[float]:
    values = np.asarray(values, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    valid = np.isfinite(values) & np.isfinite(targets)
    if valid.sum() < 2:
        return None
    first = _rankdata(values[valid])
    second = _rankdata(targets[valid])
    if np.std(first) <= EPS or np.std(second) <= EPS:
        return None
    return float(np.corrcoef(first, second)[0, 1])


def binary_auc(scores: Sequence[float], labels: Sequence[int]) -> Optional[float]:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    valid = np.isfinite(scores) & np.isin(labels, (0, 1))
    scores = scores[valid]
    labels = labels[valid]
    positive_count = int(labels.sum())
    negative_count = int(labels.size - positive_count)
    if positive_count == 0 or negative_count == 0:
        return None
    ranks = _rankdata(scores)
    rank_sum_positive = float(ranks[labels == 1].sum())
    statistic = rank_sum_positive - positive_count * (positive_count + 1) / 2.0
    return statistic / (positive_count * negative_count)


def reliability_summary(
    rows: Sequence[Mapping[str, object]], metric: str
) -> Dict[str, Optional[float]]:
    valid_iou_rows = [
        row
        for row in rows
        if math.isfinite(float(row.get("vdrm_iou", math.nan)))
        and math.isfinite(float(row.get(metric, math.nan)))
    ]
    scores = [float(row[metric]) for row in valid_iou_rows]
    ious = [float(row["vdrm_iou"]) for row in valid_iou_rows]

    correctness_scores = []
    correctness_labels = []
    pair_scores = []
    pair_labels = []
    for row in valid_iou_rows:
        score = float(row[metric])
        status = row.get("vdrm_status")
        if status in ("correct", "failed"):
            correctness_scores.append(score)
            correctness_labels.append(1 if status == "correct" else 0)
        comparison = row.get("pair_status")
        if comparison in ("vdrm_better", "vdrm_worse"):
            pair_scores.append(score)
            pair_labels.append(1 if comparison == "vdrm_better" else 0)

    return {
        "mean": _mean(scores),
        "spearman_with_vdrm_iou": spearman(scores, ious),
        "correct_vs_failed_auc": binary_auc(
            correctness_scores, correctness_labels
        ),
        "vdrm_better_vs_worse_auc": binary_auc(pair_scores, pair_labels),
    }


def continuous_metric_summary(
    rows: Sequence[Mapping[str, object]], metric: str
) -> Dict[str, Optional[float]]:
    """Relate a continuous diagnostic metric to VDRM tracking quality."""
    valid_rows = []
    for row in rows:
        try:
            value = float(row.get(metric, math.nan))
            iou = float(row.get("vdrm_iou", math.nan))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value) and math.isfinite(iou):
            valid_rows.append((row, value, iou))

    values = [value for _, value, _ in valid_rows]
    ious = [iou for _, _, iou in valid_rows]
    correct_values = [
        value
        for row, value, _ in valid_rows
        if row.get("vdrm_status") == "correct"
    ]
    failed_values = [
        value
        for row, value, _ in valid_rows
        if row.get("vdrm_status") == "failed"
    ]
    return {
        "mean": _mean(values),
        "correct_mean": _mean(correct_values),
        "failed_mean": _mean(failed_values),
        "spearman_with_vdrm_iou": spearman(values, ious),
    }


def summarize_pair_rows(
    rows: Sequence[Mapping[str, object]],
) -> Dict[str, object]:
    valid_rows = [
        row
        for row in rows
        if math.isfinite(float(row.get("delta_iou", math.nan)))
    ]
    delta_iou = [float(row["delta_iou"]) for row in valid_rows]
    metric_names = (
        "visual_reliability",
        "candidate_target_reliability",
        "candidate_reliability_peak",
        "candidate_reliability_mean",
        "vdrm_response_reliability",
        "combined_reliability",
    )
    residual = {
        metric: continuous_metric_summary(valid_rows, metric)
        for metric in RESIDUAL_METRICS
    }
    candidate_guidance = {
        metric: continuous_metric_summary(valid_rows, metric)
        for metric in GUIDANCE_METRICS
    }
    route_safeguard = {
        metric: continuous_metric_summary(valid_rows, metric)
        for metric in ROUTE_SAFEGUARD_METRICS
    }
    prototype = {
        metric: continuous_metric_summary(valid_rows, metric)
        for metric in PROTOTYPE_METRICS
    }
    summary = {
        "frame_count": len(rows),
        "valid_gt_frames": len(valid_rows),
        "baseline_mean_iou": _mean(
            float(row.get("baseline_iou", math.nan)) for row in valid_rows
        ),
        "vdrm_mean_iou": _mean(
            float(row.get("vdrm_iou", math.nan)) for row in valid_rows
        ),
        "mean_delta_iou": _mean(delta_iou),
        "median_delta_iou": _median(delta_iou),
        "vdrm_better_frames": sum(
            row.get("pair_status") == "vdrm_better" for row in valid_rows
        ),
        "similar_frames": sum(
            row.get("pair_status") == "similar" for row in valid_rows
        ),
        "vdrm_worse_frames": sum(
            row.get("pair_status") == "vdrm_worse" for row in valid_rows
        ),
        "catastrophic_vdrm_frames": sum(
            float(row.get("delta_iou", 0.0)) <= -0.5 for row in valid_rows
        ),
        "mean_center_only_delta_iou": _mean(
            float(row.get("center_only_delta_iou", math.nan))
            for row in valid_rows
        ),
        "mean_size_only_delta_iou": _mean(
            float(row.get("size_only_delta_iou", math.nan))
            for row in valid_rows
        ),
        "mean_normalized_backend_center_shift": _mean(
            float(row.get("backend_center_shift_normalized", math.nan))
            for row in valid_rows
        ),
        "mean_abs_log_width_ratio": _mean(
            abs(float(row.get("log_width_ratio", math.nan)))
            for row in valid_rows
        ),
        "mean_abs_log_height_ratio": _mean(
            abs(float(row.get("log_height_ratio", math.nan)))
            for row in valid_rows
        ),
        "mean_response_peak_shift": _mean(
            float(row.get("response_peak_shift_normalized", math.nan))
            for row in valid_rows
        ),
        "mean_residual_active_token_fraction": _mean(
            float(row.get("residual_active_token_fraction", math.nan))
            for row in valid_rows
        ),
        "mean_residual_spatial_entropy_normalized": _mean(
            float(
                row.get("residual_spatial_entropy_normalized", math.nan)
            )
            for row in valid_rows
        ),
        "reliability": {
            metric: reliability_summary(valid_rows, metric)
            for metric in metric_names
        },
        "residual": residual,
        "candidate_guidance": candidate_guidance,
        "route_safeguard": route_safeguard,
        "prototype": prototype,
    }
    for metric, metric_summary in residual.items():
        summary[f"mean_{metric}"] = metric_summary["mean"]
    for metric, metric_summary in candidate_guidance.items():
        summary[f"mean_{metric}"] = metric_summary["mean"]
    for metric, metric_summary in route_safeguard.items():
        summary[f"mean_{metric}"] = metric_summary["mean"]
    for metric, metric_summary in prototype.items():
        summary[f"mean_{metric}"] = metric_summary["mean"]
    return summary
