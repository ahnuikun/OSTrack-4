"""Validate paired diagnostic outputs and produce a root-cause report."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Optional, Sequence


TARGETS = (
    ("uav123", "ground_truth"),
    ("uav123", "baseline_replay"),
    ("dtb70", "ground_truth"),
    ("dtb70", "baseline_replay"),
)


def _format(value, digits=4):
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def _metric(summary, group, metric, field):
    return summary.get(group, {}).get(metric, {}).get(field)


def _validate_sequence_outputs(directory: Path, summary):
    frame_count = 0
    for sequence_name, sequence_summary in summary["sequences"].items():
        csv_path = directory / f"{sequence_name}.csv"
        json_path = directory / f"{sequence_name}_summary.json"
        if not csv_path.is_file():
            raise FileNotFoundError(csv_path)
        if not json_path.is_file():
            raise FileNotFoundError(json_path)
        with csv_path.open(newline="", encoding="utf-8") as handle:
            frame_count += sum(1 for _ in csv.DictReader(handle))
        saved = json.loads(json_path.read_text(encoding="utf-8"))
        if saved.get("frame_count") != sequence_summary.get("frame_count"):
            raise RuntimeError(
                f"sequence summary mismatch for {directory}/{sequence_name}"
            )
    if frame_count != summary["frame_count"]:
        raise RuntimeError(
            f"CSV frame count {frame_count} != summary {summary['frame_count']} "
            f"in {directory}"
        )


def _load_target(
    run_root: Path,
    dataset_name: str,
    anchor_mode: str,
    baseline_checkpoint: Optional[Path],
    vdrm_checkpoint: Optional[Path],
    prefer_full: bool = False,
):
    relative = Path(dataset_name) / anchor_mode
    if prefer_full:
        relative = Path("full") / relative
    directory = run_root / relative
    summary_path = directory / "dataset_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("dataset") != dataset_name:
        raise RuntimeError(f"dataset mismatch in {summary_path}")
    if summary.get("anchor_mode") != anchor_mode:
        raise RuntimeError(f"anchor mode mismatch in {summary_path}")
    if summary.get("sequence_count") != len(summary.get("sequences", {})):
        raise RuntimeError(f"sequence count mismatch in {summary_path}")
    if baseline_checkpoint is not None and Path(
        summary["baseline_checkpoint"]
    ).resolve() != baseline_checkpoint.resolve():
        raise RuntimeError(f"baseline checkpoint mismatch in {summary_path}")
    if vdrm_checkpoint is not None and Path(
        summary["vdrm_checkpoint"]
    ).resolve() != vdrm_checkpoint.resolve():
        raise RuntimeError(f"VDRM checkpoint mismatch in {summary_path}")
    _validate_sequence_outputs(directory, summary)
    return directory, summary


def _recommendations(rows, material_delta: float):
    by_key = {
        (row["dataset"], row["anchor_mode"]): row for row in rows
    }
    recommendations = []
    for dataset_name in ("uav123", "dtb70"):
        gt = by_key[(dataset_name, "ground_truth")]
        replay = by_key[(dataset_name, "baseline_replay")]
        if gt["mean_delta_iou"] <= -material_delta:
            cause = (
                "VDRM regresses under the shared ground-truth crop, which "
                "points to representation or residual-injection damage."
            )
            action = (
                "Prioritize part matching, candidate identity, and spatial "
                "residual injection."
            )
        elif replay["mean_delta_iou"] <= -material_delta:
            cause = (
                "Ground-truth crops are stable but baseline-replay crops "
                "regress, indicating sensitivity to center/scale errors."
            )
            action = (
                "Prioritize candidate identity under realistic crop "
                "perturbations."
            )
        else:
            cause = (
                "Shared-crop averages are stable; inspect standard VDRM "
                "closed-loop feedback amplification and tail failures."
            )
            action = (
                "Require candidate-consistent residual gating before "
                "changing HNCP geometry or probability."
            )

        component_actions = []
        if min(
            gt["mean_center_only_delta_iou"],
            replay["mean_center_only_delta_iou"],
        ) <= -material_delta:
            component_actions.append(
                "Center term regresses: prioritize candidate-level target "
                "identity."
            )
        if min(
            gt["mean_size_only_delta_iou"],
            replay["mean_size_only_delta_iou"],
        ) <= -material_delta:
            component_actions.append(
                "Size term regresses: isolate residual effects on size and "
                "offset only after the center hypothesis is tested."
            )
        reliability_auc = replay["visual_correct_vs_failed_auc"]
        reliability_rho = replay["visual_spearman_with_vdrm_iou"]
        if (
            reliability_auc is None
            or reliability_auc < 0.6
            or reliability_rho is None
            or reliability_rho <= 0.0
        ):
            component_actions.append(
                "Absolute visual reliability does not separate failures "
                "well; add candidate-level target reliability."
            )

        failed_norm = replay["residual_relative_norm_failed"]
        correct_norm = replay["residual_relative_norm_correct"]
        failed_top10 = replay["residual_top10_failed"]
        correct_top10 = replay["residual_top10_correct"]
        if (
            failed_norm is not None
            and correct_norm is not None
            and failed_norm > correct_norm
        ) or (
            failed_top10 is not None
            and correct_top10 is not None
            and failed_top10 > correct_top10
        ):
            component_actions.append(
                "Residual energy increases on failures; inspect spatial "
                "concentration before considering a bound."
            )

        recommendations.append(
            {
                "dataset": dataset_name,
                "cause": cause,
                "primary_action": action,
                "component_actions": component_actions,
            }
        )
    return recommendations


def build_report(
    run_root: Path,
    baseline_checkpoint: Optional[Path] = None,
    vdrm_checkpoint: Optional[Path] = None,
    material_delta: float = 0.01,
    prefer_full: bool = False,
):
    rows = []
    losses = []
    for dataset_name, anchor_mode in TARGETS:
        directory, summary = _load_target(
            run_root,
            dataset_name,
            anchor_mode,
            baseline_checkpoint,
            vdrm_checkpoint,
            prefer_full=prefer_full,
        )
        worst_sequence, worst_summary = min(
            summary["sequences"].items(),
            key=lambda item: (
                float(item[1]["mean_delta_iou"])
                if item[1]["mean_delta_iou"] is not None
                else math.inf
            ),
        )
        visual = summary["reliability"]["visual_reliability"]
        candidate = summary.get("reliability", {}).get(
            "candidate_target_reliability", {}
        )
        row = {
            "source_scope": "full" if prefer_full else "selected",
            "dataset": dataset_name,
            "anchor_mode": anchor_mode,
            "sequence_count": summary["sequence_count"],
            "frame_count": summary["frame_count"],
            "mean_delta_iou": summary["mean_delta_iou"],
            "mean_center_only_delta_iou": summary[
                "mean_center_only_delta_iou"
            ],
            "mean_size_only_delta_iou": summary[
                "mean_size_only_delta_iou"
            ],
            "catastrophic_vdrm_frames": summary[
                "catastrophic_vdrm_frames"
            ],
            "visual_correct_vs_failed_auc": visual[
                "correct_vs_failed_auc"
            ],
            "visual_spearman_with_vdrm_iou": visual[
                "spearman_with_vdrm_iou"
            ],
            "candidate_correct_vs_failed_auc": candidate.get(
                "correct_vs_failed_auc"
            ),
            "candidate_better_vs_worse_auc": candidate.get(
                "vdrm_better_vs_worse_auc"
            ),
            "candidate_spearman_with_vdrm_iou": candidate.get(
                "spearman_with_vdrm_iou"
            ),
            "candidate_modulation_mean": _metric(
                summary,
                "candidate_guidance",
                "candidate_modulation",
                "mean",
            ),
            "candidate_modulation_factor_min_mean": _metric(
                summary,
                "candidate_guidance",
                "candidate_modulation_factor_min",
                "mean",
            ),
            "candidate_modulation_factor_max_mean": _metric(
                summary,
                "candidate_guidance",
                "candidate_modulation_factor_max",
                "mean",
            ),
            "residual_relative_norm_correct": _metric(
                summary,
                "residual",
                "residual_relative_norm_mean",
                "correct_mean",
            ),
            "residual_relative_norm_failed": _metric(
                summary,
                "residual",
                "residual_relative_norm_mean",
                "failed_mean",
            ),
            "residual_top10_correct": _metric(
                summary,
                "residual",
                "residual_top10_energy_fraction",
                "correct_mean",
            ),
            "residual_top10_failed": _metric(
                summary,
                "residual",
                "residual_top10_energy_fraction",
                "failed_mean",
            ),
            "worst_sequence": worst_sequence,
            "worst_sequence_mean_delta_iou": worst_summary[
                "mean_delta_iou"
            ],
            "worst_sequence_csv": str(
                directory / f"{worst_sequence}.csv"
            ),
        }
        rows.append(row)
        losses.append(
            {
                "source_scope": row["source_scope"],
                "dataset": dataset_name,
                "anchor_mode": anchor_mode,
                "worst_sequence": worst_sequence,
                "worst_sequence_mean_delta_iou": worst_summary[
                    "mean_delta_iou"
                ],
                "csv": row["worst_sequence_csv"],
                "largest_single_frame_loss": (
                    summary["largest_vdrm_losses"][0]
                    if summary.get("largest_vdrm_losses")
                    else None
                ),
            }
        )

    recommendations = _recommendations(rows, material_delta)
    return rows, losses, recommendations


def write_report(run_root: Path, rows, losses, recommendations):
    metrics_path = run_root / "diagnostic_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    losses_path = run_root / "largest_loss_manifest.json"
    losses_path.write_text(
        json.dumps(losses, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# VDRM Backend Pair Diagnostic Report",
        "",
        f"Source scope: `{rows[0]['source_scope']}`",
        "",
        "## Five-axis evidence",
        "",
        (
            "| Dataset | Anchor | Visual Delta IoU | Center Delta IoU | "
            "Size Delta IoU | Visual AUC | Candidate AUC | Candidate pair "
            "AUC | Residual norm correct/failed | Top10 energy "
            "correct/failed | Catastrophic |"
        ),
        (
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
        ),
    ]
    for row in rows:
        lines.append(
            "| {dataset} | {anchor_mode} | {delta} | {center} | {size} | "
            "{visual_auc} | {candidate_auc} | {candidate_pair_auc} | "
            "{norm_correct}/{norm_failed} | {top_correct}/{top_failed} | "
            "{catastrophic} |".format(
                dataset=row["dataset"],
                anchor_mode=row["anchor_mode"],
                delta=_format(row["mean_delta_iou"]),
                center=_format(row["mean_center_only_delta_iou"]),
                size=_format(row["mean_size_only_delta_iou"]),
                visual_auc=_format(row["visual_correct_vs_failed_auc"]),
                candidate_auc=_format(
                    row["candidate_correct_vs_failed_auc"]
                ),
                candidate_pair_auc=_format(
                    row["candidate_better_vs_worse_auc"]
                ),
                norm_correct=_format(
                    row["residual_relative_norm_correct"]
                ),
                norm_failed=_format(row["residual_relative_norm_failed"]),
                top_correct=_format(row["residual_top10_correct"]),
                top_failed=_format(row["residual_top10_failed"]),
                catastrophic=row["catastrophic_vdrm_frames"],
            )
        )

    lines.extend(["", "## Module-one decisions", ""])
    for recommendation in recommendations:
        lines.append(f"### {recommendation['dataset']}")
        lines.append("")
        lines.append(f"- Root-cause signal: {recommendation['cause']}")
        lines.append(
            f"- Primary direction: {recommendation['primary_action']}"
        )
        for action in recommendation["component_actions"]:
            lines.append(f"- Additional evidence: {action}")
        lines.append("")
    lines.append(
        "Module one must still pass its independent Gate; a joint "
        "V3-Primary Reliability-Conditioned M2 result does not replace it."
    )

    report_path = run_root / "diagnostic_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return metrics_path, losses_path, report_path


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(
        description="Validate four paired outputs and write a diagnosis report."
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path)
    parser.add_argument("--vdrm-checkpoint", type=Path)
    parser.add_argument("--material-delta", type=float, default=0.01)
    parser.add_argument(
        "--prefer-full",
        action="store_true",
        help="Read run_root/full/<dataset>/<anchor> instead of selected runs.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None):
    args = parse_args(argv)
    rows, losses, recommendations = build_report(
        args.run_root,
        args.baseline_checkpoint,
        args.vdrm_checkpoint,
        args.material_delta,
        prefer_full=args.prefer_full,
    )
    outputs = write_report(
        args.run_root, rows, losses, recommendations
    )
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    for output in outputs:
        print("wrote:", output)


if __name__ == "__main__":
    main()
