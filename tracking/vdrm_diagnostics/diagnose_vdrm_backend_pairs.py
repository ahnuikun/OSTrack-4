"""Compare OSTrack and VDRM on exactly the same template/search tensors.

This is an open-loop, read-only diagnostic. A search crop is sampled once and
forwarded through both backends. The VDRM prediction never controls a later
crop. In ``baseline_replay`` mode the baseline prediction supplies the next
anchor; in ``ground_truth`` mode the previous valid ground-truth box does.

No model weight, training configuration, standard tracker transition, or
existing result file is changed.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
from pathlib import Path
import sys
from typing import Dict, Mapping, Optional, Sequence
import zlib

import cv2
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.config.ostrack.config import cfg as default_cfg
from lib.config.ostrack.config import update_config_from_file
from lib.models.ostrack import build_ostrack
from lib.test.evaluation import get_dataset
from lib.test.evaluation.environment import env_settings
from lib.test.utils.hann import hann2d
from lib.train.data.processing_utils import (
    sample_target,
    transform_image_to_crop,
)
from lib.utils.box_ops import clip_box
from lib.utils.ce_utils import generate_mask_cond
from lib.utils.lmdb_utils import decode_img
from tracking.vdrm_diagnostics.backend_pair_metrics import (
    EPS,
    bbox_iou_xywh,
    mix_box,
    normalized_box_center_shift,
    normalized_center_error,
    pair_status,
    residual_spatial_statistics,
    response_statistics,
    summarize_pair_rows,
    tracking_status,
    valid_box,
)


DEFAULT_BASELINE_CONFIG = "vitb_256_mae_ce_32x4_ep300_fulltn"
DEFAULT_VDRM_CONFIG = "vitb_256_mae_ce_vdrm_v3_hncp_32x4_ep300"
ANCHOR_MODES = ("baseline_replay", "ground_truth")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run retrained OSTrack and VDRM on one shared crop per frame."
        )
    )
    parser.add_argument(
        "--baseline-config", default=DEFAULT_BASELINE_CONFIG
    )
    parser.add_argument("--vdrm-config", default=DEFAULT_VDRM_CONFIG)
    parser.add_argument("--baseline-checkpoint", default=None)
    parser.add_argument("--vdrm-checkpoint", default=None)
    parser.add_argument("--dataset-name", default="uav123")
    parser.add_argument(
        "--sequences",
        nargs="+",
        required=True,
        help="Exact sequence names exposed by the dataset adapter.",
    )
    parser.add_argument(
        "--anchor-mode", choices=ANCHOR_MODES, default="baseline_replay"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Frames after initialization; 0 evaluates the full sequence.",
    )
    parser.add_argument("--nms-radius", type=int, default=1)
    parser.add_argument("--print-interval", type=int, default=100)
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Output directory. Defaults to <save_dir>/vdrm_backend_pairs/"
            "<baseline>__vs__<vdrm>/<dataset>/<anchor_mode>."
        ),
    )
    return parser.parse_args()


def resolve_config_path(config_arg: str) -> Path:
    path = Path(config_arg)
    if path.suffix != ".yaml":
        path = PROJECT_ROOT / "experiments" / "ostrack" / f"{path.name}.yaml"
    elif not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.is_file():
        raise FileNotFoundError(f"config does not exist: {path}")
    return path.resolve()


def load_config(config_arg: str):
    path = resolve_config_path(config_arg)
    config = copy.deepcopy(default_cfg)
    update_config_from_file(str(path), base_cfg=config)
    return path.stem, path, config


def resolve_checkpoint(
    explicit_path: Optional[str], config_name: str, config, save_dir: Path
) -> Path:
    if explicit_path:
        path = Path(explicit_path).expanduser()
    else:
        checkpoint_config = config.TEST.CHECKPOINT_CONFIG or config_name
        path = (
            save_dir
            / "checkpoints"
            / "train"
            / "ostrack"
            / checkpoint_config
            / f"OSTrack_ep{int(config.TEST.EPOCH):04d}.pth.tar"
        )
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {path}")
    return path.resolve()


def assert_backend_compatibility(baseline_cfg, vdrm_cfg):
    checks = {
        "TEST.TEMPLATE_FACTOR": (
            baseline_cfg.TEST.TEMPLATE_FACTOR,
            vdrm_cfg.TEST.TEMPLATE_FACTOR,
        ),
        "TEST.TEMPLATE_SIZE": (
            baseline_cfg.TEST.TEMPLATE_SIZE,
            vdrm_cfg.TEST.TEMPLATE_SIZE,
        ),
        "TEST.SEARCH_FACTOR": (
            baseline_cfg.TEST.SEARCH_FACTOR,
            vdrm_cfg.TEST.SEARCH_FACTOR,
        ),
        "TEST.SEARCH_SIZE": (
            baseline_cfg.TEST.SEARCH_SIZE,
            vdrm_cfg.TEST.SEARCH_SIZE,
        ),
        "DATA.MEAN": (baseline_cfg.DATA.MEAN, vdrm_cfg.DATA.MEAN),
        "DATA.STD": (baseline_cfg.DATA.STD, vdrm_cfg.DATA.STD),
        "MODEL.BACKBONE.STRIDE": (
            baseline_cfg.MODEL.BACKBONE.STRIDE,
            vdrm_cfg.MODEL.BACKBONE.STRIDE,
        ),
        "MODEL.HEAD.TYPE": (
            baseline_cfg.MODEL.HEAD.TYPE,
            vdrm_cfg.MODEL.HEAD.TYPE,
        ),
    }
    mismatches = {
        name: values for name, values in checks.items() if values[0] != values[1]
    }
    if mismatches:
        details = ", ".join(
            f"{name}={first!r}/{second!r}"
            for name, (first, second) in mismatches.items()
        )
        raise ValueError(
            "paired backends must share crop and output geometry: " + details
        )
    if bool(baseline_cfg.MODEL.VDRM.ENABLED):
        raise ValueError("baseline config must have MODEL.VDRM.ENABLED=False")
    if not bool(vdrm_cfg.MODEL.VDRM.ENABLED):
        raise ValueError("VDRM config must have MODEL.VDRM.ENABLED=True")
    if baseline_cfg.MODEL.HEAD.TYPE != "CENTER":
        raise ValueError("paired diagnostic currently requires CENTER heads")


def preprocess_patch(image, attention_mask, config, device):
    image_tensor = (
        torch.from_numpy(np.ascontiguousarray(image))
        .to(device=device, dtype=torch.float32)
        .permute(2, 0, 1)
        .unsqueeze(0)
    )
    mean = torch.tensor(
        config.DATA.MEAN, device=device, dtype=torch.float32
    ).view(1, 3, 1, 1)
    std = torch.tensor(
        config.DATA.STD, device=device, dtype=torch.float32
    ).view(1, 3, 1, 1)
    normalized = (image_tensor / 255.0 - mean) / std
    mask_tensor = (
        torch.from_numpy(np.ascontiguousarray(attention_mask))
        .to(device=device, dtype=torch.bool)
        .unsqueeze(0)
    )
    return normalized, mask_tensor


def template_bbox_in_crop(init_bbox, resize_factor, template_size, device):
    box = torch.tensor(init_bbox, dtype=torch.float32)
    crop_size = torch.tensor(
        [template_size, template_size], dtype=torch.float32
    )
    transformed = transform_image_to_crop(
        box, box, resize_factor, crop_size, normalize=True
    )
    return transformed.view(1, 4).to(device)


class ResidualCapture:
    def __init__(self):
        self.last: Dict[str, float] = {}

    def __call__(self, module, args, kwargs, output):
        del module
        tokens = args[0] if args else kwargs["tokens"]
        template_length = int(kwargs["template_length"])
        output_tokens = output[0]
        self.last = residual_spatial_statistics(
            tokens, output_tokens, template_length
        )


class BackendRunner:
    def __init__(self, name, config, checkpoint: Path, device, capture_residual):
        self.name = name
        self.config = config
        self.checkpoint = checkpoint
        self.device = torch.device(device)
        network = build_ostrack(config, training=False)
        checkpoint_data = torch.load(
            checkpoint, map_location="cpu", weights_only=False
        )
        if "net" not in checkpoint_data:
            raise KeyError(f"checkpoint has no 'net' state: {checkpoint}")
        network.load_state_dict(checkpoint_data["net"], strict=True)
        self.network = network.to(self.device).eval()
        self.feature_size = (
            int(config.TEST.SEARCH_SIZE)
            // int(config.MODEL.BACKBONE.STRIDE)
        )
        self.output_window = hann2d(
            torch.tensor(
                [self.feature_size, self.feature_size], dtype=torch.long
            ),
            centered=True,
        ).to(self.device)
        self.template_tensor = None
        self.template_mask = None
        self.vdrm_template_bbox = None
        self.residual_capture = None
        self.residual_hook = None
        if capture_residual:
            if not hasattr(self.network.backbone, "vdrm"):
                raise RuntimeError("VDRM backend has no VDRM module")
            self.residual_capture = ResidualCapture()
            self.residual_hook = (
                self.network.backbone.vdrm.register_forward_hook(
                    self.residual_capture,
                    with_kwargs=True,
                )
            )

    def prepare_template(
        self, template_tensor, template_bbox, template_mask
    ):
        self.template_tensor = template_tensor
        self.template_mask = template_mask
        if bool(self.config.MODEL.VDRM.ENABLED):
            self.vdrm_template_bbox = template_bbox

    @torch.no_grad()
    def forward(self, search_tensor, anchor_bbox, resize_factor, image_shape):
        if self.template_tensor is None:
            raise RuntimeError("prepare_template must be called first")
        if self.residual_capture is not None:
            self.residual_capture.last = {}
        output = self.network.forward(
            template=self.template_tensor,
            search=search_tensor,
            ce_template_mask=self.template_mask,
            template_bbox=self.vdrm_template_bbox,
        )
        score_map = output["score_map"]
        response = self.output_window * score_map
        predicted_boxes = self.network.box_head.cal_bbox(
            response, output["size_map"], output["offset_map"]
        ).view(-1, 4)
        predicted_box = (
            predicted_boxes.mean(dim=0)
            * float(self.config.TEST.SEARCH_SIZE)
            / float(resize_factor)
        ).tolist()
        mapped_box = map_box_back_from_anchor(
            predicted_box,
            anchor_bbox,
            float(self.config.TEST.SEARCH_SIZE),
            float(resize_factor),
        )
        height, width = image_shape[:2]
        mapped_box = clip_box(mapped_box, height, width, margin=10)

        result = {
            "bbox": [float(value) for value in mapped_box],
            "score": float(score_map.max().detach().item()),
            "response_map": score_map.detach(),
        }
        candidate_map = output.get("candidate_reliability_map")
        if candidate_map is not None:
            selected_index = response.flatten(1).argmax(
                dim=1, keepdim=True
            )
            result["candidate_target_reliability"] = float(
                candidate_map.flatten(1)
                .gather(1, selected_index)
                .detach()
                .mean()
                .item()
            )
        for name in (
            "visual_reliability",
            "candidate_reliability_peak",
            "candidate_reliability_mean",
            "candidate_modulation",
            "candidate_modulation_factor_min",
            "candidate_modulation_factor_max",
            "part_route_residual_retention_mean",
            "part_route_residual_retention_min",
            "part_route_residual_retention_max",
            "vdrm_alpha",
            "vdrm_residual_clip_rate",
            "vdrm_raw_delta_relative_norm",
            "vdrm_delta_relative_norm",
        ):
            value = output.get(name)
            if value is not None:
                result[name] = float(value.detach().mean().item())
        for name in (
            "part_reliability",
            "part_peak_similarity",
            "part_hard_negative_similarity",
            "part_match_margin",
        ):
            value = output.get(name)
            if value is not None:
                value = value.detach().mean(dim=0).cpu()
                result[name] = value.tolist()
        part_valid = output.get("part_valid")
        if part_valid is not None:
            result["part_valid"] = (
                part_valid.detach().all(dim=0).cpu().tolist()
            )
        if self.residual_capture is not None:
            if not self.residual_capture.last:
                raise RuntimeError("VDRM residual hook did not run")
            result.update(self.residual_capture.last)
        return result


def map_box_back_from_anchor(
    predicted_box,
    anchor_bbox,
    search_size: float,
    resize_factor: float,
):
    anchor = np.asarray(anchor_bbox, dtype=np.float64).reshape(-1)[:4]
    center_x = anchor[0] + 0.5 * anchor[2]
    center_y = anchor[1] + 0.5 * anchor[3]
    cx, cy, width, height = [float(value) for value in predicted_box]
    half_side = 0.5 * search_size / resize_factor
    cx_real = cx + center_x - half_side
    cy_real = cy + center_y - half_side
    return [
        cx_real - 0.5 * width,
        cy_real - 0.5 * height,
        width,
        height,
    ]


def read_rgb(frame_reference):
    if isinstance(frame_reference, str):
        image = cv2.imread(frame_reference)
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif isinstance(frame_reference, list) and len(frame_reference) == 2:
        image = decode_img(frame_reference[0], frame_reference[1])
    else:
        raise TypeError(f"unsupported frame reference: {frame_reference!r}")
    if image is None:
        raise FileNotFoundError(f"could not read frame: {frame_reference}")
    return image


def prepare_shared_template(
    baseline_runner, vdrm_runner, first_image, init_bbox
):
    config = baseline_runner.config
    template_patch, resize_factor, attention_mask = sample_target(
        first_image,
        init_bbox,
        float(config.TEST.TEMPLATE_FACTOR),
        output_sz=int(config.TEST.TEMPLATE_SIZE),
    )
    template_tensor, _ = preprocess_patch(
        template_patch, attention_mask, config, baseline_runner.device
    )
    template_bbox = template_bbox_in_crop(
        init_bbox,
        resize_factor,
        int(config.TEST.TEMPLATE_SIZE),
        baseline_runner.device,
    )
    baseline_mask = None
    if config.MODEL.BACKBONE.CE_LOC:
        baseline_mask = generate_mask_cond(
            config, 1, baseline_runner.device, template_bbox
        )
    vdrm_mask = None
    if vdrm_runner.config.MODEL.BACKBONE.CE_LOC:
        vdrm_mask = generate_mask_cond(
            vdrm_runner.config, 1, vdrm_runner.device, template_bbox
        )
    if (baseline_mask is None) != (vdrm_mask is None) or (
        baseline_mask is not None and not torch.equal(baseline_mask, vdrm_mask)
    ):
        raise RuntimeError("paired backends produced different template CE masks")
    baseline_runner.prepare_template(
        template_tensor, template_bbox, baseline_mask
    )
    vdrm_runner.prepare_template(template_tensor, template_bbox, vdrm_mask)


def _safe_log_ratio(numerator: float, denominator: float) -> float:
    if numerator <= 0.0 or denominator <= 0.0:
        return math.nan
    return math.log(numerator / denominator)


def build_pair_row(
    sequence_name,
    frame_index,
    frame_reference,
    ground_truth,
    anchor_bbox,
    crop_checksum,
    baseline_output,
    vdrm_output,
    nms_radius,
):
    ground_truth = np.asarray(ground_truth, dtype=np.float64).reshape(-1)[:4]
    baseline_box = baseline_output["bbox"]
    vdrm_box = vdrm_output["bbox"]
    baseline_iou = bbox_iou_xywh(baseline_box, ground_truth)
    vdrm_iou = bbox_iou_xywh(vdrm_box, ground_truth)
    delta_iou = vdrm_iou - baseline_iou

    vdrm_center_baseline_size = mix_box(vdrm_box, baseline_box)
    baseline_center_vdrm_size = mix_box(baseline_box, vdrm_box)
    center_only_iou = bbox_iou_xywh(
        vdrm_center_baseline_size, ground_truth
    )
    size_only_iou = bbox_iou_xywh(
        baseline_center_vdrm_size, ground_truth
    )

    baseline_response = response_statistics(
        baseline_output["response_map"], nms_radius
    )
    vdrm_response = response_statistics(
        vdrm_output["response_map"], nms_radius
    )
    peak_shift = math.hypot(
        vdrm_response["p1_x"] - baseline_response["p1_x"],
        vdrm_response["p1_y"] - baseline_response["p1_y"],
    ) / max(
        math.hypot(vdrm_response["width"], vdrm_response["height"]),
        EPS,
    )
    visual_reliability = float(
        vdrm_output.get("visual_reliability", math.nan)
    )
    candidate_target_reliability = float(
        vdrm_output.get("candidate_target_reliability", math.nan)
    )
    response_uniqueness = max(
        0.0, min(1.0, 1.0 - vdrm_response["peak_ratio"])
    )
    combined_reliability = (
        visual_reliability
        * response_uniqueness
        * max(0.0, 1.0 - vdrm_response["entropy_normalized"])
    )

    row = {
        "sequence": sequence_name,
        "frame_index": frame_index,
        "frame_number": frame_index + 1,
        "image_name": Path(str(frame_reference)).name,
        "crop_checksum_crc32": crop_checksum,
        "anchor_x": float(anchor_bbox[0]),
        "anchor_y": float(anchor_bbox[1]),
        "anchor_w": float(anchor_bbox[2]),
        "anchor_h": float(anchor_bbox[3]),
        "gt_x": float(ground_truth[0]),
        "gt_y": float(ground_truth[1]),
        "gt_w": float(ground_truth[2]),
        "gt_h": float(ground_truth[3]),
        "baseline_x": float(baseline_box[0]),
        "baseline_y": float(baseline_box[1]),
        "baseline_w": float(baseline_box[2]),
        "baseline_h": float(baseline_box[3]),
        "vdrm_x": float(vdrm_box[0]),
        "vdrm_y": float(vdrm_box[1]),
        "vdrm_w": float(vdrm_box[2]),
        "vdrm_h": float(vdrm_box[3]),
        "baseline_iou": baseline_iou,
        "vdrm_iou": vdrm_iou,
        "delta_iou": delta_iou,
        "baseline_status": tracking_status(baseline_iou),
        "vdrm_status": tracking_status(vdrm_iou),
        "pair_status": pair_status(delta_iou),
        "baseline_center_error": normalized_center_error(
            baseline_box, ground_truth
        ),
        "vdrm_center_error": normalized_center_error(vdrm_box, ground_truth),
        "backend_center_shift_normalized": normalized_box_center_shift(
            baseline_box, vdrm_box, ground_truth
        ),
        "log_width_ratio": _safe_log_ratio(
            float(vdrm_box[2]), float(baseline_box[2])
        ),
        "log_height_ratio": _safe_log_ratio(
            float(vdrm_box[3]), float(baseline_box[3])
        ),
        "vdrm_center_baseline_size_iou": center_only_iou,
        "baseline_center_vdrm_size_iou": size_only_iou,
        "center_only_delta_iou": center_only_iou - baseline_iou,
        "size_only_delta_iou": size_only_iou - baseline_iou,
        "baseline_score": baseline_output["score"],
        "vdrm_score": vdrm_output["score"],
        "response_peak_shift_normalized": peak_shift,
        "visual_reliability": visual_reliability,
        "candidate_target_reliability": candidate_target_reliability,
        "candidate_reliability_peak": float(
            vdrm_output.get("candidate_reliability_peak", math.nan)
        ),
        "candidate_reliability_mean": float(
            vdrm_output.get("candidate_reliability_mean", math.nan)
        ),
        "candidate_modulation": float(
            vdrm_output.get("candidate_modulation", math.nan)
        ),
        "candidate_modulation_factor_min": float(
            vdrm_output.get("candidate_modulation_factor_min", math.nan)
        ),
        "candidate_modulation_factor_max": float(
            vdrm_output.get("candidate_modulation_factor_max", math.nan)
        ),
        "part_route_residual_retention_mean": float(
            vdrm_output.get(
                "part_route_residual_retention_mean", math.nan
            )
        ),
        "part_route_residual_retention_min": float(
            vdrm_output.get(
                "part_route_residual_retention_min", math.nan
            )
        ),
        "part_route_residual_retention_max": float(
            vdrm_output.get(
                "part_route_residual_retention_max", math.nan
            )
        ),
        "baseline_response_reliability": baseline_response[
            "response_reliability"
        ],
        "vdrm_response_reliability": vdrm_response[
            "response_reliability"
        ],
        "combined_reliability": combined_reliability,
    }
    for prefix, metrics in (
        ("baseline_response", baseline_response),
        ("vdrm_response", vdrm_response),
    ):
        for name in (
            "p1",
            "p2",
            "peak_ratio",
            "peak_margin",
            "entropy_normalized",
            "p1_x",
            "p1_y",
            "p2_x",
            "p2_y",
        ):
            row[f"{prefix}_{name}"] = metrics[name]

    for name, value in vdrm_output.items():
        if name.startswith("residual_") or name.startswith("vdrm_"):
            row[name] = value
    part_values = {
        name: vdrm_output.get(name)
        for name in (
            "part_reliability",
            "part_valid",
            "part_peak_similarity",
            "part_hard_negative_similarity",
            "part_match_margin",
        )
    }
    for name, values in part_values.items():
        if values is None:
            continue
        for index, value in enumerate(values):
            row[f"{name}_{index}"] = value
    return row


def diagnose_sequence(
    sequence,
    baseline_runner,
    vdrm_runner,
    anchor_mode,
    max_frames,
    nms_radius,
    print_interval,
):
    first_image = read_rgb(sequence.frames[0])
    initial_bbox = [float(value) for value in sequence.init_info()["init_bbox"]]
    prepare_shared_template(
        baseline_runner,
        vdrm_runner,
        first_image,
        initial_bbox,
    )
    anchor_bbox = initial_bbox
    rows = []
    frame_limit = len(sequence.frames) - 1
    if max_frames > 0:
        frame_limit = min(frame_limit, max_frames)

    for frame_index in range(1, frame_limit + 1):
        frame_reference = sequence.frames[frame_index]
        image = read_rgb(frame_reference)
        ground_truth = sequence.ground_truth_rect[frame_index]
        search_patch, resize_factor, attention_mask = sample_target(
            image,
            anchor_bbox,
            float(baseline_runner.config.TEST.SEARCH_FACTOR),
            output_sz=int(baseline_runner.config.TEST.SEARCH_SIZE),
        )
        crop_checksum = f"{zlib.crc32(search_patch.tobytes()):08x}"
        search_tensor, _ = preprocess_patch(
            search_patch,
            attention_mask,
            baseline_runner.config,
            baseline_runner.device,
        )
        baseline_output = baseline_runner.forward(
            search_tensor.clone(), anchor_bbox, resize_factor, image.shape
        )
        vdrm_output = vdrm_runner.forward(
            search_tensor.clone(), anchor_bbox, resize_factor, image.shape
        )
        rows.append(
            build_pair_row(
                sequence.name,
                frame_index,
                frame_reference,
                ground_truth,
                anchor_bbox,
                crop_checksum,
                baseline_output,
                vdrm_output,
                nms_radius,
            )
        )

        if anchor_mode == "baseline_replay":
            anchor_bbox = baseline_output["bbox"]
        else:
            ground_truth_values = np.asarray(
                ground_truth, dtype=np.float64
            ).reshape(-1)[:4]
            if valid_box(ground_truth_values):
                anchor_bbox = ground_truth_values.tolist()

        if print_interval > 0 and (
            frame_index % print_interval == 0 or frame_index == frame_limit
        ):
            print(
                f"{sequence.name}: frame={frame_index}/{frame_limit} "
                f"paired={len(rows)}"
            )
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]):
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            _json_safe(data),
            handle,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )


def _json_safe(value):
    if isinstance(value, Mapping):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _extreme_frames(rows, reverse):
    valid = [
        row
        for row in rows
        if math.isfinite(float(row.get("delta_iou", math.nan)))
    ]
    selected = sorted(
        valid, key=lambda row: float(row["delta_iou"]), reverse=reverse
    )[:10]
    return [
        {
            "sequence": row["sequence"],
            "frame_number": row["frame_number"],
            "image_name": row["image_name"],
            "baseline_iou": row["baseline_iou"],
            "vdrm_iou": row["vdrm_iou"],
            "delta_iou": row["delta_iou"],
            "visual_reliability": row["visual_reliability"],
            "candidate_target_reliability": row.get(
                "candidate_target_reliability"
            ),
            "combined_reliability": row["combined_reliability"],
        }
        for row in selected
    ]


def main():
    args = parse_args()
    if args.max_frames < 0:
        raise ValueError("--max-frames must be non-negative")
    if args.nms_radius < 0:
        raise ValueError("--nms-radius must be non-negative")
    if args.print_interval < 0:
        raise ValueError("--print-interval must be non-negative")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

    baseline_name, baseline_config_path, baseline_cfg = load_config(
        args.baseline_config
    )
    vdrm_name, vdrm_config_path, vdrm_cfg = load_config(args.vdrm_config)
    assert_backend_compatibility(baseline_cfg, vdrm_cfg)

    environment = env_settings()
    save_dir = Path(environment.save_dir)
    baseline_checkpoint = resolve_checkpoint(
        args.baseline_checkpoint, baseline_name, baseline_cfg, save_dir
    )
    vdrm_checkpoint = resolve_checkpoint(
        args.vdrm_checkpoint, vdrm_name, vdrm_cfg, save_dir
    )

    dataset = get_dataset(args.dataset_name)
    sequence_by_name = {sequence.name: sequence for sequence in dataset}
    missing = [name for name in args.sequences if name not in sequence_by_name]
    if missing:
        raise ValueError(
            f"unknown sequences for {args.dataset_name}: {missing}"
        )

    baseline_runner = BackendRunner(
        baseline_name,
        baseline_cfg,
        baseline_checkpoint,
        args.device,
        capture_residual=False,
    )
    vdrm_runner = BackendRunner(
        vdrm_name,
        vdrm_cfg,
        vdrm_checkpoint,
        args.device,
        capture_residual=True,
    )

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = (
            save_dir
            / "vdrm_backend_pairs"
            / f"{baseline_name}__vs__{vdrm_name}"
            / args.dataset_name
            / args.anchor_mode
        )

    print("baseline_config:", baseline_config_path)
    print("vdrm_config:", vdrm_config_path)
    print("baseline_checkpoint:", baseline_checkpoint)
    print("vdrm_checkpoint:", vdrm_checkpoint)
    print("anchor_mode:", args.anchor_mode)
    print("device:", args.device)

    all_rows = []
    sequence_summaries = {}
    for sequence_name in args.sequences:
        rows = diagnose_sequence(
            sequence_by_name[sequence_name],
            baseline_runner,
            vdrm_runner,
            args.anchor_mode,
            args.max_frames,
            args.nms_radius,
            args.print_interval,
        )
        summary = summarize_pair_rows(rows)
        summary.update(
            {
                "dataset": args.dataset_name,
                "sequence": sequence_name,
                "anchor_mode": args.anchor_mode,
            }
        )
        _write_csv(output_dir / f"{sequence_name}.csv", rows)
        _write_json(
            output_dir / f"{sequence_name}_summary.json", summary
        )
        all_rows.extend(rows)
        sequence_summaries[sequence_name] = summary
        print(
            f"summary {sequence_name}: "
            f"baseline_iou={summary['baseline_mean_iou']}, "
            f"vdrm_iou={summary['vdrm_mean_iou']}, "
            f"delta={summary['mean_delta_iou']}"
        )

    dataset_summary = summarize_pair_rows(all_rows)
    dataset_summary.update(
        {
            "dataset": args.dataset_name,
            "anchor_mode": args.anchor_mode,
            "baseline_config": baseline_name,
            "vdrm_config": vdrm_name,
            "baseline_checkpoint": str(baseline_checkpoint),
            "vdrm_checkpoint": str(vdrm_checkpoint),
            "nms_radius": args.nms_radius,
            "sequence_count": len(sequence_summaries),
            "sequences": sequence_summaries,
            "largest_vdrm_gains": _extreme_frames(all_rows, reverse=True),
            "largest_vdrm_losses": _extreme_frames(all_rows, reverse=False),
        }
    )
    _write_json(output_dir / "dataset_summary.json", dataset_summary)

    print("\nPaired backend diagnostic completed")
    print("sequences:", len(sequence_summaries))
    print("paired_frames:", len(all_rows))
    print("baseline_mean_iou:", dataset_summary["baseline_mean_iou"])
    print("vdrm_mean_iou:", dataset_summary["vdrm_mean_iou"])
    print("mean_delta_iou:", dataset_summary["mean_delta_iou"])
    print("output:", output_dir)
    if args.device == "cuda":
        print(
            "peak_cuda_memory_gib="
            f"{torch.cuda.max_memory_allocated() / (1024 ** 3):.3f}"
        )


if __name__ == "__main__":
    main()
