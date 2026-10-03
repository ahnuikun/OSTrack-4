"""Training-only augmentations used by VDRM experiments."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


def _normalized_box_bounds(
    box: torch.Tensor,
    height: int,
    width: int,
) -> Tuple[int, int, int, int]:
    """Convert one normalized ``xywh`` box to clipped integer bounds."""
    box = box.detach().to(dtype=torch.float32).flatten()
    if box.numel() != 4:
        raise ValueError(f"box must contain four values, got {tuple(box.shape)}")

    x0 = int(torch.floor(box[0].clamp(0.0, 1.0) * width).item())
    y0 = int(torch.floor(box[1].clamp(0.0, 1.0) * height).item())
    x1 = int(
        torch.ceil((box[0] + box[2]).clamp(0.0, 1.0) * width).item()
    )
    y1 = int(
        torch.ceil((box[1] + box[3]).clamp(0.0, 1.0) * height).item()
    )
    return x0, y0, x1, y1


@torch.no_grad()
def apply_same_class_distractor_copy_paste(
    image: torch.Tensor,
    target_box: torch.Tensor,
    distractor_image: torch.Tensor,
    distractor_box: torch.Tensor,
    min_scale: float = 0.7,
    max_scale: float = 1.3,
    invalid_mask: torch.Tensor | None = None,
    max_attempts: int = 24,
    placement_mode: str = "random",
) -> Tuple[torch.Tensor, bool, torch.Tensor]:
    """Paste one real same-class instance outside the labelled target.

    Both images are already transformed and normalized. The source is cropped
    by its ground-truth box, resized relative to the labelled target area, and
    pasted only where it does not overlap the target or padded search pixels.
    The target box and all training labels remain unchanged. The returned box
    is the normalized ``xywh`` location of the pasted distractor in ``image``;
    it is all-zero when no paste is applied. ``random`` preserves the V3/V5
    first-valid placement. ``nearest`` evaluates all sampled valid candidates
    and selects the one closest to the labelled target after size normalization.
    """
    if image.ndim != 3 or distractor_image.ndim != 3:
        raise ValueError(
            "image and distractor_image must have shape [C, H, W]"
        )
    if image.shape[0] != distractor_image.shape[0]:
        raise ValueError("image and distractor_image channel counts must match")
    if not 0.0 < min_scale <= max_scale:
        raise ValueError(
            "scales must satisfy 0 < min_scale <= max_scale, got "
            f"{min_scale}, {max_scale}"
        )
    if max_attempts < 1:
        raise ValueError(f"max_attempts must be positive, got {max_attempts}")
    if placement_mode not in ("random", "nearest"):
        raise ValueError(
            "placement_mode must be 'random' or 'nearest', got "
            f"{placement_mode!r}"
        )

    empty_paste_box = image.new_zeros(4)
    _, height, width = image.shape
    _, source_height, source_width = distractor_image.shape
    target_x0, target_y0, target_x1, target_y1 = _normalized_box_bounds(
        target_box, height, width
    )
    source_x0, source_y0, source_x1, source_y1 = _normalized_box_bounds(
        distractor_box, source_height, source_width
    )

    target_width = target_x1 - target_x0
    target_height = target_y1 - target_y0
    source_width_box = source_x1 - source_x0
    source_height_box = source_y1 - source_y0
    if min(
        target_width,
        target_height,
        source_width_box,
        source_height_box,
    ) < 2:
        return image.clone(), False, empty_paste_box

    source_patch = distractor_image[
        :, source_y0:source_y1, source_x0:source_x1
    ]
    source_aspect = source_width_box / source_height_box
    scale = torch.empty((), device=image.device).uniform_(
        float(min_scale), float(max_scale)
    ).item()
    desired_area = target_width * target_height * scale * scale
    paste_width = max(2, int(round((desired_area * source_aspect) ** 0.5)))
    paste_height = max(2, int(round((desired_area / source_aspect) ** 0.5)))
    if paste_width > width or paste_height > height:
        return image.clone(), False, empty_paste_box

    if invalid_mask is not None:
        invalid_mask = invalid_mask.to(device=image.device, dtype=torch.bool)
        if invalid_mask.shape != (height, width):
            raise ValueError(
                "invalid_mask must match the search image spatial size, got "
                f"{tuple(invalid_mask.shape)} and {(height, width)}"
            )

    paste_x0 = paste_y0 = None
    best_distance = None
    target_center_x = 0.5 * (target_x0 + target_x1)
    target_center_y = 0.5 * (target_y0 + target_y1)
    distance_scale_x = max(float(target_width + paste_width), 1.0)
    distance_scale_y = max(float(target_height + paste_height), 1.0)
    for _ in range(max_attempts):
        candidate_x0 = int(
            torch.randint(
                0, width - paste_width + 1, (), device=image.device
            ).item()
        )
        candidate_y0 = int(
            torch.randint(
                0, height - paste_height + 1, (), device=image.device
            ).item()
        )
        candidate_x1 = candidate_x0 + paste_width
        candidate_y1 = candidate_y0 + paste_height

        overlaps_target = not (
            candidate_x1 <= target_x0
            or candidate_x0 >= target_x1
            or candidate_y1 <= target_y0
            or candidate_y0 >= target_y1
        )
        if overlaps_target:
            continue
        if (
            invalid_mask is not None
            and invalid_mask[
                candidate_y0:candidate_y1, candidate_x0:candidate_x1
            ].any()
        ):
            continue
        if placement_mode == "random":
            paste_x0, paste_y0 = candidate_x0, candidate_y0
            break

        candidate_center_x = candidate_x0 + 0.5 * paste_width
        candidate_center_y = candidate_y0 + 0.5 * paste_height
        normalized_distance = (
            (candidate_center_x - target_center_x) / distance_scale_x
        ) ** 2 + (
            (candidate_center_y - target_center_y) / distance_scale_y
        ) ** 2
        if best_distance is None or normalized_distance < best_distance:
            best_distance = normalized_distance
            paste_x0, paste_y0 = candidate_x0, candidate_y0

    if paste_x0 is None or paste_y0 is None:
        return image.clone(), False, empty_paste_box

    resized_patch = F.interpolate(
        source_patch.unsqueeze(0),
        size=(paste_height, paste_width),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)

    # A narrow feathered boundary avoids making a rectangular cut edge a
    # shortcut while retaining the real annotated appearance of the instance.
    alpha_y = image.new_ones(paste_height)
    alpha_x = image.new_ones(paste_width)
    edge = min(paste_height, paste_width) // 10
    if edge > 0:
        ramp = torch.linspace(
            0.0,
            1.0,
            edge + 2,
            device=image.device,
            dtype=image.dtype,
        )[1:-1]
        alpha_y[:edge] = ramp
        alpha_y[-edge:] = ramp.flip(0)
        alpha_x[:edge] = ramp
        alpha_x[-edge:] = ramp.flip(0)
    alpha = (alpha_y[:, None] * alpha_x[None, :]).unsqueeze(0)

    output = image.clone()
    destination = output[
        :,
        paste_y0:paste_y0 + paste_height,
        paste_x0:paste_x0 + paste_width,
    ]
    destination.copy_(
        resized_patch.to(dtype=image.dtype) * alpha
        + destination * (1.0 - alpha)
    )
    paste_box = image.new_tensor(
        [
            paste_x0 / width,
            paste_y0 / height,
            paste_width / width,
            paste_height / height,
        ]
    )
    return output, True, paste_box


@torch.no_grad()
def apply_structured_target_occlusion(
    images: torch.Tensor,
    target_boxes: torch.Tensor,
    probability: float = 0.5,
    min_area_ratio: float = 0.2,
    max_area_ratio: float = 0.5,
    part_grid: int = 2,
    return_mask: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mask one contiguous rectangle inside each selected search target.

    Args:
        images: normalized search images, ``[B, C, H, W]``.
        target_boxes: normalized ``xywh`` target boxes, ``[B, 4]``.
        probability: per-sample augmentation probability.
        min_area_ratio: minimum occluded fraction of the target box.
        max_area_ratio: maximum occluded fraction of the target box.
        part_grid: target-local visibility grid size.

    Returns:
        occluded_images: a cloned tensor with selected rectangles filled by 0.
        part_visibility: remaining visible fraction for every target part,
            shape ``[B, part_grid ** 2]``.
        applied: samples with a valid synthetic occlusion, shape ``[B]``.
    """
    if images.ndim != 4:
        raise ValueError(f"images must have shape [B, C, H, W], got {images.shape}")
    if target_boxes.shape != (images.shape[0], 4):
        raise ValueError(
            f"target_boxes must have shape [{images.shape[0]}, 4], "
            f"got {target_boxes.shape}"
        )
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"probability must be in [0, 1], got {probability}")
    if not 0.0 < min_area_ratio <= max_area_ratio < 1.0:
        raise ValueError(
            "area ratios must satisfy 0 < min <= max < 1, got "
            f"{min_area_ratio}, {max_area_ratio}"
        )
    if part_grid < 1:
        raise ValueError(f"part_grid must be positive, got {part_grid}")

    batch_size, _, height, width = images.shape
    device = images.device
    dtype = images.dtype

    boxes = target_boxes.to(device=device, dtype=torch.float32).clamp(0.0, 1.0)
    box_x0 = boxes[:, 0] * width
    box_y0 = boxes[:, 1] * height
    box_x1 = ((boxes[:, 0] + boxes[:, 2]) * width).clamp(0.0, float(width))
    box_y1 = ((boxes[:, 1] + boxes[:, 3]) * height).clamp(0.0, float(height))
    box_w = (box_x1 - box_x0).clamp_min(0.0)
    box_h = (box_y1 - box_y0).clamp_min(0.0)

    applied = torch.rand(batch_size, device=device) < probability
    applied &= (box_w >= 2.0) & (box_h >= 2.0)

    area_ratio = torch.empty(batch_size, device=device).uniform_(
        min_area_ratio, max_area_ratio
    )
    log_aspect = torch.empty(batch_size, device=device).uniform_(
        -0.5, 0.5
    )
    aspect = log_aspect.exp()
    occ_w_ratio = torch.sqrt(area_ratio * aspect).clamp(max=0.95)
    occ_h_ratio = torch.sqrt(area_ratio / aspect).clamp(max=0.95)
    occ_w = box_w * occ_w_ratio
    occ_h = box_h * occ_h_ratio

    occ_x0 = box_x0 + torch.rand(batch_size, device=device) * (box_w - occ_w)
    occ_y0 = box_y0 + torch.rand(batch_size, device=device) * (box_h - occ_h)
    occ_x1 = occ_x0 + occ_w
    occ_y1 = occ_y0 + occ_h

    pixel_x = torch.arange(width, device=device, dtype=torch.float32) + 0.5
    pixel_y = torch.arange(height, device=device, dtype=torch.float32) + 0.5
    mask_x = (
        (pixel_x[None, :] >= occ_x0[:, None])
        & (pixel_x[None, :] < occ_x1[:, None])
    )
    mask_y = (
        (pixel_y[None, :] >= occ_y0[:, None])
        & (pixel_y[None, :] < occ_y1[:, None])
    )
    occlusion_mask = (
        mask_y[:, :, None] & mask_x[:, None, :] & applied[:, None, None]
    )

    occluded_images = images.clone()
    occluded_images.masked_fill_(occlusion_mask[:, None], 0.0)

    visibility_parts = []
    for row in range(part_grid):
        part_y0 = box_y0 + box_h * (row / part_grid)
        part_y1 = box_y0 + box_h * ((row + 1) / part_grid)
        for col in range(part_grid):
            part_x0 = box_x0 + box_w * (col / part_grid)
            part_x1 = box_x0 + box_w * ((col + 1) / part_grid)

            overlap_w = (
                torch.minimum(part_x1, occ_x1)
                - torch.maximum(part_x0, occ_x0)
            ).clamp_min(0.0)
            overlap_h = (
                torch.minimum(part_y1, occ_y1)
                - torch.maximum(part_y0, occ_y0)
            ).clamp_min(0.0)
            part_area = ((part_x1 - part_x0) * (part_y1 - part_y0)).clamp_min(
                1.0
            )
            visibility = 1.0 - (overlap_w * overlap_h / part_area)
            visibility = torch.where(
                applied, visibility, torch.ones_like(visibility)
            )
            visibility_parts.append(visibility)

    part_visibility = torch.stack(visibility_parts, dim=-1)
    result = (
        occluded_images,
        part_visibility.to(dtype=dtype),
        applied,
    )
    return result + (occlusion_mask,) if return_mask else result
