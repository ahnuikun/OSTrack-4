from . import BaseActor
from lib.utils.misc import NestedTensor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
import math
import torch
import torch.nn.functional as F
from lib.utils.merge import merge_template_search
from ...utils.heapmap_utils import generate_heatmap
from ...utils.ce_utils import generate_mask_cond, adjust_keep_rate
from ..data.vdrm_augmentation import apply_structured_target_occlusion


def select_vdrm_part_route_weight(pred_dict, visibility_weighted=True):
    """Select optional visibility weights for V8 part-route supervision."""
    if not visibility_weighted:
        return None
    return pred_dict.get('vdrm_visibility_target')


def compute_vdrm_part_rank_loss(
    part_similarity,
    search_global_index,
    gaussian_map,
    part_valid=None,
    part_weight=None,
    margin=0.1,
):
    """Rank each template part's target match above background matches.

    ``part_similarity`` is computed inside VDRM before the residual is added.
    Candidate elimination may have removed search tokens, so
    ``search_global_index`` maps the retained tokens back to the original
    score-map grid used by ``gaussian_map``.
    """
    if part_similarity.ndim != 3:
        raise ValueError(
            "part_similarity must have shape [B, K, L], got "
            f"{tuple(part_similarity.shape)}"
        )
    batch_size, num_parts, search_length = part_similarity.shape
    if search_global_index.shape != (batch_size, search_length):
        raise ValueError(
            "search_global_index must have shape [B, L], got "
            f"{tuple(search_global_index.shape)}"
        )
    if gaussian_map.ndim != 3 or gaussian_map.shape[0] != batch_size:
        raise ValueError(
            "gaussian_map must have shape [B, H, W], got "
            f"{tuple(gaussian_map.shape)}"
        )

    global_index = search_global_index.to(
        device=part_similarity.device, dtype=torch.long
    )
    flat_gaussian = gaussian_map.to(
        device=part_similarity.device, dtype=part_similarity.dtype
    ).flatten(1)
    if (
        global_index.numel() == 0
        or global_index.min() < 0
        or global_index.max() >= flat_gaussian.shape[1]
    ):
        raise ValueError("search_global_index is outside gaussian_map")

    retained_gaussian = flat_gaussian.gather(1, global_index)
    positive_mask = retained_gaussian > 0.0

    # With aggressive candidate elimination the Gaussian support can be
    # removed. Use the retained token nearest to the target peak as a stable
    # fallback instead of dropping the whole sample.
    missing_positive = ~positive_mask.any(dim=1)
    if missing_positive.any():
        full_peak_index = flat_gaussian.argmax(dim=1)
        grid_width = gaussian_map.shape[2]
        token_x = global_index.remainder(grid_width)
        token_y = global_index.div(grid_width, rounding_mode="floor")
        peak_x = full_peak_index.remainder(grid_width).unsqueeze(1)
        peak_y = full_peak_index.div(
            grid_width, rounding_mode="floor"
        ).unsqueeze(1)
        squared_distance = (token_x - peak_x).square() + (
            token_y - peak_y
        ).square()
        fallback_index = squared_distance.argmin(dim=1, keepdim=True)
        fallback_mask = torch.zeros_like(positive_mask)
        fallback_mask.scatter_(1, fallback_index, True)
        positive_mask = positive_mask | (
            missing_positive.unsqueeze(1) & fallback_mask
        )

    negative_mask = (retained_gaussian <= 0.0) & ~positive_mask
    positive_similarity = part_similarity.masked_fill(
        ~positive_mask[:, None, :], -torch.inf
    ).amax(dim=-1)
    negative_similarity = part_similarity.masked_fill(
        ~negative_mask[:, None, :], -torch.inf
    ).amax(dim=-1)

    valid = negative_mask.any(dim=1, keepdim=True).expand(
        batch_size, num_parts
    )
    if part_valid is not None:
        valid = valid & part_valid.to(device=valid.device, dtype=torch.bool)

    element_loss = F.softplus(
        float(margin) + negative_similarity - positive_similarity
    )
    loss_weight = valid.to(element_loss.dtype)
    if part_weight is not None:
        if part_weight.shape != (batch_size, num_parts):
            raise ValueError(
                "part_weight must have shape [B, K], got "
                f"{tuple(part_weight.shape)}"
            )
        loss_weight = loss_weight * part_weight.to(
            device=element_loss.device, dtype=element_loss.dtype
        ).clamp(0.0, 1.0)

    return (element_loss * loss_weight).sum() / loss_weight.sum().clamp_min(1.0)


def compute_vdrm_candidate_focal_loss(
    candidate_logits,
    search_global_index,
    gaussian_map,
    sample_valid=None,
    alpha=2.0,
    beta=4.0,
):
    """Supervise candidate consensus on retained CE search locations.

    This is the CenterNet focal objective applied before residual injection.
    It teaches the spatial gate to select the target candidate, rather than
    merely predicting whether template parts occur somewhere in the search.
    """
    if candidate_logits.ndim != 2:
        raise ValueError(
            "candidate_logits must have shape [B, L], got "
            f"{tuple(candidate_logits.shape)}"
        )
    batch_size, search_length = candidate_logits.shape
    if search_global_index.shape != (batch_size, search_length):
        raise ValueError(
            "search_global_index must have shape [B, L], got "
            f"{tuple(search_global_index.shape)}"
        )
    if gaussian_map.ndim != 3 or gaussian_map.shape[0] != batch_size:
        raise ValueError(
            "gaussian_map must have shape [B, H, W], got "
            f"{tuple(gaussian_map.shape)}"
        )

    global_index = search_global_index.to(
        device=candidate_logits.device, dtype=torch.long
    )
    flat_gaussian = gaussian_map.to(
        device=candidate_logits.device, dtype=candidate_logits.dtype
    ).flatten(1)
    if (
        global_index.numel() == 0
        or global_index.min() < 0
        or global_index.max() >= flat_gaussian.shape[1]
    ):
        raise ValueError("search_global_index is outside gaussian_map")
    retained_target = flat_gaussian.gather(1, global_index)

    # CE can remove the exact center token. Promote the nearest retained token
    # to the positive center so every valid sample retains one stable target.
    positive_mask = retained_target.eq(1.0)
    missing_positive = ~positive_mask.any(dim=1)
    if missing_positive.any():
        full_peak_index = flat_gaussian.argmax(dim=1)
        grid_width = gaussian_map.shape[2]
        token_x = global_index.remainder(grid_width)
        token_y = global_index.div(grid_width, rounding_mode="floor")
        peak_x = full_peak_index.remainder(grid_width).unsqueeze(1)
        peak_y = full_peak_index.div(
            grid_width, rounding_mode="floor"
        ).unsqueeze(1)
        squared_distance = (token_x - peak_x).square() + (
            token_y - peak_y
        ).square()
        fallback_index = squared_distance.argmin(dim=1, keepdim=True)
        fallback_mask = torch.zeros_like(retained_target, dtype=torch.bool)
        fallback_mask.scatter_(1, fallback_index, True)
        retained_target = torch.where(
            missing_positive[:, None] & fallback_mask,
            torch.ones_like(retained_target),
            retained_target,
        )
        positive_mask = retained_target.eq(1.0)

    probability = torch.sigmoid(candidate_logits).clamp(
        1e-6, 1.0 - 1e-6
    )
    negative_mask = retained_target.lt(1.0)
    negative_weight = (1.0 - retained_target).pow(float(beta))
    positive_loss = (
        probability.log()
        * (1.0 - probability).pow(float(alpha))
        * positive_mask.to(probability.dtype)
    ).sum(dim=1)
    negative_loss = (
        (1.0 - probability).log()
        * probability.pow(float(alpha))
        * negative_weight
        * negative_mask.to(probability.dtype)
    ).sum(dim=1)
    normalizer = positive_mask.sum(dim=1).clamp_min(1).to(probability.dtype)
    per_sample_loss = -(positive_loss + negative_loss) / normalizer

    if sample_valid is None:
        valid_weight = torch.ones_like(per_sample_loss)
    else:
        if sample_valid.shape != (batch_size,):
            raise ValueError(
                "sample_valid must have shape [B], got "
                f"{tuple(sample_valid.shape)}"
            )
        valid_weight = sample_valid.to(
            device=per_sample_loss.device, dtype=per_sample_loss.dtype
        )
    return (
        per_sample_loss * valid_weight
    ).sum() / valid_weight.sum().clamp_min(1.0)


def build_vdrm_part_route_targets(
    search_bbox,
    height,
    width,
    num_parts=4,
    dilation=1.0,
):
    """Build soft search-grid targets for corresponding template parts.

    The normalized target box is split using the same square layout as the
    template prototypes. A part owns its complete sub-box, and supervision
    decays outside that region over ``dilation`` feature cells. Tiny parts
    that contain no cell center are assigned their nearest cell explicitly.
    """
    if search_bbox.ndim != 2 or search_bbox.shape[1] != 4:
        raise ValueError(
            "search_bbox must have shape [B, 4], got "
            f"{tuple(search_bbox.shape)}"
        )
    part_grid = int(math.sqrt(num_parts))
    if part_grid * part_grid != num_parts:
        raise ValueError(
            f"num_parts must be a square number, got {num_parts}"
        )
    if height < 1 or width < 1:
        raise ValueError(
            f"height and width must be positive, got {height}x{width}"
        )
    if dilation < 0.0:
        raise ValueError(
            f"dilation must be non-negative, got {dilation}"
        )

    bbox = search_bbox.clamp(0.0, 1.0)
    x0, y0, box_width, box_height = bbox.unbind(dim=-1)
    x1 = (x0 + box_width).clamp(0.0, 1.0)
    y1 = (y0 + box_height).clamp(0.0, 1.0)
    grid_y, grid_x = torch.meshgrid(
        (torch.arange(height, device=bbox.device, dtype=bbox.dtype) + 0.5)
        / height,
        (torch.arange(width, device=bbox.device, dtype=bbox.dtype) + 0.5)
        / width,
        indexing='ij',
    )
    grid_x = grid_x.unsqueeze(0)
    grid_y = grid_y.unsqueeze(0)

    targets = []
    for row in range(part_grid):
        part_y0 = y0 + (y1 - y0) * (row / part_grid)
        part_y1 = y0 + (y1 - y0) * ((row + 1) / part_grid)
        for col in range(part_grid):
            part_x0 = x0 + (x1 - x0) * (col / part_grid)
            part_x1 = x0 + (x1 - x0) * ((col + 1) / part_grid)
            distance_x = (
                (part_x0[:, None, None] - grid_x).clamp_min(0.0)
                + (grid_x - part_x1[:, None, None]).clamp_min(0.0)
            ) * width
            distance_y = (
                (part_y0[:, None, None] - grid_y).clamp_min(0.0)
                + (grid_y - part_y1[:, None, None]).clamp_min(0.0)
            ) * height
            squared_distance = distance_x.square() + distance_y.square()
            if dilation > 0.0:
                target = torch.exp(
                    -0.5 * squared_distance / float(dilation) ** 2
                )
            else:
                target = squared_distance.eq(0.0).to(bbox.dtype)

            # Very small boxes can fall between feature-cell centers. Give
            # every part one unambiguous core token before CE gathers it.
            missing_core = ~target.eq(1.0).flatten(1).any(dim=1)
            if missing_core.any():
                nearest_index = squared_distance.flatten(1).argmin(
                    dim=1, keepdim=True
                )
                nearest_mask = torch.zeros_like(target).flatten(1)
                nearest_mask.scatter_(1, nearest_index, 1.0)
                nearest_mask = nearest_mask.view_as(target)
                target = torch.where(
                    missing_core[:, None, None] & nearest_mask.bool(),
                    torch.ones_like(target),
                    target,
                )
            targets.append(target)
    return torch.stack(targets, dim=1)


def compute_vdrm_part_route_loss(
    part_route_logits,
    search_global_index,
    search_bbox,
    grid_height,
    grid_width,
    part_valid=None,
    part_weight=None,
    dilation=1.0,
):
    """Supervise V8 part routing with balanced soft region targets."""
    if part_route_logits.ndim != 3:
        raise ValueError(
            "part_route_logits must have shape [B, K, L], got "
            f"{tuple(part_route_logits.shape)}"
        )
    batch_size, num_parts, search_length = part_route_logits.shape
    if search_global_index.shape != (batch_size, search_length):
        raise ValueError(
            "search_global_index must have shape [B, L], got "
            f"{tuple(search_global_index.shape)}"
        )

    targets = build_vdrm_part_route_targets(
        search_bbox.to(
            device=part_route_logits.device,
            dtype=part_route_logits.dtype,
        ),
        grid_height,
        grid_width,
        num_parts=num_parts,
        dilation=dilation,
    )
    global_index = search_global_index.to(
        device=part_route_logits.device, dtype=torch.long
    )
    if (
        global_index.numel() == 0
        or global_index.min() < 0
        or global_index.max() >= grid_height * grid_width
    ):
        raise ValueError("search_global_index is outside the route grid")
    retained_target = targets.flatten(2).gather(
        2, global_index[:, None, :].expand(-1, num_parts, -1)
    )

    positive_mass = retained_target.sum(dim=-1)
    negative_target = 1.0 - retained_target
    negative_mass = negative_target.sum(dim=-1)
    positive_loss = -(
        retained_target * F.logsigmoid(part_route_logits)
    ).sum(dim=-1) / positive_mass.clamp_min(1e-6)
    negative_loss = -(
        negative_target * F.logsigmoid(-part_route_logits)
    ).sum(dim=-1) / negative_mass.clamp_min(1e-6)
    per_part_loss = 0.5 * (positive_loss + negative_loss)

    valid_weight = (
        (positive_mass > 0.0) & (negative_mass > 0.0)
    ).to(per_part_loss.dtype)
    if part_valid is not None:
        if part_valid.shape != (batch_size, num_parts):
            raise ValueError(
                "part_valid must have shape [B, K], got "
                f"{tuple(part_valid.shape)}"
            )
        valid_weight = valid_weight * part_valid.to(
            device=per_part_loss.device, dtype=per_part_loss.dtype
        )
    if part_weight is not None:
        if part_weight.shape != (batch_size, num_parts):
            raise ValueError(
                "part_weight must have shape [B, K], got "
                f"{tuple(part_weight.shape)}"
            )
        valid_weight = valid_weight * part_weight.to(
            device=per_part_loss.device, dtype=per_part_loss.dtype
        ).clamp(0.0, 1.0)
    loss = (
        per_part_loss * valid_weight
    ).sum() / valid_weight.sum().clamp_min(1.0)

    probability = torch.sigmoid(part_route_logits)
    expanded_valid = valid_weight.unsqueeze(-1)
    positive_probability = (
        probability * retained_target * expanded_valid
    ).sum() / (
        retained_target * expanded_valid
    ).sum().clamp_min(1e-6)
    background_probability = (
        probability * negative_target * expanded_valid
    ).sum() / (
        negative_target * expanded_valid
    ).sum().clamp_min(1e-6)
    diagnostics = {
        "part_route_positive_probability": positive_probability,
        "part_route_background_probability": background_probability,
    }
    return loss, diagnostics


def compute_vdrm_response_rank_loss(
    score_logits,
    gaussian_map,
    distractor_boxes=None,
    distractor_applied=None,
    align_distractor=True,
):
    """Rank the target response above a hard background response.

    The original V1/V3 behavior uses the maximum response over all background
    cells. When HNCP metadata is supplied and ``align_distractor`` is true, an
    applied sample instead uses the maximum response whose cell center lies
    inside the pasted distractor box. With alignment disabled, the metadata is
    used only for read-only hardness diagnostics and the V3 loss is unchanged.
    A very small valid paste that contains no cell center is represented by the
    closest background cell, so the supervision is not silently discarded.
    """
    if score_logits.ndim != 3:
        raise ValueError(
            "score_logits must have shape [B, H, W], got "
            f"{tuple(score_logits.shape)}"
        )
    if gaussian_map.shape != score_logits.shape:
        raise ValueError(
            "gaussian_map must match score_logits, got "
            f"{tuple(gaussian_map.shape)} and {tuple(score_logits.shape)}"
        )

    batch_size, height, width = score_logits.shape
    gaussian_map = gaussian_map.to(
        device=score_logits.device, dtype=score_logits.dtype
    )
    positive_index = gaussian_map.flatten(1).argmax(dim=1, keepdim=True)
    positive_logit = score_logits.flatten(1).gather(
        dim=1, index=positive_index
    ).squeeze(1)

    negative_mask = gaussian_map <= 0.0
    has_negative = negative_mask.flatten(1).any(dim=1)
    global_negative_logit = score_logits.masked_fill(
        ~negative_mask, -torch.inf
    ).flatten(1).amax(dim=1)
    global_negative_index = score_logits.masked_fill(
        ~negative_mask, -torch.inf
    ).flatten(1).argmax(dim=1, keepdim=True)
    selected_negative_logit = global_negative_logit
    aligned_mask = torch.zeros(
        batch_size, device=score_logits.device, dtype=torch.bool
    )
    applied_mask = torch.zeros_like(aligned_mask)
    distractor_global_gap = score_logits.new_zeros(())
    distractor_hard_hit_rate = score_logits.new_zeros(())
    distractor_rank_margin = score_logits.new_zeros(())

    if distractor_boxes is not None and distractor_applied is not None:
        boxes = distractor_boxes.to(
            device=score_logits.device, dtype=score_logits.dtype
        ).reshape(-1, 4)
        applied_mask = distractor_applied.to(
            device=score_logits.device, dtype=torch.bool
        ).reshape(-1)
        if boxes.shape[0] != batch_size or applied_mask.shape[0] != batch_size:
            raise ValueError(
                "distractor metadata batch size must match score_logits, got "
                f"{boxes.shape[0]}, {applied_mask.shape[0]}, and {batch_size}"
            )

        x0 = boxes[:, 0].clamp(0.0, 1.0)
        y0 = boxes[:, 1].clamp(0.0, 1.0)
        x1 = (boxes[:, 0] + boxes[:, 2]).clamp(0.0, 1.0)
        y1 = (boxes[:, 1] + boxes[:, 3]).clamp(0.0, 1.0)
        valid_box = (x1 > x0) & (y1 > y0)
        requested_alignment = applied_mask & valid_box & has_negative

        cell_x = (
            torch.arange(width, device=score_logits.device, dtype=score_logits.dtype)
            + 0.5
        ) / width
        cell_y = (
            torch.arange(height, device=score_logits.device, dtype=score_logits.dtype)
            + 0.5
        ) / height
        box_mask = (
            (cell_x[None, None, :] >= x0[:, None, None])
            & (cell_x[None, None, :] < x1[:, None, None])
            & (cell_y[None, :, None] >= y0[:, None, None])
            & (cell_y[None, :, None] < y1[:, None, None])
        )
        distractor_negative_mask = box_mask & negative_mask

        # Tiny pasted objects can fall between score-map cell centers. Map
        # those boxes to the nearest valid background cell to retain alignment.
        missing_cell = requested_alignment & ~distractor_negative_mask.flatten(1).any(dim=1)
        if missing_cell.any():
            center_x = ((x0 + x1) * 0.5)[:, None, None]
            center_y = ((y0 + y1) * 0.5)[:, None, None]
            squared_distance = (
                (cell_x[None, None, :] - center_x).square()
                + (cell_y[None, :, None] - center_y).square()
            )
            squared_distance = squared_distance.masked_fill(
                ~negative_mask, torch.inf
            )
            nearest_index = squared_distance.flatten(1).argmin(
                dim=1, keepdim=True
            )
            fallback_mask = torch.zeros_like(
                negative_mask.flatten(1), dtype=torch.bool
            )
            fallback_mask.scatter_(1, nearest_index, True)
            distractor_negative_mask = distractor_negative_mask | (
                missing_cell[:, None, None]
                & fallback_mask.view(batch_size, height, width)
            )

        aligned_mask = requested_alignment & distractor_negative_mask.flatten(1).any(dim=1)
        distractor_negative_logit = score_logits.masked_fill(
            ~distractor_negative_mask, -torch.inf
        ).flatten(1).amax(dim=1)
        if align_distractor:
            selected_negative_logit = torch.where(
                aligned_mask,
                distractor_negative_logit,
                global_negative_logit,
            )

        if aligned_mask.any():
            hard_hit = distractor_negative_mask.flatten(1).gather(
                dim=1, index=global_negative_index
            ).squeeze(1)
            distractor_hard_hit_rate = (
                hard_hit[aligned_mask].float().mean()
            )
            distractor_global_gap = (
                global_negative_logit[aligned_mask]
                - distractor_negative_logit[aligned_mask]
            ).mean()
            distractor_rank_margin = (
                positive_logit[aligned_mask]
                - distractor_negative_logit[aligned_mask]
            ).mean()

    if has_negative.any():
        rank_loss = F.softplus(
            selected_negative_logit[has_negative]
            - positive_logit[has_negative]
        ).mean()
    else:
        rank_loss = score_logits.sum() * 0.0

    applied_count = applied_mask.float().sum()
    alignment_success_rate = (
        aligned_mask.float().sum() / applied_count.clamp_min(1.0)
    )
    diagnostics = {
        "alignment_success_rate": alignment_success_rate,
        "distractor_rank_margin": distractor_rank_margin,
        "distractor_hard_hit_rate": distractor_hard_hit_rate,
        "distractor_global_gap": distractor_global_gap,
    }
    return rank_loss, diagnostics


class OSTrackActor(BaseActor):
    """ Actor for training OSTrack models """

    def __init__(self, net, objective, loss_weight, settings, cfg=None):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.cfg = cfg

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'template', 'search', 'gt_bbox'.
            template_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)
        returns:
            loss    - the training loss
            status  -  dict containing detailed losses
        """
        # forward pass
        out_dict = self.forward_pass(data)

        # compute losses
        loss, status = self.compute_losses(out_dict, data)

        return loss, status

    def forward_pass(self, data):
        # currently only support 1 template and 1 search region
        assert len(data['template_images']) == 1
        assert len(data['search_images']) == 1

        template_list = []
        for i in range(self.settings.num_template):
            template_img_i = data['template_images'][i].view(-1,
                                                             *data['template_images'].shape[2:])  # (batch, 3, 128, 128)
            # template_att_i = data['template_att'][i].view(-1, *data['template_att'].shape[2:])  # (batch, 128, 128)
            template_list.append(template_img_i)

        search_img = data['search_images'][0].view(-1, *data['search_images'].shape[2:])  # (batch, 3, 320, 320)
        # search_att = data['search_att'][0].view(-1, *data['search_att'].shape[2:])  # (batch, 320, 320)
        template_bbox = data['template_anno'][0].view(-1, 4)

        visibility_target = None
        visibility_applied = None
        vdrm_cfg = getattr(self.cfg.MODEL, "VDRM", None)
        vdrm_enabled = bool(vdrm_cfg is not None and vdrm_cfg.ENABLED)
        occlusion_probability = getattr(
            self.cfg.DATA.SEARCH, "VDRM_OCCLUSION_PROBABILITY", 0.0
        )
        if vdrm_enabled and self.net.training and occlusion_probability > 0.0:
            search_bbox = data['search_anno'][0].view(-1, 4)
            search_img, visibility_target, visibility_applied = (
                apply_structured_target_occlusion(
                    search_img,
                    search_bbox,
                    probability=occlusion_probability,
                    min_area_ratio=self.cfg.DATA.SEARCH.VDRM_OCCLUSION_MIN_RATIO,
                    max_area_ratio=self.cfg.DATA.SEARCH.VDRM_OCCLUSION_MAX_RATIO,
                    part_grid=int(vdrm_cfg.NUM_PARTS ** 0.5),
                )
            )

        box_mask_z = None
        ce_keep_rate = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            box_mask_z = generate_mask_cond(self.cfg, template_list[0].shape[0], template_list[0].device,
                                            data['template_anno'][0])

            ce_start_epoch = self.cfg.TRAIN.CE_START_EPOCH
            ce_warm_epoch = self.cfg.TRAIN.CE_WARM_EPOCH
            ce_keep_rate = adjust_keep_rate(data['epoch'], warmup_epochs=ce_start_epoch,
                                                total_epochs=ce_start_epoch + ce_warm_epoch,
                                                ITERS_PER_EPOCH=1,
                                                base_keep_rate=self.cfg.MODEL.BACKBONE.CE_KEEP_RATIO[0])

        if len(template_list) == 1:
            template_list = template_list[0]

        out_dict = self.net(template=template_list,
                            search=search_img,
                            ce_template_mask=box_mask_z,
                            ce_keep_rate=ce_keep_rate,
                            template_bbox=template_bbox,
                            return_last_attn=False)
        if visibility_target is not None:
            out_dict['vdrm_visibility_target'] = visibility_target
            out_dict['vdrm_visibility_applied'] = visibility_applied

        return out_dict

    def compute_losses(self, pred_dict, gt_dict, return_status=True):
        # gt gaussian map
        gt_bbox = gt_dict['search_anno'][-1]  # (Ns, batch, 4) (x1,y1,w,h) -> (batch, 4)
        gt_gaussian_maps = generate_heatmap(gt_dict['search_anno'], self.cfg.DATA.SEARCH.SIZE, self.cfg.MODEL.BACKBONE.STRIDE)
        gt_gaussian_maps = gt_gaussian_maps[-1].unsqueeze(1)

        # Get boxes
        pred_boxes = pred_dict['pred_boxes']
        if torch.isnan(pred_boxes).any():
            raise ValueError("Network outputs is NAN! Stop Training")
        num_queries = pred_boxes.size(1)
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0,
                                                                                                           max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)
        # compute giou and iou
        try:
            giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        except:
            giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
        # compute l1 loss
        l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        # compute location loss
        if 'score_map' in pred_dict:
            location_loss = self.objective['focal'](pred_dict['score_map'], gt_gaussian_maps)
        else:
            location_loss = torch.tensor(0.0, device=l1_loss.device)
        visibility_loss = pred_boxes.new_zeros(())
        rank_loss = pred_boxes.new_zeros(())
        candidate_loss = pred_boxes.new_zeros(())
        part_route_loss = pred_boxes.new_zeros(())
        rank_diagnostics = None
        part_route_diagnostics = None
        aux_weight_scale = 0.0

        vdrm_cfg = getattr(self.cfg.MODEL, "VDRM", None)
        vdrm_enabled = bool(vdrm_cfg is not None and vdrm_cfg.ENABLED)
        if vdrm_enabled:
            if (
                'part_reliability' in pred_dict
                and 'vdrm_visibility_target' in pred_dict
                and 'vdrm_visibility_applied' in pred_dict
            ):
                part_reliability = pred_dict['part_reliability']
                visibility_target = pred_dict['vdrm_visibility_target'].to(
                    device=part_reliability.device,
                    dtype=part_reliability.dtype,
                )
                visibility_mask = pred_dict['vdrm_visibility_applied'][:, None]
                if 'part_valid' in pred_dict:
                    visibility_mask = visibility_mask & pred_dict['part_valid']
                visibility_element = F.binary_cross_entropy(
                    part_reliability.clamp(1e-6, 1.0 - 1e-6),
                    visibility_target,
                    reduction='none',
                )
                visibility_mask_float = visibility_mask.to(
                    visibility_element.dtype
                )
                visibility_loss = (
                    visibility_element * visibility_mask_float
                ).sum() / visibility_mask_float.sum().clamp_min(1.0)

            reliability_mode = getattr(vdrm_cfg, 'RELIABILITY_MODE', 'topk')
            if reliability_mode == 'margin':
                required_rank_outputs = {
                    'part_similarity', 'search_global_index'
                }
                missing_rank_outputs = required_rank_outputs.difference(
                    pred_dict
                )
                if missing_rank_outputs:
                    raise RuntimeError(
                        "VDRM margin rank loss is missing model outputs: "
                        f"{sorted(missing_rank_outputs)}"
                    )
                part_weight = pred_dict.get('vdrm_visibility_target')
                rank_loss = compute_vdrm_part_rank_loss(
                    pred_dict['part_similarity'],
                    pred_dict['search_global_index'],
                    gt_gaussian_maps.squeeze(1),
                    part_valid=pred_dict.get('part_valid'),
                    part_weight=part_weight,
                    margin=getattr(
                        self.cfg.TRAIN, 'VDRM_RANK_MARGIN', 0.1
                    ),
                )
            elif 'score_logits' in pred_dict:
                score_logits = pred_dict['score_logits'].squeeze(1)
                gaussian_map = gt_gaussian_maps.squeeze(1)
                align_distractor_rank = bool(
                    getattr(
                        self.cfg.TRAIN,
                        'VDRM_ALIGN_DISTRACTOR_RANK',
                        False,
                    )
                )
                log_distractor_hardness = bool(
                    getattr(
                        self.cfg.TRAIN,
                        'VDRM_LOG_DISTRACTOR_HARDNESS',
                        False,
                    )
                )
                distractor_boxes = None
                distractor_applied = None
                if align_distractor_rank or log_distractor_hardness:
                    distractor_boxes = gt_dict.get('vdrm_distractor_box')
                    distractor_applied = gt_dict.get(
                        'vdrm_distractor_applied'
                    )
                rank_loss, rank_diagnostics = (
                    compute_vdrm_response_rank_loss(
                        score_logits,
                        gaussian_map,
                        distractor_boxes=distractor_boxes,
                        distractor_applied=distractor_applied,
                        align_distractor=align_distractor_rank,
                    )
                )

            spatial_gate_mode = getattr(
                vdrm_cfg, 'SPATIAL_GATE_MODE', 'token_match'
            )
            if spatial_gate_mode == 'candidate_consensus':
                required_candidate_outputs = {
                    'candidate_identity_logits', 'search_global_index'
                }
                missing_candidate_outputs = (
                    required_candidate_outputs.difference(pred_dict)
                )
                if missing_candidate_outputs:
                    raise RuntimeError(
                        "VDRM candidate loss is missing model outputs: "
                        f"{sorted(missing_candidate_outputs)}"
                    )
                candidate_loss = compute_vdrm_candidate_focal_loss(
                    pred_dict['candidate_identity_logits'],
                    pred_dict['search_global_index'],
                    gt_gaussian_maps.squeeze(1),
                    sample_valid=pred_dict.get('candidate_consensus_valid'),
                )
            elif spatial_gate_mode == 'part_aligned':
                required_route_outputs = {
                    'part_route_logits', 'search_global_index'
                }
                missing_route_outputs = required_route_outputs.difference(
                    pred_dict
                )
                if missing_route_outputs:
                    raise RuntimeError(
                        "VDRM part-route loss is missing model outputs: "
                        f"{sorted(missing_route_outputs)}"
                    )
                part_route_loss, part_route_diagnostics = (
                    compute_vdrm_part_route_loss(
                        pred_dict['part_route_logits'],
                        pred_dict['search_global_index'],
                        gt_bbox,
                        gt_gaussian_maps.shape[-2],
                        gt_gaussian_maps.shape[-1],
                        part_valid=pred_dict.get('part_valid'),
                        part_weight=select_vdrm_part_route_weight(
                            pred_dict,
                            visibility_weighted=bool(
                                getattr(
                                    self.cfg.TRAIN,
                                    'VDRM_ROUTE_VISIBILITY_WEIGHTED',
                                    True,
                                )
                            ),
                        ),
                        dilation=getattr(
                            self.cfg.TRAIN,
                            'VDRM_PART_TARGET_DILATION',
                            1.0,
                        ),
                    )
                )

            warmup_epochs = self.cfg.TRAIN.VDRM_AUX_WARMUP_EPOCHS
            epoch = float(gt_dict.get('epoch', 1))
            if warmup_epochs <= 1:
                aux_weight_scale = 1.0
            else:
                aux_weight_scale = min(
                    1.0, max(0.0, (epoch - 1.0) / (warmup_epochs - 1.0))
                )

        # weighted sum
        loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight['focal'] * location_loss
        if vdrm_enabled:
            loss = loss + aux_weight_scale * (
                self.cfg.TRAIN.VDRM_VISIBILITY_WEIGHT * visibility_loss
                + self.cfg.TRAIN.VDRM_RANK_WEIGHT * rank_loss
                + self.cfg.TRAIN.VDRM_CANDIDATE_WEIGHT * candidate_loss
                + self.cfg.TRAIN.VDRM_PART_ROUTE_WEIGHT * part_route_loss
            )
        if return_status:
            # status for log
            mean_iou = iou.detach().mean()
            status = {"Loss/total": loss.item(),
                      "Loss/giou": giou_loss.item(),
                      "Loss/l1": l1_loss.item(),
                      "Loss/location": location_loss.item(),
                      "IoU": mean_iou.item()}
            if vdrm_enabled:
                status.update({
                    "Loss/vdrm_visibility": visibility_loss.item(),
                    "Loss/vdrm_rank": rank_loss.item(),
                    "Loss/vdrm_candidate": candidate_loss.item(),
                    "Loss/vdrm_part_route": part_route_loss.item(),
                    "VDRM/aux_weight_scale": aux_weight_scale,
                    "VDRM/alpha": pred_dict['vdrm_alpha'].detach().item()
                    if 'vdrm_alpha' in pred_dict else 0.0,
                    "VDRM/reliability": pred_dict['visual_reliability'].detach().mean().item()
                    if 'visual_reliability' in pred_dict else 0.0,
                })
                if 'candidate_target_reliability' in pred_dict:
                    status["VDRM/candidate_target_reliability"] = (
                        pred_dict['candidate_target_reliability']
                        .detach()
                        .mean()
                        .item()
                    )
                if 'candidate_reliability_mean' in pred_dict:
                    status["VDRM/candidate_reliability_mean"] = (
                        pred_dict['candidate_reliability_mean']
                        .detach()
                        .mean()
                        .item()
                    )
                if part_route_diagnostics is not None:
                    status.update({
                        "VDRM/part_route_positive_probability": (
                            part_route_diagnostics[
                                'part_route_positive_probability'
                            ].detach().item()
                        ),
                        "VDRM/part_route_background_probability": (
                            part_route_diagnostics[
                                'part_route_background_probability'
                            ].detach().item()
                        ),
                    })
                if "vdrm_distractor_applied" in gt_dict:
                    status["VDRM/distractor_applied_rate"] = (
                        gt_dict["vdrm_distractor_applied"]
                        .detach()
                        .float()
                        .mean()
                        .item()
                    )
                if rank_diagnostics is not None and bool(
                    getattr(
                        self.cfg.TRAIN,
                        'VDRM_ALIGN_DISTRACTOR_RANK',
                        False,
                    )
                ):
                    status.update({
                        "VDRM/alignment_success_rate": (
                            rank_diagnostics["alignment_success_rate"]
                            .detach()
                            .item()
                        ),
                        "VDRM/distractor_rank_margin": (
                            rank_diagnostics["distractor_rank_margin"]
                            .detach()
                            .item()
                        ),
                    })
                if rank_diagnostics is not None and bool(
                    getattr(
                        self.cfg.TRAIN,
                        'VDRM_LOG_DISTRACTOR_HARDNESS',
                        False,
                    )
                ):
                    status.update({
                        "VDRM/distractor_hard_hit_rate": (
                            rank_diagnostics["distractor_hard_hit_rate"]
                            .detach()
                            .item()
                        ),
                        "VDRM/distractor_global_gap": (
                            rank_diagnostics["distractor_global_gap"]
                            .detach()
                            .item()
                        ),
                    })
                residual_status_keys = {
                    "vdrm_residual_clip_rate": "VDRM/residual_clip_rate",
                    "vdrm_raw_delta_relative_norm": (
                        "VDRM/raw_delta_relative_norm"
                    ),
                    "vdrm_delta_relative_norm": (
                        "VDRM/delta_relative_norm"
                    ),
                    "vdrm_residual_top10_energy_fraction": (
                        "VDRM/residual_top10_energy_fraction"
                    ),
                    "vdrm_residual_spatial_entropy": (
                        "VDRM/residual_spatial_entropy"
                    ),
                }
                for output_key, status_key in residual_status_keys.items():
                    if output_key in pred_dict:
                        status[status_key] = (
                            pred_dict[output_key].detach().mean().item()
                        )
                if 'part_match_margin' in pred_dict:
                    status.update({
                        "VDRM/match_margin": pred_dict[
                            'part_match_margin'
                        ].detach().mean().item(),
                        "VDRM/peak_similarity": pred_dict[
                            'part_peak_similarity'
                        ].detach().mean().item(),
                        "VDRM/hard_negative_similarity": pred_dict[
                            'part_hard_negative_similarity'
                        ].detach().mean().item(),
                    })
            return loss, status
        else:
            return loss
