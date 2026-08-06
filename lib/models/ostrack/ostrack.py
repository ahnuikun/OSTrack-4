"""
Basic OSTrack model.
"""
import math
import os
from typing import List

import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones

from lib.models.layers.head import build_box_head
from lib.models.ostrack.vit import vit_base_patch16_224
from lib.models.ostrack.vit_ce import vit_large_patch16_224_ce, vit_base_patch16_224_ce
from lib.utils.box_ops import box_xyxy_to_cxcywh


class OSTrack(nn.Module):
    """ This is the base class for OSTrack """

    def __init__(self, transformer, box_head, aux_loss=False, head_type="CORNER",
                 vdrm_enabled=False):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.backbone = transformer
        self.box_head = box_head

        self.aux_loss = aux_loss
        self.head_type = head_type
        self.vdrm_enabled = vdrm_enabled
        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)

    def forward(self, template: torch.Tensor,
                search: torch.Tensor,
                ce_template_mask=None,
                ce_keep_rate=None,
                template_bbox=None,
                return_last_attn=False,
                ):
        x, aux_dict = self.backbone(z=template, x=search,
                                    ce_template_mask=ce_template_mask,
                                    ce_keep_rate=ce_keep_rate,
                                    template_bbox=template_bbox,
                                    return_last_attn=return_last_attn, )

        # Forward head
        feat_last = x
        if isinstance(x, list):
            feat_last = x[-1]
        out = self.forward_head(feat_last, None)

        candidate_map = aux_dict.get('candidate_reliability_map')
        score_map = out.get('score_map')
        if candidate_map is not None and score_map is not None:
            if candidate_map.shape != score_map.shape:
                raise RuntimeError(
                    "candidate reliability map must match the center score "
                    f"map, got {candidate_map.shape} and {score_map.shape}"
                )
            selected_index = score_map.flatten(1).argmax(
                dim=1, keepdim=True
            )
            aux_dict['candidate_target_reliability'] = (
                candidate_map.flatten(1).gather(1, selected_index).squeeze(1)
            )

        out.update(aux_dict)
        out['backbone_feat'] = x
        return out

    def forward_head(self, cat_feature, gt_score_map=None):
        """
        cat_feature: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
        """
        enc_opt = cat_feature[:, -self.feat_len_s:]  # encoder output for the search region (B, HW, C)
        opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)

        if self.head_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.box_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out

        elif self.head_type == "CENTER":
            # run the center head
            center_output = self.box_head(
                opt_feat,
                gt_score_map,
                return_score_logits=self.vdrm_enabled,
            )
            if self.vdrm_enabled:
                score_map_ctr, bbox, size_map, offset_map, score_logits = center_output
            else:
                score_map_ctr, bbox, size_map, offset_map = center_output
            # outputs_coord = box_xyxy_to_cxcywh(bbox)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map}
            if self.vdrm_enabled:
                out['score_logits'] = score_logits
            return out
        else:
            raise NotImplementedError


def build_ostrack(cfg, training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    pretrained_path = os.path.join(current_dir, '../../../pretrained_models')
    vdrm_cfg = getattr(cfg.MODEL, "VDRM", None)
    vdrm_enabled = bool(vdrm_cfg is not None and vdrm_cfg.ENABLED)
    if vdrm_enabled and cfg.MODEL.HEAD.TYPE != "CENTER":
        raise ValueError("VDRM-v1 requires MODEL.HEAD.TYPE='CENTER'")
    if vdrm_enabled and not cfg.MODEL.BACKBONE.TYPE.endswith("_ce"):
        raise ValueError("VDRM-v1 currently requires an OSTrack CE backbone")
    if cfg.MODEL.PRETRAIN_FILE and ('OSTrack' not in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = ''

    if cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224':
        backbone = vit_base_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    elif cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224_ce':
        if vdrm_enabled and cfg.MODEL.BACKBONE.CAT_MODE != 'direct':
            raise ValueError("VDRM-v1 requires MODEL.BACKBONE.CAT_MODE='direct'")
        backbone = vit_base_patch16_224_ce(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                           ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                           ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
                                           vdrm_enabled=vdrm_enabled,
                                           vdrm_insert_layer=vdrm_cfg.INSERT_LAYER if vdrm_enabled else 6,
                                           vdrm_num_parts=vdrm_cfg.NUM_PARTS if vdrm_enabled else 4,
                                           vdrm_topk=vdrm_cfg.TOPK if vdrm_enabled else 4,
                                           vdrm_reliability_mode=vdrm_cfg.RELIABILITY_MODE if vdrm_enabled else "topk",
                                           vdrm_nms_radius=vdrm_cfg.NMS_RADIUS if vdrm_enabled else 1,
                                           vdrm_initial_match_scale=vdrm_cfg.INITIAL_MATCH_SCALE if vdrm_enabled else 5.0,
                                           vdrm_initial_match_bias=vdrm_cfg.INITIAL_MATCH_BIAS if vdrm_enabled else -2.5,
                                           vdrm_residual_max_ratio=vdrm_cfg.RESIDUAL_MAX_RATIO if vdrm_enabled else 0.0,
                                            vdrm_spatial_gate_mode=vdrm_cfg.SPATIAL_GATE_MODE if vdrm_enabled else "token_match",
                                            vdrm_candidate_local_radius=vdrm_cfg.CANDIDATE_LOCAL_RADIUS if vdrm_enabled else 1,
                                            vdrm_candidate_consensus_parts=vdrm_cfg.CANDIDATE_CONSENSUS_PARTS if vdrm_enabled else 2,
                                            vdrm_candidate_initial_match_scale=vdrm_cfg.CANDIDATE_INITIAL_MATCH_SCALE if vdrm_enabled else 5.0,
                                            vdrm_candidate_initial_match_bias=vdrm_cfg.CANDIDATE_INITIAL_MATCH_BIAS if vdrm_enabled else -2.5,
                                            vdrm_part_route_initial_match_scale=vdrm_cfg.PART_ROUTE_INITIAL_MATCH_SCALE if vdrm_enabled else 5.0,
                                            vdrm_part_route_initial_match_bias=vdrm_cfg.PART_ROUTE_INITIAL_MATCH_BIAS if vdrm_enabled else -2.5,
                                            vdrm_part_route_residual_floor=vdrm_cfg.PART_ROUTE_RESIDUAL_FLOOR if vdrm_enabled else 1.0,
                                            vdrm_candidate_modulation_max=vdrm_cfg.CANDIDATE_MODULATION_MAX if vdrm_enabled else 0.5,
                                            vdrm_alpha_max=vdrm_cfg.ALPHA_MAX if vdrm_enabled else 0.0,
                                           )
        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    elif cfg.MODEL.BACKBONE.TYPE == 'vit_large_patch16_224_ce':
        if vdrm_enabled and cfg.MODEL.BACKBONE.CAT_MODE != 'direct':
            raise ValueError("VDRM-v1 requires MODEL.BACKBONE.CAT_MODE='direct'")
        backbone = vit_large_patch16_224_ce(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                            ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                            ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
                                            vdrm_enabled=vdrm_enabled,
                                            vdrm_insert_layer=vdrm_cfg.INSERT_LAYER if vdrm_enabled else 6,
                                            vdrm_num_parts=vdrm_cfg.NUM_PARTS if vdrm_enabled else 4,
                                            vdrm_topk=vdrm_cfg.TOPK if vdrm_enabled else 4,
                                            vdrm_reliability_mode=vdrm_cfg.RELIABILITY_MODE if vdrm_enabled else "topk",
                                            vdrm_nms_radius=vdrm_cfg.NMS_RADIUS if vdrm_enabled else 1,
                                            vdrm_initial_match_scale=vdrm_cfg.INITIAL_MATCH_SCALE if vdrm_enabled else 5.0,
                                            vdrm_initial_match_bias=vdrm_cfg.INITIAL_MATCH_BIAS if vdrm_enabled else -2.5,
                                            vdrm_residual_max_ratio=vdrm_cfg.RESIDUAL_MAX_RATIO if vdrm_enabled else 0.0,
                                            vdrm_spatial_gate_mode=vdrm_cfg.SPATIAL_GATE_MODE if vdrm_enabled else "token_match",
                                            vdrm_candidate_local_radius=vdrm_cfg.CANDIDATE_LOCAL_RADIUS if vdrm_enabled else 1,
                                            vdrm_candidate_consensus_parts=vdrm_cfg.CANDIDATE_CONSENSUS_PARTS if vdrm_enabled else 2,
                                            vdrm_candidate_initial_match_scale=vdrm_cfg.CANDIDATE_INITIAL_MATCH_SCALE if vdrm_enabled else 5.0,
                                            vdrm_candidate_initial_match_bias=vdrm_cfg.CANDIDATE_INITIAL_MATCH_BIAS if vdrm_enabled else -2.5,
                                            vdrm_part_route_initial_match_scale=vdrm_cfg.PART_ROUTE_INITIAL_MATCH_SCALE if vdrm_enabled else 5.0,
                                            vdrm_part_route_initial_match_bias=vdrm_cfg.PART_ROUTE_INITIAL_MATCH_BIAS if vdrm_enabled else -2.5,
                                            vdrm_part_route_residual_floor=vdrm_cfg.PART_ROUTE_RESIDUAL_FLOOR if vdrm_enabled else 1.0,
                                            vdrm_candidate_modulation_max=vdrm_cfg.CANDIDATE_MODULATION_MAX if vdrm_enabled else 0.5,
                                            vdrm_alpha_max=vdrm_cfg.ALPHA_MAX if vdrm_enabled else 0.0,
                                            )

        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    else:
        raise NotImplementedError

    backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)

    box_head = build_box_head(cfg, hidden_dim)

    model = OSTrack(
        backbone,
        box_head,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
        vdrm_enabled=vdrm_enabled,
    )

    if 'OSTrack' in cfg.MODEL.PRETRAIN_FILE and training:
        checkpoint = torch.load(cfg.MODEL.PRETRAIN_FILE, map_location="cpu", weights_only=False)
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
        print('Load pretrained model from: ' + cfg.MODEL.PRETRAIN_FILE)

    return model
