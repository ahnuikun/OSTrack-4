"""Dataset-free single-device smoke test for VDRM-OSTrack.

This script deliberately bypasses the project dataloader and existing path
configuration. It is intended for a local RTX 5070 (or CPU fallback) only and
does not replace the server-side DDP/data smoke test.
"""

from __future__ import annotations

import argparse
import copy
import contextlib
import io
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

import torch
from torch.nn import BCEWithLogitsLoss
from torch.nn.functional import l1_loss

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.config.ostrack.config import cfg as base_cfg
from lib.config.ostrack.config import update_config_from_file
from lib.models.ostrack import build_ostrack
from lib.train.actors import OSTrackActor
from lib.train.base_functions import get_optimizer_scheduler
from lib.utils.box_ops import giou_loss
from lib.utils.focal_loss import FocalLoss


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="experiments/ostrack/vitb_256_mae_ce_vdrm_32x4_ep300.yaml",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(0)
        torch.cuda.reset_peak_memory_stats()

    cfg = copy.deepcopy(base_cfg)
    update_config_from_file(args.config, base_cfg=cfg)
    # The smoke test validates wiring, not MAE checkpoint loading.
    cfg.MODEL.PRETRAIN_FILE = ""
    # Ensure visibility targets exist whenever an enabled loss path needs
    # them. Structure-only ablations keep their registered clean inputs.
    route_uses_visibility = bool(
        cfg.TRAIN.VDRM_PART_ROUTE_WEIGHT > 0.0
        and cfg.TRAIN.VDRM_ROUTE_VISIBILITY_WEIGHTED
    )
    if cfg.TRAIN.VDRM_VISIBILITY_WEIGHT > 0.0 or route_uses_visibility:
        cfg.DATA.SEARCH.VDRM_OCCLUSION_PROBABILITY = 1.0

    model = build_ostrack(cfg, training=False).to(device)
    model.train()
    settings = SimpleNamespace(batchsize=1, num_template=1)
    objective = {
        "giou": giou_loss,
        "l1": l1_loss,
        "focal": FocalLoss(),
        "cls": BCEWithLogitsLoss(),
    }
    loss_weight = {
        "giou": cfg.TRAIN.GIOU_WEIGHT,
        "l1": cfg.TRAIN.L1_WEIGHT,
        "focal": 1.0,
        "cls": 1.0,
    }
    actor = OSTrackActor(
        net=model,
        objective=objective,
        loss_weight=loss_weight,
        settings=settings,
        cfg=cfg,
    )

    data = {
        "template_images": torch.randn(
            1, 1, 3, cfg.DATA.TEMPLATE.SIZE, cfg.DATA.TEMPLATE.SIZE,
            device=device,
        ),
        "search_images": torch.randn(
            1, 1, 3, cfg.DATA.SEARCH.SIZE, cfg.DATA.SEARCH.SIZE,
            device=device,
        ),
        "template_anno": torch.tensor(
            [[[0.25, 0.25, 0.50, 0.50]]], device=device
        ),
        "search_anno": torch.tensor(
            [[[0.35, 0.35, 0.25, 0.25]]], device=device
        ),
        # Epoch 10 activates both auxiliary losses at a non-zero warm-up weight.
        "epoch": 10,
    }
    if bool(getattr(cfg.TRAIN, "VDRM_ALIGN_DISTRACTOR_RANK", False)) or bool(
        getattr(cfg.TRAIN, "VDRM_LOG_DISTRACTOR_HARDNESS", False)
    ):
        # Exercise V5/V6 metadata paths without requiring local datasets. The
        # synthetic distractor box is disjoint from the labelled target.
        data["vdrm_distractor_applied"] = torch.ones(
            1, 1, device=device
        )
        data["vdrm_distractor_box"] = torch.tensor(
            [[[0.05, 0.05, 0.20, 0.20]]], device=device
        )

    loss, status = actor(data)
    if not torch.isfinite(loss):
        raise RuntimeError(f"non-finite smoke loss: {loss}")
    loss.backward()

    vdrm = model.backbone.vdrm
    alpha_grad = vdrm.alpha.grad
    if alpha_grad is None or not torch.isfinite(alpha_grad).all():
        raise RuntimeError(f"invalid VDRM alpha gradient: {alpha_grad}")
    if not torch.count_nonzero(alpha_grad):
        raise RuntimeError("zero VDRM alpha gradient")

    with contextlib.redirect_stdout(io.StringIO()):
        optimizer, _ = get_optimizer_scheduler(model, cfg)
    vdrm_parameter_ids = {id(parameter) for parameter in vdrm.parameters()}
    vdrm_lrs = {
        group["lr"]
        for group in optimizer.param_groups
        if any(id(parameter) in vdrm_parameter_ids for parameter in group["params"])
    }
    if vdrm_lrs != {cfg.TRAIN.LR}:
        raise RuntimeError(
            f"VDRM must use LR {cfg.TRAIN.LR}, got {sorted(vdrm_lrs)}"
        )
    alpha_before_step = vdrm.alpha.detach().clone()
    optimizer.step()
    if torch.equal(vdrm.alpha.detach(), alpha_before_step):
        raise RuntimeError("VDRM alpha did not update after one optimizer step")

    # With alpha initialized to zero, structure-only ablations legitimately
    # give the residual branch zero gradient on the first step. Once alpha has
    # moved, every VDRM parameter must receive tracking-loss gradients.
    optimizer.zero_grad(set_to_none=True)
    second_loss, status = actor(data)
    if not torch.isfinite(second_loss):
        raise RuntimeError(f"non-finite second smoke loss: {second_loss}")
    second_loss.backward()
    for name, parameter in vdrm.named_parameters():
        if not parameter.requires_grad:
            continue
        grad = parameter.grad
        if grad is None or not torch.isfinite(grad).all():
            raise RuntimeError(f"invalid VDRM gradient for {name}: {grad}")
        if not torch.count_nonzero(grad):
            raise RuntimeError(f"zero VDRM gradient for {name}")

    with tempfile.TemporaryDirectory(prefix="ostrack_vdrm_smoke_") as tmp_dir:
        checkpoint_path = os.path.join(tmp_dir, "vdrm_only.pth")
        torch.save(vdrm.state_dict(), checkpoint_path)
        restored = copy.deepcopy(vdrm)
        restored.load_state_dict(
            torch.load(checkpoint_path, map_location=device, weights_only=True)
        )

    print("VDRM smoke test passed")
    print(f"device={device}")
    for key in sorted(status):
        print(f"{key}={status[key]}")
    print(f"vdrm_lr={next(iter(vdrm_lrs))}")
    print(f"vdrm_residual_max_ratio={vdrm.residual_max_ratio}")
    print(f"vdrm_alpha_after_step={vdrm.alpha.detach().item()}")
    if device.type == "cuda":
        peak_memory_gib = torch.cuda.max_memory_allocated() / (1024 ** 3)
        print(f"peak_cuda_memory_gib={peak_memory_gib:.3f}")


if __name__ == "__main__":
    main()
