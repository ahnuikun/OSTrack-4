"""Formal width-128 training contract, portable checkpoints and frozen-base loss."""
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
import os
import random

import numpy as np
import torch
import torch.nn.functional as F

from lib.config.ostrack.config import default_config, update_config_from_file
from lib.train.frozen_vdrm import configuration_hash, visual_hash
from lib.train.module1_learning import dense_boxes, learning_loss
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
from lib.utils.ce_utils import generate_mask_cond
from lib.test.utils.hann import hann2d

PROJECT = Path(__file__).resolve().parents[2]
CONFIG = 'vitb_256_mae_ce_vdrm_module1_w128_s42_ep300'
SCRATCH_CONFIG = 'vitb_256_mae_ce_vdrm_module1_w128_scratch_s42_ep300'
SOURCE = 'vitb_256_mae_ce_vdrm_rdisc_s42_32x4_ep300'
PREFIX = 'backbone.module1_adapter.'
HEAD_KEYS = ('score_map', 'size_map', 'offset_map')


def config(name=CONFIG):
    cfg = default_config()
    update_config_from_file(PROJECT / 'experiments/ostrack' / (name + '.yaml'), cfg)
    return cfg


def validate_config(cfg):
    if cfg.TRAIN.VDRM_EXPERIMENT_ARM == 'module1_scratch':
        from lib.train.module1_scratch import validate_scratch_config
        return validate_scratch_config(cfg)
    base = config(SOURCE)
    model = deepcopy(cfg.MODEL)
    adapter = model.pop('MODULE1_ADAPTER', {})
    if adapter != dict(ENABLED=True, WIDTH=128, MODE='frozen') or model != base.MODEL:
        raise ValueError('formal model must be the original Rdisc plus width-128 adapter')
    if (cfg.DATA != base.DATA or cfg.TEST != base.TEST or cfg.TRAIN.VDRM_EXPERIMENT_ARM != 'module1'
            or cfg.TRAIN.AMP or cfg.TRAIN.EPOCH != 300 or cfg.TRAIN.VDRM_REQUIRED_SEED != 42
            or cfg.TRAIN.LR != .001 or cfg.TRAIN.GRAD_CLIP_NORM != 1.
            or cfg.TRAIN.VDRM_PART_ROUTE_WEIGHT != 0.):
        raise ValueError('formal protocol changed: four original datasets, FP32, seed 42, fixed epoch 300')


def state_hash(state):
    # visual_hash normally excludes VDRM. Prefix ALL keys to include q/route/alpha.
    return visual_hash({'all.' + k: v for k, v in state.items()})


def frozen_hash(model):
    return state_hash({k: v for k, v in model.state_dict().items() if not k.startswith(PREFIX)})


def excluded_got_sequences():
    from lib.test.analysis.module1_learning import sequence_partition
    from lib.test.analysis.module1_reweight import confirmation_ids
    split, forbidden = sequence_partition(False)
    return sorted(set(forbidden) | set(split['audit']) | set(confirmation_ids(False)))


@contextmanager
def original_rdisc(model):
    previous = model.backbone.module1_adapter_disabled
    model.backbone.module1_adapter_disabled = True
    try:
        yield
    finally:
        model.backbone.module1_adapter_disabled = previous


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def backend():
    # Fixed data order alone does not reproduce scatter/convolution gradients.
    # Set before the first CUDA matrix operation (including source loading).
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])


def checkpoint(model, cfg, epoch, optimizer, scheduler, contract, run, initial, updates):
    if cfg.MODEL.MODULE1_ADAPTER.MODE == 'frozen' and frozen_hash(model) != contract['source_state_sha256']:
        raise RuntimeError('frozen Rdisc weights or buffers changed')
    net = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return dict(net_type='OSTrack', net=net, epoch=epoch, optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(), rng=rng_state(), module1_contract=contract,
                configuration=deepcopy(cfg), run=run, initial_adapter=initial, updates=updates,
                state_sha256=state_hash(net), initial_sha256=state_hash(initial))


def validate_checkpoint(payload, cfg, expected_epoch=None, allow_smoke=False):
    validate_config(cfg)
    if cfg.MODEL.MODULE1_ADAPTER.MODE == 'joint':
        from lib.train.module1_scratch import validate_scratch_checkpoint
        return validate_scratch_checkpoint(payload, cfg, expected_epoch, allow_smoke)
    contract = payload.get('module1_contract', {})
    epoch = payload.get('epoch')
    if (payload.get('net_type') != 'OSTrack' or type(epoch) is not int or not 1 <= epoch <= 300
            or (expected_epoch is not None and epoch != expected_epoch)
            or contract.get('version') != 1 or contract.get('source_config') != SOURCE
            or contract.get('config_sha256') != configuration_hash(cfg)
            or contract.get('excluded_got_ids') != excluded_got_sequences()
            or (payload.get('run', {}).get('smoke', True) and not allow_smoke)):
        raise ValueError('wrong or incomplete formal checkpoint/protocol; smoke is not a trained model')
    net = payload['net']
    if (state_hash(net) != payload.get('state_sha256') or
            state_hash({k: v for k, v in net.items() if not k.startswith(PREFIX)}) != contract.get('source_state_sha256')):
        raise ValueError('checkpoint weights corrupted or frozen Rdisc changed')
    initial = payload['initial_adapter']
    if (state_hash(initial) != payload.get('initial_sha256') or
            set(initial) != {'search.weight', 'template.weight', 'value.weight'} or
            initial['value.weight'].count_nonzero() or not all(torch.isfinite(v).all() for v in initial.values())):
        raise ValueError('invalid saved initial adapter; seed reinitialization is not a portable verification')
    return contract


def atomic_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def overlap(boxes, target):
    a, b = box_cxcywh_to_xyxy(boxes), box_xywh_to_xyxy(target)
    intersection = (torch.minimum(a[..., 2:], b[..., 2:]) -
                    torch.maximum(a[..., :2], b[..., :2])).clamp_min(0).prod(-1)
    area = (a[..., 2:] - a[..., :2]).clamp_min(0).prod(-1)
    target_area = (b[..., 2:] - b[..., :2]).clamp_min(0).prod(-1)
    return intersection / (area + target_area - intersection).clamp_min(1e-12)


def training_loss(model, cfg, data, device, augment=True, check_identity=False):
    """Use frozen Rdisc on identical augmented inputs to label supported boxes.

    Labels concern box geometry, not instance identity. No labels enter the
    adapter. Each image has its own norm budget, good cells and preservation.
    """
    template = data['template_images'][0].to(device)
    search = data['search_images'][0].to(device)
    template_box = data['template_anno'][0].to(device)
    target = data['search_anno'][0].to(device)
    batch = len(search)
    applied = torch.zeros(batch, dtype=torch.bool, device=device)
    if augment:
        from lib.train.data.vdrm_augmentation import apply_structured_target_occlusion
        search, _, applied = apply_structured_target_occlusion(search, target,
            cfg.DATA.SEARCH.VDRM_OCCLUSION_PROBABILITY, cfg.DATA.SEARCH.VDRM_OCCLUSION_MIN_RATIO,
            cfg.DATA.SEARCH.VDRM_OCCLUSION_MAX_RATIO)
    inputs = dict(template=template, search=search, template_bbox=template_box,
                  ce_template_mask=generate_mask_cond(cfg, batch, device, template_box))
    side = cfg.DATA.SEARCH.SIZE // cfg.MODEL.BACKBONE.STRIDE
    hann = hann2d(torch.tensor([side, side]).long(), centered=True).to(device)
    with torch.no_grad(), original_rdisc(model):
        base = model(**inputs)
        retained = torch.ones(batch, side * side, dtype=torch.bool, device=device)
        for indices in base['removed_indexes_s']:
            if indices is not None:
                retained.scatter_(1, indices.long(), False)
        padding = F.max_pool2d(data['search_att'][0].to(device).float()[:, None],
                              cfg.MODEL.BACKBONE.STRIDE).flatten(1).bool()
        iou = overlap(dense_boxes(base['size_map'], base['offset_map']), target[:, None])
        good = (iou >= .5) & retained & ~padding
        peak = (base['score_map'] * hann).flatten(1).argmax(1)
        protect = iou.gather(1, peak[:, None])[:, 0] >= .5
        base = {k: base[k].detach() for k in HEAD_KEYS}
    out = model(**inputs)
    if check_identity:
        for key in HEAD_KEYS:
            if not torch.equal(out[key], base[key]):
                raise RuntimeError('zero adapter changed native Rdisc head: ' + key)
    ratio = out['module1_norm_ratio'].detach()
    if not torch.isfinite(ratio).all() or float(ratio.max()) > 1.005:
        raise RuntimeError('adapter exceeded per-image original residual budget')
    losses, records = [], []
    for i in range(batch):
        loss, metrics = learning_loss({k: out[k][i:i+1] for k in HEAD_KEYS},
            {k: base[k][i:i+1] for k in HEAD_KEYS}, good[i], target[i], hann, bool(protect[i]))
        metrics.update(protected=int(protect[i]), supported=int(good[i].any()),
                       supported_failure=int(good[i].any() and not protect[i]),
                       occlusion_augmented=int(applied[i]), norm_ratio=float(ratio[i]))
        losses.append(loss)
        records.append(metrics)
    return torch.stack(losses).mean(), records
