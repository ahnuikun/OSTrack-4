"""End-to-end MAE-initialized training of the same width-128 architecture."""
from copy import deepcopy
from types import SimpleNamespace

import torch

from lib.train.module1_formal import (PROJECT, SOURCE, PREFIX, config, state_hash, seed_all,
                                    excluded_got_sequences)
from lib.train.frozen_vdrm import configuration_hash, file_hash


def validate_scratch_config(cfg):
    base = config(SOURCE)
    model = deepcopy(cfg.MODEL)
    adapter = model.pop('MODULE1_ADAPTER', {})
    model.PRETRAIN_FILE = ''
    if (adapter != dict(ENABLED=True, WIDTH=128, MODE='joint') or model != base.MODEL
            or cfg.MODEL.PRETRAIN_FILE != 'mae_pretrain_vit_base.pth'
            or cfg.DATA != base.DATA or cfg.TEST != base.TEST
            or cfg.TRAIN.VDRM_EXPERIMENT_ARM != 'module1_scratch'
            or cfg.TRAIN.VDRM_FROZEN_BASE_CONFIG or cfg.TRAIN.AMP
            or cfg.TRAIN.EPOCH != 300 or cfg.TRAIN.VDRM_REQUIRED_SEED != 42
            or cfg.TRAIN.LR != .0004 or cfg.TRAIN.GRAD_CLIP_NORM != .1
            or cfg.TRAIN.CE_START_EPOCH != 20 or cfg.TRAIN.CE_WARM_EPOCH != 80
            or cfg.TRAIN.VDRM_PART_ROUTE_WEIGHT != .5):
        raise ValueError('scratch requires MAE initialization and the registered joint-training protocol')


def initialize_scratch(cfg, save_dir, device):
    from lib.models.ostrack import build_ostrack
    validate_scratch_config(cfg)
    mae_path = PROJECT / 'pretrained_models' / cfg.MODEL.PRETRAIN_FILE
    if not mae_path.is_file():
        raise FileNotFoundError('MAE initialization missing: ' + str(mae_path))
    seed_all(42)
    model = build_ostrack(cfg, training=True)
    mae = torch.load(mae_path, map_location='cpu', weights_only=False)['model']
    matched = 0
    for name, value in model.backbone.state_dict().items():
        if name.startswith(('blocks.', 'patch_embed.', 'norm.')):
            if name not in mae or not torch.equal(value, mae[name]):
                raise ValueError('MAE backbone weights were not actually loaded: ' + name)
            matched += 1
    if matched < 140:
        raise ValueError('incomplete ViT-B MAE initialization')
    initial = deepcopy(model.backbone.module1_adapter.state_dict())
    contract = dict(version=1, source_config='mae_pretrain_vit_base', mode='joint',
        config_sha256=configuration_hash(cfg), source_checkpoint_sha256=file_hash(mae_path),
        source_state_sha256=state_hash(model.state_dict()), excluded_got_ids=excluded_got_sequences(),
        pretrained_backbone_tensors_verified=matched)
    return model.to(device).train(), contract, initial


def optimizer_scratch(model, cfg):
    new, pretrained = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith('backbone.') and not name.startswith(('backbone.vdrm.', PREFIX)):
            pretrained.append(parameter)
        else:
            new.append(parameter)
    if not new or not pretrained or getattr(model, 'frozen_module1', False):
        raise ValueError('scratch must train the backbone, head, VDRM and adapter')
    optimizer = torch.optim.AdamW([dict(params=new),
        dict(params=pretrained, lr=cfg.TRAIN.LR * cfg.TRAIN.BACKBONE_MULTIPLIER)],
        lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, cfg.TRAIN.LR_DROP_EPOCH,
                                               gamma=cfg.TRAIN.SCHEDULER.DECAY_RATE)
    return optimizer, scheduler


def scratch_loss(model, cfg, data, device, epoch):
    from lib.train.actors import OSTrackActor
    from lib.utils.box_ops import giou_loss
    from lib.utils.focal_loss import FocalLoss
    from lib.utils import TensorDict
    # Use the existing standard OSTrack losses and original augmentations/CE
    # warmup. Frozen-teacher geometry labels are inappropriate for a new head.
    actor = OSTrackActor(model, dict(giou=giou_loss, l1=torch.nn.functional.l1_loss,
        focal=FocalLoss(), cls=torch.nn.BCEWithLogitsLoss()),
        dict(giou=2., l1=5., focal=1., cls=1.),
        SimpleNamespace(batchsize=data['search_images'].shape[1], num_template=1), cfg)
    batch = TensorDict({k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in data.items()})
    batch['epoch'] = epoch
    out = actor.forward_pass(batch)
    ratio = out['module1_norm_ratio'].detach()
    if not torch.isfinite(ratio).all() or float(ratio.max()) > 1.005:
        raise RuntimeError('scratch residual exceeded its per-image budget')
    loss, status = actor.compute_losses(out, batch)
    if not torch.isfinite(loss):
        raise RuntimeError('nonfinite scratch loss')
    record = dict(total=float(loss.detach()), giou=status['Loss/giou'], l1=status['Loss/l1'],
                  focal=status['Loss/location'], route=status['Loss/vdrm_part_route'],
                  norm_ratio=float(ratio.max()))
    return loss, [record.copy() for _ in range(data['search_images'].shape[1])]


def validate_scratch_checkpoint(payload, cfg, expected_epoch, allow_smoke):
    contract = payload.get('module1_contract', {})
    epoch = payload.get('epoch')
    if (payload.get('net_type') != 'OSTrack' or type(epoch) is not int or not 1 <= epoch <= 300
            or (expected_epoch is not None and epoch != expected_epoch)
            or contract.get('version') != 1 or contract.get('mode') != 'joint'
            or contract.get('source_config') != 'mae_pretrain_vit_base'
            or contract.get('config_sha256') != configuration_hash(cfg)
            or contract.get('excluded_got_ids') != excluded_got_sequences()
            or (payload.get('run', {}).get('smoke', True) and not allow_smoke)):
        raise ValueError('wrong scratch checkpoint, configuration or epoch')
    if state_hash(payload['net']) != payload.get('state_sha256'):
        raise ValueError('scratch checkpoint weights corrupted')
    initial = payload['initial_adapter']
    if (state_hash(initial) != payload.get('initial_sha256')
            or set(initial) != {'search.weight', 'template.weight', 'value.weight'}
            or initial['value.weight'].count_nonzero()
            or not all(torch.isfinite(v).all() for v in initial.values())):
        raise ValueError('invalid saved scratch adapter initialization')
    return contract
