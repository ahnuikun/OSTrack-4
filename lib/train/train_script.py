import os
# loss function related
from lib.utils.box_ops import giou_loss
from torch.nn.functional import l1_loss
from torch.nn import BCEWithLogitsLoss
# train pipeline related
from lib.train.trainers import LTRTrainer
# distributed training related
from torch.nn.parallel import DistributedDataParallel as DDP
# some more advanced functions
from .base_functions import *
# network related
from lib.models.ostrack import build_ostrack
# forward propagation related
from lib.train.actors import OSTrackActor
# for import modules
import importlib

from ..utils.focal_loss import FocalLoss
from .frozen_vdrm import is_frozen_run, prepare_frozen_models, audit_training_loader


def run(settings):
    settings.description = 'Training script for STARK-S, STARK-ST stage1, and STARK-ST stage2'

    # update the default configs with config file
    if not os.path.exists(settings.cfg_file):
        raise ValueError("%s doesn't exist." % settings.cfg_file)
    config_module = importlib.import_module("lib.config.%s.config" % settings.script_name)
    cfg = config_module.default_config()
    config_module.update_config_from_file(settings.cfg_file, cfg)
    validate_vdrm_experiment_contract(
        cfg, actual_seed=getattr(settings, "seed", None)
    )
    if settings.local_rank in [-1, 0]:
        print("New configuration is shown below.")
        for key in cfg.keys():
            print("%s configuration:" % key, cfg[key])
            print('\n')

    # update settings based on cfg
    update_settings(settings, cfg)

    # Record the training log
    log_dir = os.path.join(settings.save_dir, 'logs')
    if settings.local_rank in [-1, 0]:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    settings.log_file = os.path.join(log_dir, "%s-%s.log" % (settings.script_name, settings.config_name))

    frozen = is_frozen_run(cfg)
    if frozen:
        import torch.distributed as dist
        if settings.local_rank == -1 or dist.get_world_size() != 4:
            raise ValueError('Rfreeze/Rdisc formal training requires --mode multiple --nproc_per_node 4')
        net, frozen_baseline, source_path = prepare_frozen_models(
            cfg, settings.save_dir, settings.config_name,
        )
        if settings.local_rank == 0:
            print('Strict frozen Tclean source:', source_path)

    # Build dataloaders
    loader_train, loader_val = build_dataloaders(cfg, settings)

    if "RepVGG" in cfg.MODEL.BACKBONE.TYPE or "swin" in cfg.MODEL.BACKBONE.TYPE or "LightTrack" in cfg.MODEL.BACKBONE.TYPE:
        cfg.ckpt_dir = settings.save_dir

    # Create network
    if settings.script_name == "ostrack" and not frozen:
        net = build_ostrack(cfg)
    elif settings.script_name != 'ostrack':
        raise ValueError("illegal script name")

    # wrap networks to distributed one
    net.cuda()
    if frozen:
        frozen_baseline.cuda()
        report_path = os.path.join(log_dir, settings.config_name + '-frozen-preflight.json') if settings.local_rank == 0 else None
        result = audit_training_loader(net, frozen_baseline, cfg, loader_train, report_path)
        if settings.local_rank == 0:
            print('FROZEN PREFLIGHT PASSED (actual training minibatch, no optimizer step):', result)
        del frozen_baseline
        torch.cuda.empty_cache()
    if settings.local_rank != -1:
        # net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)  # add syncBN converter
        net = DDP(net, device_ids=[settings.local_rank], find_unused_parameters=True)
        settings.device = torch.device("cuda:%d" % settings.local_rank)
    else:
        settings.device = torch.device("cuda:0")
    settings.deep_sup = getattr(cfg.TRAIN, "DEEP_SUPERVISION", False)
    settings.distill = getattr(cfg.TRAIN, "DISTILL", False)
    settings.distill_loss_type = getattr(cfg.TRAIN, "DISTILL_LOSS_TYPE", "KL")
    # Loss functions and Actors
    if settings.script_name == "ostrack":
        focal_loss = FocalLoss()
        objective = {'giou': giou_loss, 'l1': l1_loss, 'focal': focal_loss, 'cls': BCEWithLogitsLoss()}
        loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT, 'focal': 1., 'cls': 1.0}
        actor = OSTrackActor(net=net, objective=objective, loss_weight=loss_weight, settings=settings, cfg=cfg)
    else:
        raise ValueError("illegal script name")

    # if cfg.TRAIN.DEEP_SUPERVISION:
    #     raise ValueError("Deep supervision is not supported now.")

    # Optimizer, parameters, and learning rates
    optimizer, lr_scheduler = get_optimizer_scheduler(net, cfg)
    use_amp = getattr(cfg.TRAIN, "AMP", False)
    loaders = [loader_train] + ([loader_val] if loader_val is not None else [])
    trainer = LTRTrainer(actor, loaders, optimizer, settings, lr_scheduler, use_amp=use_amp)

    # train process
    trainer.train(cfg.TRAIN.EPOCH, load_latest=True, fail_safe=not frozen)
