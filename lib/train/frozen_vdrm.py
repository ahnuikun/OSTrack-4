"""Fail-closed checkpoint, mode and gradient contracts for Rfreeze/Rdisc."""

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch

from lib.config.ostrack.config import default_config, update_config_from_file

PROJECT = Path(__file__).resolve().parents[2]
BASE_CONFIG = 'vitb_256_mae_ce_vdrm_v8_tclean_s42_32x4_ep300'
PREFIX = 'backbone.vdrm.'
ARMS = ('rfreeze', 'rdisc')


def is_frozen_run(cfg):
    return cfg.TRAIN.VDRM_EXPERIMENT_ARM in ARMS


def load_base_config():
    value = default_config()
    update_config_from_file(PROJECT / 'experiments' / 'ostrack' / (BASE_CONFIG + '.yaml'), value)
    return value


def validate_frozen_config(cfg):
    arm = cfg.TRAIN.VDRM_EXPERIMENT_ARM
    if arm not in ARMS:
        raise ValueError('unknown frozen VDRM experiment arm')
    if cfg.TRAIN.VDRM_FROZEN_BASE_CONFIG != BASE_CONFIG or cfg.TRAIN.VDRM_FROZEN_BASE_EPOCH != 300:
        raise ValueError('frozen arms require the registered Tclean epoch-300 baseline')
    base = load_base_config()
    if not cfg.MODEL.VDRM.ENABLED or cfg.TRAIN.BATCH_SIZE != 32:
        raise ValueError('frozen screening requires enabled VDRM and batch size 32 per GPU')
    if cfg.MODEL.PRETRAIN_FILE:
        raise ValueError('frozen arms must not load MAE or another pretrained model')
    if cfg.TRAIN.EPOCH != 300 or cfg.TEST.EPOCH != 300:
        raise ValueError('frozen screening requires training/testing at epoch 300')
    if cfg.MODEL.BACKBONE != base.MODEL.BACKBONE or cfg.MODEL.HEAD != base.MODEL.HEAD:
        raise ValueError('frozen visual architecture must exactly match Tclean')
    for section, field in (('TEMPLATE', 'SIZE'), ('SEARCH', 'SIZE')):
        if cfg.DATA[section][field] != base.DATA[section][field]:
            raise ValueError('frozen image sizes must match Tclean')
    if cfg.DATA.MEAN != base.DATA.MEAN or cfg.DATA.STD != base.DATA.STD:
        raise ValueError('frozen normalization must match Tclean')
    for field in ('INSERT_LAYER', 'NUM_PARTS', 'TOPK', 'RELIABILITY_MODE',
                  'SPATIAL_GATE_MODE', 'ALPHA_MAX', 'RESIDUAL_MAX_RATIO'):
        if cfg.MODEL.VDRM[field] != base.MODEL.VDRM[field]:
            raise ValueError('frozen V8 residual mismatch: ' + field)
    if not cfg.MODEL.VDRM.TRAIN_ALPHA:
        raise ValueError('frozen residual must train alpha')
    if cfg.MODEL.VDRM.DISCRIMINATIVE_ROUTE != (arm == 'rdisc'):
        raise ValueError('Rfreeze/Rdisc discriminator configuration mismatch')
    weights = ('VDRM_VISIBILITY_WEIGHT', 'VDRM_RANK_WEIGHT', 'VDRM_CANDIDATE_WEIGHT', 'VDRM_PART_ROUTE_WEIGHT')
    for field in weights:
        expected = .5 if arm == 'rdisc' and field == 'VDRM_PART_ROUTE_WEIGHT' else 0.
        if cfg.TRAIN[field] != expected:
            raise ValueError(f'{arm} requires {field}={expected}')
    if cfg.TRAIN.CE_START_EPOCH != 0 or cfg.TRAIN.CE_WARM_EPOCH != 0 or not cfg.TRAIN.VDRM_FROZEN_SKIP_VAL:
        raise ValueError('frozen arms require fixed CE and no training validation-set access')
    if cfg.TRAIN.VDRM_REQUIRED_SEED != 42:
        raise ValueError('frozen screening requires config seed 42')
    if cfg.TRAIN.VDRM_AUX_WARMUP_EPOCHS != 1 or cfg.TRAIN.AMP:
        raise ValueError('frozen arms require immediate route supervision and FP32 audit/training')
    if cfg.DATA.TRAIN != base.DATA.TRAIN:
        raise ValueError('frozen arms must use the Tclean training subset, never the fixed dev split')
    if cfg.TEST.CHECKPOINT_CONFIG or cfg.TEST.VDRM_ALPHA_OVERRIDE is not None or cfg.TEST.VDRM_INFERENCE_ABLATION is not None:
        raise ValueError('frozen candidates must test their own checkpoints without inference overrides')


def visual_hash(state):
    digest = hashlib.sha256()
    for name in sorted(state):
        if name.startswith(PREFIX):
            continue
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode('utf-8'))
        digest.update((str(value.dtype) + str(tuple(value.shape))).encode('ascii'))
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_frozen_checkpoint(model, checkpoint, expected=None, cfg=None, expected_epoch=None):
    epoch = checkpoint.get('epoch')
    if checkpoint.get('net_type') != 'OSTrack' or type(epoch) is not int or not 1 <= epoch <= 300:
        raise ValueError('frozen checkpoint has an invalid network identity or epoch')
    if expected_epoch is not None and epoch != expected_epoch:
        raise ValueError(f'frozen checkpoint epoch={epoch}, expected {expected_epoch}')
    contract = checkpoint.get('frozen_vdrm_contract')
    if not isinstance(contract, dict) or contract.get('version') != 1:
        raise ValueError('frozen checkpoint is missing its verified provenance contract')
    if contract.get('base_config') != BASE_CONFIG or contract.get('base_epoch') != 300:
        raise ValueError('frozen checkpoint has the wrong Tclean source')
    if expected is not None and contract != expected:
        raise ValueError('resume checkpoint source/config contract does not match this run')
    if cfg is not None and contract.get('arm') != cfg.TRAIN.VDRM_EXPERIMENT_ARM:
        raise ValueError('frozen checkpoint arm mismatch')
    if cfg is not None and contract.get('config_sha256') != configuration_hash(cfg):
        raise ValueError('frozen checkpoint configuration differs from this YAML')
    if visual_hash(checkpoint['net']) != contract.get('visual_sha256'):
        raise ValueError('frozen visual weights or BN buffers changed in checkpoint')
    return contract


def configuration_hash(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()


def prepare_frozen_models(cfg, save_dir, config_name, checkpoint_path=None):
    """Strictly load Tclean; initialize only a fresh residual, or valid resume."""
    from lib.models.ostrack import build_ostrack
    validate_frozen_config(cfg)
    source = Path(save_dir).resolve() / 'checkpoints' / 'train' / 'ostrack' / BASE_CONFIG / 'OSTrack_ep0300.pth.tar'
    if not source.is_file():
        raise FileNotFoundError(f'Tclean source checkpoint missing: {source}')
    checkpoint = torch.load(source, map_location='cpu', weights_only=False)
    settings = checkpoint.get('settings')
    source_name = settings.get('config_name') if isinstance(settings, dict) else getattr(settings, 'config_name', None)
    if source_name != BASE_CONFIG or checkpoint.get('epoch') != 300 or checkpoint.get('net_type') != 'OSTrack':
        raise ValueError('source checkpoint is not the registered Tclean epoch-300 run')
    source_state = checkpoint['net']
    if PREFIX + 'alpha' not in source_state or source_state[PREFIX + 'alpha'].item() != 0.:
        raise ValueError('Tclean source must have exactly zero alpha')
    baseline = build_ostrack(load_base_config(), training=False)
    baseline.load_state_dict(source_state, strict=True)
    baseline.requires_grad_(False).eval()
    model = build_ostrack(cfg, training=False)
    candidate_state = model.state_dict()
    expected_visual = {n for n in candidate_state if not n.startswith(PREFIX)}
    actual_visual = {n for n in source_state if not n.startswith(PREFIX)}
    if expected_visual != actual_visual:
        raise ValueError('source/candidate visual state keys differ')
    for name in expected_visual:
        if candidate_state[name].shape != source_state[name].shape or candidate_state[name].dtype != source_state[name].dtype:
            raise ValueError('source visual tensor mismatch: ' + name)
        candidate_state[name] = source_state[name]
    model.load_state_dict(candidate_state, strict=True)
    model.frozen_visual = True
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(PREFIX))
    model.train()
    contract = dict(version=1, arm=cfg.TRAIN.VDRM_EXPERIMENT_ARM, config_name=config_name,
                    config_sha256=configuration_hash(cfg), base_config=BASE_CONFIG,
                    base_epoch=300, source_sha256=file_hash(source), visual_sha256=visual_hash(source_state))
    model.frozen_vdrm_contract = contract
    if visual_hash(model.state_dict()) != contract['visual_sha256']:
        raise ValueError('visual weights were not copied exactly')
    # Inspect a resume before DDP construction and audit its actual weights.
    if checkpoint_path is None:
        directory = Path(save_dir) / 'checkpoints' / 'train' / 'ostrack' / config_name
        paths = sorted(directory.glob('OSTrack_ep*.pth.tar'))
        checkpoint_path = paths[-1] if paths else None
    if checkpoint_path is not None:
        resumed = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        validate_frozen_checkpoint(model, resumed, expected=contract, cfg=cfg)
        model.load_state_dict(resumed['net'], strict=True)
    return model, baseline, source


@contextmanager
def preserve_random_state(device):
    python_state, numpy_state = random.getstate(), np.random.get_state()
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    try:
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def make_actor(model, cfg):
    from types import SimpleNamespace
    from torch.nn.functional import l1_loss
    from lib.train.actors import OSTrackActor
    from lib.utils.box_ops import giou_loss
    from lib.utils.focal_loss import FocalLoss
    return OSTrackActor(model,
        {'giou': giou_loss, 'l1': l1_loss, 'focal': FocalLoss()},
        {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT, 'focal': 1.},
        SimpleNamespace(batchsize=cfg.TRAIN.BATCH_SIZE, num_template=1), cfg)


def audit_frozen_model(model, baseline, cfg, data):
    """Read-only optimizer-free audit; restore RNG and any temporary alpha."""
    device = next(model.parameters()).device
    old_alpha = model.backbone.vdrm.alpha.detach().clone()
    old_mode = model.training
    before = visual_hash(model.state_dict())
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if not trainable or any(not n.startswith(PREFIX) for n, _ in trainable):
        raise RuntimeError('non-residual trainable parameter in frozen experiment')
    try:
        with preserve_random_state(device):
            model.train()
            actor = make_actor(model, cfg)
            with torch.no_grad():
                model.backbone.vdrm.alpha.zero_()
                output = actor.forward_pass(data, audit=True)
                reference = baseline.eval()(**output['frozen_audit_inputs'])
                checked = ('pred_boxes', 'score_map', 'size_map', 'offset_map', 'backbone_feat')
                for key in checked:
                    if not torch.equal(output[key], reference[key]):
                        error = (output[key] - reference[key]).abs().max().item()
                        raise RuntimeError(f'alpha=0 fails exact Tclean equivalence: {key}, max_error={error}')
                model.backbone.vdrm.alpha.copy_(old_alpha)
            output = actor.forward_pass(data)
            _, _, components = actor.compute_losses(output, data, return_components=True)
            gradients = {}
            for name in ('tracking', 'route'):
                loss = components[name]
                if not torch.isfinite(loss):
                    raise RuntimeError('non-finite audit loss: ' + name)
                grads = torch.autograd.grad(loss, [p for _, p in trainable],
                    retain_graph=True, allow_unused=True) if loss.requires_grad else [None] * len(trainable)
                groups = {'visual': 0., 'residual_scalars': 0., 'route_head': 0., 'alpha': 0.}
                for (parameter_name, _), grad in zip(trainable, grads):
                    if grad is None:
                        continue
                    if not torch.isfinite(grad).all():
                        raise RuntimeError('non-finite audit gradient: ' + parameter_name)
                    group = ('route_head' if '.route_head.' in parameter_name else
                             'alpha' if parameter_name == PREFIX + 'alpha' else 'residual_scalars')
                    groups[group] += grad.detach().double().square().sum().item()
                gradients[name] = {g: value ** .5 for g, value in groups.items()}
            if gradients['tracking']['alpha'] == 0.:
                raise RuntimeError('tracking loss has no alpha gradient; post-residual graph may be detached')
            if gradients['route']['alpha'] != 0. or gradients['route']['residual_scalars'] != 0.:
                raise RuntimeError('route supervision modifies original residual/calibration parameters')
            if cfg.MODEL.VDRM.DISCRIMINATIVE_ROUTE and gradients['route']['route_head'] == 0.:
                raise RuntimeError('route audit has no eligible supervised head gradient; inspect this minibatch')
    finally:
        with torch.no_grad():
            model.backbone.vdrm.alpha.copy_(old_alpha)
        model.train(old_mode)
    if visual_hash(model.state_dict()) != before or before != model.frozen_vdrm_contract['visual_sha256']:
        raise RuntimeError('audit changed frozen visual parameters or buffers')
    return dict(alpha0_exact=True, checked_outputs=list(checked), gradients_l2=gradients,
                visual_parameters_frozen=True, optimizer_steps=0,
                trainable_tensors=len(trainable), provenance=model.frozen_vdrm_contract)


def audit_training_loader(model, baseline, cfg, loader, report_path):
    """Before DDP/optimizer: audit one actual training minibatch, not noise."""
    device = next(model.parameters()).device
    with preserve_random_state(device):
        iterator = iter(loader)
        try:
            data = next(iterator).to(device)
            data['epoch'] = 1
            result = audit_frozen_model(model, baseline, cfg, data)
        finally:
            del iterator
    if report_path is not None:
        path = Path(report_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    return result
