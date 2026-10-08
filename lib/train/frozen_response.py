"""Rresp provenance, full-baseline freezing and optimizer-free gradient audit."""

import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch

from .frozen_vdrm import BASE_CONFIG, load_base_config, file_hash, configuration_hash, preserve_random_state

PREFIX = 'response_residual.'


def is_response_run(cfg):
    return cfg.TRAIN.VDRM_EXPERIMENT_ARM in ('rresp', 'rqual')


def validate_response_config(cfg):
    base = load_base_config()
    quality = cfg.TRAIN.VDRM_EXPERIMENT_ARM == 'rqual'
    epochs = 20 if quality else 300
    if not is_response_run(cfg) or not getattr(cfg.MODEL, 'RESPONSE_RESIDUAL', {}).get('ENABLED'):
        raise ValueError('Rresp requires its named arm and enabled response head')
    if cfg.MODEL.VDRM != base.MODEL.VDRM or cfg.MODEL.BACKBONE != base.MODEL.BACKBONE or cfg.MODEL.HEAD != base.MODEL.HEAD:
        raise ValueError('Rresp must preserve the entire Tclean visual/VDRM architecture (alpha frozen at zero)')
    if cfg.TRAIN.VDRM_FROZEN_BASE_CONFIG != BASE_CONFIG or cfg.TRAIN.VDRM_FROZEN_BASE_EPOCH != 300:
        raise ValueError('Rresp requires the registered Tclean epoch-300 source')
    if cfg.MODEL.PRETRAIN_FILE or cfg.TRAIN.AMP or not cfg.TRAIN.VDRM_FROZEN_SKIP_VAL:
        raise ValueError('Rresp requires strict source loading, FP32 and no validation-set access')
    if cfg.TRAIN.BATCH_SIZE != 32 or cfg.TRAIN.EPOCH != epochs or cfg.TEST.EPOCH != epochs or cfg.TRAIN.VDRM_REQUIRED_SEED != 42:
        raise ValueError(f'response screening requires 32 per GPU, epoch {epochs} and configuration seed 42')
    if cfg.TRAIN.CE_START_EPOCH != 0 or cfg.TRAIN.CE_WARM_EPOCH != 0 or cfg.DATA.TRAIN != base.DATA.TRAIN:
        raise ValueError('Rresp requires fixed CE and the registered training subset')
    for field in ('GIOU_WEIGHT', 'L1_WEIGHT', 'VDRM_VISIBILITY_WEIGHT', 'VDRM_RANK_WEIGHT',
                  'VDRM_CANDIDATE_WEIGHT', 'VDRM_PART_ROUTE_WEIGHT'):
        if cfg.TRAIN[field] != 0:
            raise ValueError('Rresp must not train regression or legacy auxiliary losses: ' + field)
    for section in ('TEMPLATE', 'SEARCH'):
        if cfg.DATA[section].SIZE != base.DATA[section].SIZE or cfg.DATA[section].FACTOR != base.DATA[section].FACTOR:
            raise ValueError('Rresp image geometry must match Tclean')
    expected_test = dict(base.TEST, EPOCH=epochs)
    if cfg.DATA.MEAN != base.DATA.MEAN or cfg.DATA.STD != base.DATA.STD or cfg.TEST != expected_test:
        raise ValueError('Rresp test geometry/overrides and normalization must match Tclean')
    model_cfg, loss_cfg = cfg.MODEL.RESPONSE_RESIDUAL, cfg.TRAIN.RESPONSE_RESIDUAL
    if quality:
        if model_cfg.TYPE != 'candidate_quality' or model_cfg.ROI_GRID_SIZE != 3 or cfg.TRAIN.LR_DROP_EPOCH != 16:
            raise ValueError('Rqual requires the registered 3x3 candidate head and 20-epoch schedule')
        expected_loss = dict(QUALITY_WEIGHT=1., RANK_WEIGHT=.5, PRESERVE_WEIGHT=.5, ABSTAIN_WEIGHT=.1,
            POSITIVE_IOU=.5, NEGATIVE_IOU=.1, PROTECT_IOU_GAP=.05, RANK_MARGIN=.2, HARD_NEGATIVE_TOPK=4)
        if dict(loss_cfg) != expected_loss or model_cfg.MAX_DELTA_LOGIT != 2.:
            raise ValueError('Rqual screening loss/bound differs from the preregistered recipe')
    elif getattr(model_cfg, 'TYPE', 'logit') != 'logit':
        raise ValueError('Rresp must retain its original logit head')
    for field in ('PROJECTION_DIM', 'HIDDEN_DIM'):
        if type(model_cfg[field]) is not int or model_cfg[field] <= 0:
            raise ValueError('response dimensions must be positive integers')
    for value in (model_cfg.MAX_DELTA_LOGIT, loss_cfg.QUALITY_WEIGHT if quality else loss_cfg.FOCAL_WEIGHT, loss_cfg.RANK_WEIGHT):
        if not math.isfinite(value) or value <= 0:
            raise ValueError('response bound and both loss weights must be finite and positive')
    if type(loss_cfg.HARD_NEGATIVE_TOPK) is not int or loss_cfg.HARD_NEGATIVE_TOPK < 1:
        raise ValueError('response hard-negative top-k must be a positive integer')
    if any(not math.isfinite(v) or v < 0 for v in (loss_cfg.RANK_MARGIN, 0 if quality else loss_cfg.NEGATIVE_GUARD_CELLS)):
        raise ValueError('invalid response ranking margin/guard')


def baseline_hash(state, exclude_response=True):
    """Unlike the old visual hash, this ALSO protects all original VDRM tensors."""
    digest = hashlib.sha256()
    for name in sorted(state):
        if exclude_response and name.startswith(PREFIX):
            continue
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode('utf-8'))
        digest.update((str(value.dtype) + str(tuple(value.shape))).encode('ascii'))
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def validate_response_checkpoint(model, checkpoint, cfg=None, expected=None, expected_epoch=None):
    epoch = checkpoint.get('epoch')
    contract = checkpoint.get('frozen_response_contract')
    if checkpoint.get('net_type') != 'OSTrack' or type(epoch) is not int or not 1 <= epoch <= 300:
        raise ValueError('invalid Rresp checkpoint identity/epoch')
    if expected_epoch is not None and epoch != expected_epoch:
        raise ValueError('Rresp checkpoint is not the requested epoch')
    if not isinstance(contract, dict) or contract.get('version') != 1 or contract.get('arm') not in ('rresp', 'rqual'):
        raise ValueError('missing Rresp provenance contract')
    if contract['arm'] == 'rqual' and (contract.get('train_epochs') != 20 or epoch > 20):
        raise ValueError('Rqual checkpoint exceeds the registered 20-epoch screen')
    if cfg is not None and contract['arm'] != cfg.TRAIN.VDRM_EXPERIMENT_ARM:
        raise ValueError('response checkpoint arm mismatch')
    if contract.get('base_config') != BASE_CONFIG or contract.get('base_epoch') != 300:
        raise ValueError('wrong Rresp Tclean source')
    if expected is not None and contract != expected:
        raise ValueError('Rresp resume source/config differs from this run')
    if cfg is not None and contract.get('config_sha256') != configuration_hash(cfg):
        raise ValueError('Rresp checkpoint configuration differs from YAML')
    model_state = model.state_dict()
    if set(checkpoint['net']) != set(model_state):
        raise ValueError('Rresp checkpoint parameter keys differ')
    for name, value in checkpoint['net'].items():
        target = model_state[name]
        if value.shape != target.shape or value.dtype != target.dtype:
            raise ValueError('Rresp checkpoint tensor shape/dtype differs: ' + name)
        if name.startswith(PREFIX) and not torch.isfinite(value).all():
            raise ValueError('non-finite Rresp checkpoint response parameter: ' + name)
    if baseline_hash(checkpoint['net']) != contract.get('baseline_sha256'):
        raise ValueError('Rresp modified frozen baseline weights/buffers')
    if checkpoint['net']['backbone.vdrm.alpha'].item() != 0:
        raise ValueError('Rresp original alpha must remain exactly zero')
    return contract


def prepare_response_models(cfg, save_dir, config_name, checkpoint_path=None):
    from lib.models.ostrack import build_ostrack
    validate_response_config(cfg)
    source = Path(save_dir).resolve() / 'checkpoints/train/ostrack' / BASE_CONFIG / 'OSTrack_ep0300.pth.tar'
    if not source.is_file():
        raise FileNotFoundError('Tclean source checkpoint missing: ' + str(source))
    checkpoint = torch.load(source, map_location='cpu', weights_only=False)
    settings = checkpoint.get('settings')
    name = settings.get('config_name') if isinstance(settings, dict) else getattr(settings, 'config_name', None)
    if name != BASE_CONFIG or checkpoint.get('epoch') != 300 or checkpoint.get('net_type') != 'OSTrack':
        raise ValueError('source is not the registered Tclean epoch-300 checkpoint')
    state = checkpoint['net']
    if state['backbone.vdrm.alpha'].item() != 0:
        raise ValueError('Tclean source alpha is not exactly zero')
    baseline = build_ostrack(load_base_config(), training=False)
    baseline.load_state_dict(state, strict=True)
    baseline.requires_grad_(False).eval()
    model = build_ostrack(cfg, training=False)
    candidate = model.state_dict()
    if {key for key in candidate if not key.startswith(PREFIX)} != set(state):
        raise ValueError('Rresp/Tclean state keys differ')
    for key, value in state.items():
        if candidate[key].shape != value.shape or candidate[key].dtype != value.dtype:
            raise ValueError('Rresp/Tclean tensor mismatch: ' + key)
        candidate[key] = value
    model.load_state_dict(candidate, strict=True)
    contract = dict(version=1, arm=cfg.TRAIN.VDRM_EXPERIMENT_ARM, config_name=config_name,
        config_sha256=configuration_hash(cfg), base_config=BASE_CONFIG, base_epoch=300,
        source_sha256=file_hash(source), baseline_sha256=baseline_hash(state))
    if cfg.TRAIN.VDRM_EXPERIMENT_ARM == 'rqual':
        contract['train_epochs'] = cfg.TRAIN.EPOCH
    model.frozen_response_contract = contract
    if checkpoint_path is None:
        directory = Path(save_dir) / 'checkpoints/train/ostrack' / config_name
        paths = sorted(directory.glob('OSTrack_ep*.pth.tar'))
        checkpoint_path = paths[-1] if paths else None
    if checkpoint_path is not None:
        resumed = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        validate_response_checkpoint(model, resumed, cfg, contract)
        model.load_state_dict(resumed['net'], strict=True)
    if baseline_hash(model.state_dict()) != contract['baseline_sha256']:
        raise ValueError('Rresp failed exact source copying')
    model.train()
    return model, baseline, source


def make_response_actor(model, cfg):
    from lib.train.actors.response_residual import ResponseResidualActor
    actor = ResponseResidualActor
    if cfg.TRAIN.VDRM_EXPERIMENT_ARM == 'rqual':
        from lib.train.actors.candidate_quality import CandidateQualityActor
        actor = CandidateQualityActor
    return actor(model, {}, {},
        SimpleNamespace(batchsize=cfg.TRAIN.BATCH_SIZE, num_template=1), cfg)


def assert_response_optimizer(model, optimizer):
    model = model.module if hasattr(model, 'module') else model
    actual = {id(p) for group in optimizer.param_groups for p in group['params']}
    expected = {id(p) for n, p in model.named_parameters() if n.startswith(PREFIX)}
    if actual != expected or not expected:
        raise RuntimeError('Rresp optimizer must contain exactly the response-head parameters')


def audit_response_model(model, baseline, cfg, data):
    device = next(model.parameters()).device
    before = baseline_hash(model.state_dict(), exclude_response=False)
    mode, disabled = model.training, model.response_residual_disabled
    flags = {name: p.requires_grad for name, p in model.named_parameters()}
    if any(value != name.startswith(PREFIX) for name, value in flags.items()):
        raise RuntimeError('Rresp has incorrectly frozen/trainable parameters')
    result = {'optimizer_steps': 0, 'baseline_sha256': baseline_hash(model.state_dict())}
    try:
        with preserve_random_state(device):
            model.train()
            if model.backbone.training or model.box_head.training or model.backbone.vdrm.training:
                raise RuntimeError('Rresp baseline must stay in eval mode during training')
            model.response_residual_disabled = False
            actor = make_response_actor(model, cfg)
            output = actor.forward_pass(data, audit=True)
            inputs = output['frozen_audit_inputs']
            occlusion_mask = output.get('disc_route_occlusion_mask')
            with torch.no_grad():
                reference = baseline.eval()(**inputs)
                model.response_residual_disabled = True
                bypass = model(**inputs)
            for key in ('pred_boxes', 'score_map', 'size_map', 'offset_map', 'backbone_feat'):
                if not torch.equal(bypass[key], reference[key]):
                    raise RuntimeError('Rresp bypass differs from Tclean: ' + key)
            for key in ('size_map', 'offset_map', 'backbone_feat'):
                if not torch.equal(output[key], reference[key]):
                    raise RuntimeError('Rresp altered frozen visual/regression output: ' + key)
            delta_key = 'response_delta_energy' if cfg.TRAIN.VDRM_EXPERIMENT_ARM == 'rqual' else 'response_delta_logits'
            if not output[delta_key].any():
                for key in ('pred_boxes', 'score_map'):
                    if not torch.equal(output[key], reference[key]):
                        raise RuntimeError('zero-initialized Rresp is not exactly Tclean: ' + key)
            # Also prove the graph boundary when baseline requires_grad flags
            # are temporarily enabled. No weights or .grad buffers are updated.
            model.response_residual_disabled = False
            for parameter in model.parameters():
                parameter.requires_grad_(True)
            output = model(**inputs)
            # Use the ORIGINAL augmented image's mask, not a second augmentation.
            if occlusion_mask is not None:
                output['disc_route_occlusion_mask'] = occlusion_mask
            _, status, components = actor.compute_losses(output, data, return_components=True)
            if status['Response/eligible_samples'] < 1:
                raise RuntimeError('preflight batch has no surviving target/negative support; cannot audit gradients')
            parameters = list(model.named_parameters())
            for key, loss in components.items():
                grads = torch.autograd.grad(loss, [p for _, p in parameters], allow_unused=True, retain_graph=True)
                norm = 0.
                for (name, _), grad in zip(parameters, grads):
                    if not name.startswith(PREFIX) and grad is not None:
                        raise RuntimeError('response supervision leaked into baseline: ' + name)
                    if grad is not None:
                        if not torch.isfinite(grad).all():
                            raise RuntimeError('non-finite response gradient')
                        norm += grad.square().sum().item()
                # Preservation/abstention can legitimately be inactive, and
                # the zero-initialized abstention regularizer is already zero.
                optional = key == 'candidate_abstain' or (
                    key == 'candidate_preserve' and status.get('Response/preservation_samples', 0) == 0) or (
                    key == 'candidate_rank' and status.get('Response/ranking_samples', 0) == 0)
                if norm <= 0 and not optional:
                    raise RuntimeError('response loss does not train its head: ' + key)
                result[key + '_head_gradient_l2'] = math.sqrt(norm)
            result.update(bypass_exact_tclean=True, frozen_regression_exact=True,
                baseline_gradient_tensors=0, alpha_gradient_tensors=0,
                eligible_samples=status['Response/eligible_samples'],
                trainable_parameters=sum(p.numel() for n, p in parameters if n.startswith(PREFIX)))
            if cfg.TRAIN.VDRM_EXPERIMENT_ARM == 'rqual':
                result['candidate_training_support'] = {k:v for k,v in status.items() if k.startswith('Response/')}
    finally:
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(flags[name])
        model.response_residual_disabled = disabled
        model.train(mode)
        if baseline_hash(model.state_dict(), exclude_response=False) != before:
            raise RuntimeError('optimizer-free Rresp audit modified model state')
    return result


def audit_response_loader(model, baseline, cfg, loader, report_path):
    device = next(model.parameters()).device
    with preserve_random_state(device):
        iterator = iter(loader)
        try:
            data = next(iterator).to(device)
            data['epoch'] = 1
            result = audit_response_model(model, baseline, cfg, data)
        finally:
            del iterator
    if report_path is not None:
        path = Path(report_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    return result
