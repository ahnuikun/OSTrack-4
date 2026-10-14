"""Train width-128 module1: frozen-Rdisc adapter or MAE-initialized whole network.

Single-GPU FP32, microbatches accumulated to the original 32x4 global batch.
Resume is from the last completed epoch; an interrupted epoch is replayed.
Smoke runs are isolated and cannot be used by the production tracker.
"""
import argparse
from copy import deepcopy
import json
import platform
from pathlib import Path
import sys
import tarfile

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

import torch

from lib.models.ostrack import build_ostrack
from lib.train.frozen_vdrm import configuration_hash
from lib.train.module1_data import make_loader
from lib.train.module1_formal import (CONFIG, SCRATCH_CONFIG, SOURCE, PREFIX, atomic_save, backend, checkpoint,
    config, excluded_got_sequences, frozen_hash, seed_all, state_hash, training_loss,
    validate_checkpoint, validate_config)
from lib.test.analysis.frozen_failure_diagnostics import CONFIGS
from tracking.diagnose_vdrm_frozen_failures import load_models


def initialize(cfg, save_dir, device):
    paths = {arm: Path(save_dir) / 'checkpoints/train/ostrack' / CONFIGS[arm] / 'OSTrack_ep0300.pth.tar'
             for arm in ('Tclean', 'Rdisc')}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError('missing source checkpoint: ' + str(path))
    models, _, provenance = load_models(paths, 'cpu', arms=('Tclean', 'Rdisc'))
    seed_all(42)
    model = build_ostrack(cfg, training=False)
    state = model.state_dict()
    state.update(models['Rdisc'].state_dict())
    model.load_state_dict(state, strict=True)
    initial = deepcopy(model.backbone.module1_adapter.state_dict())
    contract = dict(version=1, source_config=SOURCE, config_sha256=configuration_hash(cfg),
        source_checkpoint_sha256=provenance['Rdisc']['checkpoint_sha256'],
        source_state_sha256=provenance['Rdisc']['state_sha256_before'],
        source_provenance=provenance, excluded_got_ids=excluded_got_sequences())
    # Historical helper uses a different key prefix in its digest.
    contract['source_state_sha256'] = state_hash(models['Rdisc'].state_dict())
    del models, state
    model.to(device).train()
    if frozen_hash(model) != contract['source_state_sha256']:
        raise RuntimeError('source import changed Rdisc')
    return model, contract, initial


def optimizer_for(model, cfg):
    parameters = [p for n, p in model.named_parameters() if n.startswith(PREFIX)]
    if any(p.requires_grad != n.startswith(PREFIX) for n, p in model.named_parameters()):
        raise RuntimeError('only the registered adapter may be trainable')
    optimizer = torch.optim.AdamW(parameters, lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, cfg.TRAIN.LR_DROP_EPOCH,
                                               gamma=cfg.TRAIN.SCHEDULER.DECAY_RATE)
    return optimizer, scheduler


def train_epoch(model, cfg, loader, optimizer, device, epoch, effective_batch, updates):
    joint = cfg.MODEL.MODULE1_ADAPTER.MODE == 'joint'
    seed_all(42 + epoch)  # Epoch-boundary replay, including worker seeds and augmentations.
    model.train()
    optimizer.zero_grad(set_to_none=True)
    processed = group_seen = 0
    total_samples = len(loader.dataset)
    aggregate = {}
    datasets = {}
    gradient_audit = []
    for batch_index, data in enumerate(loader):
        batch = data['search_images'].shape[1]
        if group_seen == 0:
            group_size = min(effective_batch, total_samples - processed)
        if group_seen + batch > group_size:
            raise RuntimeError('microbatch must divide effective batch')
        if joint:
            from lib.train.module1_scratch import scratch_loss
            loss, rows = scratch_loss(model, cfg, data, device, epoch)
        else:
            loss, rows = training_loss(model, cfg, data, device, check_identity=(updates == 0))
        (loss * batch / group_size).backward()
        processed += batch
        group_seen += batch
        for name in data['dataset']:
            datasets[name] = datasets.get(name, 0) + 1
        if 'vdrm_distractor_applied' in data:
            aggregate['distractor_augmented'] = aggregate.get('distractor_augmented', 0.) + float(data['vdrm_distractor_applied'].sum())
        for row in rows:
            for key, value in row.items():
                if key == 'norm_ratio':
                    aggregate['max_norm_ratio'] = max(aggregate.get('max_norm_ratio', 0.), value)
                else:
                    aggregate[key] = aggregate.get(key, 0.) + value
        if group_seen == group_size:
            if joint and updates < 3:
                groups = dict(backbone='backbone.blocks.', head='box_head.', alpha='backbone.vdrm.alpha',
                    route='backbone.vdrm.route_head.', adapter_value=PREFIX+'value.',
                    adapter_search=PREFIX+'search.', adapter_template=PREFIX+'template.')
                audit = {'update': updates + 1}
                for label, prefix in groups.items():
                    gradients = [p.grad.detach().square().sum() for name, p in model.named_parameters()
                                 if name.startswith(prefix) and p.grad is not None]
                    audit[label] = float(torch.stack(gradients).sum().sqrt()) if gradients else 0.
                gradient_audit.append(audit)
            parameters = model.parameters() if joint else model.backbone.module1_adapter.parameters()
            grad = torch.nn.utils.clip_grad_norm_(parameters,
                cfg.TRAIN.GRAD_CLIP_NORM, error_if_nonfinite=True)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            updates += 1
            group_seen = 0
            if updates % cfg.TRAIN.PRINT_INTERVAL == 0:
                print(f'EPOCH {epoch} samples={processed}/{total_samples} updates={updates} '
                      f'loss={aggregate["total"]/processed:.5f} grad={float(grad):.5f}', flush=True)
    if processed != total_samples or group_seen:
        raise RuntimeError('incomplete epoch or uncommitted gradients')
    if not joint and any(p.grad is not None for n, p in model.named_parameters() if not n.startswith(PREFIX)):
        raise RuntimeError('gradient reached frozen Rdisc')
    return updates, dict(epoch=epoch, samples=processed, updates=updates, datasets=datasets, gradient_audit=gradient_audit,
        **{k: v if k == 'max_norm_ratio' else v / processed for k, v in aggregate.items()})


def run(args):
    import cv2
    cv2.setNumThreads(0)
    joint = args.stage == 'scratch'
    config_name = SCRATCH_CONFIG if joint else CONFIG
    cfg = config(config_name)
    validate_config(cfg)
    if args.micro_batch < 1 or 128 % args.micro_batch or args.workers < 0:
        raise ValueError('--micro_batch must divide 128; --workers must be nonnegative')
    if not torch.cuda.is_available():
        raise RuntimeError('formal training/smoke requires CUDA')
    torch.set_num_threads(2)
    backend()
    seed_all(42)
    device = torch.device('cuda:0')
    root = Path(args.save_dir).resolve()
    smoke_name = 'module1_scratch_smoke' if joint else 'module1_formal_smoke'
    folder = root / ('analysis/' + smoke_name if args.smoke else 'checkpoints/train/ostrack/' + config_name)
    latest = folder / 'latest.pth.tar'
    if latest.exists() and not args.resume and not args.precheck:
        raise FileExistsError(f'{latest} already exists; use --resume to avoid overwriting a run')
    if joint:
        from lib.train.module1_scratch import initialize_scratch, optimizer_scratch
        model, contract, initial = initialize_scratch(cfg, root, device)
        optimizer, scheduler = optimizer_scratch(model, cfg)
    else:
        model, contract, initial = initialize(cfg, root, device)
        optimizer, scheduler = optimizer_for(model, cfg)
    loader, datasets = make_loader(cfg, root, args.micro_batch, args.workers, args.smoke or args.precheck)
    run_spec = dict(stage=args.stage, smoke=bool(args.smoke), micro_batch=args.micro_batch, effective_batch=128,
        workers=args.workers, datasets=datasets, epochs=(3 if joint else 1) if args.smoke else cfg.TRAIN.EPOCH,
        checkpoint_selection='fixed_last_epoch', precision='FP32', seed=42,
        initializer=dict(torch=torch.__version__, python=platform.python_version()),
        loss=('original OSTrack GIoU/L1/focal + discriminative route; per-image 1x residual norm' if joint else
              'Step08 learning_loss, uniform sample weights, per-image 1x total residual norm'),
        deterministic_algorithms=True, cudnn_deterministic=True, cublas_workspace=':4096:8')
    if args.precheck:
        if joint:
            audits = []
            updates = 0
            for epoch in range(1, 4):
                updates, metrics = train_epoch(model, cfg, loader, optimizer, device, epoch, 128, updates)
                audits += metrics['gradient_audit']
            check_scratch_gradients(audits)
            print('SCRATCH PRECHECK PASSED: verified MAE, four datasets, three real updates, all gradient paths', flush=True)
            return
        # Actual loader, real gradients and native zero-initialized head parity.
        seen = set()
        for batch in loader:
            seen.update(batch['dataset'])
            loss, metrics = training_loss(model, cfg, batch, device, check_identity=True)
            loss.backward()
        if len(seen) != 4:
            raise RuntimeError('precheck did not process every training dataset')
        if not any(p.grad is not None and p.grad.count_nonzero() for p in model.backbone.module1_adapter.parameters()):
            raise RuntimeError('precheck sampled no useful training gradient; inspect data before training')
        print('FORMAL PRECHECK PASSED: sources, four datasets, actual loader, zero identity, backward', flush=True)
        return
    start, updates = 1, 0
    if args.resume:
        if not latest.is_file():
            raise FileNotFoundError('no completed epoch to resume: ' + str(latest))
        saved = torch.load(latest, map_location='cpu', weights_only=False)
        validate_checkpoint(saved, cfg, allow_smoke=args.smoke)
        expected_contract = dict(contract)
        if joint:
            # Initial random head tensors need not regenerate bit-for-bit on
            # another CPU. Resume uses saved tensors; MAE/config hashes remain
            # strict. Preserve the original run's initialization provenance.
            expected_contract['source_state_sha256'] = saved['module1_contract']['source_state_sha256']
        if saved['module1_contract'] != expected_contract or saved['run'] != run_spec:
            raise ValueError('resume source, runtime, data paths or run settings changed')
        contract = saved['module1_contract']
        model.load_state_dict(saved['net'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        initial = saved['initial_adapter']
        start, updates = saved['epoch'] + 1, saved['updates']
        del saved
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'run.json').write_text(json.dumps(dict(run=run_spec, contract=contract), indent=2), encoding='utf-8')
    for epoch in range(start, run_spec['epochs'] + 1):
        updates, metrics = train_epoch(model, cfg, loader, optimizer, device, epoch, 128, updates)
        if not joint and (not metrics['supported'] or not model.backbone.module1_adapter.value.weight.count_nonzero()):
            raise RuntimeError('null training: no supported boxes or adapter update')
        scheduler.step()
        payload = checkpoint(model, cfg, epoch, optimizer, scheduler, contract, run_spec, initial, updates)
        validate_checkpoint(payload, cfg, allow_smoke=args.smoke)
        atomic_save(payload, latest)
        if epoch % 20 == 0 or epoch == run_spec['epochs']:
            atomic_save(payload, folder / f'OSTrack_ep{epoch:04d}.pth.tar')
        del payload
        (folder / f'epoch_{epoch:04d}.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
        print('COMPLETED', json.dumps(metrics), flush=True)
    if args.smoke:
        # Reload all weights and optimizer through the same contract after real updates.
        saved = torch.load(latest, map_location='cpu', weights_only=False)
        validate_checkpoint(saved, cfg, expected_epoch=run_spec['epochs'], allow_smoke=True)
        model.load_state_dict(saved['net'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        if joint:
            audits = []
            for path in sorted(folder.glob('epoch_*.json')):
                audits += json.loads(path.read_text())['gradient_audit']
            check_scratch_gradients(audits)
        report = dict(passed=True, performance_pass=False, real_data=True,
            datasets=datasets, epoch=saved['epoch'], updates=saved['updates'],
            source_state_sha256=contract['source_state_sha256'],
            final_state_sha256=saved['state_sha256'], initial_sha256=saved['initial_sha256'],
            frozen_weights_verified=not joint, zero_initialization_verified=not joint, checkpoint_reload_verified=True,
            torch=torch.__version__, gpu=torch.cuda.get_device_name())
        (folder / 'smoke_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        archive = folder.parent / (smoke_name + '.tar.gz')
        with tarfile.open(archive, 'w:gz') as bundle:
            for path in sorted(folder.glob('*.json')):
                bundle.add(path, arcname=path.name)
        print('FORMAL SMOKE PASSED: training/checkpoint/reload; no performance claim', flush=True)
        print('SMOKE REPORT:', archive, flush=True)
    print('CHECKPOINT:', latest, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--save_dir', default='./output')
    parser.add_argument('--stage', choices=['adapter', 'scratch'], default='adapter')
    parser.add_argument('--micro_batch', type=int, default=None)
    parser.add_argument('--workers', type=int, default=10)
    parser.add_argument('--precheck', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.micro_batch is None:
        args.micro_batch = 32 if args.stage == 'scratch' else 4
    if args.precheck and args.resume:
        parser.error('--precheck and --resume cannot be combined')
    run(args)


def check_scratch_gradients(audits):
    if not audits or audits[0]['alpha'] <= 0:
        raise RuntimeError('zero initialization blocked alpha learning')
    for name in ('backbone', 'head', 'route', 'adapter_value', 'adapter_search', 'adapter_template'):
        if not any(row[name] > 0 for row in audits):
            raise RuntimeError('scratch training did not activate gradient path: ' + name)


if __name__ == '__main__':
    main()
