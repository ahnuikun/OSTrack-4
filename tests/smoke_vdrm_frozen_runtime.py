"""Developer verification only: synthetic Tclean fixture, never real weights."""

import argparse
import contextlib
from datetime import timedelta
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / 'tests'))

from test_vdrm_frozen import candidate_config, tiny_model
from lib.models.ostrack import build_ostrack
from lib.train.base_functions import get_optimizer_scheduler
from lib.train.frozen_vdrm import (BASE_CONFIG, PREFIX, prepare_frozen_models,
    load_base_config, audit_frozen_model, make_actor, visual_hash)
from lib.train.trainers.base_trainer import BaseTrainer


def source_fixture(root, factory):
    torch.manual_seed(42)
    net = factory(load_base_config(), training=False)
    path = Path(root) / 'checkpoints/train/ostrack' / BASE_CONFIG / 'OSTrack_ep0300.pth.tar'
    path.parent.mkdir(parents=True)
    torch.save(dict(net=net.state_dict(), net_type='OSTrack', epoch=300,
                    settings=SimpleNamespace(config_name=BASE_CONFIG)), path)


def batch(size, device):
    torch.manual_seed(42)
    return dict(template_images=torch.randn(1, size, 3, 128, 128, device=device),
        search_images=torch.randn(1, size, 3, 256, 256, device=device),
        template_anno=torch.tensor([[[.25, .25, .5, .5]] * size], device=device),
        search_anno=torch.tensor([[[.25, .25, .5, .5]] * size], device=device),
        search_att=torch.zeros(1, size, 256, 256, device=device, dtype=torch.bool), epoch=1)


def developer_checkpoint(net, optimizer, root, name):
    trainer = object.__new__(BaseTrainer)
    trainer.actor = SimpleNamespace(net=net)
    trainer.optimizer = optimizer
    trainer.epoch = 2
    trainer.stats = {}
    trainer.settings = SimpleNamespace(project_path='train/ostrack/' + name,
                                      config_name=name, local_rank=0)
    trainer._checkpoint_dir = str(Path(root) / 'checkpoints')
    trainer.save_checkpoint()
    return Path(root) / 'checkpoints/train/ostrack' / name / 'OSTrack_ep0002.pth.tar'


def gpu_smoke(root, size):
    if not torch.cuda.is_available():
        raise RuntimeError('GPU smoke requires CUDA; use --mode ddp for the CPU distributed check')
    torch.set_num_threads(2)
    source_fixture(root, build_ostrack)
    results = {}
    for arm in ('rfreeze', 'rdisc'):
        torch.manual_seed(42)
        name, cfg = candidate_config(arm)
        net, base, _ = prepare_frozen_models(cfg, root, name)
        net, base = net.cuda(), base.cuda()
        data = batch(size, 'cuda')
        report = audit_frozen_model(net, base, cfg, data)
        del base
        torch.cuda.empty_cache()
        with contextlib.redirect_stdout(io.StringIO()):
            optimizer, _ = get_optimizer_scheduler(net, cfg)
        actor = make_actor(net, cfg)
        torch.cuda.reset_peak_memory_stats()
        for _ in range(2):
            loss, status = actor(data)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.TRAIN.GRAD_CLIP_NORM)
            optimizer.step()
            if any(p.grad is not None for n, p in net.named_parameters() if not n.startswith(PREFIX)):
                raise RuntimeError('GPU smoke modified the frozen gradient contract')
        assert net.backbone.vdrm.alpha.item() != 0.
        assert visual_hash(net.state_dict()) == net.frozen_vdrm_contract['visual_sha256']
        path = developer_checkpoint(net, optimizer, root, name)
        # A separate helper load checks actual trainer serialization/resume.
        resumed, base, _ = prepare_frozen_models(cfg, root, name, path)
        resumed, base = resumed.cuda(), base.cuda()
        post = audit_frozen_model(resumed, base, cfg, batch(2, 'cuda'))
        for key, value in net.state_dict().items():
            assert torch.equal(value, resumed.state_dict()[key]), key
        results[arm] = dict(initial_audit=report, resumed_audit=post,
            final_alpha=net.backbone.vdrm.alpha.item(), batch_size=size,
            peak_training_memory_gib=torch.cuda.max_memory_allocated() / 1024 ** 3)
        print('REAL VIT-B GPU SMOKE PASSED:', arm, 'batch', size, flush=True)
        del net, base, resumed, actor, optimizer, data
        torch.cuda.empty_cache()
    return results


def ddp_worker(rank, root):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=(Path(root) / 'rendezvous').as_uri(),
                            rank=rank, world_size=4, timeout=timedelta(seconds=120))
    try:
        with patch('lib.models.ostrack.build_ostrack', side_effect=tiny_model):
            for arm in ('rfreeze', 'rdisc'):
                torch.manual_seed(42 + rank)
                name, cfg = candidate_config(arm)
                net, base, _ = prepare_frozen_models(cfg, root, name)
                data = batch(2, 'cpu')
                audit_frozen_model(net, base, cfg, data)
                wrapped = DDP(net, find_unused_parameters=True)
                with contextlib.redirect_stdout(io.StringIO()):
                    optimizer, _ = get_optimizer_scheduler(wrapped, cfg)
                actor = make_actor(wrapped, cfg)
                for _ in range(3):
                    wrapped.train()
                    loss, _ = actor(data)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                for p in net.parameters():
                    if p.requires_grad:
                        reference = p.detach().clone()
                        dist.broadcast(reference, src=0)
                        if not torch.equal(reference, p):
                            raise RuntimeError('DDP parameters differ across ranks')
                if visual_hash(net.state_dict()) != net.frozen_vdrm_contract['visual_sha256']:
                    raise RuntimeError('DDP changed frozen visual parameters or BN buffers')
                if rank == 0:
                    print('FOUR-RANK CPU GLOO SMOKE PASSED:', arm, flush=True)
                dist.barrier()
    finally:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['gpu', 'ddp'], required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='vdrm_frozen_developer_smoke_') as root:
        if args.mode == 'gpu':
            result = gpu_smoke(root, args.batch_size)
            out = PROJECT / 'output/analysis/frozen_vdrm_gpu_smoke.json'
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
            print('Smoke report:', out)
        else:
            source_fixture(root, tiny_model)
            mp.spawn(ddp_worker, args=(root,), nprocs=4, join=True)
            print('FOUR-RANK DISTRIBUTED VERIFICATION COMPLETE; CPU/Gloo, not four-GPU NCCL')


if __name__ == '__main__':
    main()
