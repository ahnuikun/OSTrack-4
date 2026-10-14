"""Original four-dataset pipeline with explicit exclusion of diagnostic audits."""
import random
from copy import copy
from pathlib import Path

import numpy as np
import torch

from lib.train.admin.settings import Settings
from lib.train.base_functions import build_dataloaders, update_settings
from lib.train.data import LTRLoader
from lib.train.module1_formal import excluded_got_sequences


class FourDatasetSmoke(torch.utils.data.Dataset):
    """Force actual processing from every source in the short server check."""
    def __init__(self, sampler, count):
        self.sampler = sampler
        self.count = max(8, count)

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        sampler = copy(self.sampler)
        sampler.datasets = [self.sampler.datasets[index % 4]]
        sampler.p_datasets = [1.]
        return sampler[index]


def worker_seed(worker_id):
    import cv2
    cv2.setNumThreads(0)
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)


def make_loader(cfg, save_dir, micro_batch, workers, smoke=False):
    settings = Settings()
    settings.local_rank = -1
    settings.use_lmdb = False
    settings.save_dir = str(Path(save_dir).resolve())
    update_settings(settings, cfg)
    original, _ = build_dataloaders(cfg, settings)
    dataset = original.dataset
    dataset.max_sampling_attempts = 100
    got = next(d for d in dataset.datasets if d.get_name().lower() == 'got10k')
    names = (Path(got.root) / 'list.txt').read_text().splitlines()
    excluded = excluded_got_sequences()
    if len(names) <= max(excluded) or len(names) != len(set(names)):
        raise ValueError('GOT list does not match the registered sequence IDs')
    banned = {names[i] for i in excluded}
    got.sequence_list = [n for n in got.sequence_list if n not in banned]
    got.seq_per_class = got._build_seq_per_class()
    got.class_list = sorted(got.seq_per_class)
    manifest = []
    for source in dataset.datasets:
        count = source.get_num_sequences()
        if not count:
            raise ValueError('empty training dataset: ' + source.get_name())
        # Direct decode cannot be hidden by TrackingSampler's retry loop.
        info = source.get_sequence_info(0)
        frames, _, _ = source.get_frames(0, [0], info)
        if not frames or frames[0] is None or not np.isfinite(frames[0]).all():
            raise ValueError('unreadable first image: ' + source.get_name())
        manifest.append(dict(dataset=source.get_name(), root=str(source.root), sequences=count))
    if smoke:
        dataset = FourDatasetSmoke(dataset, 2 * micro_batch)
    loader = LTRLoader('train', dataset, training=True, batch_size=micro_batch, shuffle=True,
        num_workers=workers, drop_last=False, stack_dim=1, pin_memory=True,
        timeout=180 if workers else 0, worker_init_fn=worker_seed)
    return loader, manifest
