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


def precheck_source_images(source, max_sequences=100):
    """Find a decodable image without requiring the first sequence to exist.

    This is a bounded availability probe, not a full dataset integrity audit.
    It changes neither the sequence/class lists nor the sampler's random state;
    missing images encountered during training still use its normal retry path.
    """
    count = source.get_num_sequences()
    if not count:
        raise ValueError('empty training dataset: ' + source.get_name())
    if max_sequences < 1:
        raise ValueError('max_sequences must be positive')
    limit = min(count, max_sequences)
    last_error = None
    for seq_id in range(limit):
        info = source.get_sequence_info(seq_id)
        sequence = source.sequence_list[seq_id]
        try:
            frames, _, _ = source.get_frames(seq_id, [0], info)
            if (not frames or frames[0] is None or np.size(frames[0]) == 0
                    or not np.isfinite(frames[0]).all()):
                raise ValueError('decoded image is missing, empty or non-finite')
        except Exception as error:
            # Match the sampler's image-read retry behavior. Annotation loading
            # above remains strict; do not hide invalid annotation/config errors.
            last_error = error
            print(f'PRECHECK WARNING {source.get_name()}: sequence={sequence!r}, frame=0: '
                  f'{type(error).__name__}: {error}; trying next sequence', flush=True)
            continue
        print(f'DATA PRECHECK {source.get_name()}: decoded sequence={sequence!r}, frame=0; '
              f'failed_probes={seq_id} (sample check only)', flush=True)
        return
    raise ValueError(f'no readable image in {source.get_name()}: tried {limit}/{count} '
                     f'sequences at frame 0; root={source.root}') from last_error


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
        precheck_source_images(source)
        manifest.append(dict(dataset=source.get_name(), root=str(source.root), sequences=count))
    if smoke:
        dataset = FourDatasetSmoke(dataset, 2 * micro_batch)
    loader = LTRLoader('train', dataset, training=True, batch_size=micro_batch, shuffle=True,
        num_workers=workers, drop_last=False, stack_dim=1, pin_memory=True,
        timeout=180 if workers else 0, worker_init_fn=worker_seed)
    return loader, manifest
