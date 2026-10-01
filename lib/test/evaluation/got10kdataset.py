import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text
import os
from pathlib import Path


def _vdrm_dev_sequence_ids():
    """152 fixed GOT-10k train IDs excluded from both VOT train and val."""
    spec_dir = Path(__file__).resolve().parents[2] / 'train' / 'data_specs'

    def read_ids(name):
        return {
            int(line)
            for line in (spec_dir / name).read_text(encoding='utf-8').splitlines()
            if line.strip()
        }

    val_ids = read_ids('got10k_val_split.txt')
    vot_val_ids = read_ids('got10k_vot_val_split.txt')
    vot_train_ids = read_ids('got10k_vot_train_split.txt')
    dev_ids = val_ids - vot_val_ids
    if not vot_val_ids <= val_ids or dev_ids & (vot_train_ids | vot_val_ids):
        raise ValueError('GOT-10k VDRM dev split overlaps train or held-out val')
    if len(dev_ids) != 152:
        raise ValueError(
            f'Expected 152 fixed GOT-10k VDRM dev sequences, got {len(dev_ids)}'
        )
    return sorted(dev_ids)


class GOT10KDataset(BaseDataset):
    """ GOT-10k dataset.

    Publication:
        GOT-10k: A Large High-Diversity Benchmark for Generic Object Tracking in the Wild
        Lianghua Huang, Xin Zhao, and Kaiqi Huang
        arXiv:1810.11981, 2018
        https://arxiv.org/pdf/1810.11981.pdf

    Download dataset from http://got-10k.aitestunion.com/downloads
    """
    def __init__(self, split):
        super().__init__()
        # vdrm_dev is a fixed subset of the official train directory, but
        # its sequence IDs are excluded from VDRM's VOT train and val lists.
        if split == 'test' or split == 'val':
            self.base_path = os.path.join(self.env_settings.got10k_path, split)
        else:
            self.base_path = os.path.join(self.env_settings.got10k_path, 'train')

        self.sequence_list = self._get_sequence_list(split)
        self.split = split

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        anno_path = '{}/{}/groundtruth.txt'.format(self.base_path, sequence_name)

        ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64)

        frames_path = '{}/{}'.format(self.base_path, sequence_name)
        frame_list = [frame for frame in os.listdir(frames_path) if frame.endswith(".jpg")]
        frame_list.sort(key=lambda f: int(f[:-4]))
        frames_list = [os.path.join(frames_path, frame) for frame in frame_list]

        return Sequence(sequence_name, frames_list, 'got10k', ground_truth_rect.reshape(-1, 4))

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self, split):
        with open('{}/list.txt'.format(self.base_path)) as f:
            sequence_list = f.read().splitlines()

        if split == 'ltrval':
            with open('{}/got10k_val_split.txt'.format(self.env_settings.dataspec_path)) as f:
                seq_ids = f.read().splitlines()

            sequence_list = [sequence_list[int(x)] for x in seq_ids]
        elif split == 'vdrm_dev':
            dev_ids = _vdrm_dev_sequence_ids()
            if dev_ids[-1] >= len(sequence_list):
                raise ValueError('GOT-10k train/list.txt is shorter than dev IDs')
            sequence_list = [sequence_list[index] for index in dev_ids]
        return sequence_list
