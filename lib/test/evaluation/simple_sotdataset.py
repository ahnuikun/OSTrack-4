import os

import numpy as np

from lib.test.evaluation.data import BaseDataset, Sequence, SequenceList
from lib.test.utils.load_text import load_text


def _load_bbox_file(path):
    gt = load_text(path, delimiter=(',', '\t', ' '), dtype=np.float64, backend='numpy')
    gt = np.asarray(gt, dtype=np.float64)
    if gt.ndim == 1:
        gt = gt.reshape(1, -1)
    if gt.shape[1] < 4:
        raise ValueError("Expected at least four bbox columns in {}".format(path))
    return gt[:, :4]


def _list_images(path):
    image_exts = ('.jpg', '.jpeg', '.png', '.bmp')
    frames = [os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith(image_exts)]
    return sorted(frames)


class _FolderSOTDataset(BaseDataset):
    def __init__(self, dataset_name, env_path_attr):
        super().__init__()
        self.dataset_name = dataset_name
        self.base_path = getattr(self.env_settings, env_path_attr)
        self.sequence_list = self._discover_sequences()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def __len__(self):
        return len(self.sequence_list)

    def _discover_sequences(self):
        raise NotImplementedError

    def _construct_sequence(self, sequence_info):
        frames = _list_images(sequence_info['frame_dir'])
        ground_truth_rect = _load_bbox_file(sequence_info['anno_path'])
        if len(frames) != ground_truth_rect.shape[0]:
            usable_len = min(len(frames), ground_truth_rect.shape[0])
            frames = frames[:usable_len]
            ground_truth_rect = ground_truth_rect[:usable_len, :]
        return Sequence(sequence_info['name'], frames, self.dataset_name, ground_truth_rect)


class VisDroneSOTDataset(_FolderSOTDataset):
    def __init__(self, split='train'):
        self.split = split
        super().__init__('visdrone', 'visdrone_path')

    def _discover_sequences(self):
        split_path = os.path.join(self.base_path, self.split)
        sequences_path = os.path.join(split_path, 'sequences')
        anno_path = os.path.join(split_path, 'annotations')
        sequence_names = sorted(os.listdir(sequences_path))
        return [
            {
                'name': name,
                'frame_dir': os.path.join(sequences_path, name),
                'anno_path': os.path.join(anno_path, name + '.txt'),
            }
            for name in sequence_names
        ]


class UAVDTDataset(_FolderSOTDataset):
    def __init__(self):
        super().__init__('uavdt', 'uavdt_path')

    def _discover_sequences(self):
        sequences_path = os.path.join(self.base_path, 'sequences')
        anno_path = os.path.join(self.base_path, 'anno')
        sequence_names = sorted(os.listdir(sequences_path))
        return [
            {
                'name': name,
                'frame_dir': os.path.join(sequences_path, name),
                'anno_path': os.path.join(anno_path, name + '_gt.txt'),
            }
            for name in sequence_names
        ]


class DTB70Dataset(_FolderSOTDataset):
    def __init__(self):
        super().__init__('dtb70', 'dtb70_path')

    def _discover_sequences(self):
        sequence_names = sorted(
            name for name in os.listdir(self.base_path)
            if os.path.isdir(os.path.join(self.base_path, name))
        )
        sequence_info = []
        for name in sequence_names:
            seq_path = os.path.join(self.base_path, name)
            anno_path = self._find_annotation(seq_path)
            sequence_info.append({
                'name': name,
                'frame_dir': os.path.join(seq_path, 'img'),
                'anno_path': anno_path,
            })
        return sequence_info

    @staticmethod
    def _find_annotation(seq_path):
        for filename in ('groundtruth_rect.txt', 'groundtruth.txt', 'groundtruth_rect.1.txt'):
            path = os.path.join(seq_path, filename)
            if os.path.isfile(path):
                return path
        raise FileNotFoundError("Could not find DTB70 annotation under {}".format(seq_path))
