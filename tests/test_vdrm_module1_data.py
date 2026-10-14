"""Exercise missing TrackingNet images through the actual folder loader."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import pytest
import torch

from lib.train import module1_data
from lib.train.data.image_loader import opencv_loader
from lib.train.data.sampler import TrackingSampler
from lib.train.dataset.tracking_net import TrackingNet
from lib.train.module1_formal import config


MISSING = '-3TIfnTSM6c_2'
PRESENT = 'Nf1aqv5Fg5o_0'


@pytest.fixture
def trackingnet(tmp_path):
    root = tmp_path / 'trackingnet'
    anno = root / 'TRAIN_0/anno'
    anno.mkdir(parents=True)
    for name in (MISSING, PRESENT):
        (anno / (name + '.txt')).write_text('60,60,24,24\n' * 24)
    frames = root / 'TRAIN_0/frames' / PRESENT
    frames.mkdir(parents=True)
    image = np.random.default_rng(42).integers(0, 256, (160, 160, 3), dtype=np.uint8)
    for index in range(24):
        assert cv2.imwrite(str(frames / f'{index}.jpg'), image)
    source = TrackingNet(str(root), image_loader=opencv_loader, set_ids=[0])
    # Filesystem enumeration order differs across servers. Reproduce the
    # reported first-sequence failure regardless of this machine's order.
    source.sequence_list = [(0, MISSING), (0, PRESENT)]
    source.seq_to_class_map, source.seq_per_class = source._load_class_info()
    return source


class OtherSource:
    """Small in-memory stand-ins for the three unaffected datasets."""
    def __init__(self, root, name):
        self.root, self.name = str(root), name
        self.sequence_list = ['GOT-10k_Train_%06d' % (i + 1) for i in range(9335)] if name == 'got10k' else ['sample']
        self.seq_per_class = self._build_seq_per_class()
        self.class_list = ['object']

    def _build_seq_per_class(self): return {'object': list(range(len(self.sequence_list)))}
    def get_name(self): return self.name
    def get_num_sequences(self): return len(self.sequence_list)
    def is_video_sequence(self): return self.name != 'coco'
    def has_class_info(self): return False
    def get_sequence_info(self, index):
        return dict(bbox=torch.tensor([[60., 60., 24., 24.]] * 24),
                    visible=torch.ones(24, dtype=torch.bool), valid=torch.ones(24, dtype=torch.bool))
    def get_frames(self, seq_id, frame_ids, info):
        return ([np.full((160, 160, 3), 128, dtype=np.uint8) for _ in frame_ids],
                {'bbox': [info['bbox'][f].clone() for f in frame_ids]}, {'object_class_name': 'object'})


@pytest.mark.parametrize('stage', ['adapter', 'scratch'])
def test_loader_accepts_missing_first_trackingnet_sequence(trackingnet, tmp_path, monkeypatch, capsys, stage):
    from lib.train.module1_formal import SCRATCH_CONFIG
    others = [OtherSource(tmp_path, name) for name in ('lasot', 'got10k', 'coco')]
    (tmp_path / 'list.txt').write_text('\n'.join(others[1].sequence_list))
    sources = others + [trackingnet]
    monkeypatch.setattr('lib.train.base_functions.names2datasets', lambda *args: sources)
    before = list(trackingnet.sequence_list)
    classes = {key: list(value) for key, value in trackingnet.seq_per_class.items()}
    cfg = config(SCRATCH_CONFIG) if stage == 'scratch' else config()
    loader, manifest = module1_data.make_loader(cfg, tmp_path, micro_batch=2, workers=0, smoke=True)
    assert trackingnet.sequence_list == before
    assert trackingnet.seq_per_class == classes
    assert manifest[-1] == dict(dataset='trackingnet', root=trackingnet.root, sequences=2)
    assert loader.dataset.sampler.p_datasets == [.25] * 4
    torch.manual_seed(42)
    module1_data.random.seed(42)
    np.random.seed(42)
    seen = set()
    for batch in loader:
        seen.update(batch['dataset'])
        assert torch.isfinite(batch['search_images']).all()
    assert seen == {'lasot', 'got10k', 'coco', 'trackingnet'}
    assert 'PRECHECK WARNING trackingnet' in capsys.readouterr().out


def test_sampler_retries_missing_sequence_with_real_images(trackingnet, monkeypatch):
    def accept(data):
        data['valid'] = all(frame is not None for frame in data['search_images'])
        return data
    sampler = TrackingSampler([trackingnet], [1], samples_per_epoch=1, max_gap=10,
                              num_search_frames=1, processing=accept)
    sampler.max_sampling_attempts = 2
    sequence_ids = iter([0, 1])
    monkeypatch.setattr('lib.train.data.sampler.random.randint', lambda *args: next(sequence_ids))
    sample = sampler[0]
    assert sample['valid']
    assert sample['dataset'] == 'trackingnet'
    assert sample['template_images'][0].shape == (160, 160, 3)


def test_precheck_rejects_wholly_unreadable_trackingnet(trackingnet):
    # Keep both annotations but point the second item at the same absent folder.
    trackingnet.sequence_list[1] = trackingnet.sequence_list[0]
    with pytest.raises(ValueError, match='no readable image.*trackingnet.*2'):
        module1_data.precheck_source_images(trackingnet)


def test_precheck_retries_corrupt_image_and_obeys_limit(trackingnet):
    folder = Path(trackingnet.root) / 'TRAIN_0/frames' / MISSING
    folder.mkdir(parents=True)
    (folder / '0.jpg').write_bytes(b'not a jpeg')
    with pytest.raises(ValueError, match='no readable image.*1'):
        module1_data.precheck_source_images(trackingnet, max_sequences=1)
    module1_data.precheck_source_images(trackingnet)
