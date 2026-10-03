import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch

from tracking import evaluate_vdrm_frozen_routes as offline


class FakeRoutingModel(torch.nn.Module):
    def __init__(self, discriminative):
        super().__init__()
        self.discriminative = discriminative
        self.inputs = []

    def forward(self, template, search, **kwargs):
        self.inputs.append(search.detach().clone())
        index = torch.arange(256, device=search.device).reshape(1, -1)
        logits = torch.zeros(1, 4, 256, device=search.device)
        if self.discriminative:
            near = ((index % 16 >= 6) & (index % 16 <= 9)
                    & (index // 16 >= 6) & (index // 16 <= 9))
            logits.masked_fill_(near[:, None], 3.)
        return dict(part_route_logits=logits, search_global_index=index,
                    part_valid=torch.ones(1, 4, device=search.device, dtype=torch.bool))


class OfflineRouteTest(unittest.TestCase):
    def test_full_152_sequence_offline_pipeline_pairs_identical_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frames_dir = root / 'raw'
            frames_dir.mkdir()
            cv2.imwrite(str(frames_dir / '1.jpg'), np.full((256, 256, 3), 120, dtype=np.uint8))
            cv2.imwrite(str(frames_dir / '2.jpg'), np.full((256, 256, 3), 150, dtype=np.uint8))
            (frames_dir / 'cover.label').write_text('8\n8\n')
            (frames_dir / 'absence.label').write_text('0\n0\n')
            box = np.array([[88., 88., 80., 80.]] * 2)
            dataset = [SimpleNamespace(name=f'GOT-10k_Train_{i:06d}', dataset='got10k',
                frames=[str(frames_dir / '1.jpg'), str(frames_dir / '2.jpg')], ground_truth_rect=box)
                for i in range(152)]
            names = [f'vitb_256_mae_ce_vdrm_{arm}_s42_32x4_ep300' for arm in ('rfreeze', 'rdisc')]
            for name in names:
                checkpoint = root / 'checkpoints/train/ostrack' / name / 'OSTrack_ep0300.pth.tar'
                checkpoint.parent.mkdir(parents=True)
                torch.save({'net': {}}, checkpoint)
            results = root / 'results' / 'got10k'
            results.mkdir(parents=True)
            for sequence in dataset:
                np.savetxt(results / (sequence.name + '.txt'), box)
            models = []

            def factory(cfg, training=False):
                model = FakeRoutingModel(cfg.MODEL.VDRM.DISCRIMINATIVE_ROUTE)
                models.append(model)
                return model

            args = ['evaluate_vdrm_frozen_routes.py', '--tracker_params', *names,
                    '--save_dir', str(root), '--device', 'cpu']
            contract = dict(source_sha256='shared-source', visual_sha256='shared-visual')
            with patch.object(sys, 'argv', args), patch.object(offline, 'get_dataset', return_value=dataset), \
                 patch.object(offline, 'trackerlist', return_value=[SimpleNamespace(results_dir=str(results.parent))]), \
                 patch.object(offline, 'build_ostrack', side_effect=factory), \
                 patch.object(offline, 'validate_frozen_checkpoint', return_value=contract), \
                 contextlib.redirect_stdout(io.StringIO()):
                offline.main()
            report = json.loads((root / 'analysis/frozen_vdrm/route_offline.json').read_text())
            self.assertEqual(report['total_sequences'], 152)
            self.assertEqual(report['eligible_sequences'], 152)
            self.assertEqual(report['paired_rank_delta_pp'], 100.)
            self.assertEqual(len(models[0].inputs), 152)
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(models[0].inputs, models[1].inputs)))

    def test_rank_statistics_excludes_missing_positive_parts(self):
        logits = torch.tensor([[[1., 0.], [100., -100.]]])
        masks = dict(positive=torch.tensor([[[True, False], [False, False]]]),
                     negative=torch.tensor([[[False, True], [True, True]]]),
                     eligible=torch.tensor([[True, False]]))
        stats = offline.ranking_statistics(logits, masks)
        self.assertEqual(stats, dict(eligible_parts=1, rank_correct=1, logit_gap_sum=1.))


if __name__ == '__main__':
    unittest.main()
