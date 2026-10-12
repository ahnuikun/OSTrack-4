"""Semantic evidence checks, independent production decode and CLI contracts."""

from copy import deepcopy
import io
import json
from pathlib import Path
import re
import shlex
import sys
import tarfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import cv2
import pytest
import torch

from lib.models.layers.head import CenterPredictor
from lib.test.analysis.module1_evidence import (
    ARMS, PLAN, RECORDS, STEPS, check_bundle, decode_candidates, frame_metrics, validate_arrays, visual_geometry)
from lib.test.analysis.frozen_failure_diagnostics import CONFIGS, dump_json, write_bundle
from lib.test.tracker.ostrack import OSTrack
from lib.utils.box_ops import clip_box
from tracking import diagnose_vdrm_module1_evidence as entry
from tracking import analyze_vdrm_module1_evidence as analyzer
from tracking.diagnose_vdrm_native_float import npz_bytes

PROJECT = Path(__file__).resolve().parents[1]


def arrays_fixture():
    a = dict(template_tokens=np.zeros((1, 64, 768), np.float32),
        search_tokens=np.zeros((1, 1, 768), np.float32), part_masks=np.zeros((1, 4, 64), np.float32),
        prototypes=np.zeros((1, 4, 768), np.float32), similarity=np.zeros((1, 4, 1), np.float32),
        q=np.zeros((1, 4), np.float32), part_valid=np.zeros((1, 4), bool),
        route=np.zeros((1, 4, 1), np.float32), route_logits=np.zeros((1, 4, 1), np.float32),
        indices=np.zeros((1, 1), np.int64), delta=np.zeros((1, 1, 768), np.float32),
        alpha=np.asarray(0., np.float32), match_scale=np.asarray(5., np.float32), match_bias=np.asarray(-2.5, np.float32),
        score_map=np.full((1, 1, 16, 16), .5, np.float32), size_map=np.full((1, 2, 16, 16), .1, np.float32),
        offset_map=np.zeros((1, 2, 16, 16), np.float32), hann=np.ones((1, 1, 16, 16), np.float32),
        final_kept=np.zeros((1, 256), bool), padding=np.zeros((256, 256), bool),
        template_bbox=np.asarray([[.2, .2, .6, .6]], np.float32),
        search_input=np.zeros((1, 3, 256, 256), np.float32), template_input=np.zeros((1, 3, 128, 128), np.float32))
    a['final_kept'][0, 0] = True
    return a


def bundle_fixture(path, mutate=None):
    a = arrays_fixture()
    encoded = npz_bytes(a)
    traces, records, payloads, images = [], [], {}, {}
    png = cv2.imencode('.png', np.zeros((500, 500, 3), np.uint8))[1].tobytes()
    for name, (start, stop) in PLAN.items():
        gt = [[100., 100., 20., 20.]] * stop
        native = {}
        for arm in ARMS:
            rows = [dict(frame_1based=1, bbox_xywh=gt[0])]
            for frame in range(2, stop + 1):
                anchor = rows[-1]['bbox_xywh']
                geometry = visual_geometry(anchor, [500, 500], gt[frame-1])
                row = dict(frame_1based=frame, anchor_xywh=anchor,
                           resize_factor=256 / geometry['square_xywh'][2], exact_replay=True)
                box = decode_candidates(a, row, [500, 500])[0].tolist()
                row.update(bbox_xywh=box, iou=0.)
                rows.append(row)
            native[arm] = rows
        traces.append(dict(sequence=name, gt_xywh=gt, cover=[8.] * stop, absence=[0.] * stop, native=native))
        for frame in range(start, stop + 1):
            arms = {}
            for arm in ARMS:
                member = f'arrays/{name}/{frame:06d}_{arm}.npz'
                if arm == 'Rdisc':
                    base = {'base_' + key: a[key] for key in ('score_map', 'size_map', 'offset_map')}
                    payloads[member] = npz_bytes(dict(a, **base))
                else:
                    payloads[member] = encoded
                arms[arm] = dict(arrays_member=member,
                    geometry=visual_geometry(native[arm][frame-1]['anchor_xywh'], [500, 500], gt[frame-1]))
            records.append(dict(sequence=name, frame_1based=frame, gt_xywh=gt[frame-1], cover=8., absence=0.,
                image_shape_hw=[500, 500], image_member=f'images/{name}/{frame:06d}.png',
                arms=arms, alpha0_exact_Tclean=True))
            images[f'images/{name}/{frame:06d}.png'] = png
        images[f'images/{name}/000001.png'] = png
    provenance = {arm: dict(state_sha256_before='fixture', state_sha256_after='fixture') for arm in ARMS}
    manifest = dict(kind='vdrm_module1_step01', version=1, plan={k: list(v) for k, v in PLAN.items()},
        arms=list(ARMS), records=RECORDS, native_steps=STEPS, optimizer_steps=0, provenance=provenance,
        configurations=CONFIGS, dataset='got10k_vdrm_dev')
    if mutate:
        mutate(manifest, records, traces, payloads)
    write_bundle(path, manifest, {'records.json': dump_json(records), 'trajectories.json': dump_json(traces), **payloads, **images})


def test_fixed_protocol():
    assert (RECORDS, STEPS) == (499, 1497)
    assert PLAN['GOT-10k_Train_007878'] == (2, 130)


def test_decode_all_cells_matches_production():
    a = arrays_fixture()
    rng = np.random.default_rng(42)
    a['size_map'] = rng.uniform(.01, .9, (1, 2, 16, 16)).astype(np.float32)
    a['offset_map'] = rng.uniform(-1, 1, (1, 2, 16, 16)).astype(np.float32)
    row = dict(anchor_xywh=[150.25, -5.125, 31.375, 50.25], resize_factor=256 / 159)
    normalized = CenterPredictor.cal_bbox(SimpleNamespace(feat_sz=16),
        torch.eye(256).reshape(256, 1, 16, 16), torch.from_numpy(a['size_map']).repeat(256, 1, 1, 1),
        torch.from_numpy(a['offset_map']).repeat(256, 1, 1, 1))
    tracker = SimpleNamespace(state=row['anchor_xywh'], params=SimpleNamespace(search_size=256))
    boxes = [clip_box(OSTrack.map_box_back(tracker, box, row['resize_factor']), 200, 300, margin=10)
             for box in (normalized * 256 / row['resize_factor']).tolist()]
    np.testing.assert_array_equal(decode_candidates(a, row, [200, 300]), boxes)


@pytest.mark.parametrize('key', ['q', 'prototypes', 'route', 'delta', 'indices', 'part_valid'])
def test_semantic_tampering_rejected(key):
    a = arrays_fixture()
    a[key].flat[0] = 1 if key != 'indices' else 256
    with pytest.raises(ValueError):
        validate_arrays(a)


def test_full_archive_and_cpu_analysis(tmp_path):
    bundle = tmp_path / 'probe.tar.gz'
    bundle_fixture(bundle)
    summary, frames, _ = check_bundle(bundle)
    assert len(frames) == 1497 and summary['records'] == 499
    with patch.object(torch.cuda, 'is_available', side_effect=AssertionError('no CUDA access')):
        result = analyzer.run(bundle, tmp_path / 'analysis.tar.gz')
    assert result == summary
    with pytest.raises(FileExistsError):
        analyzer.run(bundle, tmp_path / 'analysis.tar.gz')


@pytest.mark.parametrize('kind', ['state', 'anchor', 'labels', 'bypass', 'missing', 'duplicate', 'geometry', 'prefix_iou', 'image'])
def test_rehashed_archive_corruption_rejected(tmp_path, kind):
    def mutate(manifest, records, traces, payloads):
        if kind == 'state':
            manifest['provenance']['Rdisc']['state_sha256_after'] = 'changed'
        elif kind == 'anchor':
            traces[0]['native']['Rdisc'][1]['anchor_xywh'] = [0., 0., 20., 20.]
        elif kind == 'labels':
            records[0]['cover'] = 0.
        elif kind == 'bypass':
            records[0]['alpha0_exact_Tclean'] = False
        elif kind == 'missing':
            payloads.pop(next(iter(payloads)))
        elif kind == 'duplicate':
            records[0] = deepcopy(records[1])
        elif kind == 'geometry':
            records[0]['arms']['Rdisc']['geometry']['gt_fraction_in_visual_support'] = .12345
        elif kind == 'prefix_iou':
            traces[0]['native']['Rdisc'][1]['iou'] = 1.
        else:
            records[0]['image_member'] = 'images/missing.png'
    bundle = tmp_path / 'bad.tar.gz'
    bundle_fixture(bundle, mutate)
    with pytest.raises(ValueError):
        check_bundle(bundle)




def test_existing_output_rejected_before_cuda_or_data(tmp_path):
    output = tmp_path / 'exists.tar.gz'
    output.write_bytes(b'existing')
    with patch.object(entry, 'get_dataset', side_effect=AssertionError('unexpected dataset')):
        with pytest.raises(FileExistsError):
            entry.run(tmp_path, output)
    assert output.read_bytes() == b'existing'


def test_run_uses_real_loader_return_contract(tmp_path):
    # load_models returns models, ONE shared cfg, provenance; not a cfg map.
    models = {arm: object() for arm in ARMS}
    shared_cfg = entry.config_for(CONFIGS['Tclean'])
    provenance = {arm: dict(state_sha256_before='unchanged') for arm in ARMS}
    with patch.object(torch.cuda, 'is_available', return_value=True), \
            patch.object(entry, 'checkpoint_paths', return_value={}) as paths, \
            patch.object(entry, 'get_dataset', return_value='dataset'), \
            patch.object(entry, 'prepare_cases', return_value=['case']) as prepare, \
            patch.object(entry, 'load_models', return_value=(models, shared_cfg, provenance)) as loader, \
            patch.object(entry, 'collect', return_value={}) as collect, \
            patch.object(entry, 'state_hash', return_value='unchanged'), \
            patch.object(entry, 'write_bundle') as write, \
            patch.object(entry, 'check_bundle') as check:
        output = tmp_path / 'probe.tar.gz'
        entry.run(tmp_path, output)
    loader.assert_called_once_with({}, 'cuda')
    prepare.assert_called_once_with('dataset', plan=PLAN)
    configs = collect.call_args.args[2]
    assert set(configs) == set(ARMS)
    assert configs['Tclean'].TRAIN.VDRM_EXPERIMENT_ARM == 'tclean'
    assert configs['Rfreeze'].TRAIN.VDRM_EXPERIMENT_ARM == 'rfreeze'
    assert configs['Rdisc'].MODEL.VDRM.DISCRIMINATIVE_ROUTE is True
    assert write.call_args.args[1]['records'] == 499
    check.assert_called_once_with(output)
