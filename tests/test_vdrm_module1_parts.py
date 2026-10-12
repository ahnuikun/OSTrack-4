"""Semantic tests for static readouts, support exclusions, source and real CLI."""

from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import shlex
import re
import sys
import tarfile
from unittest.mock import patch

import numpy as np
import pytest

from lib.test.analysis.module1_parts import (
    METHODS, inside, part_frame, rank_metrics, read_annotations, readouts, retained_points, summarize)
from lib.test.analysis.module1_evidence import ARMS, PLAN, _file_hash
from tracking import analyze_vdrm_module1_parts as entry


def feature_fixture():
    rng = np.random.default_rng(42)
    template = rng.normal(size=(1, 8, 6)).astype(np.float32)
    masks = np.zeros((1, 4, 8), np.float32)
    for i in range(4): masks[0, i, 2*i:2*i+2] = 1
    p = template.reshape(1, 4, 2, 6).mean(2)
    search = rng.normal(size=(1, 256, 6)).astype(np.float32)
    pn = p / np.linalg.norm(p, axis=-1, keepdims=True)
    sn = search / np.linalg.norm(search, axis=-1, keepdims=True)
    return dict(template_tokens=template, part_masks=masks, prototypes=p,
                search_tokens=search, part_valid=np.ones((1, 4), bool),
                q=np.asarray([[.2, .4, .6, .8]], np.float32), route=np.ones((1, 4, 256), np.float32),
                similarity=np.einsum('bkc,blc->bkl', pn, sn),
                indices=np.arange(256)[None], padding=np.zeros((256, 256), bool))


def record_fixture():
    return dict(sequence='fixture', frame_1based=2, gt_xywh=[64., 64., 64., 64.], cover=8., absence=0.,
                arms={arm: dict(geometry=dict(square_xywh=[0, 0, 256, 256])) for arm in ARMS})


def test_pair_auc_ties_and_direction():
    pos, neg = np.array([True, True, False, False]), np.array([False, False, True, True])
    assert rank_metrics([0, 1, 0, 1], pos, neg)['pair_auc'] == .5
    assert rank_metrics([2, 3, 0, 1], pos, neg)['pair_auc'] == 1
    assert rank_metrics([0, 1, 2, 3], pos, neg)['pair_auc'] == 0
    assert rank_metrics([0, 1, 2, 3], np.array([True, False, False, False]), neg) is None


def test_readouts_ignore_joint_part_permutation_and_flat_q_equals_mean():
    a = feature_fixture()
    reference = readouts(a)
    altered = deepcopy(a)
    order = [3, 1, 0, 2]
    for key in ('q', 'part_valid', 'part_masks', 'similarity', 'prototypes', 'route'):
        altered[key] = altered[key][:, order]
    for method in METHODS:
        np.testing.assert_allclose(readouts(altered)[method], reference[method], atol=1e-7)
    a['q'].fill(.7)
    np.testing.assert_allclose(readouts(a)['part_qmean'], readouts(a)['part_mean'], atol=1e-7)


def test_whole_mean_weights_tokens_not_part_sizes():
    a = feature_fixture()
    a['part_masks'][0, 0, 2] = 1
    a['part_masks'][0, 1, 2] = 0
    template = a['template_tokens'][0, a['part_masks'][0].any(0)]
    whole = template.mean(0)
    whole /= np.linalg.norm(whole)
    search = a['search_tokens'][0]
    expected = (search / np.linalg.norm(search, axis=-1, keepdims=True)) @ whole
    # Float32 BLAS dot and einsum reduce in different orders (a few ULPs).
    np.testing.assert_allclose(readouts(a)['whole_mean'], expected, rtol=1e-6, atol=3e-7)
    equal_part_mean = np.mean([a['template_tokens'][0, m.astype(bool)].mean(0) for m in a['part_masks'][0]], axis=0)
    equal_part_mean /= np.linalg.norm(equal_part_mean)
    wrong = (search / np.linalg.norm(search, axis=-1, keepdims=True)) @ equal_part_mean
    assert np.max(np.abs(expected - wrong)) > .01


def test_padding_and_retained_indices_resolve_original_grid():
    a = feature_fixture()
    a['indices'] = np.asarray([[0, 17, 255]])
    a['padding'][16, 16] = True
    points, support = retained_points(a, dict(square_xywh=[100, 200, 512, 512]))
    np.testing.assert_array_equal(points, [[116, 216], [148, 248], [596, 696]])
    np.testing.assert_array_equal(support, [True, False, True])


@pytest.mark.parametrize('mode', ['outside', 'tiny', 'padding', 'invisible', 'no_parts'])
def test_unsupported_frames_do_not_get_nearest_positive(mode):
    a, r = feature_fixture(), record_fixture()
    if mode == 'outside': r['gt_xywh'] = [400, 400, 32, 32]
    elif mode == 'tiny': r['gt_xywh'] = [15, 15, 1, 1]
    elif mode == 'padding': a['padding'][:] = True
    elif mode == 'invisible': r['cover'] = 0
    else:
        a['part_valid'][:] = False
        a['part_masks'][:] = 0
    result = part_frame(a, r, 'Rdisc')
    assert result['exclusion_reason'] is not None
    assert result['outside_gt_readouts'] is None


def test_identity_overlap_removed_and_both_scales_preserved():
    a, r = feature_fixture(), record_fixture()
    identical = part_frame(a, r, 'Rdisc', dict(distractor_xywh=r['gt_xywh']))
    assert all(x['readouts'] is None and x['positive_tokens'] == 0 for x in identical['identity'])
    separate = part_frame(a, r, 'Rdisc', dict(distractor_xywh=[144, 64, 64, 64]))
    assert [x['roi_scale'] for x in separate['identity']] == [1., .8]
    assert all(x['readouts'] is not None for x in separate['identity'])


def test_sequence_macro_does_not_reweight_long_sequence():
    metric = lambda v: {method: dict(pair_auc=v, mean_gap=v, peak_gap=v) for method in METHODS}
    frames = [dict(sequence='long', arm='Rdisc', exclusion_reason=None,
                   outside_gt_readouts=metric(1.), identity=[]) for _ in range(100)]
    frames += [dict(sequence='short', arm='Rdisc', exclusion_reason=None, outside_gt_readouts=metric(0.), identity=[])]
    assert summarize(frames)['arms']['Rdisc']['outside_gt']['sequence_macro']['whole_mean']['pair_auc'] == .5


def test_annotation_source_and_duplicates_rejected(tmp_path):
    value = json.loads(entry.DEFAULT_ANNOTATIONS.read_text(encoding='utf-8'))
    path = tmp_path / 'rois.json'
    path.write_text(json.dumps(value), encoding='utf-8')
    with pytest.raises(ValueError, match='source'): read_annotations(path, 'wrong')
    value['entries'].append(value['entries'][0])
    path.write_text(json.dumps(value), encoding='utf-8')
    with pytest.raises(ValueError, match='duplicate'): read_annotations(path, value['source_sha256'])


def test_actual_entry_full_fixture_and_output_protection(tmp_path):
    from test_vdrm_module1_evidence import bundle_fixture
    source = tmp_path / 'source.tar.gz'
    bundle_fixture(source)
    with tarfile.open(source) as archive:
        manifest = json.load(archive.extractfile('manifest.json'))
        records = json.load(archive.extractfile('records.json'))
    record = records[0]
    annotations = dict(kind='vdrm_module1_identity_rois', version=1, posthoc=True,
        source_sha256=_file_hash(source), entries=[dict(sequence=record['sequence'], frame_1based=record['frame_1based'],
        image_member=record['image_member'], image_sha256=manifest['members_sha256'][record['image_member']],
        distractor_xywh=[200, 200, 40, 40], reviewed_by='assistant_visual_inspection',
        identity_relation='different_same_class_instance')])
    path = tmp_path / 'annotations.json'
    path.write_text(json.dumps(annotations), encoding='utf-8')
    output = tmp_path / 'result.tar.gz'
    with patch.object(sys, 'argv', ['analyze_vdrm_module1_parts.py', '--bundle', str(source),
                                  '--output', str(output), '--annotations', str(path)]):
        entry.main()
    with tarfile.open(output) as archive:
        summary = json.load(archive.extractfile('summary.json'))
        assert summary['records'] == 1497 and summary['optimizer_steps'] == 0
        assert summary['arms']['Rdisc']['outside_gt']['eligible_frames'] == 0
    previous = output.read_bytes()
    with pytest.raises(FileExistsError): entry.run(source, output, path)
    assert output.read_bytes() == previous
