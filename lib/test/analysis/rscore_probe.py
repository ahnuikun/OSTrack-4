"""Fixed posthoc development cases and read-only Rscore bundle checks."""

import hashlib
import io
import json
from pathlib import PurePosixPath
import tarfile

import numpy as np

from .frozen_failure_diagnostics import CONFIGS
from .native_float_audit import array_hash

PROBE_CONFIG = 'vitb_256_mae_ce_vdrm_rscore_probe_s42_32x4_ep300'
# Fixed BEFORE the new closed-loop experiment. Preserve the two previous
# failure windows and protect three previous gains; no outcome-driven edits.
FOCUS_PLAN = {
    'GOT-10k_Train_003240': (88, 100),
    'GOT-10k_Train_008333': (54, 74),
    'GOT-10k_Train_001664': (65, 100),
    'GOT-10k_Train_007878': (68, 130),
    'GOT-10k_Train_008311': (55, 100),
}
ARMS = tuple(CONFIGS) + ('Rscore',)
SELECTED_FRAMES = sum(b - a + 1 for a, b in FOCUS_PLAN.values())
NATIVE_STEPS = sum(b - 1 for _, b in FOCUS_PLAN.values()) * len(ARMS)


def check_bundle(path):
    """One archive-order checksum pass; do not extract files or load models."""
    with tarfile.open(path, 'r:gz') as archive:
        members = archive.getmembers()
        names = [m.name for m in members]
        if (len(set(names)) != len(names) or any(not m.isfile() for m in members)
                or any(PurePosixPath(n).is_absolute() or '..' in PurePosixPath(n).parts or '\\' in n for n in names)):
            raise ValueError('unsafe/duplicate Rscore bundle members')
        manifest = json.load(archive.extractfile('manifest.json'))
        if (manifest.get('kind') != 'vdrm_rscore_probe' or manifest.get('version') != 1
                or manifest.get('dataset') != 'got10k_vdrm_dev' or manifest.get('optimizer_steps') != 0
                or manifest.get('configurations') != CONFIGS
                or manifest.get('probe_parameter') != PROBE_CONFIG
                or manifest.get('plan') != {n: list(b) for n, b in FOCUS_PLAN.items()}
                or manifest.get('records') != SELECTED_FRAMES or manifest.get('native_tracking_steps') != NATIVE_STEPS):
            raise ValueError('wrong Rscore probe protocol/counts')
        digests, contents = manifest['members_sha256'], {}
        if set(names) != set(digests) | {'manifest.json'}:
            raise ValueError('Rscore bundle member set mismatch')
        for member in members:
            if member.name == 'manifest.json':
                continue
            data = archive.extractfile(member).read()
            if hashlib.sha256(data).hexdigest() != digests[member.name]:
                raise ValueError('Rscore checksum mismatch: ' + member.name)
            if member.name.endswith(('.json', '.npz')):
                contents[member.name] = data
    for arm in CONFIGS:
        p = manifest['provenance'][arm]
        if p['state_sha256_before'] != p['state_sha256_after']:
            raise ValueError('Rscore changed model state: ' + arm)
    trajectories = json.loads(contents['trajectories.json'])
    if len(trajectories) != len(FOCUS_PLAN) or {t['sequence'] for t in trajectories} != set(FOCUS_PLAN):
        raise ValueError('incomplete/duplicate Rscore trajectories')
    indexed = {}
    for trajectory in trajectories:
        name = trajectory['sequence']
        start, stop = FOCUS_PLAN[name]
        gt = np.asarray(trajectory['gt_xywh'])
        if (gt.shape != (stop, 4) or not np.isfinite(gt).all() or (gt[:, 2:] < 0).any()
                or len(trajectory['cover']) != stop or len(trajectory['absence']) != stop
                or len(trajectory['source_frames']) != stop or set(trajectory['native']) != set(ARMS)):
            raise ValueError('invalid Rscore GT/history')
        for arm, rows in trajectory['native'].items():
            if len(rows) != stop or [r['frame_1based'] for r in rows] != list(range(1, stop + 1)):
                raise ValueError('missing Rscore native steps')
            if rows[0]['bbox_xywh'] != trajectory['gt_xywh'][0]:
                raise ValueError('invalid Rscore initialization')
            for i, row in enumerate(rows):
                bbox = np.asarray(row['bbox_xywh'])
                if bbox.shape != (4,) or not np.isfinite(bbox).all() or (bbox[2:] <= 0).any():
                    raise ValueError('invalid native Rscore bbox')
                if i and (row['anchor_xywh'] != rows[i - 1]['bbox_xywh'] or not row['exact_replay']):
                    raise ValueError('Rscore history is not closed-loop/exact')
                indexed[name, i + 1, arm] = row
    records = json.loads(contents['records.json'])
    expected = {(n, f) for n, (a, b) in FOCUS_PLAN.items() for f in range(a, b + 1)}
    actual = [(r['sequence'], r['frame_1based']) for r in records]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError('missing/duplicate Rscore diagnostic records')
    for row in records:
        name, frame = row['sequence'], row['frame_1based']
        if not row['alpha0_exact'] or set(row['arms']) != set(ARMS):
            raise ValueError('invalid Rscore equivalence/arms')
        for arm, artifacts in row['arms'].items():
            native = indexed[name, frame, arm]
            if artifacts['bbox_xywh'] != native['bbox_xywh']:
                raise ValueError('Rscore map record differs from native bbox')
            if artifacts['maps_member'] not in contents or artifacts['crop_member'] not in digests:
                raise ValueError('missing Rscore maps/crop')
            with np.load(io.BytesIO(contents[artifacts['maps_member']]), allow_pickle=False) as maps:
                if any(not np.isfinite(maps[k]).all() for k in maps.files):
                    raise ValueError('non-finite Rscore maps')
                score = maps['score_map']
                if (score.ndim != 4 or score.shape[:2] != (1, 1) or score.shape[2] != score.shape[3]
                        or maps['size_map'].shape != (1, 2, *score.shape[2:])
                        or maps['offset_map'].shape != maps['size_map'].shape):
                    raise ValueError('invalid Rscore map shape')
                if arm == 'Rscore':
                    for hybrid, source in (('score_map', 'Rdisc_score_map'), ('size_map', 'Tclean_size_map'),
                                            ('offset_map', 'Tclean_offset_map')):
                        if not np.array_equal(maps[hybrid], maps[source]):
                            raise ValueError('wrong Rscore component: ' + hybrid)
                    if not np.array_equal(maps['alpha0_score_map'], maps['Tclean_score_map']):
                        raise ValueError('Rscore alpha0 score differs from Tclean')
                    actual_same_peak = np.argmax(maps['hann_window'] * maps['Tclean_score_map']) == np.argmax(
                        maps['hann_window'] * score)
                    if actual_same_peak != row['same_hann_peak']:
                        raise ValueError('wrong Rscore same-peak claim')
        probe_row = indexed[name, frame, 'Rscore']
        if row['same_hann_peak'] and row['base_samecrop_bbox'] != probe_row['bbox_xywh']:
            raise ValueError('same-peak Rscore bbox differs from Tclean')
        if row['alpha0_bbox'] != row['base_samecrop_bbox']:
            raise ValueError('Rscore alpha0 bbox differs from Tclean')
        with np.load(io.BytesIO(contents[row['inputs_member']]), allow_pickle=False) as inputs:
            for key, digest in probe_row['input_sha256'].items():
                if (digest is None and key in inputs.files) or (digest is not None and (
                        key not in inputs.files or array_hash(inputs[key]) != digest)):
                    raise ValueError('wrong Rscore actual input hash: ' + key)
    return manifest
