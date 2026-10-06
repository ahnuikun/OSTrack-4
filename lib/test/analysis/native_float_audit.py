"""Registered two-case protocol and standalone integrity/closure checker."""

import hashlib
import io
import json
from pathlib import PurePosixPath
import tarfile

import numpy as np

from .frozen_failure_diagnostics import CONFIGS

# 1-based frames: (fork/audit start, native prefix stop). These predate the
# newly generated traces; never adapt windows to make counterfactuals win.
AUDIT_PLAN = {'GOT-10k_Train_003240': (88, 100), 'GOT-10k_Train_008333': (54, 74)}


def array_hash(array):
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(tuple(array.shape)).encode())
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def check_bundle(path):
    """Check every archive member without extraction; no model/GPU access."""
    with tarfile.open(path, 'r:gz') as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        if (len(names) != len(set(names)) or any(not m.isfile() for m in members)
                or any(PurePosixPath(n).is_absolute() or '..' in PurePosixPath(n).parts or '\\' in n for n in names)):
            raise ValueError('unsafe/duplicate/non-file bundle member')
        manifest = json.load(archive.extractfile('manifest.json'))
        plan = {name: list(bounds) for name, bounds in AUDIT_PLAN.items()}
        selected = sum(stop - start + 1 for start, stop in AUDIT_PLAN.values())
        steps = sum(stop - 1 for _, stop in AUDIT_PLAN.values()) * 3
        if (manifest.get('kind') != 'vdrm_native_float' or manifest.get('version') != 1
                or manifest.get('dataset') != 'got10k_vdrm_dev' or manifest.get('plan') != plan
                or manifest.get('configurations') != CONFIGS or manifest.get('optimizer_steps') != 0
                or manifest.get('selected_frames') != selected or manifest.get('native_tracking_steps') != steps
                or manifest.get('records') != selected * 3 or manifest.get('fork_records') != selected * 3):
            raise ValueError('wrong native-float bundle protocol/counts')
        expected = manifest['members_sha256']
        if set(names) != set(expected) | {'manifest.json'}:
            raise ValueError('bundle member set differs from manifest')
        # Read in ARCHIVE order exactly once. Extracting inputs, then an
        # earlier map, for every row would repeatedly rewind gzip and turn a
        # large server bundle into O(records * archive_size) decompression.
        # Cache only compressed NPZ/JSON bytes, not the large RGB PNGs.
        contents = {}
        for member in members:
            name = member.name
            if name == 'manifest.json':
                continue
            data = archive.extractfile(member).read()
            if hashlib.sha256(data).hexdigest() != expected[name]:
                raise ValueError('bundle checksum mismatch: ' + name)
            if name.endswith(('.json', '.npz')):
                contents[name] = data
        for arm in CONFIGS:
            provenance = manifest['provenance'][arm]
            if provenance['state_sha256_before'] != provenance['state_sha256_after']:
                raise ValueError('changed model state: ' + arm)
        trajectories = json.loads(contents['trajectories.json'])
        if len(trajectories) != 2 or {r['sequence'] for r in trajectories} != set(AUDIT_PLAN):
            raise ValueError('missing/duplicate native trajectories')
        native = {}
        for trajectory in trajectories:
            name = trajectory['sequence']
            start, stop = AUDIT_PLAN[name]
            if trajectory['fork_frame_1based'] != start or trajectory['stop_frame_1based'] != stop:
                raise ValueError('wrong trajectory bounds')
            gt = np.asarray(trajectory['gt_xywh'])
            if (gt.shape != (stop, 4) or not np.isfinite(gt).all()
                    or len(trajectory['source_frames']) != stop or len(trajectory['cover']) != stop
                    or len(trajectory['absence']) != stop or set(trajectory['native']) != set(CONFIGS)):
                raise ValueError('incomplete native input history')
            for arm, rows in trajectory['native'].items():
                if len(rows) != stop or [r['frame_1based'] for r in rows] != list(range(1, stop + 1)):
                    raise ValueError('missing/duplicate native float frames')
                if rows[0]['bbox_xywh'] != trajectory['gt_xywh'][0]:
                    raise ValueError('wrong native initialization box')
                for index, row in enumerate(rows):
                    box = np.asarray(row['bbox_xywh'])
                    if box.shape != (4,) or not np.isfinite(box).all() or (box[2:] <= 0).any():
                        raise ValueError('invalid native float bbox')
                    if index and (not row['exact_replay'] or row['anchor_xywh'] != rows[index - 1]['bbox_xywh']):
                        raise ValueError('broken native float closed-loop history')
                    native[name, row['frame_1based'], arm] = row
        records = json.loads(contents['records.json'])
        forks = json.loads(contents['forks.json'])
        keys = {(n, frame, arm) for n, (start, stop) in AUDIT_PLAN.items()
                for frame in range(start, stop + 1) for arm in CONFIGS}
        actual = [(r['sequence'], r['frame_1based'], r['anchor']) for r in records]
        fork_keys = {(n, frame, mode) for n, (start, stop) in AUDIT_PLAN.items()
                     for frame in range(start, stop + 1) for mode in ('control', 'route0', 'alpha0')}
        actual_forks = [(r['sequence'], r['frame_1based'], r['mode']) for r in forks]
        if len(actual) != len(keys) or set(actual) != keys or len(actual_forks) != len(fork_keys) or set(actual_forks) != fork_keys:
            raise ValueError('missing/duplicate shared-crop or fork records')
        branch_rows = {(r['sequence'], r['frame_1based'], r['mode']): r for r in forks}
        for row in records:
            source = native[row['sequence'], row['frame_1based'], row['anchor']]
            if (not row['native_replay_exact'] or not row['alpha0_exact']
                    or row['native_float_bbox'] != source['bbox_xywh']
                    or row['anchor_xywh'] != source['anchor_xywh']
                    or row['input_sha256'] != source['input_sha256']
                    or row['outputs'][row['anchor']]['bbox_xywh'] != source['bbox_xywh']
                    or set(row['outputs']) != set(CONFIGS) | {'Rdisc_route0', 'Rdisc_alpha0'}):
                raise ValueError('broken same-crop native replay contract')
            if (row['outputs']['Rdisc_alpha0']['scalars']['vdrm_alpha_raw'] != 0.
                    or row['outputs']['Rdisc_alpha0']['actual_injected_residual']['residual_relative_norm_max'] != 0.
                    or row['outputs']['Rdisc_route0']['correction_abs_max'] != 0.):
                raise ValueError('intervention not applied on same-crop record')
            check_artifacts(contents, expected, row, shared=True)
        for row in forks:
            name, frame, mode = row['sequence'], row['frame_1based'], row['mode']
            start, _ = AUDIT_PLAN[name]
            previous = (native[name, start - 1, 'Rdisc'] if frame == start
                        else branch_rows[name, frame - 1, mode])
            if (not row['exact_replay'] or row['fork_frame_1based'] != start
                    or row['anchor_xywh'] != previous['bbox_xywh']
                    or row['bbox_xywh'] != row['outputs']['bbox_xywh']):
                raise ValueError('broken counterfactual fork history')
            if mode == 'control':
                source = native[name, frame, 'Rdisc']
                if (not row['control_exact'] or row['bbox_xywh'] != source['bbox_xywh']
                        or row['input_sha256'] != source['input_sha256']):
                    raise ValueError('unmodified fork differs from native Rdisc')
            if mode == 'alpha0' and (row['outputs']['scalars']['vdrm_alpha_raw'] != 0.
                    or row['outputs']['actual_injected_residual']['residual_relative_norm_max'] != 0.):
                raise ValueError('alpha0 not applied in fork')
            if mode == 'route0' and row['outputs']['correction_abs_max'] != 0.:
                raise ValueError('route0 not applied in fork')
            check_artifacts(contents, expected, row, shared=False)
    return manifest


def check_artifacts(contents, expected, row, shared):
    for key in ('crop_member', 'maps_member', 'inputs_member') + (('image_member',) if shared else ()):
        if row[key] not in expected:
            raise ValueError('missing referenced artifact: ' + key)
    if not row['inputs_member'].endswith('.npz') or not row['maps_member'].endswith('.npz'):
        raise ValueError('native inputs/maps must be NPZ artifacts')
    with np.load(io.BytesIO(contents[row['inputs_member']]), allow_pickle=False) as inputs:
        required = {k for k, v in row['input_sha256'].items() if v is not None}
        if set(inputs.files) != required or required != {'template', 'search', 'template_bbox', 'ce_template_mask'}:
            raise ValueError('missing exact native input tensors')
        for key in required:
            if not np.isfinite(inputs[key]).all() or array_hash(inputs[key]) != row['input_sha256'][key]:
                raise ValueError('native input tensor hash mismatch: ' + key)
    with np.load(io.BytesIO(contents[row['maps_member']]), allow_pickle=False) as maps:
        if any(not np.isfinite(maps[key]).all() for key in maps.files):
            raise ValueError('non-finite audit map')
        labels = row['outputs'] if shared else ('',)
        for label in labels:
            prefix = label + '_' if label else ''
            score, size, offset = (maps[prefix + key] for key in ('score_map', 'size_map', 'offset_map'))
            route, indices = (maps[prefix + key] for key in ('part_route_logits', 'search_global_index'))
            if (score.ndim != 4 or score.shape[:2] != (1, 1) or score.shape[2] != score.shape[3]
                    or size.shape != (1, 2, *score.shape[2:]) or offset.shape != size.shape
                    or route.ndim != 3 or route.shape[:2] != (1, 4) or indices.shape != (1, route.shape[-1])):
                raise ValueError('wrong native map shape')
        if shared:
            for key in ('score_map', 'size_map', 'offset_map'):
                if not np.array_equal(maps['Tclean_' + key], maps['Rdisc_alpha0_' + key]):
                    raise ValueError('alpha0 native maps differ from Tclean')
