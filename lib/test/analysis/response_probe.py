"""Fixed five-case Rresp protocol and model-free bundle verification."""

import hashlib
import io
import json
from pathlib import PurePosixPath
import struct
import tarfile

import numpy as np

from .frozen_failure_diagnostics import CONFIGS as BASE_CONFIGS
from .native_float_audit import array_hash
from .rscore_probe import FOCUS_PLAN

CONFIG = 'vitb_256_mae_ce_vdrm_rresp_s42_32x4_ep300'
CONFIGS = dict(BASE_CONFIGS, Rresp=CONFIG)
SELECTED_FRAMES = sum(stop - start + 1 for start, stop in FOCUS_PLAN.values())
NATIVE_STEPS = sum(stop - 1 for _, stop in FOCUS_PLAN.values()) * len(CONFIGS)
EARLY_PLAN = dict(FOCUS_PLAN, **{
    'GOT-10k_Train_007878': (2, 130),
    'GOT-10k_Train_008311': (2, 100),
})
PROFILES = {'focus': FOCUS_PLAN, 'early_failure': EARLY_PLAN}


def protocol(profile='focus'):
    if profile not in PROFILES:
        raise ValueError('unknown response profile: ' + str(profile))
    plan = dict(PROFILES[profile])
    return plan, sum(b - a + 1 for a, b in plan.values()), sum(b - 1 for _, b in plan.values()) * len(CONFIGS)


def check_bundle(path, *, include_payloads=False):
    """Verify either registered profile; optionally retain analysis inputs.

    Legacy version-1 bundles without a profile are the original focus probe.
    Images are checksummed but only their dimensions are kept in memory.
    """
    with tarfile.open(path, 'r:gz') as archive:
        members = archive.getmembers()
        names = [m.name for m in members]
        if (len(set(names)) != len(names) or any(not m.isfile() for m in members)
                or any(PurePosixPath(n).is_absolute() or '..' in PurePosixPath(n).parts or '\\' in n for n in names)):
            raise ValueError('unsafe/duplicate response bundle members')
        manifest = json.load(archive.extractfile('manifest.json'))
        profile = manifest.get('profile', 'focus')
        plan, selected_frames, native_steps = protocol(profile)
        if (manifest.get('version') != 1 or manifest.get('kind') != 'vdrm_response_probe'
                or manifest.get('dataset') != 'got10k_vdrm_dev' or manifest.get('configurations') != CONFIGS
                or manifest.get('plan') != {n: list(b) for n, b in plan.items()}
                or manifest.get('records') != selected_frames or manifest.get('selected_frames') != selected_frames
                or manifest.get('native_tracking_steps') != native_steps or manifest.get('optimizer_steps') != 0):
            raise ValueError('wrong Rresp protocol/counts')
        expected = manifest['members_sha256']
        if set(names) != set(expected) | {'manifest.json'}:
            raise ValueError('response bundle member set differs')
        contents, image_shapes = {}, {}
        # One archive-order pass; never repeatedly rewind/decompress gzip.
        for member in members:
            if member.name == 'manifest.json':
                continue
            data = archive.extractfile(member).read()
            if hashlib.sha256(data).hexdigest() != expected[member.name]:
                raise ValueError('response bundle checksum mismatch: ' + member.name)
            if member.name.endswith(('.json', '.npz')):
                contents[member.name] = data
            elif member.name.endswith('_image.png'):
                if len(data) < 24 or data[:8] != b'\x89PNG\r\n\x1a\n' or data[12:16] != b'IHDR':
                    raise ValueError('invalid response source PNG header')
                width, height = struct.unpack('>II', data[16:24])
                if min(width, height) <= 0:
                    raise ValueError('invalid response image dimensions')
                image_shapes[member.name] = (height, width)
        for arm in CONFIGS:
            p = manifest['provenance'][arm]
            if p['state_sha256_before'] != p['state_sha256_after']:
                raise ValueError('response probe changed model state: ' + arm)
        traces = json.loads(contents['trajectories.json'])
        if len(traces) != len(plan) or {t['sequence'] for t in traces} != set(plan):
            raise ValueError('wrong response trajectory set')
        indexed = {}
        for trace in traces:
            name = trace['sequence']
            stop = plan[name][1]
            if (len(trace['gt_xywh']) != stop or len(trace['cover']) != stop or len(trace['absence']) != stop
                    or len(trace['source_frames']) != stop or set(trace['native']) != set(CONFIGS)):
                raise ValueError('incomplete response source history')
            for arm, rows in trace['native'].items():
                if len(rows) != stop or [r['frame_1based'] for r in rows] != list(range(1, stop + 1)):
                    raise ValueError('incomplete response trajectory')
                if rows[0]['bbox_xywh'] != trace['gt_xywh'][0]:
                    raise ValueError('wrong response initialization')
                for index, row in enumerate(rows):
                    box = np.asarray(row['bbox_xywh'])
                    if box.shape != (4,) or not np.isfinite(box).all() or (box[2:] <= 0).any():
                        raise ValueError('invalid response float bbox')
                    if index and (row['anchor_xywh'] != rows[index - 1]['bbox_xywh'] or not row['exact_replay']):
                        raise ValueError('broken response closed-loop history')
                    indexed[name, row['frame_1based'], arm] = row
        records = json.loads(contents['records.json'])
        keys = {(n, f) for n, (start, stop) in plan.items() for f in range(start, stop + 1)}
        actual = [(r['sequence'], r['frame_1based']) for r in records]
        if len(actual) != len(set(actual)) or set(actual) != keys:
            raise ValueError('missing/duplicate selected response frames')
        for record in records:
            name, frame = record['sequence'], record['frame_1based']
            if not record['bypass_exact'] or not record['frozen_regression_exact'] or set(record['arms']) != set(CONFIGS):
                raise ValueError('invalid response isolation proof')
            if record['image_member'] not in image_shapes:
                raise ValueError('missing response original image')
            for arm, item in record['arms'].items():
                if item['crop_member'] not in expected or item['maps_member'] not in contents:
                    raise ValueError('missing response crop/maps')
                if item['bbox_xywh'] != indexed[name, frame, arm]['bbox_xywh']:
                    raise ValueError('response map/native bbox mismatch')
                with np.load(io.BytesIO(contents[item['maps_member']]), allow_pickle=False) as maps:
                    if any(not np.isfinite(maps[k]).all() for k in maps.files):
                        raise ValueError('non-finite response maps')
                    score = maps['score_map']
                    if (score.shape != (1, 1, 16, 16) or maps['size_map'].shape != (1, 2, 16, 16)
                            or maps['offset_map'].shape != maps['size_map'].shape):
                        raise ValueError('wrong response map shape')
                    if arm == 'Rresp':
                        if not np.array_equal(maps['bypass_score_map'], maps['Tclean_score_map']):
                            raise ValueError('response bypass score differs')
                        for key in ('size_map', 'offset_map'):
                            if not np.array_equal(maps[key], maps['Tclean_' + key]):
                                raise ValueError('response changed frozen regression: ' + key)
                        peak = np.argmax(maps['hann_window'] * score)
                        same = peak == np.argmax(maps['hann_window'] * maps['Tclean_score_map'])
                        if bool(same) != record['same_hann_peak']:
                            raise ValueError('wrong response same-peak assertion')
                        if np.abs(maps['response_delta_logits']).max() > 2.000001:
                            raise ValueError('response correction exceeded preregistered bound')
                        if profile == 'early_failure':
                            kept = maps['final_retained_mask']
                            padding = maps['search_padding_mask']
                            if (kept.shape != score.shape or kept.dtype != np.bool_
                                    or padding.shape != (256, 256) or padding.dtype != np.bool_
                                    or maps['Tclean_score_logits'].shape != score.shape):
                                raise ValueError('wrong early response CE/padding/logit shape')
                            if np.any(maps['response_delta_logits'][~kept] != 0):
                                raise ValueError('CE-removed response cell was modified')
                            if not np.allclose(maps['score_logits'], maps['Tclean_score_logits'] +
                                               maps['response_delta_logits'], rtol=0, atol=2e-6):
                                raise ValueError('response logits do not equal base plus correction')
            candidate = indexed[name, frame, 'Rresp']
            if record['same_hann_peak'] and record['base_samecrop_bbox'] != candidate['bbox_xywh']:
                raise ValueError('same-peak response changed regression bbox')
            if record['bypass_bbox'] != record['base_samecrop_bbox']:
                raise ValueError('response bypass changed decoded bbox')
            with np.load(io.BytesIO(contents[record['inputs_member']]), allow_pickle=False) as inputs:
                if set(inputs.files) != {k for k, v in candidate['input_sha256'].items() if v is not None}:
                    raise ValueError('wrong response actual input keys')
                for key in inputs.files:
                    if array_hash(inputs[key]) != candidate['input_sha256'][key]:
                        raise ValueError('response actual input hash differs: ' + key)
    return (manifest, contents, image_shapes) if include_payloads else manifest
