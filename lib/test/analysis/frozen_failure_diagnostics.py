"""Pure planning, geometry and bundle checks for frozen VDRM diagnostics."""

import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import tarfile

import numpy as np
import torch

from .extract_results import calc_seq_err_robust
from .frozen_vdrm_stages import visibility_protocol, recovery_delay

CONFIGS = {
    'Tclean': 'vitb_256_mae_ce_vdrm_v8_tclean_s42_32x4_ep300',
    'Rfreeze': 'vitb_256_mae_ce_vdrm_rfreeze_s42_32x4_ep300',
    'Rdisc': 'vitb_256_mae_ce_vdrm_rdisc_s42_32x4_ep300',
}
PLAN = {
    'GOT-10k_Train_008333': ('reappearance', 'Tclean'),
    'GOT-10k_Train_003240': ('loss', 'Tclean'),
    'GOT-10k_Train_001664': ('loss', 'Rfreeze'),
    'GOT-10k_Train_007878': ('gain', 'Rfreeze'),
    'GOT-10k_Train_008311': ('gain', 'Tclean'),
}


def overlapping_fraction(target, region):
    """Intersection / target area, not IoU; GT never chooses a search crop."""
    target, region = np.asarray(target, dtype=float), np.asarray(region, dtype=float)
    if target.shape != (4,) or region.shape != (4,) or not np.isfinite([target, region]).all():
        raise ValueError('coverage requires finite xywh boxes')
    if (target[2:] <= 0).any():
        return None
    if (region[2:] <= 0).any():
        return 0.
    size = np.maximum(np.minimum(target[:2] + target[2:], region[:2] + region[2:])
                      - np.maximum(target[:2], region[:2]), 0.)
    return float(size.prod() / target[2:].prod())


def checked_predictions(prediction, gt, name):
    prediction = np.asarray(prediction, dtype=float)
    if prediction.shape != gt.shape or not np.isfinite(prediction).all() or (prediction[:, 2:] <= 0).any():
        raise ValueError('prediction/GT length or values mismatch: ' + name)
    # Native result writer uses fmt='%d', including the initialization box.
    if not (np.array_equal(prediction[0], gt[0]) or np.array_equal(prediction[0], np.trunc(gt[0]))):
        raise ValueError('initial prediction does not match GT: ' + name)
    return prediction


def make_window_plan(name, gt, cover, absence, predictions):
    """Bounded posthoc failure/gain windows; preserve the full trajectory."""
    if name not in PLAN:
        raise ValueError('unregistered diagnostic sequence: ' + name)
    gt = np.asarray(gt, dtype=float)
    if gt.ndim != 2 or gt.shape[1] != 4 or not np.isfinite(gt).all() or len(gt) < 6:
        raise ValueError('invalid GT trajectory: ' + name)
    cover, absence = np.asarray(cover), np.asarray(absence)
    if cover.shape != (len(gt),):
        raise ValueError('visibility/GT length mismatch: ' + name)
    masks, events = visibility_protocol(cover, absence)
    overlaps = {}
    for arm in CONFIGS:
        prediction = checked_predictions(predictions[arm], gt, name + ':' + arm)
        values, _, _, valid = calc_seq_err_robust(torch.tensor(prediction), torch.tensor(gt), 'got10k')
        overlaps[arm] = values.numpy()
    visible = valid.numpy() & (cover > 0) & (absence == 0)
    visible[0] = False
    purpose, reference = PLAN[name]
    difference = overlaps['Rdisc'] - overlaps[reference]
    selected, windows = set(), []

    def add(first, last, reason):
        first, last = max(1, int(first)), min(len(gt) - 1, int(last))
        selected.update(range(first, last + 1))
        windows.append(dict(first_frame_1based=first + 1, last_frame_1based=last + 1, reason=reason))

    if purpose == 'reappearance':
        eligible = np.flatnonzero(masks['reappearance'] & visible)
        if not len(eligible):
            raise ValueError('no registered reappearance frames: ' + name)
        worst = int(eligible[np.argmin(difference[eligible])])
        add(worst - 10, worst + 10, 'worst Rdisc-vs-Tclean reappearance frame with context')
    else:
        signed = difference if purpose == 'gain' else -difference
        candidates = [i for i in range(1, len(gt) - 4) if visible[i:i + 5].all()]
        if not candidates:
            raise ValueError('no five-frame visible comparison run: ' + name)
        worst_or_best = max(candidates, key=lambda i: float(signed[i:i + 5].mean()))
        add(worst_or_best - 3, worst_or_best + 7, 'extreme five-frame visible ' + purpose)
        sustained = [i for i in range(1, len(gt) - 2)
                     if visible[i:i + 3].all() and (signed[i:i + 3] >= .1).all()]
        if sustained:
            add(sustained[0] - 5, sustained[0] + 5, 'first sustained differential ' + purpose)
        if name.endswith('007878'):
            for event in events:
                if (recovery_delay(overlaps[reference], event) is None
                        and recovery_delay(overlaps['Rdisc'], event) is not None):
                    delay = recovery_delay(overlaps['Rdisc'], event)
                    add(event['occlusion_start_1based'] - 6, event['start'] + delay + 5,
                        'Rdisc restores missed reference recovery; five-frame criterion')
    return dict(sequence=name, purpose=purpose, reference=reference, windows=windows,
        selected_frames_1based=[i + 1 for i in sorted(selected)],
        trajectory=dict(gt_xywh=gt.tolist(), cover=cover.tolist(), absence=absence.tolist(),
            predictions={a: np.asarray(predictions[a]).tolist() for a in CONFIGS},
            iou={a: v.tolist() for a, v in overlaps.items()}),
        recovery_events=[dict(**e, delay_frames={a: recovery_delay(v, e) for a, v in overlaps.items()})
                         for e in events])


def dump_json(value):
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode('utf-8')


def write_bundle(output, manifest, payloads):
    """Never overwrite old bundles; no input file is edited or extracted."""
    output = Path(output)
    if output.exists():
        raise FileExistsError('refusing to overwrite diagnostic bundle: ' + str(output))
    manifest = dict(manifest)
    if 'manifest.json' in payloads:
        raise ValueError('manifest is reserved')
    for name, payload in payloads.items():
        path = PurePosixPath(name)
        if path.is_absolute() or '..' in path.parts or '\\' in name or not isinstance(payload, bytes):
            raise ValueError('unsafe bundle member: ' + name)
    manifest['members_sha256'] = {k: hashlib.sha256(v).hexdigest() for k, v in payloads.items()}
    # Assemble first; failures during inference/image reading never leave an
    # apparently complete final archive. Exclusive creation handles races.
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode='w:gz') as archive:
        for name, data in {'manifest.json': dump_json(manifest), **payloads}.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as target:
        target.write(stream.getbuffer())


def check_bundle(path):
    """Read and checksum every member without filesystem extraction."""
    with tarfile.open(path, mode='r:gz') as archive:
        members = archive.getmembers()
        names = [m.name for m in members]
        if len(names) != len(set(names)) or any(not m.isfile() for m in members):
            raise ValueError('bundle contains duplicate or non-file members')
        if any(PurePosixPath(n).is_absolute() or '..' in PurePosixPath(n).parts or '\\' in n for n in names):
            raise ValueError('unsafe bundle paths')
        manifest = json.load(archive.extractfile('manifest.json'))
        if manifest.get('version') != 1 or manifest.get('dataset') != 'got10k_vdrm_dev':
            raise ValueError('unknown diagnostic bundle version/dataset')
        expected = manifest['members_sha256']
        if set(names) != set(expected) | {'manifest.json'}:
            raise ValueError('bundle member set differs from manifest')
        for name, digest in expected.items():
            if hashlib.sha256(archive.extractfile(name).read()).hexdigest() != digest:
                raise ValueError('bundle checksum mismatch: ' + name)
        records = json.load(archive.extractfile('records.json'))
        plan = json.load(archive.extractfile('windows.json'))
        expected_keys = {(p['sequence'], f, a) for p in plan for f in p['selected_frames_1based'] for a in CONFIGS}
        actual_keys = [(r['sequence'], r['frame_1based'], r['anchor']) for r in records]
        if len(actual_keys) != len(set(actual_keys)) or set(actual_keys) != expected_keys:
            raise ValueError('missing/duplicate diagnostic frame-anchor records')
        if (len(plan) != len(PLAN) or {p['sequence'] for p in plan} != set(PLAN)
                or manifest['selected_frames'] != sum(len(p['selected_frames_1based']) for p in plan)
                or manifest['selected_frames'] * 3 != len(records) or manifest['records'] != len(records)
                or manifest.get('configurations') != CONFIGS or manifest.get('optimizer_steps') != 0):
            raise ValueError('wrong selected sequence/frame count')
        for arm in CONFIGS:
            provenance = manifest['provenance'][arm]
            if provenance['state_sha256_before'] != provenance['state_sha256_after']:
                raise ValueError('bundle reports changed model state: ' + arm)
        for row in records:
            if set(row['outputs']) != set(CONFIGS) | {'Rdisc_route0', 'Rdisc_alpha0'}:
                raise ValueError('missing diagnostic intervention')
            if not row['alpha0_exact']:
                raise ValueError('failed alpha-zero check in bundle')
            for key in ('image_member', 'crop_member', 'maps_member'):
                if row[key] not in expected:
                    raise ValueError('missing referenced bundle artifact')
            with np.load(io.BytesIO(archive.extractfile(row['maps_member']).read()), allow_pickle=False) as maps:
                if any(not np.isfinite(maps[k]).all() for k in maps.files):
                    raise ValueError('non-finite diagnostic map')
                for arm in row['outputs']:
                    score, route = maps[arm + '_score'], maps[arm + '_route_logits']
                    indices = maps[arm + '_search_global_index']
                    if (score.ndim != 4 or score.shape[:2] != (1, 1) or score.shape[2] != score.shape[3]
                            or route.ndim != 3 or route.shape[:2] != (1, 4)
                            or indices.shape != (1, route.shape[-1])):
                        raise ValueError('wrong score/route/CE map shape')
                if not np.array_equal(maps['Tclean_score'], maps['Rdisc_alpha0_score']):
                    raise ValueError('alpha0 map differs from Tclean')
    return manifest
