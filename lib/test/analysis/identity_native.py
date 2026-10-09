"""GT-free selections and CPU-recomputed evidence on a native baseline trajectory."""
import hashlib
import json
from pathlib import PurePosixPath
import tarfile

import numpy as np
import torch
import torch.nn.functional as F

from lib.models.layers.candidate_quality import dense_center_boxes
from lib.train.identity_offline import ARMS, IdentityMatcher, roi_cells
from lib.test.evaluation.got10kdataset import _vdrm_dev_sequence_ids
from lib.test.analysis.identity_offline import check_bundle as check_screen
from lib.test.analysis.candidate_cues import read_npz
from lib.test.analysis.got10k_annotations import validate_native_annotations

PROTOCOL = dict(version=1, dataset='got10k_vdrm_dev', pools=['top5', 'all256'],
                minimum_cells=3, success_iou=0.5, minimum_hard_frames=30,
                minimum_hard_sequences=10, bootstrap_samples=5000, bootstrap_seed=42,
                normal_cover_minimum=7, reappearance_window=10,
                feedback='Tclean_only', current_gt_in_selection=False,
                optimizer_steps=0, candidate_heads_closed_loop=False)


def load_screen_heads(path, device='cpu'):
    manifest, _ = check_screen(path)
    with tarfile.open(path, 'r:gz') as archive:
        states = {a: read_npz(archive.extractfile('weights_'+a+'.npz').read()) for a in ARMS}
    heads = {}
    for arm, (layout, _) in ARMS.items():
        head = IdentityMatcher(manifest['feature_dim'], layout)
        head.load_state_dict({k: torch.from_numpy(v) for k, v in states[arm].items()}, strict=True)
        heads[arm] = head.requires_grad_(False).to(device).eval()
    return heads, manifest


@torch.no_grad()
def extract_cells(output, template_box, template_padding, search_padding):
    """The v1 raw ROI representation, now for ALL candidates; never takes GT."""
    feature = output['backbone_feat']
    feature = feature[-1] if isinstance(feature, list) else feature
    if feature.ndim != 3 or feature.shape[:2] != (1, 320):
        raise ValueError('requires native 8x8 template and 16x16 search grids')
    dimension = feature.shape[-1]
    z = feature[:, :64].transpose(1, 2).reshape(1, dimension, 8, 8)
    x = feature[:, 64:].transpose(1, 2).reshape(1, dimension, 16, 16)
    kept = torch.ones((1, 256), dtype=torch.bool, device=feature.device)
    for removed in output.get('removed_indexes_s', ()):
        if removed is not None:
            kept.scatter_(1, removed.long(), False)
    support = kept.reshape(1, 1, 16, 16) & ~F.adaptive_max_pool2d(search_padding[:, None].float(), (16, 16)).bool()
    zsupport = ~F.adaptive_max_pool2d(template_padding[:, None].float(), (8, 8)).bool()
    boxes = dense_center_boxes(output['size_map'], output['offset_map'])
    zbox = torch.cat((template_box[:, :2]+template_box[:, 2:]/2, template_box[:, 2:]), 1)[:, None]
    template, zm = roi_cells(z, zbox, zsupport.to(z))
    candidates, xm = roi_cells(x, boxes, support.to(x))
    return boxes, template[:, 0], candidates, zm[:, 0], xm


def candidate_pools(base, common):
    """Stable score order, no GT filtering and no injected positive candidate."""
    order = np.argsort(-base, axis=1, kind='stable')
    top = np.zeros_like(common, dtype=bool)
    np.put_along_axis(top, order[:, :5], True, axis=1)
    supported = common >= PROTOCOL['minimum_cells']
    return dict(top5=top & supported, all256=supported)


def choose_candidates(base, scores, eligible):
    """No GT input. Prefer native on ties; unsupported rows fall back to native."""
    native = base.argmax(1)
    masked = np.where(eligible, scores, -np.inf)
    selected = masked.argmax(1)
    empty = ~eligible.any(1)
    maximum = masked.max(1)
    tied = (masked == maximum[:, None]) & eligible
    prefer_native = tied[np.arange(len(base)), native]
    selected[empty | prefer_native] = native[empty | prefer_native]
    return selected, empty, tied.sum(1) > 1


def overlaps(boxes, gt):
    # Invalid GT has no IoU. Do not turn two negative dimensions into an area,
    # and do not overwrite the raw annotations saved for independent replay.
    valid = (gt[:, 2:] > 0).all(1)
    geometry = np.where(valid[:, None], gt, 0)
    end = np.minimum(boxes[..., :2]+boxes[..., 2:], geometry[:, None, :2]+geometry[:, None, 2:])
    start = np.maximum(boxes[..., :2], geometry[:, None, :2])
    area = np.maximum(end-start, 0).prod(-1)
    union = boxes[..., 2:].prod(-1)+geometry[:, None, 2:].prod(-1)-area
    return np.divide(area, union, out=np.zeros_like(area), where=(union > 0) & valid[:, None])


def sequence_evidence(name, arrays):
    """Validate closure and recompute selections/metrics solely from saved outputs."""
    n = len(arrays['native_boxes'])
    shapes = dict(native_boxes=(n, 4), anchors=(n, 4), boxes=(n, 256, 4),
                  base_scores=(n, 256), common_cells=(n, 256), gt=(n+1, 4),
                  cover=(n+1,), absence=(n+1,))
    shapes.update({a+'_scores': (n, 256) for a in ARMS})
    if n < 1 or set(arrays) != set(shapes):
        raise ValueError('invalid native sequence fields/count')
    for key, shape in shapes.items():
        if arrays[key].shape != shape or not np.isfinite(arrays[key]).all():
            raise ValueError('invalid native array: '+key)
    gt = arrays['gt']
    masks, _ = validate_native_annotations(gt, arrays['cover'], arrays['absence'], name)
    if (arrays['boxes'][..., 2:] <= 0).any():
        raise ValueError('invalid box geometry')
    common = arrays['common_cells']
    if not np.isin(common, np.arange(10)).all() or (arrays['base_scores'] < 0).any():
        raise ValueError('invalid candidate support/base response')
    if any((np.abs(arrays[a+'_scores']) > 1.00001).any() for a in ARMS):
        raise ValueError('identity score outside cosine range')
    expected_anchors = np.concatenate((gt[:1], arrays['native_boxes'][:-1]))
    if not np.array_equal(arrays['anchors'], expected_anchors):
        raise ValueError('broken native float trajectory; GT reset or state mutation')
    base = arrays['base_scores']
    native = base.argmax(1)
    if not np.array_equal(arrays['boxes'][np.arange(n), native], arrays['native_boxes']):
        raise ValueError('all-candidate decode differs from actual native output')
    visible = ((gt[1:, 2:] > 0).all(1) & (arrays['cover'][1:] > 0) & (arrays['absence'][1:] == 0))
    stages = dict(visible=visible, normal=visible & (arrays['cover'][1:] >= 7),
                  severe_partial=visible & masks['severe_partial'][1:],
                  reappearance=visible & masks['reappearance'][1:])
    iou = overlaps(arrays['boxes'], gt[1:])
    base_iou = iou[np.arange(n), native]
    base_correct = base_iou >= .5
    pools = candidate_pools(base, common)
    good = iou >= .5
    all_good = good.any(1)
    supported_good = (good & pools['all256']).any(1)
    top_good = (good & pools['top5']).any(1)
    counts = dict(frames=n, visible=int(visible.sum()), invisible_or_invalid=int((~visible).sum()),
        baseline_wrong=int((visible & ~base_correct).sum()),
        no_good_box=int((visible & ~all_good).sum()),
        good_box_but_no_supported_good=int((visible & all_good & ~supported_good).sum()),
        supported_good_outside_top5=int((visible & supported_good & ~top_good).sum()))
    result = dict(sequence=name, counts=counts, pools={})
    for pool, eligible in pools.items():
        hard = visible & ~base_correct & (good & eligible).any(1)
        pool_result = dict(recoverable_hard=int(hard.sum()), heads={})
        for arm in ARMS:
            selected, fallback, ties = choose_candidates(base, arrays[arm+'_scores'], eligible)
            chosen_iou = iou[np.arange(n), selected]
            correct = chosen_iou >= .5
            repaired = visible & ~base_correct & correct
            damaged = visible & base_correct & ~correct
            def frame_ids(mask):
                return (np.flatnonzero(mask)+2).tolist()
            stage_values = {}
            for stage, mask in stages.items():
                stage_values[stage] = dict(frames=int(mask.sum()),
                    base_ao=float(base_iou[mask].mean()) if mask.any() else None,
                    head_ao=float(chosen_iou[mask].mean()) if mask.any() else None,
                    delta_ao_pp=float((chosen_iou[mask]-base_iou[mask]).mean()*100) if mask.any() else None)
            pool_result['heads'][arm] = dict(stages=stage_values,
                repairs=frame_ids(repaired), damages=frame_ids(damaged),
                hard_repaired=int((hard & correct).sum()),
                fallbacks=int(fallback.sum()), ties=int(ties.sum()),
                selected_indices=selected.tolist())
        result['pools'][pool] = pool_result
    return result


def interval(values):
    if not values:
        return dict(sequences=0, delta_pp=None, ci95_pp=None)
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(PROTOCOL['bootstrap_seed'])
    draws = rng.choice(values, (PROTOCOL['bootstrap_samples'], len(values)), replace=True).mean(1)
    return dict(sequences=len(values), delta_pp=float(values.mean()),
                ci95_pp=np.quantile(draws, [.025, .975]).tolist() if len(values) > 1 else None)


def summarize(records):
    totals = {k: sum(r['counts'][k] for r in records) for k in records[0]['counts']}
    report = dict(sequences=len(records), counts=totals, pools={}, candidate_heads_closed_loop=False,
                  automatic_integration_approved=False)
    for pool in PROTOCOL['pools']:
        hard = sum(r['pools'][pool]['recoverable_hard'] for r in records)
        hard_sequences = sum(r['pools'][pool]['recoverable_hard'] > 0 for r in records)
        entry = dict(recoverable_hard=hard, hard_sequences=hard_sequences,
            hard_evidence_sufficient=hard >= PROTOCOL['minimum_hard_frames'] and hard_sequences >= PROTOCOL['minimum_hard_sequences'], heads={})
        for arm in ARMS:
            heads = [r['pools'][pool]['heads'][arm] for r in records]
            stages = {}
            for stage in heads[0]['stages']:
                values = [h['stages'][stage] for h in heads if h['stages'][stage]['frames']]
                stages[stage] = dict(frames=sum(v['frames'] for v in values),
                    base_ao_percent=float(np.mean([v['base_ao'] for v in values])*100) if values else None,
                    head_ao_percent=float(np.mean([v['head_ao'] for v in values])*100) if values else None,
                    **interval([v['delta_ao_pp'] for v in values]))
            entry['heads'][arm] = dict(stages=stages, repairs=sum(len(h['repairs']) for h in heads),
                damages=sum(len(h['damages']) for h in heads), hard_repaired=sum(h['hard_repaired'] for h in heads),
                fallbacks=sum(h['fallbacks'] for h in heads), ties=sum(h['ties'] for h in heads))
        entry['comparisons'] = {}
        for first, second in (('mean_id', 'mean_rank'), ('spatial_id', 'mean_id'), ('spatial_id', 'mean_rank')):
            stages = {}
            for stage in ('visible', 'normal', 'severe_partial', 'reappearance'):
                values = []
                for record in records:
                    a = record['pools'][pool]['heads'][first]['stages'][stage]
                    b = record['pools'][pool]['heads'][second]['stages'][stage]
                    if a['frames']:
                        values.append(a['delta_ao_pp']-b['delta_ao_pp'])
                stages[stage] = interval(values)
            entry['comparisons'][first+'_minus_'+second] = stages
        report['pools'][pool] = entry
    return report


def render_report(summary):
    lines = ['# Native-trajectory identity probe', '',
        'Tclean alone advances state. Identity heads are SAME-CROP counterfactuals, NOT closed-loop trackers.',
        'All frames retained. GT used for initialization and post-inference labels only.',
        f"Sequences: {summary['sequences']}/152; counts: {summary['counts']}", '']
    for pool, data in summary['pools'].items():
        lines += [f"## {pool}", '', f"Recoverable hard: {data['recoverable_hard']} frames / {data['hard_sequences']} sequences; sufficient: {data['hard_evidence_sufficient']}", '',
                  '| Head | Repairs | Damages | Visible AO delta pp [95% CI] | Normal AO delta pp |',
                  '| --- | ---: | ---: | --- | --- |']
        for arm, value in data['heads'].items():
            stage = value['stages']
            lines.append(f"| {arm} | {value['repairs']} | {value['damages']} | {stage['visible']['delta_pp']} {stage['visible']['ci95_pp']} | {stage['normal']['delta_pp']} |")
        lines.append('')
        for name, stages in data['comparisons'].items():
            lines.append(f"{name}, visible AO: {stages['visible']}")
        lines.append('')
    lines.append('No automatic integration, new training, identity-switch claim, or closed-loop recovery claim.\n')
    return '\n'.join(lines)


def check_bundle(path):
    """Stream sequence shards; no original checkpoint/dataset/GPU access."""
    with tarfile.open(path, 'r:gz') as archive:
        members = archive.getmembers()
        names = [m.name for m in members]
        if len(names) != len(set(names)) or any(not m.isfile() or PurePosixPath(m.name).is_absolute() or
                '..' in PurePosixPath(m.name).parts or '\\' in m.name for m in members):
            raise ValueError('unsafe native identity archive')
        manifest = json.load(archive.extractfile('manifest.json'))
        ids = _vdrm_dev_sequence_ids()
        sequences = manifest['sequences']
        if (manifest.get('kind') != 'vdrm_identity_native' or manifest.get('protocol') != PROTOCOL or
                [s['sequence_id'] for s in sequences] != ids or
                any(s['name'] != f'GOT-10k_Train_{s["sequence_id"]+1:06d}' for s in sequences) or
                manifest.get('optimizer_steps') != 0 or manifest.get('parameter_gradient_tensors') != 0 or
                set(manifest['states_before']) != {'Tclean', *ARMS} or
                manifest['states_before'] != manifest['states_after']):
            raise ValueError('native identity provenance/partition contract differs')
        for key in ('source_screen_sha256', 'checkpoint_sha256', 'data_list_sha256'):
            digest = manifest[key]
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
                raise ValueError('invalid source hash')
        by_member = {s['member']: s for s in sequences}
        expected = set(by_member) | {'records.json', 'summary.json', 'report.md'}
        if len(by_member) != len(ids) or set(names) != expected | {'manifest.json'} or set(manifest['members_sha256']) != expected:
            raise ValueError('native identity member set differs')
        records, documents = [], {}
        for member in members:
            if member.name == 'manifest.json':
                continue
            value = archive.extractfile(member).read()
            if hashlib.sha256(value).hexdigest() != manifest['members_sha256'][member.name]:
                raise ValueError('native identity checksum differs: '+member.name)
            if member.name in by_member:
                spec = by_member[member.name]
                arrays = read_npz(value)
                if len(arrays['native_boxes']) != spec['frames']-1 or spec['exact_replay_frames'] != [2]:
                    raise ValueError('incomplete native frame coverage/replay')
                records.append(sequence_evidence(spec['name'], arrays))
            else:
                documents[member.name] = value
        records.sort(key=lambda r: r['sequence'])
        summary = summarize(records)
        if records != json.loads(documents['records.json']) or summary != json.loads(documents['summary.json']):
            raise ValueError('native identity independent metric recomputation differs')
        if render_report(summary) != documents['report.md'].decode('utf-8'):
            raise ValueError('native identity report differs')
        if manifest['native_tracking_steps'] != summary['counts']['frames']:
            raise ValueError('native tracking count differs')
    return manifest, summary
