"""Standalone identity screen on frozen ROI caches, never a tracking policy."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from lib.test.evaluation.got10kdataset import _vdrm_dev_sequence_ids
from lib.train.frozen_vdrm import file_hash

PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = PROJECT / 'experiments/vdrm_identity/offline_v1.json'
ARMS = {'mean_rank': ('mean', False), 'mean_id': ('mean', True), 'spatial_id': ('spatial', True)}


def load_protocol(path=DEFAULT_PROTOCOL):
    return validate_protocol(json.loads(Path(path).read_text(encoding='utf-8')))


def validate_protocol(value):
    required = set(json.loads(DEFAULT_PROTOCOL.read_text(encoding='utf-8')))
    if set(value) != required or any(not isinstance(v, (int, float)) or isinstance(v, bool)
                                   or not np.isfinite(v) for v in value.values()):
        raise ValueError('invalid identity protocol keys/values')
    if any(value[k] != v for k, v in dict(version=1, seed=42, candidates=5, roi_grid=3,
                                         projection_dim=32, epochs=20).items()):
        raise ValueError('identity screen requires version 1, internal seed 42 and registered dimensions/budget')
    integers = ('version', 'seed', 'candidates', 'roi_grid', 'projection_dim', 'epochs',
                'train_sequences', 'frames_per_sequence', 'batch_size', 'minimum_cells',
                'minimum_train_pairs', 'minimum_dev_pairs', 'minimum_dev_sequences', 'minimum_hard_pairs')
    if any(type(value[k]) is not int or value[k] < 1 for k in integers):
        raise ValueError('positive integer identity counts required')
    if value['batch_size'] < 2 or not 1 <= value['minimum_cells'] <= 9:
        raise ValueError('invalid identity batch/cell count')
    if not 0 <= value['negative_iou'] < value['positive_iou'] <= 1:
        raise ValueError('invalid candidate IoU thresholds')
    if any(value[k] <= 0 for k in ('learning_rate', 'temperature', 'identity_weight')):
        raise ValueError('positive learning rate, temperature and identity weight required')
    if value['weight_decay'] < 0 or not 0 <= value['minimum_template_accuracy'] <= 1:
        raise ValueError('invalid identity regularization/gate')
    if min(value['normal_tolerance_pp'], value['minimum_hard_gain_pp']) < 0:
        raise ValueError('nonnegative performance margins required')
    return value


def protocol_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def registered_ids(protocol):
    spec = PROJECT / 'lib/train/data_specs/got10k_vot_train_split.txt'
    train = sorted({int(x) for x in spec.read_text().split()})
    dev = _vdrm_dev_sequence_ids()
    if set(train) & set(dev) or protocol['train_sequences'] > len(train):
        raise ValueError('invalid/overlapping identity sequence partition')
    rng = np.random.default_rng(protocol['seed'])
    return sorted(rng.choice(train, protocol['train_sequences'], replace=False).tolist()), dev


@torch.no_grad()
def roi_cells(features, boxes, support):
    """Keep all nine normalized spatial samples instead of averaging them."""
    axis = (torch.arange(3, device=features.device, dtype=features.dtype) + .5) / 3
    yy, xx = torch.meshgrid(axis, axis, indexing='ij')
    points = torch.stack((xx, yy), -1).reshape(1, 1, 9, 2)
    grid = (boxes[:, :, None, :2] + (points-.5)*boxes[:, :, None, 2:])*2-1
    weights = F.grid_sample(support, grid, align_corners=False)
    samples = F.grid_sample(features.detach()*support, grid, align_corners=False)
    values = (samples/weights.clamp_min(1e-6)).permute(0, 2, 3, 1)
    masks = weights[:, 0] > 1e-6
    return values.masked_fill(~masks[..., None], 0).detach(), masks


class IdentityMatcher(nn.Module):
    def __init__(self, dimension, mode, projection_dim=32):
        super().__init__()
        if mode not in ('mean', 'spatial'):
            raise ValueError('unknown identity mode')
        self.mode = mode
        self.project = nn.Linear(dimension, projection_dim, bias=False)

    def encode(self, cells, valid):
        cells, valid = cells.detach(), valid.detach()
        if cells.shape[-2] != 9 or valid.shape != cells.shape[:-1]:
            raise ValueError('identity descriptors require nine aligned cells')
        if self.mode == 'mean':
            cells = (cells*valid[..., None]).sum(-2, keepdim=True)/valid.sum(-1, keepdim=True)[..., None].clamp_min(1)
            valid = valid.any(-1, keepdim=True)
        features = F.layer_norm(cells, (cells.shape[-1],))
        return F.normalize(self.project(features), dim=-1), valid

    @staticmethod
    def compare(template, candidate, template_mask, candidate_mask):
        mask = template_mask & candidate_mask
        return ((template*candidate).sum(-1)*mask).sum(-1)/mask.sum(-1).clamp_min(1)

    def forward(self, template, candidates, template_mask, candidate_mask):
        z, zm = self.encode(template, template_mask)
        x, xm = self.encode(candidates, candidate_mask)
        return self.compare(z[:, None], x, zm[:, None], xm)


def losses(model, batch, protocol, include_identity=True):
    scores = model(batch['template'], batch['candidates'], batch['template_mask'], batch['candidate_mask'])
    rank = F.cross_entropy(scores/protocol['temperature'], torch.zeros(len(scores), dtype=torch.long, device=scores.device))
    z, zm = model.encode(batch['template'], batch['template_mask'])
    x, xm = model.encode(batch['candidates'][:, 0], batch['candidate_mask'][:, 0])
    cross = model.compare(z[:, None], x[None], zm[:, None], xm[None])
    # A different crop from the SAME sequence must never be a negative.
    same = batch['sequence_id'][:, None] == batch['sequence_id'][None]
    diagonal = torch.eye(len(scores), device=scores.device, dtype=torch.bool)
    supported = (batch['template_mask'][:, None] & batch['candidate_mask'][None,:,0]).sum(-1)>=protocol['minimum_cells']
    allowed = ((~same) & supported) | diagonal
    active = (allowed & ~diagonal).any(1)
    if active.any():
        logits = (cross/protocol['temperature']).masked_fill(~allowed, -torch.inf)
        target = torch.arange(len(scores), device=scores.device)
        # Symmetric negatives use their respective eligible query rows.
        reverse_active = (allowed.T & ~diagonal).any(1)
        identity = (F.cross_entropy(logits[active], target[active]) +
                    F.cross_entropy(logits.T[reverse_active], target[reverse_active]))/2
    else:
        identity = scores.sum()*0
    total = rank + (protocol['identity_weight'] if include_identity else 0.)*identity
    if not torch.isfinite(total):
        raise ValueError('nonfinite identity objective')
    return total, dict(rank=float(rank.detach()), identity=float(identity.detach()),
                      identity_queries=int(active.sum()))


def read_cache(directory):
    directory = Path(directory)
    index = json.loads((directory/'index.json').read_text(encoding='utf-8'))
    protocol = load_protocol(directory/'protocol.json')
    train_ids, dev_ids = registered_ids(protocol)
    if (index.get('kind') != 'vdrm_identity_cache' or index.get('version') != 1 or
            index.get('protocol_sha256') != protocol_hash(protocol) or
            index.get('train_ids') != train_ids or index.get('dev_ids') != dev_ids or
            index.get('optimizer_steps') != 0 or index.get('state_before') != index.get('state_after')):
        raise ValueError('identity cache provenance/partition differs')
    rows = index['rows']
    keys = [(r['sequence_id'], r['frame_1based']) for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError('duplicate cached pair')
    shards = {}
    for member, digest in index['members_sha256'].items():
        if Path(member).name != member or '/' in member or '\\' in member or not member.endswith('.npz'):
            raise ValueError('unsafe identity cache member')
        if file_hash(directory/member) != digest:
            raise ValueError('identity cache checksum differs: '+member)
        with np.load(directory/member, allow_pickle=False) as archive:
            shards[member] = {k: archive[k] for k in archive.files}
    tensors = dict(template=[], candidates=[], template_mask=[], candidate_mask=[])
    used = set()
    for row in rows:
        partition = train_ids if row['split']=='train' else dev_ids if row['split']=='dev' else []
        if row['sequence_id'] not in partition:
            raise ValueError('cached pair violates train/dev boundary')
        key = (row['member'], row['offset'])
        if key in used:
            raise ValueError('duplicate cache row reference')
        used.add(key)
        shard = shards[row['member']]
        if set(shard) != set(tensors):
            raise ValueError('unexpected cache feature/label fields')
        dim = index['feature_dim']
        shapes = dict(template=(9,dim), candidates=(5,9,dim), template_mask=(9,), candidate_mask=(5,9))
        for name in tensors:
            value = shard[name][row['offset']]
            dtype = np.bool_ if name.endswith('mask') else np.float32
            if value.shape != shapes[name] or value.dtype != dtype or not np.isfinite(value).all():
                raise ValueError('invalid cached identity feature: '+name)
            tensors[name].append(value)
        mask = tensors['candidate_mask'][-1] & tensors['template_mask'][-1][None]
        if (mask.sum(-1) < protocol['minimum_cells']).any():
            raise ValueError('cached pair has insufficient common support')
        overlaps = np.asarray(row['ious'])
        if overlaps.shape != (5,) or not np.isfinite(overlaps).all() or (overlaps<0).any() or (overlaps>1).any() or overlaps[0] < protocol['positive_iou'] or (overlaps[1:] > protocol['negative_iou']).any():
            raise ValueError('cached positive/negative labels are inconsistent')
        base_scores = np.asarray(row['base_scores'])
        if base_scores.shape!=(5,) or not np.isfinite(base_scores).all() or (base_scores<0).any():
            raise ValueError('invalid baseline candidate scores')
        if not np.isfinite(row['cover']) or not 0<=row['cover']<=8:
            raise ValueError('invalid cached visibility label')
    if used != {(name, i) for name, shard in shards.items() for i in range(len(shard['template']))}:
        raise ValueError('unreferenced cached features')
    if not rows:
        raise ValueError('cache contains no eligible pairs; inspect index skip reasons')
    data = {k: torch.from_numpy(np.stack(v)) for k,v in tensors.items()}
    data['sequence_id'] = torch.tensor([r['sequence_id'] for r in rows])
    return index, protocol, data


def strict_top1(scores):
    """Ties are failures, never a benefit of storing the positive first."""
    return scores[:, 0] > scores[:, 1:].max(1).values


def sequence_mean(values, rows, mask=None):
    values = np.asarray(values, dtype=float)
    mask = np.ones(len(rows), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    ids = np.array([r['sequence_id'] for r in rows])
    groups = [values[(ids==s)&mask].mean() for s in sorted(set(ids[mask]))]
    return float(np.mean(groups)) if groups else None


def paired_interval(first, second, rows, mask=None):
    ids = np.array([r['sequence_id'] for r in rows])
    mask = np.ones(len(rows),dtype=bool) if mask is None else np.asarray(mask,dtype=bool)
    delta = np.asarray(first,dtype=float)-np.asarray(second,dtype=float)
    grouped = np.array([delta[(ids==s)&mask].mean() for s in sorted(set(ids[mask]))])
    if not len(grouped):
        return dict(sequences=0, delta_pp=None, ci95_pp=None)
    rng = np.random.default_rng(42)
    samples = rng.choice(grouped, (5000,len(grouped)), replace=True).mean(1)
    return dict(sequences=len(grouped),delta_pp=float(grouped.mean()*100),
                ci95_pp=(np.quantile(samples,[.025,.975])*100).tolist() if len(grouped)>=2 else None)


def summarize_evaluation(rows, outcomes, protocol, train_count):
    base = np.array([r['base_scores'][0] > max(r['base_scores'][1:]) for r in rows])
    normal = np.array([r['cover']>=7 for r in rows])
    hard = ~base
    enough = (train_count>=protocol['minimum_train_pairs'] and len(rows)>=protocol['minimum_dev_pairs'] and
              len({r['sequence_id'] for r in rows})>=protocol['minimum_dev_sequences'] and
              int(hard.sum())>=protocol['minimum_hard_pairs'] and normal.any())
    report = dict(development_pairs=len(rows), development_sequences=len({r['sequence_id'] for r in rows}),
                  hard_pairs=int(hard.sum()), normal_pairs=int(normal.sum()), evidence_sufficient=bool(enough),
                  baseline_accuracy=sequence_mean(base, rows), heads={}, closed_loop=False)
    for name, data in outcomes.items():
        correct = np.asarray(data['candidate_correct'],dtype=bool)
        donor = np.asarray(data['template_correct'],dtype=bool)
        donor_valid = np.asarray(data['template_eligible'],dtype=bool)
        identity = sequence_mean(donor,rows,donor_valid)
        delta = paired_interval(correct,base,rows)
        normal_delta = paired_interval(correct,base,rows,normal)
        hard_delta = paired_interval(correct,base,rows,hard)
        donor_sequences = len({r['sequence_id'] for r,valid in zip(rows,donor_valid) if valid})
        supported = (enough and donor_valid.sum()>=protocol['minimum_dev_pairs'] and
                     donor_sequences>=protocol['minimum_dev_sequences'])
        passed = bool(supported and identity is not None and identity>=protocol['minimum_template_accuracy'] and
                      delta['delta_pp']>=-protocol['normal_tolerance_pp'] and
                      normal_delta['delta_pp']>=-protocol['normal_tolerance_pp'] and
                      hard_delta['delta_pp']>=protocol['minimum_hard_gain_pp'])
        report['heads'][name] = dict(candidate_accuracy=sequence_mean(correct,rows),template_accuracy=identity,
            template_queries=int(donor_valid.sum()), template_sequences=donor_sequences,
            versus_baseline=delta, normal_versus_baseline=normal_delta,
            hard_versus_baseline=hard_delta, offline_gate_passed=passed)
    comparison = paired_interval(outcomes['spatial_id']['candidate_correct'],outcomes['mean_id']['candidate_correct'],rows)
    normal_comparison = paired_interval(outcomes['spatial_id']['candidate_correct'],outcomes['mean_id']['candidate_correct'],rows,normal)
    identity_common = np.asarray(outcomes['mean_rank']['template_eligible']) & np.asarray(outcomes['mean_id']['template_eligible'])
    identity_comparison = paired_interval(outcomes['mean_id']['template_correct'],outcomes['mean_rank']['template_correct'],rows,identity_common)
    identity_candidate_comparison = paired_interval(outcomes['mean_id']['candidate_correct'],outcomes['mean_rank']['candidate_correct'],rows)
    identity_normal_comparison = paired_interval(outcomes['mean_id']['candidate_correct'],outcomes['mean_rank']['candidate_correct'],rows,normal)
    report.update(spatial_minus_mean=comparison, spatial_minus_mean_normal=normal_comparison,
        identity_minus_rank_template=identity_comparison, identity_minus_rank_candidate=identity_candidate_comparison,
        identity_minus_rank_normal=identity_normal_comparison,
        identity_supervision_supported=bool(enough and report['heads']['mean_id']['offline_gate_passed'] and
            identity_comparison['ci95_pp'] is not None and identity_comparison['ci95_pp'][0]>0 and
            identity_candidate_comparison['delta_pp']>=-protocol['normal_tolerance_pp'] and
            identity_normal_comparison['delta_pp']>=-protocol['normal_tolerance_pp']),
        spatial_complexity_supported=bool(enough and report['heads']['spatial_id']['offline_gate_passed'] and
            comparison['ci95_pp'] is not None and comparison['ci95_pp'][0]>0 and
            normal_comparison['delta_pp']>=-protocol['normal_tolerance_pp']))
    return report
