"""Model-free checks of projected descriptors from the offline identity screen."""
import hashlib
import json
from pathlib import PurePosixPath
import tarfile

import numpy as np
import torch

from lib.train.identity_offline import (IdentityMatcher, strict_top1, summarize_evaluation, protocol_hash,
                                       validate_protocol, registered_ids, ARMS)
from .candidate_cues import read_npz


def donor_indices(rows):
    first = {}
    for index, row in enumerate(rows):
        first.setdefault(row['sequence_id'],index)
    identities = sorted(first)
    output = []
    for row in rows:
        if len(identities)<5:
            output.append([-1]*4)
            continue
        position = identities.index(row['sequence_id'])
        output.append([first[identities[(position+k)%len(identities)]] for k in range(1,5)])
    return np.asarray(output,dtype=np.int64)


def evaluate_embeddings(rows, arrays, protocol):
    outcomes, evidence = {}, {}
    donors = donor_indices(rows)
    for mode in ARMS:
        values = {k:torch.from_numpy(arrays[mode+'_'+k]) for k in ('z','x','zm','xm')}
        z,x,zm,xm = (values[k] for k in ('z','x','zm','xm'))
        scores = IdentityMatcher.compare(z[:,None],x,zm[:,None],xm)
        wrong = torch.full((len(rows),4),-torch.inf)
        eligible = torch.ones(len(rows),dtype=torch.bool)
        raw_zm,raw_xm = arrays['template_mask'],arrays['candidate_mask']
        for i,selected in enumerate(donors):
            if (selected<0).any():
                eligible[i] = False
                continue
            for k,j in enumerate(selected):
                if int((raw_zm[j] & raw_xm[i,0]).sum())<protocol['minimum_cells']:
                    eligible[i] = False
                wrong[i,k] = IdentityMatcher.compare(z[j],x[i,0],zm[j],xm[i,0])
        correct = strict_top1(scores)
        template_correct = (scores[:,0] > wrong.max(1).values) & eligible
        outcomes[mode] = dict(candidate_correct=correct.tolist(),template_correct=template_correct.tolist(),
                              template_eligible=eligible.tolist())
        evidence[mode] = dict(scores=scores.tolist(),donor_scores=wrong.tolist())
    # Null rather than non-standard JSON infinities for unavailable donor sets.
    for value in evidence.values():
        value['donor_scores'] = [[v if np.isfinite(v) else None for v in row] for row in value['donor_scores']]
    return outcomes,evidence,donors.tolist()


def render_report(summary):
    lines = ['# Offline identity screen','',
        'GT-assisted cached pairs; NOT closed-loop tracking, reliability calibration, or an identity-switch benchmark.',
        'Ties count as failures. Ratios are sequence macro means; intervals bootstrap sequences.', '',
        f"Dev: {summary['development_pairs']} pairs / {summary['development_sequences']} eligible sequences; "
        f"hard pairs: {summary['hard_pairs']}; sufficient evidence: {summary['evidence_sufficient']}", '',
        '| Head | Candidate accuracy | Own-template top1 | Delta vs base pp | Offline gate |',
        '| --- | ---: | ---: | ---: | --- |']
    def fmt(value):
        return 'N/A' if value is None else f'{value:.4f}'
    for mode,row in summary['heads'].items():
        lines.append(f"| {mode} | {fmt(row['candidate_accuracy'])} | {fmt(row['template_accuracy'])} | "
                     f"{fmt(row['versus_baseline']['delta_pp'])} | {row['offline_gate_passed']} |")
    lines += ['',f"Identity-supervision minus rank-only template retrieval: {summary['identity_minus_rank_template']}",
        f"Evidence for identity supervision: {summary['identity_supervision_supported']}",
        f"Spatial-minus-mean (same identity supervision): {summary['spatial_minus_mean']}",
        f"Evidence for added spatial complexity: {summary['spatial_complexity_supported']}",
        'No automatic tracker integration, epoch extension, or joint-model approval.','']
    return '\n'.join(lines)


def check_bundle(path):
    with tarfile.open(path,'r:gz') as archive:
        members = archive.getmembers()
        names = [m.name for m in members]
        if len(names)!=len(set(names)) or any(not m.isfile() or PurePosixPath(m.name).is_absolute() or
                '..' in PurePosixPath(m.name).parts or '\\' in m.name for m in members):
            raise ValueError('unsafe identity result bundle')
        payloads = {m.name:archive.extractfile(m).read() for m in members}
    manifest = json.loads(payloads.pop('manifest.json'))
    expected = {'protocol.json','rows.json','embeddings.npz','outcomes.json','evidence.json','donors.json',
                'summary.json','report.md','training.json'} | {'weights_'+arm+'.npz' for arm in ARMS}
    if set(payloads)!=expected or manifest.get('kind')!='vdrm_identity_screen' or manifest.get('version')!=1:
        raise ValueError('wrong identity result contract')
    if manifest['members_sha256']!={k:hashlib.sha256(v).hexdigest() for k,v in payloads.items()}:
        raise ValueError('identity result checksum differs')
    protocol = validate_protocol(json.loads(payloads['protocol.json']))
    if manifest['protocol_sha256']!=protocol_hash(protocol):
        raise ValueError('identity result protocol differs')
    train_ids,dev_ids = registered_ids(protocol)
    if (manifest['train_ids']!=train_ids or manifest['dev_ids']!=dev_ids or manifest['epochs']!=20 or
            manifest.get('backbone_loaded_during_training') is not False):
        raise ValueError('identity result training/partition contract differs')
    rows = json.loads(payloads['rows.json'])
    if not rows or any(r['split']!='dev' or r['sequence_id'] not in manifest['dev_ids'] for r in rows):
        raise ValueError('identity evaluation rows are not development data')
    if set(manifest['train_ids']) & set(manifest['dev_ids']):
        raise ValueError('training/development overlap')
    if len(rows)!=manifest['dev_pairs'] or len({(r['sequence_id'],r['frame_1based']) for r in rows})!=len(rows):
        raise ValueError('identity result duplicate/missing development rows')
    for mode in ARMS:
        state = read_npz(payloads['weights_'+mode+'.npz'])
        if (set(state)!={'project.weight'} or state['project.weight'].shape!=(32,manifest['feature_dim']) or
                state['project.weight'].dtype!=np.float32 or not np.isfinite(state['project.weight']).all()):
            raise ValueError('invalid exported identity head')
    training = json.loads(payloads['training.json'])
    if set(training)!=set(ARMS) or len({r['initial_sha256'] for r in training.values()})!=1:
        raise ValueError('identity initialization provenance differs')
    if len({r['sample_order_sha256'] for r in training.values()})!=1:
        raise ValueError('identity sample order differs')
    steps = ((manifest['train_pairs']+protocol['batch_size']-1)//protocol['batch_size'])*protocol['epochs']
    for mode,stats in training.items():
        audit = stats['audit']
        if (stats['include_identity']!=ARMS[mode][1] or [h['epoch'] for h in stats['history']]!=list(range(1,21)) or stats['optimizer_steps']!=steps or
                audit['feature_gradient_tensors']!=0 or audit['optimizer_steps']!=0 or
                not np.isfinite(audit['head_gradient_l2']) or audit['head_gradient_l2']<=0 or audit['identity_queries']<2):
            raise ValueError('identity training/audit budget differs')
    arrays = read_npz(payloads['embeddings.npz'])
    keys = {m+'_'+k for m in ARMS for k in ('z','x','zm','xm')} | {'template_mask','candidate_mask'}
    if set(arrays)!=keys:
        raise ValueError('unexpected identity descriptors')
    n = len(rows)
    for mode,(layout,_) in ARMS.items():
        cells = 1 if layout=='mean' else 9
        shapes = dict(z=(n,cells,32),x=(n,5,cells,32),zm=(n,cells),xm=(n,5,cells))
        for key,shape in shapes.items():
            value = arrays[mode+'_'+key]
            if value.shape!=shape or not np.isfinite(value).all() or value.dtype!=(np.bool_ if key.endswith('m') else np.float32):
                raise ValueError('invalid identity descriptor: '+mode+'_'+key)
    for name,shape in (('template_mask',(n,9)),('candidate_mask',(n,5,9))):
        if arrays[name].shape!=shape or arrays[name].dtype!=np.bool_:
            raise ValueError('invalid identity support mask')
    for prefix,raw in (('z','template_mask'),('x','candidate_mask')):
        if (not np.array_equal(arrays['spatial_id_'+prefix+'m'],arrays[raw]) or
                any(not np.array_equal(arrays[arm+'_'+prefix+'m'],arrays[raw].any(-1,keepdims=True)) for arm in ('mean_rank','mean_id'))):
            raise ValueError('projected support differs from frozen support')
    outcomes,evidence,donors = evaluate_embeddings(rows,arrays,protocol)
    summary = summarize_evaluation(rows,outcomes,protocol,manifest['train_pairs'])
    for name,value in (('outcomes.json',outcomes),('evidence.json',evidence),('donors.json',donors),('summary.json',summary)):
        if value!=json.loads(payloads[name]):
            raise ValueError('independent identity recomputation differs: '+name)
    if render_report(summary)!=payloads['report.md'].decode('utf-8'):
        raise ValueError('identity report differs')
    return manifest,summary
