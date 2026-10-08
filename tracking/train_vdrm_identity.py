"""Train three controlled identity matchers on TRAIN caches; never load a tracker."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))

from lib.train.identity_offline import read_cache, IdentityMatcher, losses, protocol_hash, ARMS
from lib.train.frozen_vdrm import file_hash
from lib.test.analysis.candidate_cues import npz_bytes
from lib.test.analysis.frozen_failure_diagnostics import write_bundle, dump_json
from lib.test.analysis.identity_offline import evaluate_embeddings, check_bundle, render_report
from lib.train.identity_offline import summarize_evaluation


def audit_head(model,batch,protocol,include_identity=True):
    inputs = {k:v.detach().clone().requires_grad_(True) if k in ('template','candidates') else v for k,v in batch.items()}
    value,status = losses(model,inputs,protocol,include_identity=include_identity)
    params = list(model.parameters())
    gradients = torch.autograd.grad(value,[inputs['template'],inputs['candidates'],*params],allow_unused=True)
    if any(g is not None for g in gradients[:2]) or not any(g is not None and g.abs().sum()>0 for g in gradients[2:]):
        raise RuntimeError('identity head gradient isolation failed')
    if status['identity_queries']<2:
        raise RuntimeError('preflight lacks supported different-sequence identity negatives')
    return dict(feature_gradient_tensors=0,head_gradient_l2=float(sum(g.square().sum() for g in gradients[2:] if g is not None).sqrt()),
                identity_queries=status['identity_queries'],optimizer_steps=0)


def train_head(mode,data,ids,protocol,dimension,device):
    torch.manual_seed(protocol['seed'])
    layout,include_identity = ARMS[mode]
    model = IdentityMatcher(dimension,layout,protocol['projection_dim']).to(device)
    initial = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    optimizer = torch.optim.AdamW(model.parameters(),lr=protocol['learning_rate'],weight_decay=protocol['weight_decay'])
    expected_ids = set(data['sequence_id'][ids].tolist())
    def batch(indices):
        value = {k:v[indices].to(device) for k,v in data.items()}
        if not set(value['sequence_id'].tolist())<=expected_ids:
            raise RuntimeError('development data entered identity optimizer')
        return value
    audit = audit_head(model,batch(ids[:protocol['batch_size']]),protocol,include_identity)
    generator = torch.Generator().manual_seed(protocol['seed'])
    history,steps = [],0
    sample_order = hashlib.sha256()
    for epoch in range(1,protocol['epochs']+1):
        order = ids[torch.randperm(len(ids),generator=generator)]
        totals = dict(loss=0.,rank=0.,identity=0.,identity_queries=0,samples=0)
        for indices in order.split(protocol['batch_size']):
            sample_order.update(indices.numpy().tobytes())
            optimizer.zero_grad(set_to_none=True)
            value,status = losses(model,batch(indices),protocol,include_identity=include_identity)
            value.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise RuntimeError('nonfinite identity gradients')
            optimizer.step();steps+=1
            count = len(indices)
            totals['loss']+=float(value.detach())*count
            for key in ('rank','identity'):
                totals[key]+=status[key]*count
            totals['identity_queries']+=status['identity_queries'];totals['samples']+=count
        history.append(dict(epoch=epoch,**{k:v/len(ids) if k in ('loss','rank','identity') else v for k,v in totals.items()}))
        print(f"{mode} epoch {epoch:02d}/{protocol['epochs']}: loss={history[-1]['loss']:.6f}",flush=True)
    optimizer.zero_grad(set_to_none=True)
    return model.eval().requires_grad_(False),dict(audit=audit,history=history,optimizer_steps=steps,include_identity=include_identity,
        sample_order_sha256=sample_order.hexdigest(),
        initial_sha256=hashlib.sha256(b''.join(initial[k].numpy().tobytes() for k in sorted(initial))).hexdigest())


@torch.no_grad()
def export_embeddings(model,data,ids,device,batch_size):
    output = {k:[] for k in ('z','x','zm','xm')}
    for selected in ids.split(batch_size):
        z,zm = model.encode(data['template'][selected].to(device),data['template_mask'][selected].to(device))
        x,xm = model.encode(data['candidates'][selected].to(device),data['candidate_mask'][selected].to(device))
        for key,value in dict(z=z,x=x,zm=zm,xm=xm).items():
            output[key].append(value.cpu().numpy())
    return {k:np.concatenate(v) for k,v in output.items()}


def run(cache_dir,output_dir,device='cuda'):
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError('refusing to overwrite identity screen: '+str(output_dir))
    cache,protocol,data = read_cache(cache_dir)
    train = torch.tensor([i for i,r in enumerate(cache['rows']) if r['split']=='train'],dtype=torch.long)
    dev = torch.tensor([i for i,r in enumerate(cache['rows']) if r['split']=='dev'],dtype=torch.long)
    if len(train)<protocol['minimum_train_pairs'] or not len(dev) or len(set(data['sequence_id'][train].tolist()))<2:
        raise ValueError('insufficient eligible training pairs/sequences or missing development pairs; inspect cache index')
    train_ids = set(data['sequence_id'][train].tolist())
    if train_ids & set(data['sequence_id'][dev].tolist()):
        raise RuntimeError('training and development pairs overlap')
    output_dir.mkdir(parents=True,exist_ok=False)
    arrays,training,payloads = {},{},{}
    source_hash = file_hash(Path(cache_dir)/'index.json')
    for mode in ARMS:
        model,training[mode] = train_head(mode,data,train,protocol,cache['feature_dim'],device)
        state = {k:v.cpu() for k,v in model.state_dict().items()}
        checkpoint = dict(kind='vdrm_identity_head',arm=mode,mode=ARMS[mode][0],include_identity=ARMS[mode][1],epoch=protocol['epochs'],net=state,
                          protocol=protocol,cache_index_sha256=source_hash,feature_dim=cache['feature_dim'])
        torch.save(checkpoint,output_dir/f'Identity_{mode}_ep0020.pth')
        payloads['weights_'+mode+'.npz'] = npz_bytes({k:v.numpy() for k,v in state.items()})
        arrays.update({mode+'_'+k:v for k,v in export_embeddings(model,data,dev,device,protocol['batch_size']).items()})
    if len({r['initial_sha256'] for r in training.values()})!=1:
        raise RuntimeError('paired identity heads did not share initialization')
    if len({r['sample_order_sha256'] for r in training.values()})!=1:
        raise RuntimeError('paired identity heads did not share sample order')
    arrays.update(template_mask=data['template_mask'][dev].numpy(),candidate_mask=data['candidate_mask'][dev].numpy())
    rows = [cache['rows'][i] for i in dev.tolist()]
    outcomes,evidence,donors = evaluate_embeddings(rows,arrays,protocol)
    summary = summarize_evaluation(rows,outcomes,protocol,len(train))
    if source_hash!=file_hash(Path(cache_dir)/'index.json') or any(file_hash(Path(cache_dir)/name)!=digest
            for name,digest in cache['members_sha256'].items()):
        raise RuntimeError('source cache changed during training')
    payloads.update({'protocol.json':dump_json(protocol),'rows.json':dump_json(rows),'embeddings.npz':npz_bytes(arrays),
        'outcomes.json':dump_json(outcomes),'evidence.json':dump_json(evidence),'donors.json':dump_json(donors),
        'summary.json':dump_json(summary),'training.json':dump_json(training),'report.md':render_report(summary).encode()})
    manifest = dict(kind='vdrm_identity_screen',version=1,protocol_sha256=protocol_hash(protocol),
        train_ids=cache['train_ids'],dev_ids=cache['dev_ids'],train_pairs=len(train),dev_pairs=len(dev),
        cache_index_sha256=source_hash,checkpoint_sha256=cache['checkpoint_sha256'],backbone_loaded_during_training=False,
        epochs=protocol['epochs'],feature_dim=cache['feature_dim'])
    target = output_dir/'identity_offline_results.tar.gz'
    write_bundle(target,manifest,payloads)
    check_bundle(target)
    print(render_report(summary));print('IDENTITY SCREEN COMPLETE:',target.resolve())
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache_dir',default='output/analysis/identity_offline_v1/cache')
    parser.add_argument('--output_dir',default='output/analysis/identity_offline_v1/screen')
    parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.device=='cuda' and not torch.cuda.is_available():
        parser.error('CUDA unavailable; --device cpu is supported for cached-head training')
    run(args.cache_dir,args.output_dir,args.device)


if __name__=='__main__':main()
