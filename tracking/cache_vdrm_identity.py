"""One-time frozen Tclean ROI cache for a GT-assisted OFFLINE identity screen."""
import argparse
import json
from pathlib import Path
import re
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from lib.train.identity_offline import DEFAULT_PROTOCOL, load_protocol, protocol_hash, registered_ids, roi_cells
from lib.models.layers.candidate_quality import dense_center_boxes
from lib.train.actors.candidate_quality import candidate_iou
from lib.train.data.processing_utils import sample_target, transform_image_to_crop
from lib.train.frozen_vdrm import BASE_CONFIG, file_hash
from lib.test.utils.hann import hann2d
from lib.utils.ce_utils import generate_mask_cond
from tracking.diagnose_vdrm_frozen_failures import load_models, state_hash


def plan_sequences(root, protocol):
    root = Path(root).resolve()
    # Validate dataset identity from list contents, not the resolved directory
    # basename: a legitimate train symlink may point at a differently named mount.
    names = (root/'list.txt').read_text(encoding='utf-8').splitlines()
    if len(names)!=len(set(names)) or any(not re.fullmatch(r'GOT-10k_Train_\d{6}', n) for n in names):
        raise ValueError('invalid GOT-10k training list')
    train_ids, dev_ids = registered_ids(protocol)
    if max(train_ids+dev_ids)>=len(names):
        raise ValueError('GOT train list shorter than registered sequence IDs')
    plans = []
    for split, ids in (('train',train_ids),('dev',dev_ids)):
        for identity in ids:
            folder = root/names[identity]
            gt = np.loadtxt(folder/'groundtruth.txt', delimiter=',', ndmin=2)
            cover = np.loadtxt(folder/'cover.label', ndmin=1)
            absence = np.loadtxt(folder/'absence.label', ndmin=1)
            if (gt.shape != (len(cover),4) or absence.shape!=cover.shape or
                    not np.isfinite(gt).all() or not np.isfinite(cover).all() or
                    not np.isfinite(absence).all() or len(gt)<2 or (gt[0,2:]<=0).any()):
                raise ValueError('invalid training annotations: '+str(folder))
            if ((cover<0)|(cover>8)).any() or not np.isin(absence,[0,1]).all():
                raise ValueError('invalid visibility labels: '+str(folder))
            valid = (gt[:,2:]>0).all(1)
            visible = valid & (cover>0) & (absence==0)
            eligible = np.flatnonzero(visible & np.roll(visible,1) & (np.arange(len(gt))>0))
            if not visible[0]:
                eligible = np.array([],dtype=int)
            slots = np.unique(np.linspace(0,len(eligible)-1,min(len(eligible),protocol['frames_per_sequence']),dtype=int)) if len(eligible) else []
            frames = eligible[slots].tolist() if len(slots) else []
            for frame in [0,*frames]:
                if not (folder/f'{frame+1:08d}.jpg').is_file():
                    raise FileNotFoundError(folder/f'{frame+1:08d}.jpg')
            plans.append(dict(split=split, sequence_id=identity, sequence=names[identity],
                              folder=folder, gt=gt, cover=cover, frames=frames,initial_visible=bool(visible[0])))
    return plans, train_ids, dev_ids


def crop(path, anchor, size, factor, cfg, device):
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError('image cannot be decoded: '+str(path))
    patch, resize, padding = sample_target(cv2.cvtColor(image,cv2.COLOR_BGR2RGB),anchor,factor,size)
    tensor = torch.from_numpy(patch).to(device).float().permute(2,0,1)[None]/255
    mean = tensor.new_tensor(cfg.DATA.MEAN)[None,:,None,None]
    std = tensor.new_tensor(cfg.DATA.STD)[None,:,None,None]
    return (tensor-mean)/std, resize, torch.from_numpy(padding).to(device)[None]


def crop_box(box, anchor, resize, size, device):
    return transform_image_to_crop(torch.tensor(box,dtype=torch.float32),torch.tensor(anchor,dtype=torch.float32),
        resize,torch.tensor([size,size]),normalize=True).to(device)[None]


def jitter_anchor(box, identity, frame, seed):
    rng = np.random.default_rng(np.random.SeedSequence([seed,identity,frame]))
    anchor = np.asarray(box,dtype=float).copy()
    center = anchor[:2]+anchor[2:]/2 + rng.uniform(-.5,.5,2)*np.sqrt(anchor[2:].prod())
    anchor[2:] *= np.exp(rng.uniform(-.15,.15,2))
    anchor[:2] = center-anchor[2:]/2
    return anchor


@torch.no_grad()
def collect_pair(output, template_box, target, template_padding, search_padding, protocol):
    features = output['backbone_feat']
    features = features[-1] if isinstance(features,list) else features
    batch, count, dim = features.shape
    if batch!=1 or count!=320:
        raise ValueError('identity cache requires native 8x8/16x16 feature grids')
    z = features[:,:64].transpose(1,2).reshape(1,dim,8,8)
    x = features[:,64:].transpose(1,2).reshape(1,dim,16,16)
    kept = torch.ones((1,256),dtype=torch.bool,device=features.device)
    for removed in output.get('removed_indexes_s',()):
        if removed is not None:
            kept.scatter_(1,removed.long(),False)
    support = kept.reshape(1,1,16,16) & ~F.adaptive_max_pool2d(search_padding[:,None].float(),(16,16)).bool()
    zsupport = ~F.adaptive_max_pool2d(template_padding[:,None].float(),(8,8)).bool()
    boxes = dense_center_boxes(output['size_map'],output['offset_map'])
    zbox = torch.cat((template_box[:,:2]+template_box[:,2:]/2,template_box[:,2:]),1)[:,None]
    template, template_mask = roi_cells(z,zbox,zsupport.to(z))
    candidates, masks = roi_cells(x,boxes,support.to(x))
    common = (masks & template_mask).sum(-1)[0]
    overlap = candidate_iou(boxes,target)[0]
    valid = common>=protocol['minimum_cells']
    good, bad = valid & (overlap>=protocol['positive_iou']), valid & (overlap<=protocol['negative_iou'])
    if not good.any():
        return None, 'no_supported_positive'
    if int(bad.sum())<4:
        return None, 'fewer_than_four_supported_negatives'
    base = (output['score_map']*hann2d(torch.tensor([16,16]),centered=True).to(features)).flatten()
    # Keep the strongest supported GOOD baseline candidate. Selecting only
    # max-IoU could discard the baseline's correct winner and invent hard cases.
    positive = base.masked_fill(~good,-torch.inf).argmax()
    negatives = base.masked_fill(~bad,-torch.inf).topk(4).indices
    selected = torch.cat((positive[None],negatives))
    arrays = dict(template=template[0,0].cpu().numpy(),template_mask=template_mask[0,0].cpu().numpy(),
                  candidates=candidates[0,selected].cpu().numpy(),candidate_mask=masks[0,selected].cpu().numpy())
    labels = dict(ious=overlap[selected].cpu().tolist(),base_scores=base[selected].cpu().tolist(),
                  candidate_indices=selected.cpu().tolist())
    return (arrays,labels), None


def run(root, save_dir, directory, protocol_path=DEFAULT_PROTOCOL, device='cuda'):
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError('refusing to overwrite identity cache: '+str(directory))
    protocol = load_protocol(protocol_path)
    path = Path(save_dir)/'checkpoints/train/ostrack'/BASE_CONFIG/'OSTrack_ep0300.pth.tar'
    if not path.is_file():
        raise FileNotFoundError('missing Tclean epoch-300 checkpoint: '+str(path))
    plans, train_ids, dev_ids = plan_sequences(root,protocol)
    models, cfg, provenance = load_models({'Tclean':path},torch.device(device),arms=('Tclean',))
    model = models['Tclean']
    before = state_hash(model)
    directory.mkdir(parents=True,exist_ok=False)
    (directory/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    index = dict(kind='vdrm_identity_cache',version=1,protocol_sha256=protocol_hash(protocol),
        train_ids=train_ids,dev_ids=dev_ids,data_list_sha256=file_hash(Path(root)/'list.txt'),
        checkpoint_sha256=provenance['Tclean']['checkpoint_sha256'],state_before=before,
        optimizer_steps=0,rows=[],sequences=[],members_sha256={},feature_dim=None,
        crop_protocol='GT-assisted previous-GT plus deterministic jitter; offline, NOT closed-loop tracking',
        candidate_protocol='highest base-Hann supported IoU>=0.5 positive plus four highest supported IoU<=0.1 negatives')
    try:
        with torch.no_grad():
            for number, plan in enumerate(plans):
                gt, folder = plan['gt'], plan['folder']
                z, resize, zp = crop(folder/'00000001.jpg',gt[0],128,2.,cfg,device)
                zbox = crop_box(gt[0],gt[0],resize,128,device)
                zm = generate_mask_cond(cfg,1,torch.device(device),zbox)
                saved, skipped = [], {} if plan['initial_visible'] else {'invisible_initial_template':1}
                member = plan['split']+'_'+plan['sequence']+'.npz'
                for frame in plan['frames']:
                    anchor = jitter_anchor(gt[frame-1],plan['sequence_id'],frame,protocol['seed'])
                    search, resize, padding = crop(folder/f'{frame+1:08d}.jpg',anchor,256,4.,cfg,device)
                    output = model(template=z,search=search,ce_template_mask=zm,template_bbox=zbox)
                    # Current GT is first used after the frozen visual forward.
                    target = crop_box(gt[frame],anchor,resize,256,device)
                    pair, reason = collect_pair(output,zbox,target,zp,padding,protocol)
                    if reason:
                        skipped[reason] = skipped.get(reason,0)+1
                        continue
                    arrays, labels = pair
                    index['feature_dim'] = arrays['template'].shape[-1]
                    index['rows'].append(dict(split=plan['split'],sequence_id=plan['sequence_id'],sequence=plan['sequence'],
                        frame_1based=frame+1,cover=float(plan['cover'][frame]),member=member,offset=len(saved),**labels))
                    saved.append(arrays)
                if saved:
                    np.savez_compressed(directory/member,**{k:np.stack([r[k] for r in saved]) for k in saved[0]})
                    index['members_sha256'][member] = file_hash(directory/member)
                index['sequences'].append(dict(split=plan['split'],sequence=plan['sequence'],sequence_id=plan['sequence_id'],
                    planned_frames_1based=[f+1 for f in plan['frames']],eligible=len(saved),skipped=skipped))
                print(f"Cache {number+1}/{len(plans)} {plan['split']} {plan['sequence']}: {len(saved)}/{len(plan['frames'])} eligible",flush=True)
    finally:
        after = state_hash(model)
        if before!=after or any(p.grad is not None or p.requires_grad for p in model.parameters()) or any(m.training for m in model.modules()):
            raise RuntimeError('identity cache changed frozen model state/mode/gradients')
    index['state_after'] = after
    (directory/'index.json').write_text(json.dumps(index,indent=2,allow_nan=False),encoding='utf-8')
    print('IDENTITY CACHE COMPLETE:',directory.resolve(),'pairs:',len(index['rows']))
    return index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data_root',default='./data/got10k/train')
    parser.add_argument('--save_dir',default='./output')
    parser.add_argument('--config',default=str(DEFAULT_PROTOCOL))
    parser.add_argument('--output_dir',default='output/analysis/identity_offline_v1/cache')
    args = parser.parse_args()
    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        parser.error('frozen backbone cache requires CUDA')
    run(args.data_root,args.save_dir,args.output_dir,args.config)


if __name__=='__main__':main()
