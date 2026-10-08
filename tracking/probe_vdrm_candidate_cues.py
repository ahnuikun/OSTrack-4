"""Export/replay frozen Rqual head cues from an existing 298-frame probe."""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import platform
import sys

import numpy as np
import torch
import torch.nn.functional as F

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))

from lib.test.analysis.candidate_cues import (npz_bytes,read_npz,evaluate_controls,
    frame_metrics,check_native,summarize,render_report,check_bundle)
from lib.test.analysis.response_probe import check_bundle as check_source, QUALITY_CONFIG as CONFIG
from lib.test.analysis.response_failure_audit import candidate_boxes
from lib.test.analysis.frozen_failure_diagnostics import dump_json,write_bundle
from lib.train.frozen_response import prepare_response_models,validate_response_checkpoint
from lib.train.frozen_vdrm import file_hash
from tracking.diagnose_vdrm_frozen_failures import config_for,state_hash
from tracking.diagnose_vdrm_native_float import INPUT_KEYS


@contextmanager
def capture_cues(head):
    values=dict(projected=[],cues=[])
    def projected(module,args,output):
        values['projected'].append(F.normalize(output.detach(),dim=-1).clone())
    def cues(module,args):
        values['cues'].append(args[0].detach().clone())
    hooks=[head.project.register_forward_hook(projected),head.correct.register_forward_pre_hook(cues)]
    try:
        yield values
    finally:
        for hook in hooks:hook.remove()


def run(bundle,save_dir,output,device):
    output=Path(output)
    if output.exists():raise FileExistsError('refusing to overwrite cue bundle: '+str(output))
    if output.suffixes[-2:]!=['.tar','.gz']:raise ValueError('output must end in .tar.gz')
    source,payloads,shapes=check_source(bundle,include_payloads=True)
    if source.get('candidate_arm')!='Rqual' or source.get('profile')!='early_failure':
        raise ValueError('requires the registered Rqual early_failure probe')
    path=Path(save_dir)/'checkpoints/train/ostrack'/CONFIG/'OSTrack_ep0020.pth.tar'
    if not path.is_file():raise FileNotFoundError('missing Rqual epoch-20 checkpoint: '+str(path))
    if file_hash(path)!=source['provenance']['Rqual']['checkpoint_sha256']:
        raise ValueError('checkpoint is not the one used for the supplied probe')
    cfg=config_for(CONFIG)
    model,baseline,_=prepare_response_models(cfg,save_dir,CONFIG,path)
    validate_response_checkpoint(model,torch.load(path,map_location='cpu',weights_only=False),cfg,expected_epoch=20)
    del baseline
    model=model.to(device).requires_grad_(False).eval()
    before=state_hash(model)
    if before!=source['provenance']['Rqual']['state_sha256_before']:
        raise ValueError('Rqual state differs from source probe')
    traces={t['sequence']:t for t in json.loads(payloads['trajectories.json'])}
    records=json.loads(payloads['records.json'])
    exported=[];donors={};max_replay_error=0.
    try:
        with torch.no_grad():
            for index,record in enumerate(records):
                name,frame=record['sequence'],record['frame_1based']
                raw=read_npz(payloads[record['inputs_member']])
                if set(raw)-INPUT_KEYS:raise ValueError('unexpected model inputs; current GT is forbidden')
                inputs={k:torch.from_numpy(v).to(device) for k,v in raw.items()}
                with capture_cues(model.response_residual) as captured:
                    out=model(**inputs)
                if len(captured['projected'])!=2 or len(captured['cues'])!=1:
                    raise RuntimeError('candidate cue capture layout changed')
                maps=read_npz(payloads[record['arms']['Rqual']['maps_member']])
                for key in ('candidate_quality_logits','candidate_boxes','candidate_valid_mask','response_delta_energy',
                            'response_base_score_map','score_map','size_map','offset_map'):
                    expected=torch.from_numpy(maps[key]).to(device)
                    actual=out[key]
                    if expected.dtype==torch.bool:
                        if not torch.equal(actual,expected):raise RuntimeError('replay validity changed')
                    else:
                        max_replay_error=max(max_replay_error,float((actual-expected).abs().max()))
                        if not torch.allclose(actual,expected,rtol=1e-5,atol=2e-5):
                            raise RuntimeError('saved-input replay differs: '+key)
                hann=torch.from_numpy(maps['hann_window']).to(device)
                if (out['score_map']*hann).argmax().item()!=int((maps['score_map']*maps['hann_window']).argmax()):
                    raise RuntimeError('saved-input replay changes native peak')
                # No GT reaches either forward or cue construction. Labels below
                # annotate already-computed candidates only.
                trace=traces[name];row=trace['native']['Rqual'][frame-1]
                candidates,template=captured['projected']
                arrays=dict(cues=captured['cues'][0].cpu().numpy(),candidate_embedding=candidates.cpu().numpy(),
                    template_embedding=template.cpu().numpy(),valid=maps['candidate_valid_mask'],
                    base_score=maps['response_base_score_map'],hann=maps['hann_window'],
                    boxes_xywh=candidate_boxes(maps,row,shapes[record['image_member']]))
                donors.setdefault(name,arrays['template_embedding'])
                exported.append((dict(sequence=name,frame_1based=frame,gt_xywh=trace['gt_xywh'][frame-1],
                    cover=trace['cover'][frame-1],absence=trace['absence'][frame-1],
                    source_inputs_sha256=row['input_sha256'],source_native_iou=row['iou'],
                    source_native_peak=int((maps['score_map']*maps['hann_window']).argmax()),
                    source_native_bbox=row['bbox_xywh']),arrays))
                if (index+1)%50==0 or index+1==len(records):print(f'Frozen input replays: {index+1}/{len(records)}',flush=True)
            new_payloads={'head.npz':npz_bytes({k:v.cpu().numpy() for k,v in model.response_residual.correct.state_dict().items()}),
                          'donors.npz':npz_bytes(donors)}
            outputs=[];metrics=[]
            for row,arrays in exported:
                controls=evaluate_controls(model.response_residual.correct,torch.from_numpy(arrays['cues']).to(device),
                    torch.from_numpy(arrays['candidate_embedding']).to(device),
                    {n:torch.from_numpy(v).to(device) for n,v in donors.items() if n!=row['sequence']},
                    torch.from_numpy(arrays['valid']).to(device),torch.from_numpy(arrays['base_score']).to(device))
                row['variants']=list(controls)
                arrays['logits']=np.stack([v['logits'].cpu().numpy() for v in controls.values()])
                arrays['scores']=np.stack([v['score'].cpu().numpy() for v in controls.values()])
                row['member']=f"frames/{row['sequence']}_{row['frame_1based']:06d}.npz"
                metric=frame_metrics(row,arrays)
                check_native(row, arrays, metric)
                new_payloads[row['member']]=npz_bytes(arrays);outputs.append(row);metrics.append(metric)
    finally:
        after=state_hash(model)
        if before!=after or any(p.grad is not None or p.requires_grad for p in model.parameters()) or any(m.training for m in model.modules()):
            raise RuntimeError('cue audit altered model state/mode or produced parameter gradients')
    summary=summarize(metrics)
    new_payloads.update({'records.json':dump_json(outputs),'per_frame.json':dump_json(metrics),
        'summary.json':dump_json(summary),'report.md':render_report(summary).encode('utf-8')})
    manifest=dict(kind='vdrm_candidate_cues',version=1,dataset='got10k_vdrm_dev',config=CONFIG,
        records=len(outputs),optimizer_steps=0,source_bundle_sha256=file_hash(bundle),
        checkpoint_sha256=file_hash(path),state_sha256_before=before,state_sha256_after=after,
        maximum_saved_input_replay_error=max_replay_error,
        donor_rule='first saved frame per other sequence; all four other-sequence donors, never GT-selected',
        environment=dict(python=platform.python_version(),torch=torch.__version__,device=str(device)))
    write_bundle(output,manifest,new_payloads)
    verified, _, _ = check_bundle(output)
    print(render_report(summary));print('RQUAL CUE AUDIT PASSED (integrity only):',output.resolve())
    return verified


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--bundle',help='Existing Rqual 298-frame probe; only this mode needs weights/CUDA')
    mode.add_argument('--check_bundle',help='CPU independent recomputation, no weights/data/GPU needed')
    parser.add_argument('--save_dir',default='./output')
    parser.add_argument('--output',default='output/analysis/frozen_quality/rqual_cue_audit.tar.gz')
    args=parser.parse_args()
    torch.set_num_threads(2)
    if args.check_bundle:
        manifest,summary,_=check_bundle(args.check_bundle)
        print(render_report(summary));print('RQUAL CUE BUNDLE CHECK PASSED:',manifest['records'],'records')
    else:
        if not torch.cuda.is_available():parser.error('frozen backbone replay requires CUDA')
        run(args.bundle,args.save_dir,args.output,'cuda')


if __name__=='__main__':main()
