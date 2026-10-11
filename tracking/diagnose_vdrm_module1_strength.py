"""Step05: fixed-input signed residual sweep, with native and raw head readouts.

Gains are a predeclared stress test, NOT tuned tracking hyperparameters. The
location family uses current GT. No trajectory, optimizer or checkpoint changes.
"""

import argparse
import io
from pathlib import Path
import platform
import sys
import tarfile

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

import numpy as np
import torch

from lib.test.analysis.frozen_failure_diagnostics import CONFIGS, dump_json, write_bundle
from lib.test.analysis.module1_gates import ARMS, protocol_plan, read_source
from lib.test.analysis.module1_location import SCALES, support_mask
from lib.test.analysis.module1_strength import GAINS, check_bundle, schedule
from tracking.diagnose_vdrm_frozen_failures import SpatialCapture, config_for, load_models, state_hash
from tracking.diagnose_vdrm_module1_gates import evidence_arrays
from tracking.diagnose_vdrm_module1_location import cached_inputs, selected_records
from tracking.diagnose_vdrm_native_float import NativeTracker, clone_output, decode_native, input_hashes, npz_bytes
from tracking.vdrm_diagnostics.backend_pair_metrics import bbox_iou_xywh
from tracking.vdrm_diagnostics.module1_strength_interventions import strength_intervention


def collect(bundle, selected, models, cfg):
    model = models['Rdisc']
    tracker, capture = NativeTracker(model,cfg), SpatialCapture()
    hook = model.backbone.vdrm.register_forward_hook(capture,with_kwargs=True)
    device = next(model.parameters()).device
    cases, rows, payloads = [], [], {}
    try:
        with torch.no_grad(), tarfile.open(bundle,'r|gz') as archive:
            for member in archive:
                if member.name not in selected: continue
                with np.load(io.BytesIO(archive.extractfile(member).read()),allow_pickle=False) as raw:
                    source = {k:raw[k] for k in raw.files}
                case = dict(selected[member.name])
                inputs = cached_inputs(source,cfg,device)
                if input_hashes(inputs) != case['input_sha256']:
                    raise RuntimeError('strength cached input hashes differ')
                if not np.array_equal(tracker.output_window.cpu().numpy(),source['hann']):
                    raise RuntimeError('native Hann differs from source')
                masks = {s:support_mask(source,case,s) for s in SCALES}
                case['eligibility'] = {str(s):dict(reason=reason,target_tokens=int(mask.sum())) for s,(mask,reason) in masks.items()}
                for family,gain,scale in schedule(case['eligibility']):
                    mask = None if scale is None else masks[scale][0]
                    with strength_intervention(model,family,gain,mask) as values:
                        out = clone_output(model.forward(**inputs))
                        a = evidence_arrays(out,tracker,dict(padding=source['padding']),values,capture)
                        box = decode_native(tracker,out,case['anchor_xywh'],case['resize_factor'],case['image_shape_hw'])
                    if scale is None:
                        a.update({k:source[k].copy() for k in ('similarity','match_scale','match_bias')})
                        a['target_mask'] = np.zeros_like(source['indices'],dtype=bool)
                    for k in ('indices','part_valid'):
                        if not np.array_equal(a[k],source[k]): raise RuntimeError('changed strength prefix: '+k)
                    for saved,actual in [('q','raw_q'),('route','raw_route'),('similarity','similarity')]:
                        if not np.allclose(source[saved],a[actual],rtol=1e-6,atol=2e-6):
                            raise RuntimeError('strength prefix mismatch: '+saved)
                    control = family == 'original' and gain == 1
                    if control:
                        for k in ('score_map','size_map','offset_map','final_kept'):
                            if not np.array_equal(a[k],source[k]): raise RuntimeError('control does not exactly reproduce source: '+k)
                        if box != case['source_bbox_xywh']: raise RuntimeError('control float box mismatch')
                    if gain == 0:
                        for k in ('score_map','size_map','offset_map'):
                            if not np.array_equal(a[k],source['base_'+k]): raise RuntimeError('zero gain does not exactly reproduce Tclean: '+k)
                            a['source_'+k] = source['base_'+k].copy()
                    if input_hashes(inputs) != case['input_sha256']:
                        raise RuntimeError('strength intervention modified cached inputs')
                    path = f'arrays/{case["sequence"]}/{case["frame_1based"]:06d}_{family}_{gain:g}_{scale}.npz'
                    payloads[path] = npz_bytes(a)
                    gt = np.asarray(case['gt_xywh'])
                    rows.append(dict(case,family=family,strength=gain,roi_scale=scale,
                        mode='control' if family=='original' else 'route_target_norm',scope='shared_residual_strength',
                        arrays_member=path,bbox_xywh=box,iou=bbox_iou_xywh(box,gt) if (gt[2:]>0).all() else None,
                        source_maps_exact=control,saved_Tclean_exact=gain==0))
                cases.append(case)
                if len(cases)%10==0 or len(cases)==len(selected):
                    print(f'STRENGTH: {len(cases)}/{len(selected)} cached frames; {len(rows)} forwards; controls exact',flush=True)
    finally:
        hook.remove()
    if len(cases)!=len(selected): raise RuntimeError('missing strength cached inputs')
    return {'cases.json':dump_json(cases),'rows.json':dump_json(rows),**payloads},len(rows)


def run(save_dir,source_path,output,smoke=False):
    output = Path(output)
    analysis_output = output.with_name(output.name.removesuffix('.tar.gz')+'_analysis.tar.gz')
    for path in (output,analysis_output):
        if path.exists(): raise FileExistsError('refusing to overwrite strength artifact: '+str(path))
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    paths = {a:Path(save_dir)/'checkpoints/train/ostrack'/CONFIGS[a]/'OSTrack_ep0300.pth.tar' for a in ARMS}
    for path in paths.values():
        if not path.is_file(): raise FileNotFoundError('missing checkpoint: '+str(path))
    print('Checking step01 cached evidence on CPU...',flush=True)
    source = read_source(source_path)
    plan = protocol_plan(smoke)
    selected = selected_records(source,plan)
    models,_,provenance = load_models(paths,'cuda',arms=ARMS)
    cfg = config_for(CONFIGS['Rdisc'])
    for arm in ARMS:
        old = source['manifest']['provenance'][arm]
        if any(provenance[arm][k]!=old[k] for k in ('checkpoint_sha256','state_sha256_before')):
            raise ValueError('strength checkpoint differs from source: '+arm)
    print(f'STEP05 PRECHECK PASSED: smoke={smoke}, {len(selected)} cached frames; signed gain diagnosis only',flush=True)
    try:
        payloads,count = collect(source_path,selected,models,cfg)
    finally:
        for arm,model in models.items():
            after = state_hash(model)
            if after!=provenance[arm]['state_sha256_before'] or any(p.grad is not None for p in model.parameters()) or any(m.training for m in model.modules()):
                raise RuntimeError('strength experiment changed model state, mode or gradients')
            provenance[arm]['state_sha256_after'] = after
    manifest = dict(kind='vdrm_module1_step05_strength',version=1,smoke_only=smoke,
        plan={k:list(v) for k,v in plan.items()},gains=list(GAINS),scales=list(SCALES),optimizer_steps=0,
        configurations={a:CONFIGS[a] for a in ARMS},fixed_input_only=True,source_verified=True,
        source_sha256=source['sha256'],source_provenance={a:source['manifest']['provenance'][a] for a in ARMS},
        provenance=provenance,forward_calls=count,
        environment=dict(python=platform.python_version(),torch=torch.__version__,cuda=torch.version.cuda,
                         gpu=torch.cuda.get_device_name(),matmul_tf32=torch.backends.cuda.matmul.allow_tf32))
    write_bundle(output,manifest,payloads)
    summary,frames = check_bundle(output)
    write_bundle(analysis_output,dict(kind='vdrm_module1_step05_strength_analysis',version=1,
        evidence_sha256=summary['evidence_sha256'],source_sha256=source['sha256'],optimizer_steps=0),
        {'summary.json':dump_json(summary),'per_frame.json':dump_json(frames)})
    marker = 'STEP05 SMOKE PASSED' if smoke else 'STEP05 STRENGTH CHECK PASSED'
    print(f'{marker}: {summary["cases"]} cached frames; {count} forwards; diagnosis only, no performance pass',flush=True)
    print('Download:',output.resolve(),flush=True)
    print('Download:',analysis_output.resolve(),flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--save_dir',default='./output')
    parser.add_argument('--source')
    parser.add_argument('--output')
    parser.add_argument('--smoke',action='store_true')
    args = parser.parse_args()
    root = Path(args.save_dir)/'analysis'
    source = args.source or str(root/'module1_step01/module1_evidence.tar.gz')
    output = args.output or str(root/'module1_step05'/('module1_strength_smoke.tar.gz' if args.smoke else 'module1_strength.tar.gz'))
    run(args.save_dir,source,output,args.smoke)


if __name__ == '__main__': main()
