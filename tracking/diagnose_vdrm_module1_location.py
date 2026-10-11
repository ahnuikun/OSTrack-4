"""Step04 GT-location proxies on cached inputs; diagnostic GT use is explicit."""

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
from lib.test.analysis.module1_location import PROXIES, SCALES, check_bundle, support_mask
from lib.utils.ce_utils import generate_mask_cond
from tracking.diagnose_vdrm_frozen_failures import SpatialCapture, config_for, load_models, state_hash
from tracking.diagnose_vdrm_module1_gates import evidence_arrays
from tracking.diagnose_vdrm_native_float import NativeTracker, clone_output, decode_native, input_hashes, npz_bytes
from tracking.vdrm_diagnostics.backend_pair_metrics import bbox_iou_xywh
from tracking.vdrm_diagnostics.module1_gate_interventions import gate_intervention
from tracking.vdrm_diagnostics.module1_location_interventions import location_intervention


def cached_inputs(a, cfg, device):
    inputs = {key: torch.from_numpy(a[source]).to(device) for key,source in
              [('template','template_input'),('search','search_input'),('template_bbox','template_bbox')]}
    inputs['ce_template_mask'] = generate_mask_cond(cfg,1,device,inputs['template_bbox'])
    return inputs


def selected_records(source, plan):
    traces = {t['sequence']: t for t in source['trajectories']}
    result = {}
    for r in source['records']:
        name, frame = r['sequence'],r['frame_1based']
        if name not in plan or not plan[name][0] <= frame <= plan[name][1]: continue
        native = traces[name]['native']['Rdisc'][frame-1]
        row = {k:r[k] for k in ('sequence','frame_1based','gt_xywh','cover','absence','image_shape_hw')}
        row.update({k:native[k] for k in ('anchor_xywh','resize_factor','input_sha256')})
        row['source_bbox_xywh'] = native['bbox_xywh']
        row['image_sha256'] = source['manifest']['members_sha256'][r['image_member']]
        result[r['arms']['Rdisc']['arrays_member']] = row
    return result


def collect(bundle, selected, models, cfg):
    model = models['Rdisc']
    tracker = NativeTracker(model,cfg)
    capture = SpatialCapture()
    hook = model.backbone.vdrm.register_forward_hook(capture,with_kwargs=True)
    cases, rows, payloads = [], [], {}
    device = next(model.parameters()).device
    try:
        with torch.no_grad(), tarfile.open(bundle,'r|gz') as archive:
            for member in archive:
                if member.name not in selected: continue
                with np.load(io.BytesIO(archive.extractfile(member).read()),allow_pickle=False) as raw:
                    a = {key:raw[key] for key in raw.files}
                case = dict(selected[member.name])
                inputs = cached_inputs(a,cfg,device)
                if input_hashes(inputs) != case['input_sha256']:
                    raise RuntimeError('reconstructed inputs differ from saved native inputs')
                if not np.array_equal(tracker.output_window.cpu().numpy(),a['hann']):
                    raise RuntimeError('native Hann window differs from cached source')
                masks = {scale:support_mask(a,case,scale) for scale in SCALES}
                case['eligibility'] = {str(scale):dict(reason=reason,target_tokens=int(mask.sum()))
                                       for scale,(mask,reason) in masks.items()}
                schedule = [('control',None),('alpha0',None)] + [
                    (mode,scale) for scale,(mask,reason) in masks.items() if reason is None for mode in PROXIES]
                for mode,scale in schedule:
                    context = gate_intervention(model,mode) if scale is None else location_intervention(model,mode,masks[scale][0])
                    with context as values:
                        out = clone_output(model.forward(**inputs))
                        arrays = evidence_arrays(out,tracker,dict(padding=a['padding']),values,capture)
                        box = decode_native(tracker,out,case['anchor_xywh'],case['resize_factor'],case['image_shape_hw'])
                    if scale is None:
                        arrays.update({key:a[key].copy() for key in ('similarity','match_scale','match_bias')})
                        arrays['target_mask'] = np.zeros_like(a['indices'],dtype=bool)
                    for key in ('indices','part_valid'):
                        if not np.array_equal(arrays[key],a[key]):
                            raise RuntimeError('location intervention changed insertion prefix: '+key)
                    for saved,actual in [('q','raw_q'),('route','raw_route'),('similarity','similarity')]:
                        if not np.allclose(a[saved],arrays[actual],rtol=1e-6,atol=2e-6):
                            raise RuntimeError('cached insertion evidence mismatch: '+saved)
                    if mode == 'control':
                        for key in ('score_map','size_map','offset_map','final_kept'):
                            if not np.array_equal(arrays[key],a[key]):
                                raise RuntimeError('control does not reproduce saved source: '+key)
                        if box != case['source_bbox_xywh']:
                            raise RuntimeError('control float decode differs from native source')
                    if mode == 'alpha0':
                        for key in ('score_map','size_map','offset_map'):
                            if not np.array_equal(arrays[key],a['base_'+key]):
                                raise RuntimeError('alpha0 differs from saved same-input Tclean: '+key)
                            arrays['source_'+key] = a['base_'+key].copy()
                    if input_hashes(inputs) != case['input_sha256']:
                        raise RuntimeError('location proxy changed cached inputs')
                    name = f'arrays/{case["sequence"]}/{case["frame_1based"]:06d}_{mode}_{scale}.npz'
                    payloads[name] = npz_bytes(arrays)
                    gt = np.asarray(case['gt_xywh'])
                    rows.append(dict(case, mode=mode, roi_scale=scale, scope='shared_GT_location_proxy', arrays_member=name,
                        bbox_xywh=box, iou=bbox_iou_xywh(box,gt) if (gt[2:]>0).all() else None,
                        source_maps_exact=mode=='control', saved_Tclean_exact=mode=='alpha0'))
                cases.append(case)
                if len(cases) % 10 == 0 or len(cases) == len(selected):
                    print(f'LOCATION PROXY: {len(cases)}/{len(selected)} cached frames; {len(rows)} forwards; controls exact',flush=True)
    finally:
        hook.remove()
    if len(cases) != len(selected): raise RuntimeError('missing cached inputs')
    return {'cases.json':dump_json(cases),'rows.json':dump_json(rows),**payloads},len(rows)


def run(save_dir, source_path, output, smoke=False):
    if Path(output).exists(): raise FileExistsError('refusing to overwrite location evidence: '+str(output))
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
        if any(provenance[arm][k] != old[k] for k in ('checkpoint_sha256','state_sha256_before')):
            raise ValueError('checkpoint differs from step01: '+arm)
    print(f'STEP04 PRECHECK PASSED: smoke={smoke}, {len(selected)} cached frames; diagnostic GT location use',flush=True)
    try:
        payloads,count = collect(source_path,selected,models,cfg)
    finally:
        for arm,model in models.items():
            after = state_hash(model)
            if (after != provenance[arm]['state_sha256_before'] or any(p.grad is not None for p in model.parameters())
                    or any(m.training for m in model.modules())):
                raise RuntimeError('location diagnostic changed model state/mode or created gradients')
            provenance[arm]['state_sha256_after'] = after
    manifest = dict(kind='vdrm_module1_step04_location',version=1,smoke_only=smoke,
        plan={k:list(v) for k,v in plan.items()},proxies=list(PROXIES),scales=list(SCALES),optimizer_steps=0,
        configurations={a:CONFIGS[a] for a in ARMS},gt_location_proxy_only=True,source_verified=True,
        source_sha256=source['sha256'],source_provenance={a:source['manifest']['provenance'][a] for a in ARMS},
        provenance=provenance,forward_calls=count,
        environment=dict(python=platform.python_version(),torch=torch.__version__,cuda=torch.version.cuda,
                         gpu=torch.cuda.get_device_name(),matmul_tf32=torch.backends.cuda.matmul.allow_tf32))
    write_bundle(output,manifest,payloads)
    checked,_ = check_bundle(output)
    marker = 'STEP04 SMOKE PASSED' if smoke else 'STEP04 LOCATION CHECK PASSED'
    print(f'{marker}: {checked["cases"]} cached frames; GT-location diagnosis only',flush=True)
    print('Download:',Path(output).resolve(),flush=True)
    return checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--save_dir',default='./output')
    parser.add_argument('--source')
    parser.add_argument('--output')
    parser.add_argument('--smoke',action='store_true')
    args = parser.parse_args()
    root = Path(args.save_dir)/'analysis'
    source = args.source or str(root/'module1_step01/module1_evidence.tar.gz')
    output = args.output or str(root/'module1_step04'/('module1_location_smoke.tar.gz' if args.smoke else 'module1_location.tar.gz'))
    run(args.save_dir,source,output,args.smoke)


if __name__ == '__main__': main()
