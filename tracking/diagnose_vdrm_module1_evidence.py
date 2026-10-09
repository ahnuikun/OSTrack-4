"""Step01: original frozen VDRM internals on exact native float trajectories."""

import argparse
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

import numpy as np
import torch
import torch.nn.functional as F

from lib.test.analysis.module1_evidence import ARMS, PLAN, RECORDS, STEPS, check_bundle
from lib.test.analysis.frozen_failure_diagnostics import CONFIGS, dump_json, write_bundle
from lib.test.evaluation import get_dataset
from tracking.diagnose_vdrm_frozen_failures import (
    SpatialCapture, checkpoint_paths, config_for, crop_geometry, intervention, load_models, png_bytes, state_hash)
from tracking.diagnose_vdrm_native_float import (
    NativeTracker, assert_exact, clone_output, native_step, npz_bytes, prepare_cases, read_rgb)


class PartCapture(SpatialCapture):
    """Read-only capture; recompute masked means without changing forward."""

    def __call__(self, module, args, kwargs, output):
        super().__call__(module, args, kwargs, output)
        tokens = args[0] if args else kwargs['tokens']
        start = kwargs['template_length']
        if module.spatial_gate_mode != 'part_aligned' or module.residual_max_ratio != 0:
            raise ValueError('step01 requires the registered unbounded V8 part_aligned path')
        template, search = tokens[:, :start], tokens[:, start:]
        masks, valid = module._build_part_masks(kwargs['template_bbox'], start)
        masks = masks.to(template.dtype)
        prototypes = torch.einsum('bkl,blc->bkc', masks, template) / masks.sum(-1, keepdim=True).clamp_min(1)
        prototypes *= valid[..., None]
        similarity = torch.einsum('bkc,blc->bkl', F.normalize(prototypes, dim=-1, eps=module.eps),
                                  F.normalize(search, dim=-1, eps=module.eps))
        diagnostics = output[1]
        values = dict(template_tokens=template, search_tokens=search, part_masks=masks,
            prototypes=prototypes, similarity=similarity, q=diagnostics['part_reliability'],
            part_valid=valid, route=diagnostics['part_route_gate'], route_logits=diagnostics['part_route_logits'],
            indices=kwargs['search_global_index'], delta=output[0][:, start:] - search,
            alpha=diagnostics['vdrm_alpha'], template_bbox=kwargs['template_bbox'],
            match_scale=F.softplus(module.log_match_scale), match_bias=module.match_bias)
        self.arrays = {k: v.detach().cpu().numpy().copy() for k, v in values.items()}


def collect(cases, models, configs):
    captures = {arm: PartCapture() for arm in ARMS}
    hooks = [models[arm].backbone.vdrm.register_forward_hook(captures[arm], with_kwargs=True) for arm in ARMS]
    arrays_payloads, image_payloads, records, traces = {}, {}, [], []
    try:
        with torch.no_grad():
            for sequence, gt, cover, absence, start, stop in cases:
                trackers = {arm: NativeTracker(models[arm], configs[arm]) for arm in ARMS}
                initial = read_rgb(sequence.frames[0])
                prefix = 'images/' + sequence.name
                image_payloads[prefix + '/000001.png'] = png_bytes(initial)
                for tracker in trackers.values():
                    tracker.initialize(initial, dict(init_bbox=gt[0].tolist()))
                native = {arm: [dict(frame_1based=1, bbox_xywh=gt[0].tolist())] for arm in ARMS}
                for frame in range(2, stop + 1):
                    image = read_rgb(sequence.frames[frame - 1])
                    saved = {}
                    for arm, tracker in trackers.items():
                        snapshot, row = native_step(tracker, image, gt[frame - 1], frame, captures[arm])
                        native[arm].append(row)
                        if frame < start:
                            continue
                        internal = dict(captures[arm].arrays)
                        out = snapshot['output']
                        final_kept = torch.ones(1, 256, dtype=torch.bool, device=out['score_map'].device)
                        for removed in out['removed_indexes_s']:
                            if removed is not None:
                                final_kept.scatter_(1, removed.long(), False)
                        for key in ('score_map', 'size_map', 'offset_map'):
                            internal[key] = out[key].detach().cpu().numpy()
                        internal.update(hann=tracker.output_window.detach().cpu().numpy(),
                            final_kept=final_kept.cpu().numpy(), padding=np.asarray(snapshot['padding']),
                            search_input=snapshot['inputs']['search'].detach().cpu().numpy(),
                            template_input=snapshot['inputs']['template'].detach().cpu().numpy())
                        if arm == 'Rdisc':
                            reference = clone_output(models['Tclean'].forward(**snapshot['inputs']))
                            with intervention(models[arm], 'alpha0'):
                                bypass = clone_output(models[arm].forward(**snapshot['inputs']))
                            assert_exact(reference, bypass, keys=('pred_boxes', 'score_map', 'size_map', 'offset_map', 'backbone_feat'))
                            for key in ('score_map', 'size_map', 'offset_map'):
                                internal['base_' + key] = reference[key].detach().cpu().numpy()
                        member = f'arrays/{sequence.name}/{frame:06d}_{arm}.npz'
                        arrays_payloads[member] = npz_bytes(internal)
                        saved[arm] = dict(arrays_member=member, geometry=crop_geometry(row['anchor_xywh'],
                            configs[arm].TEST.SEARCH_FACTOR, image.shape, gt[frame-1]))
                    if frame >= start:
                        member = f'{prefix}/{frame:06d}.png'
                        image_payloads[member] = png_bytes(image)
                        records.append(dict(sequence=sequence.name, frame_1based=frame,
                            image_shape_hw=list(image.shape[:2]), image_member=member,
                            gt_xywh=gt[frame-1].tolist(), cover=float(cover[frame-1]), absence=float(absence[frame-1]),
                            arms=saved, alpha0_exact_Tclean=True))
                    if frame % 25 == 0:
                        print(f'{sequence.name}: native prefix {frame}/{stop}', flush=True)
                traces.append(dict(sequence=sequence.name, gt_xywh=gt[:stop].tolist(), cover=cover[:stop].tolist(),
                                   absence=absence[:stop].tolist(), native=native))
    finally:
        for hook in hooks:
            hook.remove()
    # Metadata first: CPU verification streams large arrays instead of retaining all.
    return {'records.json': dump_json(records), 'trajectories.json': dump_json(traces),
            **arrays_payloads, **image_payloads}


def run(save_dir, output):
    if Path(output).exists():
        raise FileExistsError('refusing to overwrite evidence: ' + str(output))
    if not torch.cuda.is_available():
        raise RuntimeError('native tracker requires CUDA')
    paths = checkpoint_paths(save_dir)
    cases = prepare_cases(get_dataset('got10k_vdrm_dev'), plan=PLAN)
    models, _, provenance = load_models(paths, 'cuda')
    configs = {arm: config_for(CONFIGS[arm]) for arm in ARMS}
    print(f'PRECHECK PASSED: {len(cases)} cases, {STEPS} native steps, {RECORDS} saved frames', flush=True)
    payloads = collect(cases, models, configs)
    for arm in ARMS:
        provenance[arm]['state_sha256_after'] = state_hash(models[arm])
        if provenance[arm]['state_sha256_before'] != provenance[arm]['state_sha256_after']:
            raise RuntimeError('read-only collection changed state: ' + arm)
    manifest = dict(kind='vdrm_module1_step01', version=1, dataset='got10k_vdrm_dev',
        configurations=CONFIGS, arms=list(ARMS), plan={k: list(v) for k, v in PLAN.items()},
        records=RECORDS, native_steps=STEPS, optimizer_steps=0, provenance=provenance,
        full_dev_gate_passed=False, automatic_identity_verdict=False)
    write_bundle(output, manifest, payloads)
    check_bundle(output)
    print(f'STEP01 EVIDENCE CHECK PASSED: {STEPS} native steps, {RECORDS} records; NOT a performance pass')
    print('Download:', Path(output).resolve())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--save_dir', default='./output')
    parser.add_argument('--output')
    parser.add_argument('--check_bundle', help='CPU-only verification of an existing archive')
    args = parser.parse_args()
    if args.check_bundle:
        summary, _, _ = check_bundle(args.check_bundle)
        print(f"STEP01 EVIDENCE CHECK PASSED: {summary['records']} records; NOT a performance pass")
        return
    run(args.save_dir, args.output or str(Path(args.save_dir) / 'analysis/module1_step01/module1_evidence.tar.gz'))


if __name__ == '__main__':
    main()
