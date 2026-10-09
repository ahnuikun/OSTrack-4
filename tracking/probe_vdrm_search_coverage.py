"""Frozen visual candidate probe on uniformly sampled saved native anchors."""
import argparse
from pathlib import Path
import sys
import tarfile

import numpy as np
import torch
import torch.nn.functional as F

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from lib.models.layers.candidate_quality import dense_center_boxes
from lib.test.analysis.candidate_cues import read_npz
from lib.test.analysis.frozen_failure_diagnostics import dump_json
from lib.test.analysis.identity_guard_bundle import check_bundle as check_guard
from lib.test.analysis.search_coverage import (PROTOCOL, selected_ids, crop_specs, sequence_evidence,
    summarize, render_report, check_bundle)
from lib.train.data.processing_utils import sample_target
from lib.train.frozen_vdrm import BASE_CONFIG, file_hash
from tracking.diagnose_vdrm_frozen_failures import load_models, state_hash, EXACT_KEYS
from tracking.diagnose_vdrm_native_float import NativeTracker, capture_native, assert_exact, read_rgb, input_hashes
from tracking.probe_vdrm_identity_native import map_candidates
from tracking.screen_vdrm_identity_guard import plan_sequences


def decode_candidates(tracker, output, padding, anchor, resize, shape):
    """Actual CENTER boxes, native mapping/Hann, CE and nonpadding support."""
    boxes = dense_center_boxes(output['size_map'], output['offset_map'])
    mapped = map_candidates(tracker, boxes, anchor, resize, shape)
    scores = (tracker.output_window*output['score_map']).flatten().cpu().numpy()
    kept = torch.ones((1, 256), dtype=torch.bool, device=output['score_map'].device)
    for removed in output.get('removed_indexes_s', ()):
        if removed is not None: kept.scatter_(1, removed.long(), False)
    padding = torch.as_tensor(padding, device=kept.device)[None, None].float()
    supported = (kept & ~F.adaptive_max_pool2d(padding, (16, 16)).bool().flatten(1))[0].cpu().numpy()
    return mapped, scores, supported


@torch.no_grad()
def collect_sequence(plan, model, cfg):
    tracker = NativeTracker(model, cfg)
    tracker.initialize(read_rgb(str(plan['frames'][0])), {'init_bbox': plan['gt'][0].tolist()})
    expected = plan['expected']; replay_frames = []
    values = {k:[] for k in ('native_boxes','anchors','boxes','scores','supported','crop_anchors','crop_squares','image_hw','frame_indices')}
    for i in range(0, len(plan['frames'])-1, PROTOCOL['sampling_stride']):
        frame = i+2; image = read_rgb(str(plan['frames'][i+1]))
        # Replay recorded prediction state, never current GT or an oracle box.
        tracker.state = expected['anchors'][i].tolist(); tracker.frame_id = i
        anchor = np.asarray(tracker.state, dtype=np.float64)
        with capture_native(tracker) as snapshot: tracker.track(image)
        output = snapshot['output']; native = np.asarray(tracker.state, dtype=np.float64)
        if tracker.frame_id != i+1 or not np.array_equal(anchor, expected['anchors'][i]) or not np.array_equal(native, expected['native_boxes'][i]):
            raise RuntimeError(f'source guard native trajectory differs: {plan["name"]}:{frame}')
        if frame == 2:
            assert_exact(output, model.forward(**snapshot['inputs']), keys=EXACT_KEYS)
            replay_frames.append(frame)
        centers, factors, squares = crop_specs(anchor)
        boxes, scores, supported = [], [], []
        center = decode_candidates(tracker, output, snapshot['padding'], anchor, snapshot['resize'], image.shape)
        if not np.array_equal(center[0], expected['boxes'][i]) or not np.array_equal(center[1], expected['base_scores'][i]):
            raise RuntimeError(f'source guard center output differs: {plan["name"]}:{frame}')
        boxes.append(center[0]); scores.append(center[1]); supported.append(center[2])
        unchanged = (list(tracker.state), tracker.frame_id)
        for crop in range(1, len(centers)):
            patch, resize, padding = sample_target(image, centers[crop].tolist(), float(factors[crop]),
                output_sz=tracker.params.search_size)
            search = tracker.preprocessor.process(patch, padding)
            inputs = dict(template=tracker.z_dict1.tensors, search=search.tensors,
                ce_template_mask=tracker.box_mask_z, template_bbox=tracker.vdrm_template_bbox)
            before = input_hashes(inputs)
            alternate = model.forward(**inputs)
            if before != input_hashes(inputs): raise RuntimeError('extra crop changed model inputs')
            b, s, kept = decode_candidates(tracker, alternate, padding, centers[crop], resize, image.shape)
            boxes.append(b); scores.append(s); supported.append(kept)
        if unchanged != (list(tracker.state), tracker.frame_id) or snapshot['input_sha256'] != input_hashes(snapshot['inputs']):
            raise RuntimeError('extra visual crops modified native state/template/inputs')
        row = dict(native_boxes=native, anchors=anchor, boxes=np.stack(boxes), scores=np.stack(scores),
            supported=np.stack(supported), crop_anchors=centers, crop_squares=squares, image_hw=np.array(image.shape[:2]), frame_indices=np.asarray(i))
        for key, value in row.items(): values[key].append(value)
    # All current GT and visibility enter saved evidence only after inference.
    arrays = {k:np.stack(v) for k,v in values.items()}
    arrays.update({k:plan[k] for k in ('gt','cover','absence')})
    arrays.update(source_boxes=expected['boxes'], source_scores=expected['base_scores'],
        source_native_boxes=expected['native_boxes'], source_anchors=expected['anchors'])
    return arrays, replay_frames


def run(guard_bundle, data_root, save_dir, output_dir, device='cuda'):
    directory = Path(output_dir)
    if directory.exists(): raise FileExistsError('refusing to overwrite coverage output: '+str(directory))
    checkpoint = Path(save_dir)/'checkpoints/train/ostrack'/BASE_CONFIG/'OSTrack_ep0300.pth.tar'
    if not checkpoint.is_file(): raise FileNotFoundError('missing Tclean ep300: '+str(checkpoint))
    source, decision = check_guard(guard_bundle)
    if decision['next_action'] != 'stop_existing_identity_reranking_path':
        raise ValueError('coverage redesign requires the completed failed identity screen')
    ids = selected_ids(source['screen_train_ids'])
    plans = plan_sequences(data_root, dict(fit=ids, audit=[]))
    checkpoint_hash, guard_hash = file_hash(checkpoint), file_hash(guard_bundle)
    list_hash = file_hash(Path(data_root)/'list.txt')
    if checkpoint_hash != source['checkpoint_sha256'] or list_hash != source['data_list_sha256']:
        raise ValueError('coverage checkpoint/data list differs from source guard')
    specs = {s['sequence_id']:s for s in source['sequences'] if s['split'] == 'fit'}
    with tarfile.open(guard_bundle, 'r:gz') as archive:
        for plan in plans:
            expected = read_npz(archive.extractfile(specs[plan['sequence_id']]['member']).read())
            if any(not np.array_equal(plan[k], expected[k]) for k in ('gt','cover','absence')):
                raise ValueError('coverage annotations differ from source guard: '+plan['name'])
            plan['expected'] = {k:expected[k] for k in ('anchors','native_boxes','boxes','base_scores')}
    models, cfg, _ = load_models({'Tclean':checkpoint}, torch.device(device), arms=('Tclean',))
    if cfg.TEST.SEARCH_FACTOR != 4 or cfg.TEST.SEARCH_SIZE != 256:
        raise ValueError('coverage requires the registered factor4/size256 baseline')
    model = models['Tclean']; before = state_hash(model)
    if before != source['states_before']['Tclean']:
        raise ValueError('coverage model state differs from frozen guard source')
    directory.mkdir(parents=True, exist_ok=False); (directory/'sequences').mkdir()
    records, specifications, members = [], [], []
    try:
        for number, plan in enumerate(plans, 1):
            arrays, replay_frames = collect_sequence(plan, model, cfg)
            records.append(sequence_evidence(plan['name'], arrays))
            member = 'sequences/'+plan['name']+'.npz'
            with (directory/member).open('xb') as stream: np.savez_compressed(stream, **arrays)
            members.append(member)
            specifications.append(dict(sequence_id=plan['sequence_id'], name=plan['name'], frames=len(plan['frames']),
                member=member, exact_replay_frames=replay_frames))
            print(f'Coverage {number}/{len(plans)} {plan["name"]}: {len(arrays["native_boxes"])} sampled steps / {len(plan["frames"])-1} saved history steps', flush=True)
    finally:
        after = state_hash(model)
        if before != after or any(p.grad is not None or p.requires_grad for p in model.parameters()) or any(m.training for m in model.modules()):
            raise RuntimeError('coverage probe modified frozen state/mode/gradients')
    if file_hash(checkpoint) != checkpoint_hash or file_hash(guard_bundle) != guard_hash or file_hash(Path(data_root)/'list.txt') != list_hash:
        raise RuntimeError('coverage source files changed during run')
    result = summarize(records)
    documents = {'records.json':dump_json(records), 'summary.json':dump_json(result), 'report.md':render_report(result).encode('utf-8')}
    for name, value in documents.items(): (directory/name).write_bytes(value)
    members.extend(documents)
    steps = sum(s['frames']-1 for s in specifications)
    replayed = sum((s['frames']-2)//PROTOCOL['sampling_stride']+1 for s in specifications)
    manifest = dict(kind='vdrm_search_coverage', protocol=PROTOCOL, sequences=specifications,
        source_screen_train_ids=source['screen_train_ids'], source_guard_sha256=guard_hash,
        checkpoint_sha256=checkpoint_hash, data_list_sha256=list_hash, state_before=before, state_after=after,
        optimizer_steps=0, parameter_gradient_tensors=0, source_history_steps=steps,
        replayed_native_steps=replayed, extra_crop_forwards=replayed*5,
        members_sha256={name:file_hash(directory/name) for name in members})
    (directory/'manifest.json').write_bytes(dump_json(manifest))
    partial = directory/'search_coverage_results.tar.gz.partial'
    with partial.open('xb') as stream, tarfile.open(fileobj=stream, mode='w:gz') as archive:
        for name in ['manifest.json', *members]: archive.add(directory/name, arcname=name, recursive=False)
    check_bundle(partial)
    target = directory/'search_coverage_results.tar.gz'
    if target.exists(): raise FileExistsError(target)
    partial.rename(target)
    print(render_report(result)); print('SEARCH COVERAGE PROBE COMPLETE:', target.resolve())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--guard_bundle', required=True)
    parser.add_argument('--data_root', default='./data/got10k/train')
    parser.add_argument('--save_dir', default='./output')
    parser.add_argument('--output_dir', default='output/analysis/search_coverage_v1')
    args = parser.parse_args(); torch.set_num_threads(2)
    if not torch.cuda.is_available(): parser.error('native frozen Tclean inference requires CUDA')
    run(args.guard_bundle, args.data_root, args.save_dir, args.output_dir)


if __name__ == '__main__': main()
