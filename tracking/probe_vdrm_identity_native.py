"""Frozen identity heads on the complete native Tclean development trajectory."""
import argparse
from pathlib import Path
import sys
import tarfile

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from lib.test.analysis.identity_native import (PROTOCOL, load_screen_heads, extract_cells,
    sequence_evidence, summarize, render_report, check_bundle)
from lib.test.analysis.frozen_failure_diagnostics import dump_json
from lib.test.analysis.got10k_annotations import validate_native_annotations
from lib.test.evaluation.got10kdataset import _vdrm_dev_sequence_ids
from lib.train.frozen_vdrm import BASE_CONFIG, file_hash
from lib.train.identity_offline import ARMS
from lib.utils.box_ops import clip_box
from tracking.diagnose_vdrm_frozen_failures import load_models, state_hash, EXACT_KEYS
from tracking.diagnose_vdrm_native_float import (NativeTracker, capture_native, assert_exact,
                                                read_rgb, input_hashes)


def plan_sequences(root):
    root = Path(root).resolve()
    names = (root/'list.txt').read_text(encoding='utf-8').splitlines()
    if names != [f'GOT-10k_Train_{i+1:06d}' for i in range(len(names))]:
        raise ValueError('requires ordered GOT-10k training list, not val/test or a filtered list')
    ids = _vdrm_dev_sequence_ids()
    if not names or max(ids) >= len(names):
        raise ValueError('GOT-10k training list shorter than fixed development IDs')
    plans = []
    for identity in ids:
        folder = root/names[identity]
        gt = np.loadtxt(folder/'groundtruth.txt', delimiter=',', ndmin=2)
        cover = np.loadtxt(folder/'cover.label', ndmin=1)
        absence = np.loadtxt(folder/'absence.label', ndmin=1)
        validate_native_annotations(gt, cover, absence, folder)
        frames = [folder/f'{i+1:08d}.jpg' for i in range(len(gt))]
        actual = sorted(p.name for p in folder.glob('*.jpg'))
        if actual != [p.name for p in frames]:
            raise ValueError('missing/extra development frames: '+str(folder))
        plans.append(dict(sequence_id=identity, name=names[identity], frames=frames,
                          gt=gt, cover=cover, absence=absence))
    return plans


def map_candidates(tracker, boxes, anchor, resize, shape):
    """Use exact native scalar arithmetic/clip; restore state even on failure."""
    state = tracker.state
    try:
        tracker.state = list(anchor)
        scaled = (boxes[0]*tracker.params.search_size/resize).tolist()
        return np.asarray([clip_box(tracker.map_box_back(box, resize), *shape[:2], margin=10)
                           for box in scaled], dtype=np.float64)
    finally:
        tracker.state = state


@torch.no_grad()
def collect_sequence(plan, model, heads, cfg):
    tracker = NativeTracker(model, cfg)
    tracker.initialize(read_rgb(str(plan['frames'][0])), {'init_bbox': plan['gt'][0].tolist()})
    arrays = {k: [] for k in ('native_boxes', 'anchors', 'boxes', 'base_scores', 'common_cells')}
    arrays.update({a+'_scores': [] for a in ARMS})
    replay_frames = []
    for frame, path in enumerate(plan['frames'][1:], start=2):
        image = read_rgb(str(path))
        anchor = list(tracker.state)
        with capture_native(tracker) as snapshot:
            result = tracker.track(image)  # No current GT, info, or head result.
        out = snapshot['output']
        if tracker.frame_id != frame-1 or list(result['target_bbox']) != list(tracker.state):
            raise RuntimeError('native state/frame contract differs')
        if frame == 2:
            replay = model.forward(**snapshot['inputs'])
            assert_exact(out, replay, keys=EXACT_KEYS)
            if snapshot['input_sha256'] != input_hashes(snapshot['inputs']):
                raise RuntimeError('replay changed native inputs')
            replay_frames.append(frame)
        boxes, z, x, zm, xm = extract_cells(out, tracker.vdrm_template_bbox,
            tracker.z_dict1.mask, torch.from_numpy(snapshot['padding']).to(out['score_map'].device)[None])
        common = (zm[:, None] & xm).sum(-1)[0].cpu().numpy().astype(np.int16)
        base = (tracker.output_window*out['score_map']).flatten().cpu().numpy()
        mapped = map_candidates(tracker, boxes, anchor, snapshot['resize'], image.shape)
        native = np.asarray(tracker.state, dtype=np.float64)
        if not np.array_equal(mapped[int(base.argmax())], native):
            raise RuntimeError('dense candidate geometry differs from native tracker')
        unchanged = (list(tracker.state), tracker.frame_id)
        for arm, head in heads.items():
            scores = head(z, x, zm, xm)[0].cpu().numpy()
            if scores.shape != (256,) or not np.isfinite(scores).all():
                raise RuntimeError('nonfinite/wrong-sized identity output')
            arrays[arm+'_scores'].append(scores)
        if (list(tracker.state), tracker.frame_id) != unchanged:
            raise RuntimeError('identity head changed native trajectory')
        for key, value in dict(native_boxes=native, anchors=np.asarray(anchor), boxes=mapped,
                               base_scores=base, common_cells=common).items():
            arrays[key].append(value)
    # Current annotations first enter result labeling after inference completes.
    arrays = {k: np.stack(v) for k, v in arrays.items()}
    arrays.update({k: plan[k] for k in ('gt', 'cover', 'absence')})
    return arrays, replay_frames


def package(directory, manifest, members):
    manifest = dict(manifest, members_sha256={n: file_hash(directory/n) for n in members})
    (directory/'manifest.json').write_bytes(dump_json(manifest))
    target = directory/'identity_native_results.tar.gz'
    partial = directory/'identity_native_results.tar.gz.partial'
    with partial.open('xb') as stream, tarfile.open(fileobj=stream, mode='w:gz') as archive:
        for name in ['manifest.json', *members]:
            archive.add(directory/name, arcname=name, recursive=False)
    check_bundle(partial)
    # Exclusive completion marker: never replace any existing final artifact.
    if target.exists():
        raise FileExistsError(target)
    partial.rename(target)
    return target


def run(screen_bundle, data_root, save_dir, output_dir, device='cuda'):
    directory = Path(output_dir)
    if directory.exists():
        raise FileExistsError('refusing to overwrite native identity output: '+str(directory))
    source = Path(save_dir)/'checkpoints/train/ostrack'/BASE_CONFIG/'OSTrack_ep0300.pth.tar'
    if not source.is_file():
        raise FileNotFoundError('missing registered Tclean ep300: '+str(source))
    plans = plan_sequences(data_root)
    heads, screen = load_screen_heads(screen_bundle)
    source_hash, screen_hash = file_hash(source), file_hash(screen_bundle)
    if source_hash != screen['checkpoint_sha256']:
        raise ValueError('Tclean checkpoint does not match identity feature source')
    models, cfg, _ = load_models({'Tclean': source}, torch.device(device), arms=('Tclean',))
    model = models['Tclean']
    heads = {a: h.to(device) for a, h in heads.items()}
    all_models = dict(Tclean=model, **heads)
    before = {a: state_hash(m) for a, m in all_models.items()}
    directory.mkdir(parents=True, exist_ok=False)
    (directory/'sequences').mkdir()
    records, specifications, members = [], [], []
    try:
        for number, plan in enumerate(plans, start=1):
            arrays, replay_frames = collect_sequence(plan, model, heads, cfg)
            record = sequence_evidence(plan['name'], arrays)
            member = 'sequences/'+plan['name']+'.npz'
            with (directory/member).open('xb') as stream:
                np.savez_compressed(stream, **arrays)
            members.append(member)
            records.append(record)
            specifications.append(dict(name=plan['name'], sequence_id=plan['sequence_id'],
                frames=len(plan['frames']), member=member, exact_replay_frames=replay_frames))
            print(f"Native identity {number}/{len(plans)} {plan['name']}: {len(plan['frames'])-1} steps; "
                  f"top5 recoverable hard={record['pools']['top5']['recoverable_hard']}", flush=True)
    finally:
        after = {a: state_hash(m) for a, m in all_models.items()}
        if before != after or any(p.grad is not None or p.requires_grad for m in all_models.values() for p in m.parameters()) or any(s.training for m in all_models.values() for s in m.modules()):
            raise RuntimeError('native identity probe changed frozen state/mode/gradients')
    if source_hash != file_hash(source) or screen_hash != file_hash(screen_bundle):
        raise RuntimeError('source files changed during probe')
    summary = summarize(records)
    documents = {'records.json': dump_json(records), 'summary.json': dump_json(summary),
                 'report.md': render_report(summary).encode('utf-8')}
    for name, value in documents.items():
        (directory/name).write_bytes(value)
        members.append(name)
    manifest = dict(kind='vdrm_identity_native', protocol=PROTOCOL, sequences=specifications,
        native_tracking_steps=summary['counts']['frames'], optimizer_steps=0, parameter_gradient_tensors=0,
        source_screen_sha256=screen_hash, checkpoint_sha256=source_hash,
        data_list_sha256=file_hash(Path(data_root)/'list.txt'), states_before=before, states_after=after)
    target = package(directory, manifest, members)
    print(render_report(summary))
    print('IDENTITY NATIVE PROBE COMPLETE:', target.resolve())
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--screen_bundle', required=True)
    parser.add_argument('--data_root', default='./data/got10k/train')
    parser.add_argument('--save_dir', default='./output')
    parser.add_argument('--output_dir', default='output/analysis/identity_native_v1')
    args = parser.parse_args()
    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        parser.error('native tracker requires CUDA')
    run(args.screen_bundle, args.data_root, args.save_dir, args.output_dir)


if __name__ == '__main__':
    main()
