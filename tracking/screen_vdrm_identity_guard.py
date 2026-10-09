"""One bounded TRAIN-domain calibration/audit; fixed existing heads, no training."""
import argparse
from pathlib import Path
import sys
import tarfile

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from lib.train.frozen_vdrm import BASE_CONFIG, file_hash
from lib.test.analysis.identity_native import load_screen_heads
from lib.test.analysis.identity_guard import PROTOCOL, partition, grid_record, fit_policy, decide
from lib.test.analysis.identity_guard_bundle import check_bundle, render_report
from lib.test.analysis.frozen_failure_diagnostics import dump_json
from lib.test.analysis.frozen_vdrm_stages import visibility_protocol
from tracking.diagnose_vdrm_frozen_failures import load_models, state_hash
from tracking.probe_vdrm_identity_native import collect_sequence


def plan_sequences(root, parts):
    root = Path(root).resolve()
    names = (root/'list.txt').read_text(encoding='utf-8').splitlines()
    if not names or names != [f'GOT-10k_Train_{i+1:06d}' for i in range(len(names))]:
        raise ValueError('requires ordered complete GOT-10k training list')
    if max(parts['fit']+parts['audit']) >= len(names):
        raise ValueError('GOT training list shorter than registered guard IDs')
    plans = []
    for split in ('fit', 'audit'):
        for identity in parts[split]:
            folder = root/names[identity]
            gt = np.loadtxt(folder/'groundtruth.txt', delimiter=',', ndmin=2)
            cover = np.loadtxt(folder/'cover.label', ndmin=1)
            absence = np.loadtxt(folder/'absence.label', ndmin=1)
            visibility_protocol(cover, absence)
            if (gt.shape != (len(cover), 4) or len(gt) < 2 or not np.isfinite(gt).all() or
                    (gt[:, 2:] < 0).any() or (gt[0, 2:] <= 0).any()):
                raise ValueError('invalid guard annotations: '+str(folder))
            frames = [folder/f'{i+1:08d}.jpg' for i in range(len(gt))]
            if sorted(p.name for p in folder.glob('*.jpg')) != [p.name for p in frames]:
                raise ValueError('missing/extra guard frames: '+str(folder))
            plans.append(dict(sequence_id=identity, name=names[identity], split=split,
                frames=frames, gt=gt, cover=cover, absence=absence))
    return plans


def run(screen_bundle, data_root, save_dir, output_dir, device='cuda'):
    directory = Path(output_dir)
    if directory.exists():
        raise FileExistsError('refusing to overwrite guard output: '+str(directory))
    source = Path(save_dir)/'checkpoints/train/ostrack'/BASE_CONFIG/'OSTrack_ep0300.pth.tar'
    if not source.is_file():
        raise FileNotFoundError('missing Tclean ep300: '+str(source))
    heads, screen = load_screen_heads(screen_bundle)
    parts = partition(screen['train_ids'])
    plans = plan_sequences(data_root, parts)
    source_hash, screen_hash = file_hash(source), file_hash(screen_bundle)
    if source_hash != screen['checkpoint_sha256']:
        raise ValueError('Tclean differs from identity training source')
    models, cfg, _ = load_models({'Tclean': source}, torch.device(device), arms=('Tclean',))
    heads = {a: h.to(device) for a, h in heads.items()}
    all_models = dict(Tclean=models['Tclean'], **heads)
    before = {a: state_hash(m) for a, m in all_models.items()}
    directory.mkdir(parents=True, exist_ok=False); (directory/'sequences').mkdir()
    members, records, specs, choice = [], [], [], None
    try:
        for split in ('fit', 'audit'):
            if split == 'audit':
                choice, _ = fit_policy(records)
                # Persist the exact choice BEFORE any held-out visual inference.
                (directory/'frozen_choice.json').write_bytes(dump_json(choice))
                if choice['threshold'] == PROTOCOL['disabled_threshold']:
                    print('FIT SELECTED DISABLED BASELINE: skip held-out GPU work; stop existing reranking path.', flush=True)
                    break
            selected_plans = [p for p in plans if p['split'] == split]
            for number, plan in enumerate(selected_plans, start=1):
                arrays, replay_frames = collect_sequence(plan, models['Tclean'], heads, cfg)
                records.append(grid_record(plan['name'], split, arrays, selected=choice))
                member = 'sequences/'+split+'_'+plan['name']+'.npz'
                with (directory/member).open('xb') as stream: np.savez_compressed(stream, **arrays)
                members.append(member)
                specs.append(dict(name=plan['name'], split=split, sequence_id=plan['sequence_id'],
                    frames=len(plan['frames']), member=member, exact_replay_frames=replay_frames))
                print(f"Guard {split} {number}/{len(selected_plans)} {plan['name']}: {len(plan['frames'])-1} native steps", flush=True)
    finally:
        after = {a: state_hash(m) for a, m in all_models.items()}
        if before != after or any(p.grad is not None or p.requires_grad for m in all_models.values() for p in m.parameters()) or any(s.training for m in all_models.values() for s in m.modules()):
            raise RuntimeError('guard screen modified frozen state/mode/gradients')
    if source_hash != file_hash(source) or screen_hash != file_hash(screen_bundle):
        raise RuntimeError('guard source files changed during run')
    result = decide(records)
    if result['selected'] != choice:
        raise RuntimeError('held-out work modified frozen choice')
    documents = {'records.json': dump_json(records), 'decision.json': dump_json(result),
                 'report.md': render_report(result).encode('utf-8')}
    for name, value in documents.items(): (directory/name).write_bytes(value)
    members += ['frozen_choice.json', *documents]
    manifest = dict(kind='vdrm_identity_guard', protocol=PROTOCOL, partition=parts,
        screen_train_ids=screen['train_ids'], frozen_choice=choice, sequences=specs,
        native_tracking_steps=sum(s['frames']-1 for s in specs), optimizer_steps=0, parameter_gradient_tensors=0,
        states_before=before, states_after=after, source_screen_sha256=screen_hash,
        checkpoint_sha256=source_hash, data_list_sha256=file_hash(Path(data_root)/'list.txt'),
        members_sha256={name: file_hash(directory/name) for name in members})
    (directory/'manifest.json').write_bytes(dump_json(manifest))
    partial = directory/'identity_guard_results.tar.gz.partial'
    with partial.open('xb') as stream, tarfile.open(fileobj=stream, mode='w:gz') as archive:
        for name in ['manifest.json', *members]: archive.add(directory/name, arcname=name, recursive=False)
    check_bundle(partial)
    target = directory/'identity_guard_results.tar.gz'
    if target.exists(): raise FileExistsError(target)
    partial.rename(target)
    print(render_report(result)); print('IDENTITY GUARD SCREEN COMPLETE:', target.resolve())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--screen_bundle', required=True)
    parser.add_argument('--data_root', default='./data/got10k/train')
    parser.add_argument('--save_dir', default='./output')
    parser.add_argument('--output_dir', default='output/analysis/identity_guard_v1')
    args = parser.parse_args(); torch.set_num_threads(2)
    if not torch.cuda.is_available(): parser.error('native Tclean collection requires CUDA')
    run(args.screen_bundle, args.data_root, args.save_dir, args.output_dir)


if __name__ == '__main__': main()
