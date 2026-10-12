"""Step11: task-loss versus oracle-residual fitting, single-frame versus shared.

Only previously selected TRAINING frames are fitted. GT-assisted teacher and
post-hoc case selection make this a memorization diagnostic, not a deployable
tracker or a generalization experiment. No new dataset replay is required.
"""

import argparse
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
import numpy as np
import torch

from lib.train.module1_learning import ResidualAdapter, learning_loss
from lib.train.frozen_vdrm import file_hash
from lib.test.analysis.module1_numerics import read_archive, load_np
from lib.test.analysis.module1_learning import check_bundle as check_learning
from lib.test.analysis.module1_fit_audit import check_bundle as check_fit, HEADS, geometry
from lib.test.analysis.module1_learnability import ARMS, CONTROLS, protocol, select_cases, teacher_error, check_bundle
from lib.test.analysis.module1_direction import decode_metrics
from lib.test.analysis.frozen_failure_diagnostics import CONFIGS, dump_json, write_bundle
from tracking.diagnose_vdrm_frozen_failures import load_models, state_hash
from tracking.diagnose_vdrm_native_float import npz_bytes
from tracking.diagnose_vdrm_module1_numerics import backend_state
from tracking.train_vdrm_module1_learning import torch_prefix, suffix_forward
from tracking.vdrm_diagnostics.module1_numeric_suffix import CachedSuffix


def prepare(a, c, vectors, device):
    return dict(a=a, c=c, p=torch_prefix(a, device), vectors=vectors,
                teacher=torch.from_numpy(vectors['oracle_full']).to(device),
                base={k: torch.from_numpy(a[k]).to(device) for k in HEADS[:3]},
                good=torch.from_numpy(a['good']).to(device), target=torch.from_numpy(a['target']).to(device))


def task_loss(out, case):
    return learning_loss(out, case['base'], case['good'], case['target'], case['p']['hann'], case['c']['protect'])


def adapter_delta(adapter, case):
    p = case['p']
    return adapter(p['search_tokens'], p['prototypes'], p['part_valid'], p['mixture'], p['delta'])


@torch.no_grad()
def record(case, suffix, delta, arm, step, payloads, references=None):
    out = suffix_forward(suffix, case['p'], delta)
    arrays = {k: out[k].cpu().numpy() for k in HEADS}
    actual = ((case['p']['search_tokens'] + delta) - case['p']['search_tokens']).cpu().numpy()
    if references is not None and any(not np.array_equal(arrays[k], references[k]) for k in HEADS):
        raise RuntimeError('source replay or zero-initialization parity failed')
    geom = geometry(actual, case['a']['delta'])
    if any(not np.isfinite(v) for v in geom.values()) or geom['norm_ratio'] > 1.005:
        raise RuntimeError('nonfinite residual or norm budget exceeded')
    arrays['actual_delta'] = actual
    member = f'outputs/{case["c"]["case_id"]}_{arm}_{step:04d}.npz'
    payloads[member] = npz_bytes(arrays)
    metrics, _ = decode_metrics(arrays, case['c'], case['a']['hann'])
    _, loss = task_loss(out, case)
    return dict(case_id=case['c']['case_id'], arm=arm, step=step, arrays_member=member,
                geometry=geom, loss=loss, teacher_error=teacher_error(actual, case['vectors']['oracle_full'], case['a']['delta']), **metrics)


def fit_run(cases, suffix, arm, spec, payloads, source_initial):
    """A shared step accumulates the MEAN gradient, with one clip and Adam step.

    Each frame has exactly the same number of exposures in single/shared arms.
    Teacher fitting skips the expensive suffix except at fixed checkpoints.
    """
    torch.manual_seed(spec['seed'])
    adapter = ResidualAdapter(width=spec['width'], space='full').to(cases[0]['p']['delta'].device)
    initial = state_hash(adapter)
    if initial != source_initial:
        raise ValueError('adapter initialization differs from Step08')
    run_id = arm if arm.startswith('shared_') else arm + '/' + cases[0]['c']['case_id']
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=spec['learning_rate'], weight_decay=spec['weight_decay'])
    rows = [record(c, suffix, adapter_delta(adapter, c), arm, 0, payloads, c['a']) for c in cases]
    trace = []
    for step in range(1, spec['steps'] + 1):
        optimizer.zero_grad(set_to_none=True); objective = 0.
        for c in cases:
            delta = adapter_delta(adapter, c)
            if arm.endswith('_teacher'):
                loss = (delta - c['teacher']).square().sum() / c['p']['delta'].square().sum().clamp_min(1e-30)
            else:
                loss, _ = task_loss(suffix_forward(suffix, c['p'], delta), c)
            if not torch.isfinite(loss):
                raise RuntimeError('nonfinite fitting objective')
            (loss / len(cases)).backward(); objective += float(loss.detach()) / len(cases)
        norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), spec['gradient_clip'], error_if_nonfinite=True)
        previous = [p.detach().clone() for p in adapter.parameters()]
        optimizer.step()
        change = torch.sqrt(sum((p.detach() - before).square().sum() for p, before in zip(adapter.parameters(), previous)))
        trace.append(dict(step=step, objective=objective, gradient_norm=float(norm), parameter_step_norm=float(change)))
        if step in spec['checkpoints']:
            rows.extend(record(c, suffix, adapter_delta(adapter, c), arm, step, payloads) for c in cases)
            print(f'LEARNABILITY {run_id}: {step}/{spec["steps"]}, objective={objective:.6f}', flush=True)
    adapter.zero_grad(set_to_none=True); adapter.eval().requires_grad_(False)
    final = state_hash(adapter)
    if initial == final:
        raise RuntimeError('adapter did not learn')
    # Replay saved final weights through the actual adapter and actual head.
    replay = ResidualAdapter(width=spec['width'], space='full').to(cases[0]['p']['delta'].device).eval().requires_grad_(False)
    replay.load_state_dict(adapter.state_dict())
    for c in cases:
        row = next(r for r in rows if r['case_id'] == c['c']['case_id'] and r['step'] == spec['steps'])
        reference = load_np(payloads[row['arrays_member']])
        with torch.no_grad():
            delta = adapter_delta(replay, c); out = suffix_forward(suffix, c['p'], delta)
        if any(not np.array_equal(out[k].cpu().numpy(), reference[k]) for k in HEADS):
            raise RuntimeError('saved final weights replay differs')
    member = 'weights/' + run_id.replace('/', '_') + '.npz'
    payloads[member] = npz_bytes({k: v.cpu().numpy() for k, v in adapter.state_dict().items()})
    training = dict(run_id=run_id, case_ids=[c['c']['case_id'] for c in cases], initial_sha256=initial, final_sha256=final,
                    weights_member=member, updates=spec['steps'], frame_exposures=len(cases) * spec['steps'],
                    parameter_count=sum(p.numel() for p in adapter.parameters()), trace=trace)
    return training, rows


def run(save_dir, learning_dir, fit_path, output, smoke=False):
    output = Path(output); analysis = output.with_name(output.name.removesuffix('.tar.gz') + '_analysis.tar.gz')
    if output.exists() or analysis.exists():
        raise FileExistsError('refusing to overwrite learnability evidence')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    print('Checking Step08/09 evidence and selected training-cache hashes...', flush=True)
    learning_dir = Path(learning_dir); learning_path = learning_dir / 'module1_learning.tar.gz'
    check_learning(learning_path); lm, ld = read_archive(learning_path)
    check_fit(fit_path); fm, fd = read_archive(fit_path)
    if fm['learning_sha256'] != file_hash(learning_path) or fm['source_sha256'] != lm['source_sha256']:
        raise ValueError('Step08/09 source chain differs')
    if not smoke and (lm['protocol']['smoke_only'] or fm['smoke_only']):
        raise ValueError('formal fitting requires formal Step08/09 evidence')
    if backend_state() != lm['backend'] or backend_state() != fm['backend']:
        raise ValueError('backend differs from source')
    source_cases = json.loads(fd['cases.json']); source_rows = json.loads(fd['rows.json'])
    selected = select_cases(source_cases, source_rows, smoke)
    source_training = json.loads(ld['training.json'])['full']
    if file_hash(learning_dir / 'cache/index.json') != source_training['train_cache_sha256']:
        raise ValueError('training cache index mismatch')
    index = {r['member']: r for r in json.loads(ld['train_index.json'])['rows']}
    prepared = []; payloads = {}; cases = []
    for old in selected:
        c = dict(old); path = learning_dir / 'cache' / c['member']
        if any(c[k] != v for k, v in index[c['member']].items()) or file_hash(path) != c['sha256']:
            raise ValueError('selected training cache mismatch')
        a = load_np(path.read_bytes()); vectors = load_np(fd[c['vectors_member']])
        c['cache_member'] = 'cache/' + c['case_id'] + '.npz'
        c['source_vectors_member'] = 'source_vectors/' + c['case_id'] + '.npz'
        payloads[c['cache_member']] = path.read_bytes(); payloads[c['source_vectors_member']] = fd[c['vectors_member']]
        cases.append(c); prepared.append(prepare(a, c, vectors, 'cuda'))
    paths = {a: Path(save_dir) / 'checkpoints/train/ostrack' / CONFIGS[a] / 'OSTrack_ep0300.pth.tar' for a in ('Tclean', 'Rdisc')}
    models, _, provenance = load_models(paths, 'cuda', arms=('Tclean', 'Rdisc'))
    for arm in models:
        for k in ('checkpoint_sha256', 'state_sha256_before'):
            if provenance[arm][k] != lm['provenance'][arm][k] or provenance[arm][k] != fm['provenance'][arm][k]:
                raise ValueError('checkpoint differs from Step08/09')
    suffix = CachedSuffix(models['Rdisc']).cuda(); del models
    before = state_hash(suffix); spec = protocol(smoke); rows = []; training = []
    if before != fm['suffix_before']:
        raise ValueError('frozen suffix differs from Step09')
    source_lookup = {(r['case_id'], r['mode']): r for r in source_rows}
    for c in prepared:
        for arm in CONTROLS:
            old = source_lookup[c['c']['case_id'], arm]; payloads[old['arrays_member']] = fd[old['arrays_member']]
            reference = load_np(fd[old['arrays_member']])
            delta = torch.from_numpy(c['vectors'][arm]).cuda()
            rows.append(record(c, suffix, delta, arm, 0, payloads, reference))
    print(f'STEP11 PRECHECK PASSED: {len(cases)} training frames; {spec["steps"]} exposures/frame/arm; memorization only', flush=True)
    for arm in ARMS:
        groups = [[c] for c in prepared] if arm.startswith('single_') else [prepared]
        for group in groups:
            report, outputs = fit_run(group, suffix, arm, spec, payloads, source_training['initial_sha256'])
            training.append(report); rows.extend(outputs)
    after = state_hash(suffix)
    if before != after or any(p.grad is not None or p.requires_grad for p in suffix.parameters()):
        raise RuntimeError('frozen suffix changed or accumulated gradients')
    manifest = dict(kind='vdrm_module1_step11_learnability', version=1, protocol=spec,
                    learning_sha256=file_hash(learning_path), fit_sha256=file_hash(fit_path),
                    source_initial_sha256=source_training['initial_sha256'], source_control_parity=True,
                    suffix_before=before, suffix_after=after, provenance=provenance, backend=backend_state())
    payloads.update({'cases.json': dump_json(cases), 'rows.json': dump_json(rows), 'training.json': dump_json(training),
                     'source_cases.json': fd['cases.json'], 'source_rows.json': fd['rows.json']})
    write_bundle(output, manifest, payloads); summary, _, _ = check_bundle(output)
    write_bundle(analysis, dict(kind='vdrm_module1_step11_learnability_analysis', version=1, evidence_sha256=file_hash(output)),
                 {'summary.json': dump_json(summary)})
    print('STEP11 LEARNABILITY CHECK PASSED; no generalization or tracking-performance claim', flush=True)
    print('Download:', output.resolve(), flush=True)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--save_dir', default='./output'); p.add_argument('--learning_dir'); p.add_argument('--fit')
    p.add_argument('--output'); p.add_argument('--smoke', action='store_true')
    args = p.parse_args(); root = Path(args.save_dir) / 'analysis'; torch.set_num_threads(2)
    run(args.save_dir, args.learning_dir or root / 'module1_step08', args.fit or root / 'module1_step09/module1_fit.tar.gz',
        args.output or root / ('module1_step11_smoke' if args.smoke else 'module1_step11') / 'module1_learnability.tar.gz', args.smoke)


if __name__ == '__main__':
    main()
