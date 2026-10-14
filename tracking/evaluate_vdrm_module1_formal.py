"""Original five-benchmark acceptance plus the registered five-case regression.

Uses the production tracker and existing benchmark metric implementation.
Evaluation never selects checkpoints or supplies current GT to the network.
"""
import argparse
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tarfile

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

import numpy as np
import torch

from lib.test.evaluation import get_dataset, trackerlist
from lib.test.evaluation.environment import env_settings
from lib.test.evaluation.result_paths import resolve_result_bbox_path
from lib.test.evaluation.running import run_dataset
from lib.test.analysis.plot_results import print_results, check_and_load_precomputed_results
from lib.test.analysis.module1_evidence import ious
from lib.test.analysis.module1_gates import protocol_plan
from lib.test.analysis.module1_learning import summarize
from lib.test.analysis.frozen_failure_diagnostics import CONFIGS
from lib.train.frozen_vdrm import file_hash
from lib.train.module1_formal import CONFIG, SCRATCH_CONFIG, backend, config, validate_checkpoint
from tracking.test_uav_suite import DATASETS

NAMES = {'Tclean': CONFIGS['Tclean'], 'Rdisc': CONFIGS['Rdisc'], 'Module1': CONFIG}


def benchmark_data():
    datasets = {}
    for name in DATASETS:
        sequences = get_dataset(name)
        if not len(sequences):
            raise ValueError('empty test dataset: ' + name)
        for seq in sequences:
            if (len(seq.frames) != len(seq.ground_truth_rect) or not seq.frames
                    or np.asarray(seq.ground_truth_rect).shape != (len(seq.frames), 4)):
                raise ValueError('invalid test sequence: ' + seq.name)
            missing = next((p for p in seq.frames if not Path(p).is_file()), None)
            if missing:
                raise FileNotFoundError('test image missing: ' + missing)
        datasets[name] = sequences
        print(f'PRECHECK {name}: {len(sequences)} sequences, all frame paths present', flush=True)
    return datasets


def regression_data():
    sequences = {seq.name: seq for seq in get_dataset('got10k_vdrm_dev')}
    result = {}
    for name, (_, stop) in protocol_plan(False).items():
        if name not in sequences:
            raise ValueError('missing regression sequence: ' + name)
        seq = sequences[name]
        folder = Path(seq.frames[0]).parent
        cover = np.loadtxt(folder / 'cover.label', ndmin=1)
        absence = np.loadtxt(folder / 'absence.label', ndmin=1)
        if (len(seq.frames) < stop or len(seq.ground_truth_rect) != len(seq.frames)
                or len(cover) != len(seq.frames) or len(absence) != len(seq.frames)):
            raise ValueError('incomplete regression sequence/labels: ' + name)
        for path in seq.frames[:stop]:
            if not Path(path).is_file():
                raise FileNotFoundError('regression image missing: ' + path)
        result[name] = seq, cover, absence
    return result


def suite_names(stage):
    return dict(NAMES, Module1=SCRATCH_CONFIG if stage == 'scratch' else CONFIG)


def checkpoint_paths(save_dir, stage='adapter'):
    return {arm: Path(save_dir) / 'checkpoints/train/ostrack' / name / 'OSTrack_ep0300.pth.tar'
            for arm, name in suite_names(stage).items()}


def precheck(save_dir, stage='adapter'):
    env = env_settings()
    if Path(env.save_dir).resolve() != Path(save_dir).resolve():
        raise ValueError('--save_dir must match lib/test/evaluation/local.py save_dir')
    paths = checkpoint_paths(save_dir, stage)
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError('required fixed epoch-300 checkpoint missing: ' + str(path))
    payload = torch.load(paths['Module1'], map_location='cpu', weights_only=False)
    contract = validate_checkpoint(payload, config(suite_names(stage)['Module1']), expected_epoch=300)
    if stage == 'adapter' and contract['source_checkpoint_sha256'] != file_hash(paths['Rdisc']):
        raise ValueError('Rdisc comparison is not the source of this adapter')
    if stage == 'adapter' and contract['source_provenance']['Tclean']['checkpoint_sha256'] != file_hash(paths['Tclean']):
        raise ValueError('Tclean comparison is not the registered source')
    fingerprint = {arm: file_hash(path) for arm, path in paths.items()}
    del payload
    regression_data()
    return benchmark_data(), fingerprint


def checked_results(trackers, dataset):
    for tracker in trackers:
        for seq in dataset:
            path = resolve_result_bbox_path(tracker.results_dir, seq)
            boxes = np.loadtxt(path, ndmin=2)
            if boxes.shape != (len(seq.frames), 4) or not np.isfinite(boxes).all() or (boxes[:, 2:] <= 0).any():
                raise ValueError('incomplete/invalid results; rerun with --force: ' + path)


def stamp_results(tracker, dataset, name, checkpoint_sha256, can_run):
    """Reuse unchanged controls across stages, with explicit weight provenance."""
    directory = Path(tracker.results_dir)
    stamp = directory / ('.module1_' + name + '.json')
    expected = dict(checkpoint_sha256=checkpoint_sha256, torch=torch.__version__,
                    deterministic=True, config=tracker.parameter_name)
    if stamp.exists():
        if json.loads(stamp.read_text()) != expected:
            raise ValueError('cached results belong to different weights/runtime: ' + str(stamp))
        return
    if any(Path(resolve_result_bbox_path(directory, seq)).exists() for seq in dataset):
        raise ValueError('existing results have no module1 provenance: ' + str(directory))
    if not can_run:
        raise FileNotFoundError('run evaluation before analysis: ' + str(stamp))
    directory.mkdir(parents=True, exist_ok=True)
    stamp.write_text(json.dumps(expected, indent=2), encoding='utf-8')


def regression(save_dir, output, stage='adapter'):
    from lib.models.ostrack import build_ostrack
    from tracking.diagnose_vdrm_frozen_failures import load_models
    from tracking.diagnose_vdrm_native_float import NativeTracker, capture_native, decode_native, read_rgb
    paths = checkpoint_paths(save_dir, stage)
    models, _, _ = load_models(paths, 'cuda', arms=('Tclean', 'Rdisc'))
    del models['Tclean']
    cfg = config(suite_names(stage)['Module1'])
    payload = torch.load(paths['Module1'], map_location='cpu', weights_only=False)
    validate_checkpoint(payload, cfg, expected_epoch=300)
    models['Module1'] = build_ostrack(cfg, training=False).cuda().eval()
    models['Module1'].load_state_dict(payload['net'], strict=True)
    del payload
    dataset = regression_data()
    rows = []
    for name, (_, stop) in protocol_plan(False).items():
        seq, cover, absence = dataset[name]
        trackers = {arm: NativeTracker(model, cfg) for arm, model in models.items()}
        first = read_rgb(seq.frames[0])
        for tracker in trackers.values():
            tracker.initialize(first, dict(init_bbox=seq.ground_truth_rect[0].tolist()))
        with torch.no_grad():
            for index in range(1, stop):
                image = read_rgb(seq.frames[index])
                reference = trackers['Rdisc']
                anchor = list(reference.state)
                with capture_native(reference) as captured:
                    base = reference.track(image)['target_bbox']
                fixed = decode_native(trackers['Module1'], models['Module1'](**captured['inputs']),
                    anchor, captured['resize'], image.shape)
                native = trackers['Module1'].track(image)['target_bbox']
                gt = seq.ground_truth_rect[index]
                if index % 10 == 0 or index in (60, 61, 73, 74, stop-1):
                    import cv2
                    canvas = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    for label, box, color in [('GT', gt, (0, 220, 0)), ('Rdisc', base, (0, 140, 255)),
                                              ('Module1', native, (255, 180, 0))]:
                        x, y, w, h = np.asarray(box).astype(int)
                        cv2.rectangle(canvas, (x, y), (x+w, y+h), color, 2)
                        cv2.putText(canvas, label, (x, max(15, y-5)), cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1)
                    frames = output / 'regression_frames'
                    frames.mkdir(exist_ok=True)
                    if not cv2.imwrite(str(frames / f'{name}_{index+1:06d}.png'), canvas):
                        raise RuntimeError('failed to export regression visualization')
                for scope, arm, box in [('closed', 'Rdisc', base), ('shared', 'Rdisc', base),
                                        ('closed', 'Module1', native), ('shared', 'Module1', fixed)]:
                    rows.append(dict(population='regression', sequence=name, frame_1based=index+1,
                        scope=scope, arm=arm, bbox_xywh=box, gt_xywh=gt.tolist(),
                        iou=float(ious(np.asarray([box]), gt)[0]) if min(gt[2:]) > 0 else None,
                        visible=bool(cover[index] > 0 and absence[index] == 0 and min(gt[2:]) > 0),
                        cover=float(cover[index]), absence=float(absence[index])))
        print('REGRESSION completed:', name, flush=True)
    report = summarize(rows, populations=('regression',), modes=('Rdisc', 'Module1'))
    (output / 'regression_rows.json').write_text(json.dumps(rows), encoding='utf-8')
    (output / 'regression_summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    # Keep each original positive case visible, rather than hiding it in a mean.
    per_sequence = {}
    for name in protocol_plan(False):
        per_sequence[name] = summarize([r for r in rows if r['sequence'] == name],
            populations=('regression',), modes=('Rdisc', 'Module1'))
    (output / 'regression_per_sequence.json').write_text(json.dumps(per_sequence, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--save_dir', default='./output')
    parser.add_argument('--stage', choices=['adapter', 'scratch'], default='adapter')
    parser.add_argument('--precheck', action='store_true')
    parser.add_argument('--data_only', action='store_true', help='Check five test datasets before training; no new weights required')
    parser.add_argument('--run', action='store_true', help='Run/resume all three models on the original five datasets')
    parser.add_argument('--force', action='store_true', help='Rerun existing benchmark results')
    parser.add_argument('--regression_only', action='store_true')
    parser.add_argument('--threads', type=int, default=0)
    parser.add_argument('--num_gpus', type=int, default=1)
    args = parser.parse_args()
    if args.num_gpus < 1 or args.threads < 0 or (args.force and not args.run):
        parser.error('invalid worker/GPU settings; --force requires --run')
    torch.set_num_threads(2)
    backend()
    if args.data_only:
        benchmark_data()
        regression_data()
        print('FIVE-DATASET AND REGRESSION PREFLIGHT PASSED; no checkpoint required')
        return
    datasets, fingerprint = precheck(args.save_dir, args.stage)
    if args.precheck:
        print('FORMAL ACCEPTANCE PRECHECK PASSED')
        return
    output_name = 'module1_scratch_acceptance' if args.stage == 'scratch' else 'module1_formal_acceptance'
    output = Path(args.save_dir).resolve() / 'analysis' / output_name
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / 'manifest.json'
    run_id = 12843 if args.stage == 'scratch' else 12842
    protocol = dict(version=1, stage=args.stage, checkpoint_sha256=fingerprint, run_id=run_id, baseline_run_id=12842,
                    datasets=list(DATASETS), checkpoint_selection='fixed_epoch_300',
                    automatic_performance_pass=False, torch=torch.__version__)
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != protocol:
        raise ValueError('acceptance checkpoint/runtime changed; preserve old results before a new run')
    manifest_path.write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    if not args.regression_only:
        for name, dataset in datasets.items():
            trackers = [trackerlist('ostrack', parameter_name, name,
                                   run_ids=run_id if arm == 'Module1' else 12842, display_name=arm)[0]
                        for arm, parameter_name in suite_names(args.stage).items()]
            for arm, tracker in zip(NAMES, trackers):
                stamp_results(tracker, dataset, name, fingerprint[arm], args.run)
            if args.run:
                run_dataset(dataset, trackers, debug=False, threads=args.threads, num_gpus=args.num_gpus, force=args.force)
            checked_results(trackers, dataset)
            stream = io.StringIO()
            with redirect_stdout(stream):
                print_results(trackers, dataset, output_name + '_' + name, merge_results=False,
                    plot_types=('success', 'norm_prec', 'prec'), force_evaluation=True, skip_missing_seq=False)
            print(stream.getvalue(), flush=True)
            (output / (name + '_metrics.txt')).write_text(stream.getvalue(), encoding='utf-8')
            evaluation = check_and_load_precomputed_results(trackers, dataset, output_name + '_' + name)
            # Export actual per-sequence curves and scores for independent local analysis.
            def serializable(value):
                if isinstance(value, (torch.Tensor, np.ndarray)):
                    return value.tolist()
                if isinstance(value, np.generic):
                    return value.item()
                raise TypeError(type(value).__name__)
            (output / (name + '_evaluation.json')).write_text(json.dumps(evaluation, default=serializable), encoding='utf-8')
            result_index = []
            for arm, tracker in zip(NAMES, trackers):
                for seq in dataset:
                    path = Path(resolve_result_bbox_path(tracker.results_dir, seq))
                    result_index.append(dict(arm=arm, sequence=seq.name, frames=len(seq.frames),
                        path=str(path), sha256=file_hash(path)))
            (output / (name + '_result_index.json')).write_text(json.dumps(result_index), encoding='utf-8')
    regression(args.save_dir, output, args.stage)
    archive = output.parent / (output_name + '.tar.gz')
    with tarfile.open(archive, 'w:gz') as bundle:
        for path in sorted(output.rglob('*')):
            if path.is_file():
                bundle.add(path, arcname=path.relative_to(output).as_posix())
    print('ACCEPTANCE EVIDENCE:', archive)
    print('Completed evaluation; effectiveness requires reviewing all five benchmarks and regression cases.')


if __name__ == '__main__':
    main()
