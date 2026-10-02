"""Compare fixed GOT-10k VDRM dev results from multiple tracker configs.

This script reads existing tracking result TXT files; it does not train or
rerun tracking. All arms are evaluated together so the saved cache contains
their paired per-sequence scores instead of silently replacing one arm.
"""

import argparse
import os
import pickle
import sys

import numpy as np

project_path = os.path.join(os.path.dirname(__file__), '..')
if project_path not in sys.path:
    sys.path.append(project_path)

from lib.test.analysis.plot_results import print_per_sequence_results, print_results
from lib.test.evaluation import get_dataset, trackerlist
from lib.test.evaluation.environment import env_settings
from lib.test.evaluation.result_paths import result_bbox_path, resolve_result_bbox_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tracker_params', nargs='+', required=True,
                        help='Config names whose result TXT files already exist.')
    parser.add_argument('--per_sequence', action='store_true',
                        help='Also print each sequence mean overlap.')
    parser.add_argument('--reference', required=True,
                        help='Config name used as the paired reference arm.')
    args = parser.parse_args()

    dataset_name = 'got10k_vdrm_dev'
    dataset = get_dataset(dataset_name)
    if len(dataset) != 152:
        raise ValueError(f'Expected 152 fixed dev sequences, got {len(dataset)}')
    trackers = [
        trackerlist(name='ostrack', parameter_name=name,
                    dataset_name=dataset_name, run_ids=None,
                    display_name=name)[0]
        for name in args.tracker_params
    ]
    if args.reference not in args.tracker_params:
        parser.error('--reference must also appear in --tracker_params')
    missing_by_arm = {}
    layout_by_arm = {}
    for tracker in trackers:
        missing = []
        nested = 0
        flat = 0
        for seq in dataset:
            path = resolve_result_bbox_path(tracker.results_dir, seq)
            if not os.path.isfile(path):
                missing.append(path)
            elif path == result_bbox_path(tracker.results_dir, seq):
                nested += 1
            else:
                flat += 1
        layout_by_arm[tracker.parameter_name] = (nested, flat)
        if missing:
            missing_by_arm[tracker.parameter_name] = missing
    if missing_by_arm:
        details = '\n'.join(
            f'{name}: missing {len(paths)}/152, first: {paths[0]}'
            for name, paths in missing_by_arm.items()
        )
        raise FileNotFoundError(
            'Complete every tracking arm before paired analysis:\n' + details
        )
    for name, (nested, flat) in layout_by_arm.items():
        print(f'{name}: {nested} nested + {flat} flat GOT-10k result files')
    report_name = 'vdrm_module1_dev'
    print(f'Analyzing fixed GOT-10k VDRM dev: {len(dataset)}/152 sequences')
    print_results(trackers, dataset, report_name, merge_results=True,
                  plot_types=('success', 'norm_prec', 'prec'),
                  force_evaluation=True, skip_missing_seq=False)
    if args.per_sequence:
        print_per_sequence_results(trackers, dataset, report_name,
                                   merge_results=True, force_evaluation=False,
                                   skip_missing_seq=False)

    cache_path = os.path.join(
        env_settings().result_plot_path, report_name, 'eval_data.pkl'
    )
    with open(cache_path, 'rb') as file:
        evaluated = pickle.load(file)
    if not all(evaluated['valid_sequence']):
        raise ValueError('Paired analysis requires all 152 dev sequences')
    names = [row['param'] for row in evaluated['trackers']]
    mean_overlap = np.asarray(evaluated['avg_overlap_all'], dtype=np.float64)
    reference_index = names.index(args.reference)
    rng = np.random.default_rng(42)
    resample = rng.integers(0, len(dataset), size=(5000, len(dataset)))
    print('\nPaired per-sequence mean-overlap differences vs ' + args.reference)
    print('Arm | Delta AO (pp) | 95% sequence-bootstrap CI (pp) | Improved/Worsened/Tied')
    for index, name in enumerate(names):
        if index == reference_index:
            continue
        delta = 100.0 * (mean_overlap[:, index] - mean_overlap[:, reference_index])
        bounds = np.quantile(delta[resample].mean(axis=1), [0.025, 0.975])
        improved = int(np.count_nonzero(delta > 1e-9))
        worsened = int(np.count_nonzero(delta < -1e-9))
        tied = len(delta) - improved - worsened
        print(f'{name} | {delta.mean():+.3f} | '
              f'[{bounds[0]:+.3f}, {bounds[1]:+.3f}] | '
              f'{improved}/{worsened}/{tied}')


if __name__ == '__main__':
    main()
