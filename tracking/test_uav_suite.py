import argparse
import os
import sys

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

from tracking.test import run_tracker


DATASETS = ['visdrone', 'uav123', 'uavdt', 'dtb70', 'lasot']


def parse_args():
    parser = argparse.ArgumentParser(description='Run OSTrack tests on the UAV/LaSOT evaluation suite.')
    parser.add_argument('--tracker_name', type=str, default='ostrack', help='Name of tracking method.')
    parser.add_argument('--tracker_param', type=str, required=True,
                        help='Name of tracker parameter/config file. The checkpoint is selected from this config.')
    parser.add_argument('--dataset', type=str, default='all', choices=['all'] + DATASETS,
                        help='Dataset to test. Use "all" to test all configured datasets.')
    parser.add_argument('--runid', type=int, default=None, help='The run id.')
    parser.add_argument('--sequence', type=str, default=None,
                        help='Sequence number or name. Only valid when testing one dataset.')
    parser.add_argument('--debug', type=int, default=0, help='Debug level.')
    parser.add_argument('--threads', type=int, default=0, help='Number of parallel workers.')
    parser.add_argument('--num_gpus', type=int, default=4, help='Number of GPUs available for testing.')
    return parser.parse_args()


def main():
    args = parse_args()
    datasets = DATASETS if args.dataset == 'all' else [args.dataset]

    if args.sequence is not None and len(datasets) > 1:
        raise ValueError('--sequence can only be used when --dataset is a single dataset name.')

    for dataset_name in datasets:
        print('=' * 80)
        print('Testing dataset: {}'.format(dataset_name))
        print('=' * 80)
        run_tracker(args.tracker_name, args.tracker_param, args.runid, dataset_name,
                    args.sequence, args.debug, args.threads, num_gpus=args.num_gpus)


if __name__ == '__main__':
    main()
