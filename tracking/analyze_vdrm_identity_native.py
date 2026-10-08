"""CPU-only integrity, trajectory and metric checks for the native identity probe."""
import argparse
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.test.analysis.identity_native import check_bundle, render_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    _, summary = check_bundle(args.bundle)
    print(render_report(summary))
    print('IDENTITY NATIVE BUNDLE CHECK PASSED:', summary['sequences'],
          'sequences; same-crop counterfactuals, NOT a tracking pass')


if __name__ == '__main__':
    main()
