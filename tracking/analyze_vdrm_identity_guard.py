"""CPU verification of train-only threshold selection and a fixed held-out decision."""
import argparse
from pathlib import Path
import sys
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.test.analysis.identity_guard_bundle import check_bundle, render_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', required=True)
    args = parser.parse_args(); torch.set_num_threads(2)
    manifest, result = check_bundle(args.bundle)
    print(render_report(result))
    print('IDENTITY GUARD BUNDLE CHECK PASSED:', len(manifest['sequences']), 'sequences; NOT deployment approval')


if __name__ == '__main__': main()
