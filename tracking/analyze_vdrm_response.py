"""CPU-only Rresp candidate/early-failure audit; no datasets or weights needed."""

import argparse
from pathlib import Path
import sys

import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from lib.test.analysis.frozen_failure_diagnostics import dump_json, write_bundle
from lib.test.analysis.response_failure_audit import analyze_bundle
from lib.train.frozen_vdrm import file_hash


def run(bundle, output):
    bundle, output = Path(bundle), Path(output)
    if output.exists():
        raise FileExistsError('refusing to overwrite response analysis: ' + str(output))
    if not bundle.is_file():
        raise FileNotFoundError('response probe missing: ' + str(bundle))
    if output.suffixes[-2:] != ['.tar', '.gz']:
        raise ValueError('analysis output must end in .tar.gz')
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(min(previous_threads, 2))
        summary, frames, report = analyze_bundle(bundle)
    finally:
        torch.set_num_threads(previous_threads)
    source_hash = file_hash(bundle)
    summary['source_bundle_sha256'] = source_hash
    manifest = dict(kind='vdrm_response_failure_analysis', version=1, dataset='got10k_vdrm_dev',
        source_bundle=str(bundle.resolve()), source_bundle_sha256=source_hash,
        profile=summary['profile'], records=summary['records'], optimizer_steps=0,
        full_dev_gate_passed=False, automatic_identity_verdict=False)
    write_bundle(output, manifest, {'summary.json': dump_json(summary), 'per_frame.json': dump_json(frames),
                                    'report.md': report.encode('utf-8')})
    print(report)
    print(f"RRESP ANALYSIS COMPLETE: {summary['records']} records; no performance/identity verdict")
    print('Analysis bundle:', output.resolve())
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', required=True, help='Verified Rresp focus or early_failure .tar.gz')
    parser.add_argument('--output', help='New analysis .tar.gz path; existing files are never overwritten')
    args = parser.parse_args()
    output = args.output or str(Path(args.bundle).with_name(Path(args.bundle).name.removesuffix('.tar.gz') + '_analysis.tar.gz'))
    run(args.bundle, output)


if __name__ == '__main__':
    main()
