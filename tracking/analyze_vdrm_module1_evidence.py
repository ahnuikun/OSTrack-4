"""CPU-only independent step01 evidence analysis; no datasets/checkpoints."""

import argparse
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from lib.test.analysis.module1_evidence import check_bundle


def run(bundle, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError('refusing to overwrite analysis: ' + str(output))
    summary, frames, report = check_bundle(bundle)
    # Import writing helpers only after validation. This remains CPU-only.
    from lib.test.analysis.frozen_failure_diagnostics import dump_json, write_bundle
    manifest = dict(kind='vdrm_module1_step01_analysis', version=1,
                    source_sha256=summary['source_sha256'], optimizer_steps=0)
    write_bundle(output, manifest, {'summary.json': dump_json(summary),
        'per_frame.json': dump_json(frames), 'report.md': report.encode('utf-8')})
    print(report)
    print('STEP01 ANALYSIS COMPLETE; no automatic identity/performance verdict')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run(args.bundle, args.output)


if __name__ == '__main__':
    main()
