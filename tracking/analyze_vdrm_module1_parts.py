"""CPU-only step02 part readouts on existing step01 evidence."""

import argparse
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from lib.test.analysis.module1_parts import analyze

DEFAULT_ANNOTATIONS = PROJECT / 'tracking/vdrm_diagnostics/module1_identity_rois_v1.json'


def run(bundle, output, annotations=DEFAULT_ANNOTATIONS):
    output = Path(output)
    if output.exists():
        raise FileExistsError('refusing to overwrite step02 output: ' + str(output))
    summary, frames, report = analyze(bundle, annotations, progress=print)
    from lib.test.analysis.frozen_failure_diagnostics import dump_json, write_bundle
    manifest = dict(kind='vdrm_module1_step02_parts', version=1,
                    source_sha256=summary['source_sha256'], annotations_sha256=summary['annotations_sha256'],
                    optimizer_steps=0, tracking_performance_passed=False)
    write_bundle(output, manifest, {'summary.json': dump_json(summary), 'per_frame.json': dump_json(frames),
        'annotations.json': Path(annotations).read_bytes(), 'report.md': report.encode('utf-8')})
    print(report)
    print('STEP02 PARTS ANALYSIS COMPLETE; static readouts, not tracking performance')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--annotations', default=str(DEFAULT_ANNOTATIONS))
    args = parser.parse_args()
    run(args.bundle, args.output, args.annotations)


if __name__ == '__main__':
    main()
