"""CPU verification of step04 GT-location proxy evidence."""

import argparse
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))

from lib.test.analysis.module1_location import check_bundle,report_text


def run(bundle,output):
    if Path(output).exists(): raise FileExistsError('refusing to overwrite location analysis: '+str(output))
    summary,frames = check_bundle(bundle)
    from lib.test.analysis.frozen_failure_diagnostics import dump_json,write_bundle
    write_bundle(output,dict(kind='vdrm_module1_step04_location_analysis',version=1,
        source_sha256=summary['source_sha256'],evidence_sha256=summary['evidence_sha256'],optimizer_steps=0),
        {'summary.json':dump_json(summary),'per_frame.json':dump_json(frames),'report.md':report_text(summary).encode('utf-8')})
    print(f'STEP04 ANALYSIS COMPLETE: smoke={summary["smoke_only"]}; GT-location proxy only')
    print('Download:',Path(output).resolve())
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',required=True)
    parser.add_argument('--output',required=True)
    args = parser.parse_args()
    run(args.bundle,args.output)


if __name__ == '__main__': main()
