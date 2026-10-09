"""CPU-only decomposition of existing native identity results; no new inference."""
import argparse
import json
from pathlib import Path
import sys
import tarfile

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.test.analysis.identity_native import check_bundle
from lib.test.analysis.identity_guard import diagnose_sequence, diagnose_summary
from lib.test.analysis.candidate_cues import read_npz
from lib.train.frozen_vdrm import file_hash


def review(bundle):
    manifest, original = check_bundle(bundle)
    specs = {s['member']: s for s in manifest['sequences']}
    records = []
    with tarfile.open(bundle, 'r:gz') as archive:
        for member in archive:
            if member.name in specs:
                arrays = read_npz(archive.extractfile(member).read())
                records.append(diagnose_sequence(specs[member.name]['name'], arrays))
    return dict(source_sha256=file_hash(bundle), original=original,
        interpretation='posthoc mechanism diagnosis; NOT a new independent validation',
        summary=diagnose_summary(records), per_sequence=records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    path = Path(args.output)
    if path.exists():
        raise FileExistsError('refusing to overwrite native review: '+str(path))
    result = review(args.bundle)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
    print('NATIVE IDENTITY REVIEW COMPLETE:', path.resolve())
    for arm, evidence in result['summary']['heads'].items():
        print(arm, 'localization-only rollback (POSTHOC):', evidence['localization_lock'])


if __name__ == '__main__':
    main()
