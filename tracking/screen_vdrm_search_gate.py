"""CPU-only cached gate screen. Refuses overwrites; never loads tracker weights."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tarfile

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.test.analysis.candidate_cues import read_npz, npz_bytes
from lib.test.analysis.search_coverage import check_bundle
from lib.test.analysis.search_gate import (PROTOCOL, dataset_rows, proposal_diagnostics,
                                         cross_validate, evaluate, render_report)


def file_sha(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_source(bundle):
    manifest, _ = check_bundle(bundle)
    rows, proposal = [], {}
    with tarfile.open(bundle, 'r:gz') as archive:
        for spec in manifest['sequences']:
            arrays = read_npz(archive.extractfile(spec['member']).read())
            rows.append(dataset_rows(spec['sequence_id'], arrays))
            for key, value in proposal_diagnostics(arrays).items():
                proposal[key] = proposal.get(key, 0)+value
    joined = {key: np.concatenate([r[key] for r in rows]) for key in rows[0]}
    return manifest, joined, proposal


def run(bundle, output_dir):
    source, destination = Path(bundle).resolve(), Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError('output already exists: '+str(destination))
    source_hash = file_sha(source)
    manifest, joined, proposal = load_source(source)
    predictions, models = cross_validate(joined)
    result, records = evaluate(joined, predictions)
    result['proposal_diagnostics'] = proposal
    # Independent fit/replay: thresholds, OOF flags and metrics must reproduce.
    repeated, repeated_models = cross_validate(joined)
    if models != repeated_models or any(not np.array_equal(predictions[k], repeated[k]) for k in predictions):
        raise ValueError('gate deterministic refit differs')
    if file_sha(source) != source_hash:
        raise ValueError('source coverage bundle changed during screen')
    document = dict(kind='vdrm_search_gate', protocol=PROTOCOL, source_sha256=source_hash,
                    source_checkpoint_sha256=manifest['checkpoint_sha256'], source_unchanged=True,
                    deterministic_refit_equal=True, tracker_optimizer_steps=0, folds=models)
    destination.mkdir(parents=True, exist_ok=False)
    for name, value in (('manifest.json', document), ('summary.json', result), ('records.json', records)):
        (destination/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    (destination/'oof_predictions.npz').write_bytes(npz_bytes(dict(**joined, **predictions)))
    (destination/'report.md').write_text(render_report(result), encoding='utf-8')
    hashes = {p.name: file_sha(p) for p in sorted(destination.iterdir())}
    (destination/'checksums.json').write_text(json.dumps(hashes, indent=2)+'\n', encoding='utf-8')
    print(render_report(result))
    print('SEARCH GATE SCREEN COMPLETE:', destination)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', required=True)
    parser.add_argument('--output_dir', required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    run(args.bundle, args.output_dir)


if __name__ == '__main__':
    main()
