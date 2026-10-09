"""Verify saved gate artifacts by reconstructing features and refitting from source."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.test.analysis.search_gate import PROTOCOL, cross_validate, evaluate, render_report
from tracking.screen_vdrm_search_gate import file_sha, load_source


def compare(expected, saved, field):
    """Tight numeric tolerance permits BLAS differences, not threshold/role changes."""
    if isinstance(expected, dict):
        if not isinstance(saved, dict) or set(expected) != set(saved):
            raise ValueError('gate fields differ: '+field)
        for key in expected:
            compare(expected[key], saved[key], field+'.'+key)
    elif isinstance(expected, list):
        if not isinstance(saved, list) or len(expected) != len(saved):
            raise ValueError('gate list differs: '+field)
        for i, (a, b) in enumerate(zip(expected, saved)):
            compare(a, b, field+f'[{i}]')
    elif isinstance(expected, float):
        if isinstance(saved, bool) or not isinstance(saved, (int, float)) or not np.isclose(expected, saved, rtol=1e-9, atol=1e-10):
            raise ValueError('gate numeric value differs: '+field)
    elif type(expected) is not type(saved) or expected != saved:
        raise ValueError('gate value differs: '+field)


def verify(bundle, result_dir):
    root = Path(result_dir).resolve()
    required = {'manifest.json', 'summary.json', 'records.json', 'oof_predictions.npz', 'report.md'}
    if set(p.name for p in root.iterdir()) != required | {'checksums.json'}:
        raise ValueError('gate output member set differs')
    hashes = json.loads((root/'checksums.json').read_text(encoding='utf-8'))
    if set(hashes) != required or any(file_sha(root/name) != digest for name, digest in hashes.items()):
        raise ValueError('gate artifact checksum differs')
    metadata = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    source_hash = file_sha(bundle)
    if metadata.get('source_sha256') != source_hash:
        raise ValueError('gate source bundle differs')
    source, joined, proposal = load_source(bundle)
    predictions, models = cross_validate(joined)
    expected = dict(kind='vdrm_search_gate', protocol=PROTOCOL, source_sha256=source_hash,
                    source_checkpoint_sha256=source['checkpoint_sha256'], source_unchanged=True,
                    deterministic_refit_equal=True, tracker_optimizer_steps=0, folds=models)
    compare(expected, metadata, 'manifest')
    arrays = dict(**joined, **predictions)
    with np.load(root/'oof_predictions.npz', allow_pickle=False) as saved:
        if set(arrays) != set(saved.files):
            raise ValueError('gate per-frame fields differ')
        for key, value in arrays.items():
            actual = saved[key]
            if value.shape != actual.shape or value.dtype != actual.dtype:
                raise ValueError('gate array shape/type differs: '+key)
            equal = (np.allclose(value, actual, rtol=1e-9, atol=1e-10) if value.dtype.kind == 'f'
                     else np.array_equal(value, actual))
            if not equal:
                raise ValueError('gate features/labels/predictions differ: '+key)
    result, records = evaluate(joined, predictions)
    result['proposal_diagnostics'] = proposal
    summary = json.loads((root/'summary.json').read_text(encoding='utf-8'))
    compare(result, summary, 'summary')
    compare(records, json.loads((root/'records.json').read_text(encoding='utf-8')), 'records')
    if (root/'report.md').read_text(encoding='utf-8') != render_report(summary):
        raise ValueError('gate report differs')
    if source_hash != file_sha(bundle):
        raise ValueError('gate source changed during verification')
    print(render_report(summary))
    print('SEARCH GATE ARTIFACT CHECK PASSED:', len(records), 'sequences; development diagnostic only')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', required=True)
    parser.add_argument('--result_dir', required=True)
    args = parser.parse_args(); torch.set_num_threads(2)
    verify(args.bundle, args.result_dir)


if __name__ == '__main__':
    main()
