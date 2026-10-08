"""Recheck a completed v1 screen, optional raw cache and saved head checkpoints."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import tarfile

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from lib.test.analysis.identity_native import load_screen_heads
from lib.test.analysis.identity_offline import check_bundle
from lib.test.analysis.candidate_cues import read_npz
from lib.train.identity_offline import ARMS, read_cache
from lib.train.frozen_vdrm import file_hash


def review(bundle, cache=None, screen_dir=None):
    manifest, summary = check_bundle(bundle)
    with tarfile.open(bundle) as archive:
        contents = {m.name: archive.extractfile(m).read() for m in archive}
    rows = json.loads(contents['rows.json'])
    outcomes = json.loads(contents['outcomes.json'])
    base = np.array([r['base_scores'][0] > max(r['base_scores'][1:]) for r in rows])
    result = dict(bundle_sha256=file_hash(bundle), summary=summary, flips={}, cases=[],
                  raw_cache_checked=False, saved_checkpoints_checked=False)
    for arm, value in outcomes.items():
        correct = np.asarray(value['candidate_correct'])
        result['flips'][arm] = dict(repairs=int((~base & correct).sum()),
            damages=int((base & ~correct).sum()), both_wrong=int((~base & ~correct).sum()))
    for i, row in enumerate(rows):
        correct = {a: bool(outcomes[a]['candidate_correct'][i]) for a in ARMS}
        if not base[i] or not all(correct.values()):
            result['cases'].append(dict(sequence=row['sequence'], frame_1based=row['frame_1based'],
                cover=row['cover'], base_correct=bool(base[i]), heads_correct=correct))
    if screen_dir is not None:
        for arm in ARMS:
            checkpoint = torch.load(Path(screen_dir)/f'Identity_{arm}_ep0020.pth', map_location='cpu', weights_only=True)
            exported = read_npz(contents['weights_'+arm+'.npz'])
            if (checkpoint.get('arm') != arm or checkpoint.get('epoch') != 20 or
                    checkpoint.get('cache_index_sha256') != manifest['cache_index_sha256'] or
                    set(checkpoint['net']) != set(exported) or
                    any(not np.array_equal(checkpoint['net'][k].numpy(), v) for k, v in exported.items())):
                raise ValueError('saved checkpoint differs from exported weight: '+arm)
        result['saved_checkpoints_checked'] = True
    if cache is not None:
        if file_hash(Path(cache)/'index.json') != manifest['cache_index_sha256']:
            raise ValueError('wrong source cache index')
        index, protocol, data = read_cache(cache)
        if index['checkpoint_sha256'] != manifest['checkpoint_sha256']:
            raise ValueError('source backbone hash differs')
        dev = [i for i, r in enumerate(index['rows']) if r['split'] == 'dev']
        if [index['rows'][i] for i in dev] != rows:
            raise ValueError('source development rows differ')
        result['coverage'] = {}
        for split in ('train', 'dev'):
            sequences = [s for s in index['sequences'] if s['split'] == split]
            selected = [r for r in index['rows'] if r['split'] == split]
            skipped = Counter()
            for sequence in sequences:
                skipped.update(sequence['skipped'])
            result['coverage'][split] = dict(sequences=len(sequences), pairs=len(selected),
                planned=sum(len(s['planned_frames_1based']) for s in sequences), skipped=dict(skipped),
                cover=dict(Counter(str(int(r['cover'])) for r in selected)))
        arrays = read_npz(contents['embeddings.npz'])
        heads, _ = load_screen_heads(bundle)
        errors = {}
        with torch.no_grad():
            for arm, head in heads.items():
                worst = 0.
                for offset in range(0, len(dev), 64):
                    ids = dev[offset:offset+64]
                    z, zm = head.encode(data['template'][ids], data['template_mask'][ids])
                    x, xm = head.encode(data['candidates'][ids], data['candidate_mask'][ids])
                    for key, value in dict(z=z, x=x, zm=zm, xm=xm).items():
                        expected = arrays[arm+'_'+key][offset:offset+len(ids)]
                        if key.endswith('m'):
                            if not np.array_equal(value.numpy(), expected):
                                raise ValueError('raw replay support differs')
                        else:
                            worst = max(worst, float(np.abs(value.numpy()-expected).max()))
                if worst > 1e-5:
                    raise ValueError('raw-feature CPU replay exceeds 1e-5 tolerance: '+arm)
                errors[arm] = worst
        result.update(raw_cache_checked=True, projected_descriptor_max_abs_error=errors)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', required=True)
    parser.add_argument('--cache_dir')
    parser.add_argument('--screen_dir')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('refusing to overwrite review: '+str(output))
    result = review(args.bundle, args.cache_dir, args.screen_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
    print('IDENTITY V1 REVIEW PASSED:', output.resolve())
    print('Repairs / damages:', result['flips'])
    print('Raw projection replay:', result.get('projected_descriptor_max_abs_error', 'not requested'))


if __name__ == '__main__':
    main()
