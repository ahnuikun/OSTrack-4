"""Standalone verification of the train-domain identity guard screening bundle."""
import hashlib
import json
from pathlib import PurePosixPath
import tarfile

from lib.train.identity_offline import ARMS
from lib.test.analysis.candidate_cues import read_npz
from lib.test.analysis.identity_guard import PROTOCOL, partition, grid_record, decide


def render_report(decision):
    lines = ['# Frozen identity guard screen', '',
        'TRAIN-domain fit/held-out audit; no official 152-sequence dev threshold fitting.',
        'Same-crop only. No optimizer, state feedback, new reliability model, or deployment approval.', '',
        'Frozen choice: '+str(decision['selected']),
        'Held-out result: '+str(decision['heldout']),
        'Held-out screen passed: '+str(decision['heldout_screen_passed']),
        'Rejection reasons: '+str(decision['rejection_reasons']),
        'Next action: '+decision['next_action'],
        'Automatic integration approved: False', '']
    return '\n'.join(lines)


def check_bundle(path):
    with tarfile.open(path, 'r:gz') as archive:
        members = archive.getmembers()
        names = [m.name for m in members]
        if len(names) != len(set(names)) or any(not m.isfile() or PurePosixPath(m.name).is_absolute() or
                '..' in PurePosixPath(m.name).parts or '\\' in m.name for m in members):
            raise ValueError('unsafe guard bundle')
        manifest = json.load(archive.extractfile('manifest.json'))
        expected_partition = partition(manifest['screen_train_ids'])
        choice = manifest['frozen_choice']
        disabled = choice['threshold'] == PROTOCOL['disabled_threshold']
        specs = manifest['sequences']
        actual = {s: [r['sequence_id'] for r in specs if r['split'] == s] for s in ('fit', 'audit')}
        expected = dict(fit=expected_partition['fit'], audit=[] if disabled else expected_partition['audit'])
        if (manifest.get('kind') != 'vdrm_identity_guard' or manifest.get('protocol') != PROTOCOL or
                manifest.get('partition') != expected_partition or actual != expected or
                len(specs) != sum(len(ids) for ids in expected.values()) or
                any(s['name'] != f'GOT-10k_Train_{s["sequence_id"]+1:06d}' for s in specs) or
                manifest.get('optimizer_steps') != 0 or manifest.get('parameter_gradient_tensors') != 0 or
                set(manifest['states_before']) != {'Tclean', *ARMS} or
                manifest['states_before'] != manifest['states_after']):
            raise ValueError('guard partition/state/budget contract differs')
        for key in ('source_screen_sha256', 'checkpoint_sha256', 'data_list_sha256'):
            digest = manifest[key]
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
                raise ValueError('invalid guard source hash')
        by_member = {s['member']: s for s in specs}
        expected_members = set(by_member) | {'records.json', 'decision.json', 'frozen_choice.json', 'report.md'}
        if (len(by_member) != len(specs) or set(names) != expected_members | {'manifest.json'} or
                set(manifest['members_sha256']) != expected_members):
            raise ValueError('guard bundle member set differs')
        records, documents = [], {}
        for member in members:
            if member.name == 'manifest.json':
                continue
            data = archive.extractfile(member).read()
            if hashlib.sha256(data).hexdigest() != manifest['members_sha256'][member.name]:
                raise ValueError('guard checksum differs')
            if member.name in by_member:
                spec = by_member[member.name]
                arrays = read_npz(data)
                if len(arrays['native_boxes']) != spec['frames']-1 or spec['exact_replay_frames'] != [2]:
                    raise ValueError('incomplete guard native trajectory')
                records.append(grid_record(spec['name'], spec['split'], arrays,
                    selected=choice if spec['split'] == 'audit' else None))
            else:
                documents[member.name] = data
        order = {r['name']: i for i, r in enumerate(specs)}
        records.sort(key=lambda r: order[r['sequence']])
        result = decide(records)
        if result['selected'] != choice or json.loads(documents['frozen_choice.json']) != choice:
            raise ValueError('held-out data changed frozen fit choice')
        if records != json.loads(documents['records.json']) or result != json.loads(documents['decision.json']):
            raise ValueError('independent guard decision recomputation differs')
        if render_report(result) != documents['report.md'].decode('utf-8'):
            raise ValueError('guard report differs')
        if manifest['native_tracking_steps'] != sum(s['frames']-1 for s in specs):
            raise ValueError('guard native step count differs')
    return manifest, result
