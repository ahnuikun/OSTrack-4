"""Tracker-result paths: canonical writer layout and legacy read compatibility."""

import filecmp
import os


_DATASET_SUBDIRS = frozenset(('got10k', 'trackingnet'))


def result_base_path(results_dir, seq):
    """Return the path stem for one sequence, without an output suffix."""
    if seq.dataset in _DATASET_SUBDIRS:
        return os.path.join(results_dir, seq.dataset, seq.name)
    return os.path.join(results_dir, seq.name)


def result_bbox_path(results_dir, seq, object_id=None):
    """Return the bounding-box TXT path written by the tracker runner."""
    base = result_base_path(results_dir, seq)
    if object_id is not None:
        base += f'_{object_id}'
    return base + '.txt'


def resolve_result_bbox_path(results_dir, seq, object_id=None):
    """Read a GOT-10k result from either layout, rejecting conflicting copies.

    New runs always write the canonical nested path. Some existing GOT-10k
    results are flat; their provenance cannot be inferred from the path alone.
    Return the canonical path when no result exists so callers can report it.
    """
    canonical = result_bbox_path(results_dir, seq, object_id)
    if seq.dataset != 'got10k':
        return canonical

    suffix = f'_{object_id}' if object_id is not None else ''
    legacy = os.path.join(results_dir, f'{seq.name}{suffix}.txt')
    canonical_exists = os.path.isfile(canonical)
    legacy_exists = os.path.isfile(legacy)
    if canonical_exists and legacy_exists:
        if not filecmp.cmp(canonical, legacy, shallow=False):
            raise ValueError(
                'Conflicting GOT-10k result copies; refuse ambiguous analysis: '
                f'{canonical} != {legacy}'
            )
        return canonical
    if legacy_exists:
        return legacy
    return canonical
