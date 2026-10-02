"""Package 60 existing dev images around the localized tracking failures.

Uses the dataset adapter's frame order rather than assuming a JPG filename
format. Reads existing images only; no checkpoint or GPU inference is needed.
"""

import argparse
import io
import json
from pathlib import Path
import sys
import tarfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_NAME = 'got10k_vdrm_dev'
# Inclusive, 1-based ranges. The selection is diagnostic, not a new eval split.
KEYFRAME_PLAN = {
    'GOT-10k_Train_001201': ((1, 3), (8, 11)),
    'GOT-10k_Train_004083': ((1, 1), (80, 83), (88, 91)),
    'GOT-10k_Train_008669': ((1, 1), (15, 18), (47, 50)),
    'GOT-10k_Train_007878': ((1, 1), (62, 66), (75, 79), (85, 88), (93, 96)),
    'GOT-10k_Train_002126': ((1, 1), (88, 92)),
    'GOT-10k_Train_007764': ((1, 1), (7, 10), (39, 43)),
}


def collect_keyframes(dataset):
    """Resolve and check every source before creating the output archive."""
    sequences = {sequence.name: sequence for sequence in dataset}
    if len(sequences) != len(dataset):
        raise ValueError('Dataset has duplicate sequence names')
    items = []
    for name, ranges in KEYFRAME_PLAN.items():
        if name not in sequences:
            raise ValueError(f'Missing registered dev sequence: {name}')
        sequence = sequences[name]
        if len(sequence.frames) != len(sequence.ground_truth_rect):
            raise ValueError(f'Image/GT frame counts differ: {name}')
        selected = sorted({frame for first, last in ranges
                           for frame in range(first, last + 1)})
        for number in selected:
            if number < 1 or number > len(sequence.frames):
                raise ValueError(f'Frame {number} is outside {name}')
            source = Path(sequence.frames[number - 1])
            if not source.is_file():
                raise FileNotFoundError(source)
            items.append({
                'sequence': name,
                'frame_1based': number,
                'source_path': str(source),
                'archive_path': f'images/{name}/{number:08d}{source.suffix.lower()}',
                'gt_xywh': [float(v) for v in sequence.ground_truth_rect[number - 1]],
            })
    return items


def write_archive(output, items):
    """Write a fresh archive containing the checked images and their manifest."""
    output = Path(output)
    manifest = {
        'dataset': DATASET_NAME,
        'frame_numbering': '1-based; follows the project dataset adapter',
        'purpose': 'Visual confirmation of localized failures, not model selection',
        'sequences': len({row['sequence'] for row in items}),
        'frames': len(items),
        'items': items,
    }
    payload = json.dumps(manifest, ensure_ascii=False, indent=2).encode('utf-8')
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation preserves an already-downloaded/generated bundle.
    with output.open('xb') as target, tarfile.open(fileobj=target, mode='w:gz') as archive:
        info = tarfile.TarInfo('manifest.json')
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
        for row in items:
            source = Path(row['source_path'])
            info = tarfile.TarInfo(row['archive_path'])
            info.size = source.stat().st_size
            with source.open('rb') as image:
                archive.addfile(info, image)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='output/vdrm_module1_keyframes.tar.gz',
                        help='New archive path; refuses to overwrite an existing file.')
    args = parser.parse_args()
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from lib.test.evaluation import get_dataset

    dataset = get_dataset(DATASET_NAME)
    if len(dataset) != 152:
        raise ValueError(f'Expected 152 fixed dev sequences, got {len(dataset)}')
    items = collect_keyframes(dataset)
    manifest = write_archive(args.output, items)
    print(f"Packed {manifest['frames']} existing frames from {manifest['sequences']} dev sequences")
    print('Archive:', Path(args.output).resolve())


if __name__ == '__main__':
    main()
