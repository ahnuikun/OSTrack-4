import json
from pathlib import Path
import tarfile
import tempfile
from types import SimpleNamespace
import unittest

from tracking.export_vdrm_module1_keyframes import (
    KEYFRAME_PLAN, collect_keyframes, write_archive,
)


class KeyframeExportTest(unittest.TestCase):
    def dataset(self, root):
        dataset = []
        for name, ranges in KEYFRAME_PLAN.items():
            size = max(end for _, end in ranges)
            folder = root / name
            folder.mkdir()
            frames = []
            for number in range(1, size + 1):
                # Filename deliberately differs from the usual GOT-10k name.
                path = folder / f'image-{number}.jpg'
                path.write_bytes(str(number).encode('ascii'))
                frames.append(str(path))
            dataset.append(SimpleNamespace(
                name=name, frames=frames,
                ground_truth_rect=[[1, 2, 3, 4] for _ in frames],
            ))
        return dataset

    def test_archive_uses_adapter_order_and_contains_all_60_selected_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            items = collect_keyframes(self.dataset(root))
            path = root / 'keyframes.tar.gz'
            write_archive(path, items)
            with tarfile.open(path) as archive:
                manifest = json.load(archive.extractfile('manifest.json'))
                self.assertEqual(manifest['frames'], 60)
                self.assertEqual(manifest['sequences'], 6)
                self.assertEqual(len(archive.getmembers()), 61)
                for name in KEYFRAME_PLAN:
                    self.assertEqual(
                        archive.extractfile(f'images/{name}/00000001.jpg').read(), b'1',
                    )
                self.assertEqual(
                    archive.extractfile('images/GOT-10k_Train_008669/00000017.jpg').read(),
                    b'17',
                )

    def test_missing_source_is_detected_before_archive_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = self.dataset(root)
            Path(dataset[0].frames[0]).unlink()
            with self.assertRaises(FileNotFoundError):
                collect_keyframes(dataset)
            self.assertFalse((root / 'keyframes.tar.gz').exists())

    def test_existing_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'keyframes.tar.gz'
            path.write_bytes(b'existing bundle')
            with self.assertRaises(FileExistsError):
                write_archive(path, [])
            self.assertEqual(path.read_bytes(), b'existing bundle')


if __name__ == '__main__':
    unittest.main()
