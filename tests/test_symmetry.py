from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import yaml

from court_keypoints.prepare_dataset import (
    canonicalize_side_camera,
    reorder_points,
    schema_coordinates,
    symmetry_index,
)
from court_keypoints.train import load_evaluation_videos


class SymmetryCanonicalizationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.coordinates = schema_coordinates(
            [
                {"index": index, "planeX": point[0], "planeY": point[1]}
                for index, point in enumerate(
                    [
                        (10, 88), (10, 62.666667), (10, 50), (10, 37.333333), (10, 12),
                        (90, 12), (90, 37.333333), (90, 50), (90, 62.666667), (90, 88),
                        (10, 79.555556), (10, 71.111111), (10, 58.444445), (10, 54.222222),
                        (10, 45.777778), (10, 41.555555), (10, 28.888889), (10, 20.444444),
                        (36.666667, 12), (63.333333, 12), (90, 20.444444), (90, 28.888889),
                        (90, 41.555555), (90, 45.777778), (90, 54.222222), (90, 58.444445),
                        (90, 71.111111), (90, 79.555556), (63.333333, 88), (36.666667, 88),
                        (36.666667, 62.666667), (63.333333, 62.666667),
                        (36.666667, 50), (63.333333, 50),
                        (36.666667, 37.333333), (63.333333, 37.333333),
                    ]
                )
            ]
        )
        cls.symmetries = {
            "identity": symmetry_index(cls.coordinates, mirror_plane_x=False, mirror_plane_y=False),
            "width": symmetry_index(cls.coordinates, mirror_plane_x=True, mirror_plane_y=False),
            "length": symmetry_index(cls.coordinates, mirror_plane_x=False, mirror_plane_y=True),
            "both": symmetry_index(cls.coordinates, mirror_plane_x=True, mirror_plane_y=True),
        }
        cls.canonical = [
            {
                "index": index,
                "x": float((100.0 - point[1]) / 100.0),
                "y": float(point[0] / 100.0),
                "visibility": 2,
                "source": "test",
            }
            for index, point in enumerate(cls.coordinates)
        ]

    def test_every_rectangle_symmetry_returns_same_canonical_order(self) -> None:
        expected = np.asarray([(point["x"], point["y"]) for point in self.canonical])
        for name, index_map in self.symmetries.items():
            with self.subTest(name=name):
                permuted = reorder_points(self.canonical, index_map)
                canonicalized, audit = canonicalize_side_camera(
                    permuted,
                    self.coordinates,
                    self.symmetries,
                )
                actual = np.asarray([(point["x"], point["y"]) for point in canonicalized])
                np.testing.assert_allclose(actual, expected)
                self.assertEqual(audit["transform"], name)
                self.assertEqual(audit["viewMode"], "sideline")
                self.assertGreater(audit["margin"], 0)

    def test_endline_view_uses_vertical_length_axis(self) -> None:
        canonical = [
            {
                "index": index,
                "x": float(point[0] / 100.0),
                "y": float(point[1] / 100.0),
                "visibility": 2,
                "source": "test",
            }
            for index, point in enumerate(self.coordinates)
        ]
        expected = np.asarray([(point["x"], point["y"]) for point in canonical])
        for name, index_map in self.symmetries.items():
            with self.subTest(name=name):
                permuted = reorder_points(canonical, index_map)
                canonicalized, audit = canonicalize_side_camera(
                    permuted,
                    self.coordinates,
                    self.symmetries,
                )
                actual = np.asarray([(point["x"], point["y"]) for point in canonicalized])
                np.testing.assert_allclose(actual, expected)
                self.assertEqual(audit["transform"], name)
                self.assertEqual(audit["viewMode"], "endline")
                self.assertGreater(audit["margin"], 0)

    def test_fixed_evaluation_manifest_has_all_three_videos(self) -> None:
        root = Path(__file__).resolve().parents[1]
        payload = yaml.safe_load((root / "eval-videos.yaml").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_root = Path(temporary_directory)
            for item in payload["videos"]:
                item["file"] = f"{item['id']}.mp4"
                (temporary_root / item["file"]).write_bytes(b"test")
            manifest = temporary_root / "eval-videos.yaml"
            manifest.write_text(yaml.safe_dump(payload), encoding="utf-8")
            videos = load_evaluation_videos(manifest)
        self.assertEqual(
            [video["id"] for video in videos],
            ["pW66S38FAQM", "rMvxEtorQhw", "gdrSr-Vso90"],
        )


if __name__ == "__main__":
    unittest.main()
