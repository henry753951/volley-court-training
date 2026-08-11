from __future__ import annotations

import unittest

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

from volleyball_synthetic.camera_modes import (
    CANONICAL_CAMERA_MODE_BY_SOURCE,
    DIAGONAL_BOTTOM_RIGHT,
    DIAGONAL_TOP_RIGHT,
    ENDLINE,
    FAR_CENTER,
    FAR_LEFT,
    FAR_RIGHT,
    LEFT_SIDE,
    NEAR_CENTER,
    NEAR_LEFT,
    NEAR_RIGHT,
    RIGHT_SIDE,
    SIDELINE,
    canonical_camera_mode,
    classify_outside_camera_position,
    classify_projected_camera_position,
    keypoint_permutation,
)
from volleyball_synthetic.constants import KEYPOINT_NAMES, KEYPOINT_WORLD
from volleyball_synthetic.labels import _apply_keypoint_permutation


class CameraModeTest(unittest.TestCase):
    def test_complete_eight_to_four_mapping(self) -> None:
        expected = {
            FAR_CENTER: ENDLINE,
            NEAR_CENTER: ENDLINE,
            LEFT_SIDE: SIDELINE,
            RIGHT_SIDE: SIDELINE,
            FAR_LEFT: DIAGONAL_BOTTOM_RIGHT,
            NEAR_RIGHT: DIAGONAL_BOTTOM_RIGHT,
            FAR_RIGHT: DIAGONAL_TOP_RIGHT,
            NEAR_LEFT: DIAGONAL_TOP_RIGHT,
        }
        self.assertEqual(CANONICAL_CAMERA_MODE_BY_SOURCE, expected)
        self.assertEqual(
            {source: canonical_camera_mode(source) for source in expected},
            expected,
        )

    def test_outside_camera_uses_real_location(self) -> None:
        cases = {
            (-3.0, -2.0, 5.0): FAR_LEFT,
            (4.5, -2.0, 5.0): FAR_CENTER,
            (12.0, -2.0, 5.0): FAR_RIGHT,
            (-3.0, 9.0, 5.0): LEFT_SIDE,
            (12.0, 9.0, 5.0): RIGHT_SIDE,
            (-3.0, 20.0, 5.0): NEAR_LEFT,
            (4.5, 20.0, 5.0): NEAR_CENTER,
            (12.0, 20.0, 5.0): NEAR_RIGHT,
        }
        for location, expected in cases.items():
            with self.subTest(location=location):
                self.assertEqual(classify_outside_camera_position(location), expected)
        self.assertIsNone(classify_outside_camera_position((4.5, 9.0, 30.0)))

    def test_inside_camera_uses_projected_long_axis(self) -> None:
        cases = {
            ((0.0, 0.0), (0.0, 100.0)): NEAR_CENTER,
            ((0.0, 100.0), (0.0, 0.0)): FAR_CENTER,
            ((100.0, 0.0), (0.0, 0.0)): LEFT_SIDE,
            ((0.0, 0.0), (100.0, 0.0)): RIGHT_SIDE,
            ((0.0, 0.0), (100.0, 100.0)): NEAR_RIGHT,
            ((100.0, 100.0), (0.0, 0.0)): FAR_LEFT,
            ((100.0, 0.0), (0.0, 100.0)): NEAR_LEFT,
            ((0.0, 100.0), (100.0, 0.0)): FAR_RIGHT,
        }
        for (far_endpoint, near_endpoint), expected in cases.items():
            with self.subTest(far_endpoint=far_endpoint, near_endpoint=near_endpoint):
                self.assertEqual(
                    classify_projected_camera_position(far_endpoint, near_endpoint),
                    expected,
                )

    def test_permutation_moves_coordinates_and_visibility_together(self) -> None:
        raw = [
            {
                "index": index,
                "name": KEYPOINT_NAMES[index],
                "world": [float(index), float(index) + 0.1, 0.04],
                "pixel": [float(index) + 0.2, float(index) + 0.3],
                "visibility": index % 3,
                "occluded_by": f"object-{index}",
            }
            for index in range(36)
        ]
        for source_camera_position in CANONICAL_CAMERA_MODE_BY_SOURCE:
            with self.subTest(source_camera_position=source_camera_position):
                permutation = keypoint_permutation(source_camera_position, KEYPOINT_WORLD)
                canonical = _apply_keypoint_permutation(raw, permutation)
                self.assertEqual(len(canonical), 36)
                self.assertEqual(sorted(permutation), list(range(36)))
                for output_index, source_index in enumerate(permutation):
                    self.assertEqual(canonical[output_index]["index"], output_index)
                    self.assertEqual(canonical[output_index]["source_index"], source_index)
                    self.assertEqual(canonical[output_index]["pixel"], raw[source_index]["pixel"])
                    self.assertEqual(canonical[output_index]["world"], raw[source_index]["world"])
                    self.assertEqual(
                        canonical[output_index]["visibility"],
                        raw[source_index]["visibility"],
                    )
                    self.assertEqual(
                        canonical[output_index]["occluded_by"],
                        raw[source_index]["occluded_by"],
                    )

    def test_opposite_physical_cameras_project_to_the_same_canonical_order(self) -> None:
        scene = bpy.context.scene
        camera_data = bpy.data.cameras.new("TEST_Court36Camera")
        camera = bpy.data.objects.new("TEST_Court36Camera", camera_data)
        scene.collection.objects.link(camera)
        camera.data.lens = 50.0
        camera.data.sensor_width = 36.0
        locations = {
            FAR_CENTER: (4.5, -12.0, 8.0),
            NEAR_CENTER: (4.5, 30.0, 8.0),
            LEFT_SIDE: (-12.0, 9.0, 8.0),
            RIGHT_SIDE: (21.0, 9.0, 8.0),
            FAR_LEFT: (-12.0, -12.0, 8.0),
            NEAR_RIGHT: (21.0, 30.0, 8.0),
            FAR_RIGHT: (21.0, -12.0, 8.0),
            NEAR_LEFT: (-12.0, 30.0, 8.0),
        }
        opposite_pairs = (
            (FAR_CENTER, NEAR_CENTER),
            (LEFT_SIDE, RIGHT_SIDE),
            (FAR_LEFT, NEAR_RIGHT),
            (FAR_RIGHT, NEAR_LEFT),
        )
        projected: dict[str, list[tuple[float, float]]] = {}
        try:
            for source_camera_position, location in locations.items():
                camera.location = Vector(location)
                camera.rotation_mode = "QUATERNION"
                camera.rotation_quaternion = (
                    Vector((4.5, 9.0, 0.04)) - camera.location
                ).to_track_quat("-Z", "Y")
                bpy.context.view_layer.update()
                raw = [
                    world_to_camera_view(scene, camera, Vector(point))
                    for point in KEYPOINT_WORLD
                ]
                permutation = keypoint_permutation(source_camera_position, KEYPOINT_WORLD)
                projected[source_camera_position] = [
                    (float(raw[source_index].x), float(raw[source_index].y))
                    for source_index in permutation
                ]
            for first, second in opposite_pairs:
                with self.subTest(first=first, second=second):
                    for first_point, second_point in zip(projected[first], projected[second]):
                        self.assertAlmostEqual(first_point[0], second_point[0], places=6)
                        self.assertAlmostEqual(first_point[1], second_point[1], places=6)
        finally:
            bpy.data.objects.remove(camera, do_unlink=True)
            bpy.data.cameras.remove(camera_data)


if __name__ == "__main__":
    unittest.main()
