"""Court36 camera-position classification and 180-degree canonicalization.

The rendered image is never used to guess an outside camera's viewpoint.  Its
real Blender location determines one of eight source positions.  Only cameras
that are inside/above the court use the already-known projected court axis as
a deterministic fallback.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


FAR_CENTER = "FAR_CENTER"
NEAR_CENTER = "NEAR_CENTER"
LEFT_SIDE = "LEFT_SIDE"
RIGHT_SIDE = "RIGHT_SIDE"
FAR_LEFT = "FAR_LEFT"
FAR_RIGHT = "FAR_RIGHT"
NEAR_LEFT = "NEAR_LEFT"
NEAR_RIGHT = "NEAR_RIGHT"

ENDLINE = "ENDLINE"
SIDELINE = "SIDELINE"
DIAGONAL_BOTTOM_RIGHT = "DIAGONAL_BOTTOM_RIGHT"
DIAGONAL_TOP_RIGHT = "DIAGONAL_TOP_RIGHT"

SOURCE_CAMERA_POSITIONS = (
    FAR_CENTER,
    NEAR_CENTER,
    LEFT_SIDE,
    RIGHT_SIDE,
    FAR_LEFT,
    FAR_RIGHT,
    NEAR_LEFT,
    NEAR_RIGHT,
)

CANONICAL_CAMERA_MODE_BY_SOURCE = {
    FAR_CENTER: ENDLINE,
    NEAR_CENTER: ENDLINE,
    LEFT_SIDE: SIDELINE,
    RIGHT_SIDE: SIDELINE,
    FAR_LEFT: DIAGONAL_BOTTOM_RIGHT,
    NEAR_RIGHT: DIAGONAL_BOTTOM_RIGHT,
    FAR_RIGHT: DIAGONAL_TOP_RIGHT,
    NEAR_LEFT: DIAGONAL_TOP_RIGHT,
}

# These are the 180-degree counterparts of the canonical reference viewpoints
# NEAR_CENTER, LEFT_SIDE, NEAR_RIGHT and NEAR_LEFT respectively.
ROTATED_SOURCE_CAMERA_POSITIONS = frozenset(
    (FAR_CENTER, RIGHT_SIDE, FAR_LEFT, FAR_RIGHT)
)


def canonical_camera_mode(source_camera_position: str) -> str:
    """Return the four-way training mode for an eight-way source position."""

    try:
        return CANONICAL_CAMERA_MODE_BY_SOURCE[source_camera_position]
    except KeyError as exc:
        raise ValueError(f"unsupported source camera position: {source_camera_position}") from exc


def classify_outside_camera_position(
    location: Sequence[float],
    *,
    court_width: float = 9.0,
    court_length: float = 18.0,
    epsilon: float = 1e-6,
) -> str | None:
    """Classify an outside camera from its real world-space x/y coordinates.

    ``None`` means the camera is vertically above or horizontally inside the
    court footprint and therefore needs projected-axis classification.
    """

    if len(location) < 2:
        raise ValueError("camera location must contain at least x and y")
    x = float(location[0])
    y = float(location[1])
    left = x < -epsilon
    right = x > court_width + epsilon
    far = y < -epsilon
    near = y > court_length + epsilon

    if far and left:
        return FAR_LEFT
    if far and right:
        return FAR_RIGHT
    if near and left:
        return NEAR_LEFT
    if near and right:
        return NEAR_RIGHT
    if far:
        return FAR_CENTER
    if near:
        return NEAR_CENTER
    if left:
        return LEFT_SIDE
    if right:
        return RIGHT_SIDE
    return None


def classify_projected_camera_position(
    far_endpoint: Sequence[float],
    near_endpoint: Sequence[float],
    *,
    epsilon: float = 1e-6,
) -> str:
    """Classify an inside/overhead camera from the projected court long axis.

    ``far_endpoint`` is the projection of world ``y=0`` and ``near_endpoint``
    is world ``y=18``.  Image y grows downwards.  The returned source position
    makes the upper endpoint the canonical far end; a nearly horizontal axis
    uses its left/right direction as the deterministic tie-breaker.
    """

    if len(far_endpoint) < 2 or len(near_endpoint) < 2:
        raise ValueError("projected endpoints must contain x and y")
    dx = float(far_endpoint[0]) - float(near_endpoint[0])
    dy = float(far_endpoint[1]) - float(near_endpoint[1])
    if math.hypot(dx, dy) <= epsilon:
        raise ValueError("cannot classify a camera whose projected court long axis has zero length")

    diagonal_boundary = math.tan(math.pi / 8.0)
    abs_x = abs(dx)
    abs_y = abs(dy)
    if abs_x <= abs_y * diagonal_boundary:
        return NEAR_CENTER if dy < 0.0 else FAR_CENTER
    if abs_y <= abs_x * diagonal_boundary:
        return LEFT_SIDE if dx > 0.0 else RIGHT_SIDE
    if dx * dy > 0.0:
        return NEAR_RIGHT if dy < 0.0 else FAR_LEFT
    return NEAR_LEFT if dy < 0.0 else FAR_RIGHT


def rotation_180_permutation(
    keypoint_world: Sequence[Sequence[float]],
    *,
    court_width: float = 9.0,
    court_length: float = 18.0,
    tolerance: float = 1e-5,
) -> tuple[int, ...]:
    """Map each canonical output index to its 180-degree source index."""

    if len(keypoint_world) != 36:
        raise ValueError(f"Court36 requires exactly 36 keypoints, got {len(keypoint_world)}")
    coordinates = [tuple(float(value) for value in point[:3]) for point in keypoint_world]
    permutation: list[int] = []
    for x, y, z in coordinates:
        target = (court_width - x, court_length - y, z)
        candidates = [
            index
            for index, point in enumerate(coordinates)
            if max(abs(point[axis] - target[axis]) for axis in range(3)) <= tolerance
        ]
        if len(candidates) != 1:
            raise ValueError(f"Court36 has no unique 180-degree pair for {target}: {candidates}")
        permutation.append(candidates[0])
    if sorted(permutation) != list(range(36)):
        raise ValueError("Court36 180-degree mapping is not a permutation")
    return tuple(permutation)


def keypoint_permutation(
    source_camera_position: str,
    keypoint_world: Sequence[Sequence[float]],
) -> tuple[int, ...]:
    """Return output-index -> source-index mapping for the source viewpoint."""

    canonical_camera_mode(source_camera_position)
    if source_camera_position in ROTATED_SOURCE_CAMERA_POSITIONS:
        return rotation_180_permutation(keypoint_world)
    if len(keypoint_world) != 36:
        raise ValueError(f"Court36 requires exactly 36 keypoints, got {len(keypoint_world)}")
    return tuple(range(36))
