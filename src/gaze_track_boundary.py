"""Shared boundary-fixation detection for gaze-track analyzers."""

import numpy as np


def boundary_region(
    points_xy: np.ndarray, from_start: bool, radius: float
) -> set[int]:
    """Return the boundary-connected run that stays near its endpoint."""
    anchor_index = 0 if from_start else len(points_xy) - 1
    indices = (
        range(len(points_xy))
        if from_start
        else range(len(points_xy) - 1, -1, -1)
    )
    region: set[int] = set()
    for index in indices:
        if np.linalg.norm(points_xy[index] - points_xy[anchor_index]) > radius:
            break
        region.add(index)
    return region


def find_boundary_regions(
    points_xy: np.ndarray,
    radius: float,
    min_points: int,
) -> tuple[set[int], set[int]]:
    """Return qualifying start/end fixation regions as point-index sets."""
    if radius <= 0:
        raise ValueError("boundary radius must be > 0")
    if min_points < 1:
        raise ValueError("minimum boundary points must be >= 1")
    start = boundary_region(points_xy, True, radius)
    end = boundary_region(points_xy, False, radius)
    if len(start) < min_points:
        start.clear()
    if len(end) < min_points:
        end.clear()
    return start, end
