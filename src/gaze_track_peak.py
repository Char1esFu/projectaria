"""Densest-point gaze analysis: an alternative to window/radius clustering.

Walks the stitched-canvas gaze track from start to end. Each point counts how
many of its temporal-window neighbors (window=9 -> 4 before + 4 after) fall
within ``radius`` pixels of it; the highest count wins, ties broken by the
smallest mean squared distance from the window neighbors to the point
(tighter local spread = denser). Candidates need at least window//2 neighbors
in radius — the same density floor the clustering method uses — so a winner
is never picked from plain sweep motion.

Boundary-connectivity check: the winner's dense segment — consecutive track
points inside its radius circle — must not reach the first or last sample.
In this setup the glasses sweep in from an irrelevant area, fixate the
target, then sweep away, so density touching either end of the track is
residue from those sweeps, not the fixated target. A rejected segment is
excluded entirely and the next-ranked candidate is tried; when no candidate
survives the caller falls back to an unweighted average.

This module also hosts ``find_gaze_center_stamp``, the single entry point
that runs either this method or the clustering one from
``src.gaze_track_cluster`` and returns the center timestamp used for
temporal score weighting.

Usage:
  python -m src.gaze_track_peak recordings/test01/20/stitched_gaze_track.json \
      --window 9 --radius 20
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np

from src.gaze_track_cluster import (
    _validate_params,
    analyze_track,
    count_neighbors_in_radius,
)

GAZE_CENTER_METHODS = ("cluster", "peak")


def _window_mean_sq_dist(points_xy: np.ndarray, window: int) -> np.ndarray:
    """Mean squared distance from each point to its temporal-window neighbors
    (self excluded). Lower = tighter local spread around that point.

    Mean *squared* distance rather than the variance of distances: a ring of
    neighbors at equal radius has zero distance variance but is not dense —
    the second moment penalizes both far and spread-out neighbors."""
    half = window // 2
    n = len(points_xy)
    msd = np.full(n, np.inf, dtype=np.float64)
    for i, point in enumerate(points_xy):
        start = max(0, i - half)
        stop = min(n, i + half + 1)
        neighbors = np.delete(points_xy[start:stop], i - start, axis=0)
        if len(neighbors):
            msd[i] = float(np.mean(np.sum((neighbors - point) ** 2, axis=1)))
    return msd


def _dense_segment(points_xy: np.ndarray, center: int, radius: float) -> tuple[int, int]:
    """Inclusive index range of consecutive track points inside the radius
    circle of points_xy[center], grown from the center in both directions."""
    p = points_xy[center]
    lo = center
    while lo > 0 and np.linalg.norm(points_xy[lo - 1] - p) <= radius:
        lo -= 1
    hi = center
    while hi < len(points_xy) - 1 and np.linalg.norm(points_xy[hi + 1] - p) <= radius:
        hi += 1
    return lo, hi


def find_peak(
    points_xy: np.ndarray, window: int, radius: float
) -> tuple[Optional[int], np.ndarray, np.ndarray, list[tuple[int, int]]]:
    """Pick the densest acceptable track point.

    Returns ``(chosen_index, counts, msd, rejected_segments)``. Candidates are
    ranked by (neighbor count desc, mean squared distance asc) and must have
    at least window//2 neighbors in radius; a candidate whose dense segment
    touches the first or last sample is rejected together with its whole
    segment. chosen_index is None when nothing survives."""
    _validate_params(window, radius)
    n = len(points_xy)
    counts = count_neighbors_in_radius(points_xy, window, radius)
    msd = _window_mean_sq_dist(points_xy, window)
    min_count = window // 2

    order = sorted(range(n), key=lambda i: (-counts[i], msd[i]))
    excluded = np.zeros(n, dtype=bool)
    rejected: list[tuple[int, int]] = []
    for i in order:
        if excluded[i] or counts[i] < min_count:
            continue
        lo, hi = _dense_segment(points_xy, i, radius)
        if lo == 0 or hi == n - 1:
            excluded[lo:hi + 1] = True
            rejected.append((lo, hi))
            continue
        return i, counts, msd, rejected
    return None, counts, msd, rejected


def _save_peak_visualization(
    points_xy: np.ndarray,
    chosen_idx: Optional[int],
    radius: float,
    rejected: list[tuple[int, int]],
    track_path: Path,
    output_path: Path,
) -> None:
    """Track polyline with rejected boundary segments in orange and the chosen
    peak in green (dot + radius circle), on stitched.png when available."""
    import cv2

    background_path = track_path.parent / "stitched.png"
    canvas = cv2.imread(str(background_path)) if background_path.exists() else None
    if canvas is None:
        width = int(np.ceil(points_xy[:, 0].max())) + 20
        height = int(np.ceil(points_xy[:, 1].max())) + 20
        canvas = np.zeros((height, width, 3), dtype=np.uint8)

    pts = np.rint(points_xy).astype(np.int32)
    if len(pts) >= 2:
        cv2.polylines(canvas, [pts], False, (150, 150, 150), 1, cv2.LINE_AA)

    rejected_mask = np.zeros(len(pts), dtype=bool)
    for lo, hi in rejected:
        rejected_mask[lo:hi + 1] = True
    for pt, is_rejected in zip(pts, rejected_mask):
        color = (0, 140, 255) if is_rejected else (160, 160, 160)
        cv2.circle(canvas, tuple(pt), 3, color, -1, cv2.LINE_AA)

    if chosen_idx is not None:
        center = tuple(pts[chosen_idx])
        cv2.circle(canvas, center, 5, (0, 255, 0), -1, cv2.LINE_AA)
        cv2.circle(canvas, center, max(8, int(round(radius))), (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(canvas, "peak", (center[0] + int(round(radius)) + 4, center[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imwrite(str(output_path), canvas)


def analyze_track_peak(
    track_path: Path, window: int, radius: float, save_png: bool = True
) -> Path:
    """Run the densest-point analysis on a stitched_gaze_track.json and write
    ``<stem>_peak.json`` (plus ``.png``) next to it. Returns the JSON path."""
    data = json.loads(track_path.read_text())
    records = data.get("points", [])
    if not records:
        raise ValueError(f"No points found in {track_path}.")

    points_xy = np.asarray([r["canvas_xy"] for r in records], dtype=np.float64)
    idx, counts, msd, rejected = find_peak(points_xy, window, radius)

    chosen = None
    if idx is not None:
        r = records[idx]
        chosen = {
            "index": idx,
            "stamp_ns": r.get("stamp_ns"),
            "t_sec": r.get("t_sec"),
            "canvas_xy": r["canvas_xy"],
            "neighbors_in_radius": int(counts[idx]),
            "mean_sq_dist_px2": round(float(msd[idx]), 2),
        }

    output_path = track_path.with_name(f"{track_path.stem}_peak.json")
    output = {
        "source": str(track_path),
        "coordinate_frame": data.get("coordinate_frame"),
        "method": "densest point with boundary-connectivity check",
        "params": {
            "window": window,
            "radius_px": radius,
            "min_neighbors": window // 2,
        },
        "count": len(records),
        "chosen": chosen,
        "rejected_boundary_segments": [
            {
                "start_index": lo,
                "end_index": hi,
                "stamp_ns_start": records[lo].get("stamp_ns"),
                "stamp_ns_end": records[hi].get("stamp_ns"),
            }
            for lo, hi in rejected
        ],
        "points": [
            {**record, "neighbors_in_radius": int(count)}
            for record, count in zip(records, counts)
        ],
    }
    output_path.write_text(json.dumps(output, indent=2))

    print(
        f"{len(records)} points, window={window}, radius={radius}px -> "
        + (
            f"peak at index {chosen['index']} (stamp_ns={chosen['stamp_ns']}, "
            f"count={chosen['neighbors_in_radius']})"
            if chosen
            else "no acceptable peak"
        )
        + f", {len(rejected)} boundary segment(s) rejected"
    )
    print(f"Saved peak JSON: {output_path}")

    if save_png:
        png_path = output_path.with_suffix(".png")
        _save_peak_visualization(points_xy, idx, radius, rejected, track_path, png_path)
        print(f"Saved peak PNG: {png_path}")

    return output_path


def find_gaze_center_stamp(
    track_path: Path, method: str, window: int, radius: float
) -> Optional[int]:
    """Analyze the gaze track with the chosen method and return the center
    stamp_ns for temporal score weighting, or None when the method finds no
    acceptable center. Both methods write their analysis JSON (+PNG) next to
    the track file."""
    if method == "cluster":
        out_path = analyze_track(track_path, window, radius)
        clusters = json.loads(out_path.read_text()).get("clusters", [])
        return clusters[0].get("center_stamp_ns") if clusters else None
    if method == "peak":
        out_path = analyze_track_peak(track_path, window, radius)
        chosen = json.loads(out_path.read_text()).get("chosen")
        return chosen.get("stamp_ns") if chosen else None
    raise ValueError(f"Unknown gaze center method: {method!r} (use 'cluster' or 'peak').")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Densest-point gaze track analysis with boundary-connectivity check."
    )
    parser.add_argument("track", type=Path, help="Path to stitched_gaze_track.json")
    parser.add_argument(
        "--window", type=int, required=True,
        help="Odd window length. Example: 9 means previous 4 + next 4 points.",
    )
    parser.add_argument(
        "--radius", type=float, required=True,
        help="Pixel radius of the circle drawn around each track point.",
    )
    parser.add_argument(
        "--no-png", action="store_true",
        help="Only write JSON; skip the visualization PNG.",
    )
    args = parser.parse_args()
    analyze_track_peak(args.track, args.window, args.radius, save_png=not args.no_png)


if __name__ == "__main__":
    main()
