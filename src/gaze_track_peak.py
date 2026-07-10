"""Densest-point gaze analysis for stitched gaze tracks.

Each point is ranked by nearby temporal neighbors, with ties broken by local
spread. Dense segments touching the track boundary are rejected as sweep-in/out
motion. ``find_gaze_center_stamp`` also dispatches to the cluster method.
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


def _window_mean_sq_dist(points_xy: np.ndarray, window: int, radius: float) -> np.ndarray:
    """Return each point's local spread within its temporal window."""
    half = window // 2
    n = len(points_xy)
    msd = np.full(n, np.inf, dtype=np.float64)
    for i, point in enumerate(points_xy):
        start = max(0, i - half)
        stop = min(n, i + half + 1)
        neighbors = np.delete(points_xy[start:stop], i - start, axis=0)
        sq_dists = np.sum((neighbors - point) ** 2, axis=1)
        sq_dists = sq_dists[sq_dists <= radius * radius]
        if len(sq_dists):
            msd[i] = float(np.mean(sq_dists))
    return msd


def _dense_segment(points_xy: np.ndarray, center: int, radius: float) -> tuple[int, int]:
    """Return the consecutive in-radius segment around center."""
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
    """Pick the densest non-boundary track point."""
    _validate_params(window, radius)
    n = len(points_xy)
    counts = count_neighbors_in_radius(points_xy, window, radius)
    msd = _window_mean_sq_dist(points_xy, window, radius)
    min_count = 1

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


def save_peak_visualization(
    points_xy: np.ndarray,
    chosen: Optional[dict],
    radius: float,
    track_path: Path,
    output_path: Path,
) -> None:
    import cv2

    background_path = track_path.parent / "stitched_trajectory.png"
    has_background = background_path.exists()
    canvas = cv2.imread(str(background_path)) if has_background else None
    if canvas is None:
        width = int(np.ceil(points_xy[:, 0].max())) + 20
        height = int(np.ceil(points_xy[:, 1].max())) + 20
        canvas = np.zeros((height, width, 3), dtype=np.uint8)

    if not has_background:
        pts = np.rint(points_xy).astype(np.int32)
        if len(pts) >= 2:
            cv2.polylines(canvas, [pts], False, (150, 150, 150), 1, cv2.LINE_AA)
        for pt in pts:
            cv2.circle(canvas, tuple(pt), 2, (160, 160, 160), -1, cv2.LINE_AA)

    if chosen is not None:
        cx, cy = (int(round(v)) for v in chosen["canvas_xy"])
        peak_color = (255, 255, 0)
        peak_radius = max(1, int(round(radius)))
        cv2.circle(canvas, (cx, cy), peak_radius, peak_color, 2, cv2.LINE_AA)
        cv2.circle(canvas, (cx, cy), 5, peak_color, -1, cv2.LINE_AA)
        cv2.putText(
            canvas,
            "peak",
            (cx + peak_radius + 4, cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            peak_color,
            2,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(output_path), canvas)


def analyze_track_peak(track_path: Path, window: int, radius: float) -> Path:
    """Write peak analysis JSON and update the trajectory visualization."""
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
            "min_neighbors": 1,
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

    png_path = track_path.parent / "stitched_trajectory.png"
    save_peak_visualization(points_xy, chosen, radius, track_path, png_path)

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
    print(f"Saved trajectory PNG: {png_path}")

    return output_path


def find_gaze_center_stamp(
    track_path: Path, method: str, window: int, radius: float
) -> Optional[int]:
    """Return the gaze-center timestamp from the selected method."""
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
    args = parser.parse_args()
    analyze_track_peak(args.track, args.window, args.radius)


if __name__ == "__main__":
    main()
