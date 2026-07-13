"""Minimum-spread gaze analysis for stitched gaze tracks.

Endpoint-connected fixation regions are removed first. The remaining point
with the smallest local mean squared distance is selected as the gaze peak.
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np

from src.gaze_rgb_config import DEFAULT_GAZE_MIN_BOUNDARY_POINTS
from src.gaze_track_boundary import find_boundary_regions

def _validate_params(window: int, radius: float) -> None:
    if window < 3 or window % 2 == 0:
        raise ValueError("window must be an odd integer >= 3")
    if radius <= 0:
        raise ValueError("radius must be > 0")


def _window_mean_sq_dist(points_xy: np.ndarray, window: int) -> np.ndarray:
    """Return each point's local spread using every point in its time window."""
    half = window // 2
    n = len(points_xy)
    msd = np.full(n, np.inf, dtype=np.float64)
    for i, point in enumerate(points_xy):
        start = max(0, i - half)
        stop = min(n, i + half + 1)
        neighbors = np.delete(points_xy[start:stop], i - start, axis=0)
        sq_dists = np.sum((neighbors - point) ** 2, axis=1)
        if len(sq_dists):
            msd[i] = float(np.mean(sq_dists))
    return msd


def find_peak(
    points_xy: np.ndarray,
    window: int,
    radius: float,
    min_boundary_points: int = DEFAULT_GAZE_MIN_BOUNDARY_POINTS,
) -> tuple[Optional[int], np.ndarray, set[int], set[int]]:
    """Pick the lowest-MSD point after removing endpoint fixation regions."""
    _validate_params(window, radius)
    msd = _window_mean_sq_dist(points_xy, window)
    start_region, end_region = find_boundary_regions(
        points_xy, radius, min_boundary_points
    )
    excluded = start_region | end_region
    eligible = [
        i for i in range(len(points_xy))
        if i not in excluded and np.isfinite(msd[i])
    ]
    if not eligible:
        return None, msd, start_region, end_region
    return min(eligible, key=lambda i: msd[i]), msd, start_region, end_region


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


def analyze_track_peak(
    track_path: Path, window: int, radius: float, visualize: bool = True
) -> Path:
    """Write peak analysis JSON and optionally update the trajectory image."""
    data = json.loads(track_path.read_text())
    records = data.get("points", [])
    if not records:
        raise ValueError(f"No points found in {track_path}.")

    points_xy = np.asarray([r["canvas_xy"] for r in records], dtype=np.float64)
    idx, msd, start_region, end_region = find_peak(points_xy, window, radius)

    chosen = None
    if idx is not None:
        r = records[idx]
        chosen = {
            "index": idx,
            "stamp_ns": r.get("stamp_ns"),
            "t_sec": r.get("t_sec"),
            "canvas_xy": r["canvas_xy"],
            "mean_sq_dist_px2": round(float(msd[idx]), 2),
        }

    output_path = track_path.with_name(f"{track_path.stem}_peak.json")
    output = {
        "source": str(track_path),
        "coordinate_frame": data.get("coordinate_frame"),
        "method": (
            "minimum local mean squared distance over all points in the temporal "
            "window after endpoint filtering"
        ),
        "params": {
            "window": window,
            "boundary_radius_px": radius,
            "min_boundary_points": DEFAULT_GAZE_MIN_BOUNDARY_POINTS,
        },
        "count": len(records),
        "chosen": chosen,
        "start_boundary_indices": sorted(start_region),
        "end_boundary_indices": sorted(end_region),
        "points": [
            {
                **record,
                "mean_sq_dist_px2": (
                    round(float(point_msd), 2) if np.isfinite(point_msd) else None
                ),
                "boundary_region": (
                    "start" if i in start_region else "end" if i in end_region else ""
                ),
            }
            for i, (record, point_msd) in enumerate(zip(records, msd))
        ],
    }
    output_path.write_text(json.dumps(output, indent=2))

    png_path = track_path.parent / "stitched_trajectory.png"
    if visualize:
        save_peak_visualization(points_xy, chosen, radius, track_path, png_path)

    print(
        f"{len(records)} points, window={window}, radius={radius}px -> "
        + (
            f"peak at index {chosen['index']} (stamp_ns={chosen['stamp_ns']}, "
            f"MSD={chosen['mean_sq_dist_px2']}px²)"
            if chosen
            else "no acceptable peak"
        )
        + f", {len(start_region)} start + {len(end_region)} end points rejected"
    )
    print(f"Saved peak JSON: {output_path}")
    if visualize:
        print(f"Saved trajectory PNG: {png_path}")

    return output_path


def find_gaze_center_stamp(
    track_path: Path, window: int, radius: float, visualize: bool = True
) -> Optional[int]:
    """Return the peak gaze-center timestamp."""
    out_path = analyze_track_peak(track_path, window, radius, visualize=visualize)
    chosen = json.loads(out_path.read_text()).get("chosen")
    return chosen.get("stamp_ns") if chosen else None


def visualize_analyzed_peak(track_path: Path, radius: float) -> None:
    """Draw a previously analyzed peak after external validation succeeds."""
    data = json.loads(track_path.read_text())
    records = data.get("points", [])
    if not records:
        raise ValueError(f"No points found in {track_path}.")
    peak_path = track_path.with_name(f"{track_path.stem}_peak.json")
    chosen = json.loads(peak_path.read_text()).get("chosen")
    points_xy = np.asarray([record["canvas_xy"] for record in records], dtype=np.float64)
    output_path = track_path.parent / "stitched_trajectory.png"
    save_peak_visualization(points_xy, chosen, radius, track_path, output_path)
    print(f"Saved validated peak to trajectory PNG: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Minimum-MSD gaze analysis after endpoint fixation filtering."
    )
    parser.add_argument("track", type=Path, help="Path to stitched_gaze_track.json")
    parser.add_argument(
        "--window", type=int, required=True,
        help="Odd window length. Example: 9 means previous 4 + next 4 points.",
    )
    parser.add_argument(
        "--radius", type=float, required=True,
        help="Pixel radius used to identify start/end fixation regions.",
    )
    args = parser.parse_args()
    analyze_track_peak(args.track, args.window, args.radius)


if __name__ == "__main__":
    main()
