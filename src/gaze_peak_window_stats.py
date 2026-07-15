"""Report per-point local MSD and highlight all low-MSD selected points.

The input can be one stitched_gaze_track.json file or an experiment directory.
For a directory, every stitched_gaze_track.json below it is analyzed.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from src.gaze_rgb_config import (
    DEFAULT_GAZE_MIN_BOUNDARY_POINTS,
    DEFAULT_GAZE_PEAK_RADIUS,
    DEFAULT_GAZE_PEAK_WINDOW,
    MSD_THRESHOLD,
)
from src.gaze_track_peak import find_stable_points


def find_tracks(path: Path) -> list[Path]:
    """Return track files represented by a file or experiment directory."""
    if path.is_file():
        return [path]
    return sorted(path.rglob("stitched_gaze_track.json"))


def analyze_window_stats(
    track_path: Path,
    window: int,
    boundary_radius: float,
    min_boundary_points: int,
) -> dict:
    """Calculate the same per-point MSD and peak used by gaze_track_peak."""
    data = json.loads(track_path.read_text())
    records = data.get("points", [])
    if not records:
        raise ValueError("track contains no gaze points")

    points_xy = np.asarray([record["canvas_xy"] for record in records], dtype=np.float64)
    selected_indices, msd, start_region, end_region = find_stable_points(
        points_xy, window, boundary_radius, MSD_THRESHOLD, min_boundary_points
    )
    half = window // 2
    selected_set = set(selected_indices)
    excluded = start_region | end_region
    eligible_indices = [
        index for index, value in enumerate(msd)
        if index not in excluded and np.isfinite(value)
    ]
    fallback_minimum_index = (
        min(eligible_indices, key=lambda index: msd[index])
        if not selected_indices and eligible_indices else None
    )

    points = []
    for index, (record, value) in enumerate(zip(records, msd)):
        window_start = max(0, index - half)
        window_stop = min(len(records), index + half + 1)
        boundary_region = (
            "start" if index in start_region else "end" if index in end_region else ""
        )
        points.append({
            "index": index,
            "stamp_ns": record.get("stamp_ns"),
            "t_sec": record.get("t_sec"),
            "canvas_x": float(points_xy[index, 0]),
            "canvas_y": float(points_xy[index, 1]),
            "window_start_index": window_start,
            "window_end_index": window_stop - 1,
            "neighbor_count": window_stop - window_start - 1,
            "mean_sq_dist_px2": float(value) if np.isfinite(value) else None,
            "boundary_region": boundary_region,
            "eligible_for_peak": bool(not boundary_region and np.isfinite(value)),
            "selected_by_msd": index in selected_set,
            "is_fallback_minimum": index == fallback_minimum_index,
        })

    return {
        "source": str(track_path),
        "method": "low-MSD selection used by gaze_track_peak.find_stable_points",
        "params": {
            "window": window,
            "boundary_radius_px": boundary_radius,
            "min_boundary_points": min_boundary_points,
            "msd_threshold_px2": MSD_THRESHOLD,
        },
        "point_count": len(points),
        "selected": [points[index].copy() for index in selected_indices],
        "selected_indices": selected_indices,
        "fallback_minimum": (
            points[fallback_minimum_index].copy()
            if fallback_minimum_index is not None else None
        ),
        "start_boundary_indices": sorted(start_region),
        "end_boundary_indices": sorted(end_region),
        "points": points,
    }


def save_csv(report: dict, path: Path) -> None:
    points = report["points"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(points[0]))
        writer.writeheader()
        writer.writerows(points)


def save_plot(report: dict, path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required unless --no-plot is used") from exc

    points = report["points"]
    indices = np.asarray([point["index"] for point in points])
    values = np.asarray([
        point["mean_sq_dist_px2"]
        if point["mean_sq_dist_px2"] is not None else np.nan
        for point in points
    ])
    selected = np.asarray([point["selected_by_msd"] for point in points])

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(indices, values, color="#8a8f98", linewidth=1.3, alpha=0.8)
    ax.scatter(indices[~selected], values[~selected], s=28,
               color="#3478b8", label="all other points", zorder=2)
    if selected.any():
        ax.scatter(indices[selected], values[selected], s=75,
                   color="#d62728", label=f"selected: MSD < {MSD_THRESHOLD:g} px²",
                   zorder=4)
    for point in report["selected"]:
        point_index = point["index"]
        point_msd = point["mean_sq_dist_px2"]
        ax.annotate(
            f"{point_msd:.2f}",
            xy=(point_index, point_msd),
            xytext=(0, 9),
            textcoords="offset points",
            fontsize=9,
            ha="center",
            color="#9f1d20",
            zorder=6,
        )

    fallback = report["fallback_minimum"]
    if fallback is not None:
        fallback_index = fallback["index"]
        fallback_msd = fallback["mean_sq_dist_px2"]
        ax.scatter(
            [fallback_index], [fallback_msd], s=165, marker="*",
            color="#ffb000", edgecolor="#7a4b00", linewidth=0.8,
            label=(f"fallback minimum (index {fallback_index}, "
                   f"MSD={fallback_msd:.2f} px²)"),
            zorder=6,
        )
        ax.annotate(
            f"Fallback minimum\nindex: {fallback_index}\nMSD: {fallback_msd:.2f} px²",
            xy=(fallback_index, fallback_msd),
            xytext=(12, 18),
            textcoords="offset points",
            fontsize=9,
            color="#7a4b00",
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "#fff7df",
                  "edgecolor": "#ffb000", "alpha": 0.95},
            arrowprops={"arrowstyle": "->", "color": "#b87800"},
            zorder=7,
        )

    for region, color, label in (
        (report["start_boundary_indices"], "#47a447", "excluded start boundary"),
        (report["end_boundary_indices"], "#9b59b6", "excluded end boundary"),
    ):
        if region:
            region_array = np.asarray(region, dtype=int)
            ax.scatter(region_array, values[region_array], marker="x", s=55,
                       color=color, label=label, zorder=5)

    params = report["params"]
    ax.set_title(
        f"Per-point local MSD (window={params['window']})\n{report['source']}"
    )
    ax.set_xlabel("gaze point index")
    ax.set_ylabel("mean squared distance (px²)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path,
                        help="stitched_gaze_track.json or experiment directory")
    parser.add_argument("--window", type=int, default=DEFAULT_GAZE_PEAK_WINDOW,
                        help="odd local window size (default: %(default)s)")
    parser.add_argument("--boundary-radius", type=float,
                        default=DEFAULT_GAZE_PEAK_RADIUS,
                        help="endpoint fixation radius in pixels (default: %(default)s)")
    parser.add_argument("--min-boundary-points", type=int,
                        default=DEFAULT_GAZE_MIN_BOUNDARY_POINTS,
                        help="minimum endpoint region size (default: %(default)s)")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    tracks = find_tracks(args.input)
    if not tracks:
        raise SystemExit(f"No stitched_gaze_track.json found under {args.input}")

    failed = 0
    for track in tracks:
        try:
            report = analyze_window_stats(
                track, args.window, args.boundary_radius, args.min_boundary_points
            )
            base = track.with_name(f"{track.stem}_window_stats")
            json_path = base.with_suffix(".json")
            csv_path = base.with_suffix(".csv")
            json_path.write_text(json.dumps(report, indent=2))
            save_csv(report, csv_path)
            if not args.no_plot:
                save_plot(report, base.with_suffix(".png"))
            print(
                f"{track}: {report['point_count']} points, "
                f"{len(report['selected'])} selected with MSD < {MSD_THRESHOLD:g}px²"
            )
            fallback = report["fallback_minimum"]
            if fallback is not None:
                print(
                    f"  Fallback minimum: index={fallback['index']}, "
                    f"MSD={fallback['mean_sq_dist_px2']:.2f}px²"
                )
            print(f"  JSON: {json_path}\n  CSV:  {csv_path}")
            if not args.no_plot:
                print(f"  Plot: {base.with_suffix('.png')}")
        except (ValueError, KeyError, json.JSONDecodeError, RuntimeError) as exc:
            failed += 1
            print(f"SKIP {track}: {exc}")

    if failed == len(tracks):
        raise SystemExit("All tracks failed analysis")


if __name__ == "__main__":
    main()
