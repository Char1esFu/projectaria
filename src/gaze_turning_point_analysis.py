"""Offline turning-point analysis for existing stitched gaze tracks.

The detector removes boundary-connected fixation regions, estimates incoming
and outgoing motion over configurable multi-sample windows, and scores robust
direction changes with sufficient displacement and a local speed dip.
"""

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np

from src.gaze_rgb_config import (
    DEFAULT_GAZE_BOUNDARY_RADIUS,
    DEFAULT_GAZE_MIN_BOUNDARY_POINTS,
)
from src.gaze_track_boundary import find_boundary_regions
from src.gaze_velocity_direction import (
    calculate_point_metrics,
    find_tracks,
    load_track,
    write_points_csv,
)


@dataclass
class AnalysisParams:
    direction_window: int = 2
    boundary_radius_px: float = DEFAULT_GAZE_BOUNDARY_RADIUS
    min_boundary_points: int = DEFAULT_GAZE_MIN_BOUNDARY_POINTS
    min_window_displacement_px: float = 15.0
    min_turn_deg: float = 90.0
    min_path_fraction: float = 0.1
    max_path_fraction: float = 0.9


def analyze_track(track_path: Path, params: AnalysisParams) -> dict:
    records, t, points = load_track(track_path)
    n = len(points)
    k = params.direction_window
    if k < 1:
        raise ValueError("direction_window must be >= 1")

    start_region, end_region = find_boundary_regions(
        points, params.boundary_radius_px, params.min_boundary_points
    )

    motion_rows = calculate_point_metrics(records, t, points, k)
    total_path = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
    rows: list[dict] = []
    candidates: list[dict] = []
    for i, motion_row in enumerate(motion_rows):
        row = {
            **motion_row,
            "boundary_region": (
                "start" if i in start_region else "end" if i in end_region else ""
            ),
            "eligible": False,
            "rejection_reason": "",
            "score": None,
        }
        if motion_row["turn_deg"] is None:
            row["rejection_reason"] = "insufficient_direction_window"
            rows.append(row)
            continue
        score = motion_row["turn_score"] * (0.3 + 0.7 * motion_row["speed_dip"])
        row["score"] = round(score, 4)

        reasons = []
        if i in start_region or i in end_region:
            reasons.append("boundary_region")
        if motion_row["incoming_displacement_px"] < params.min_window_displacement_px:
            reasons.append("low_incoming_displacement")
        if motion_row["outgoing_displacement_px"] < params.min_window_displacement_px:
            reasons.append("low_outgoing_displacement")
        if abs(motion_row["turn_deg"]) < params.min_turn_deg:
            reasons.append("turn_below_threshold")
        if not params.min_path_fraction <= motion_row["path_fraction"] <= params.max_path_fraction:
            reasons.append("outside_path_fraction")
        row["rejection_reason"] = ";".join(reasons)
        row["eligible"] = not reasons
        if row["eligible"]:
            candidates.append(row)
        rows.append(row)

    candidates.sort(key=lambda row: row["score"], reverse=True)
    return {
        "source": str(track_path),
        "params": asdict(params),
        "point_count": n,
        "total_path_px": round(total_path, 3),
        "start_boundary_indices": sorted(start_region),
        "end_boundary_indices": sorted(end_region),
        "chosen": candidates[0] if candidates else None,
        "candidates": candidates,
        "points": rows,
    }


def save_csv(report: dict, path: Path) -> None:
    write_points_csv(report, path)


def save_plot(report: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = report["points"]
    points = np.asarray([[row["x_px"], row["y_px"]] for row in rows])
    fig, (track_ax, metric_ax) = plt.subplots(1, 2, figsize=(13, 6))
    track_ax.plot(points[:, 0], points[:, 1], "-", color="#777777", linewidth=2)
    track_ax.scatter(points[:, 0], points[:, 1], s=55, c=np.arange(len(points)), cmap="plasma",
                     edgecolors="white", linewidths=1)
    for row in rows:
        track_ax.annotate(str(row["point_number"]), (row["x_px"], row["y_px"]),
                          xytext=(4, 4), textcoords="offset points", fontsize=8)
    chosen = report["chosen"]
    if chosen:
        track_ax.scatter(chosen["x_px"], chosen["y_px"], s=230, facecolors="none",
                         edgecolors="#e31a1c", linewidths=3, label="chosen")
        track_ax.legend()
    track_ax.invert_yaxis()
    track_ax.set_aspect("equal", adjustable="datalim")
    track_ax.set_title("Trajectory and selected turning point")
    track_ax.set_xlabel("canvas x (px)")
    track_ax.set_ylabel("canvas y (px, downward)")
    track_ax.grid(alpha=0.25)

    point_numbers = [row["point_number"] for row in rows]
    turns = [abs(row["turn_deg"]) if row["turn_deg"] is not None else np.nan for row in rows]
    speed_dips = [
        row["speed_dip"] if row["speed_dip"] is not None else np.nan for row in rows
    ]
    metric_ax.plot(point_numbers, turns, "o-", label="|turn| (deg)", color="#2474b5")
    metric_ax.axhline(report["params"]["min_turn_deg"], color="#2474b5", linestyle="--")
    speed_dip_ax = metric_ax.twinx()
    speed_dip_ax.plot(point_numbers, speed_dips, "s-", label="speed dip",
                      color="#e1812c", alpha=0.8)
    metric_ax.set_title("Threshold diagnostics")
    metric_ax.set_xlabel("point number (1-based)")
    metric_ax.set_ylabel("absolute direction change (deg)", color="#2474b5")
    speed_dip_ax.set_ylabel("speed dip (0=no dip, 1=full stop/reversal)", color="#e1812c")
    speed_dip_ax.set_ylim(-0.05, 1.05)
    metric_ax.grid(alpha=0.25)
    fig.suptitle(str(report["source"]))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Track JSON or recordings directory")
    parser.add_argument("--direction-window", type=int, default=2)
    parser.add_argument(
        "--boundary-radius", type=float, default=DEFAULT_GAZE_BOUNDARY_RADIUS
    )
    parser.add_argument(
        "--min-boundary-points", type=int,
        default=DEFAULT_GAZE_MIN_BOUNDARY_POINTS,
    )
    parser.add_argument("--min-displacement", type=float, default=15.0)
    parser.add_argument("--min-turn", type=float, default=90.0)
    parser.add_argument("--min-path-fraction", type=float, default=0.1)
    parser.add_argument("--max-path-fraction", type=float, default=0.9)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    params = AnalysisParams(
        direction_window=args.direction_window,
        boundary_radius_px=args.boundary_radius,
        min_boundary_points=args.min_boundary_points,
        min_window_displacement_px=args.min_displacement,
        min_turn_deg=args.min_turn,
        min_path_fraction=args.min_path_fraction,
        max_path_fraction=args.max_path_fraction,
    )

    tracks = find_tracks(args.input)
    if not tracks:
        raise SystemExit(f"No stitched_gaze_track.json found under {args.input}")
    summary = []
    for track in tracks:
        try:
            report = analyze_track(track, params)
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            print(f"SKIP {track}: {exc}")
            continue
        base = track.with_name(f"{track.stem}_turning_analysis")
        json_path = base.with_suffix(".json")
        csv_path = base.with_suffix(".csv")
        json_path.write_text(json.dumps(report, indent=2))
        save_csv(report, csv_path)
        if not args.no_plot:
            save_plot(report, base.with_suffix(".png"))
        chosen = report["chosen"]
        print(
            f"{track}: "
            + (f"point {chosen['point_number']} (index={chosen['index']}, "
               f"turn={chosen['turn_deg']}°, score={chosen['score']})" if chosen else "no candidate")
        )
        summary.append({
            "track": str(track),
            "chosen_index": chosen["index"] if chosen else None,
            "chosen_point_number": chosen["point_number"] if chosen else None,
            "turn_deg": chosen["turn_deg"] if chosen else None,
            "score": chosen["score"] if chosen else None,
            "candidate_count": len(report["candidates"]),
        })

    summary_path = (args.input if args.input.is_dir() else args.input.parent) / "gaze_turning_analysis_summary.csv"
    if summary:
        import csv
        with summary_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
            writer.writeheader()
            writer.writerows(summary)
        print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
