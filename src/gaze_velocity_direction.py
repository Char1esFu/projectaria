"""Core gaze-track kinematics plus offline JSON/CSV/plot reporting."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def load_track(track_path: Path) -> tuple[list[dict], np.ndarray, np.ndarray]:
    """Load timestamped canvas points from a stitched gaze-track JSON."""
    records = json.loads(track_path.read_text()).get("points", [])
    records = [
        record for record in records
        if record.get("t_sec") is not None and record.get("canvas_xy") is not None
    ]
    if len(records) < 3:
        raise ValueError("at least 3 timestamped gaze points are required")
    t = np.asarray([record["t_sec"] for record in records], dtype=np.float64)
    points = np.asarray([record["canvas_xy"] for record in records], dtype=np.float64)
    if np.any(np.diff(t) <= 0):
        raise ValueError("timestamps must be strictly increasing")
    return records, t, points


def calculate_motion(
    t: np.ndarray, points: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return segment midpoint times, velocity, speed, heading, turn times/angles."""
    dt = np.diff(t)
    velocity = np.diff(points, axis=0) / dt[:, None]
    speed = np.linalg.norm(velocity, axis=1)
    heading = np.degrees(np.arctan2(velocity[:, 1], velocity[:, 0]))
    turn = (np.diff(heading) + 180.0) % 360.0 - 180.0
    velocity_t = (t[:-1] + t[1:]) / 2.0
    turn_t = (velocity_t[:-1] + velocity_t[1:]) / 2.0
    return velocity_t, velocity, speed, heading, turn_t, turn


def calculate_point_metrics(
    records: list[dict], t: np.ndarray, points: np.ndarray, direction_window: int
) -> list[dict]:
    """Calculate aligned per-point motion metrics used by offline analyzers."""
    if direction_window < 1:
        raise ValueError("direction_window must be >= 1")
    n = len(points)
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative_path = np.r_[0.0, np.cumsum(segment_lengths)]
    total_path = float(cumulative_path[-1])
    path_fraction = cumulative_path / total_path if total_path > 0 else np.zeros(n)
    rows: list[dict] = []
    k = direction_window
    for i, record in enumerate(records):
        row = {
            "index": i,
            "point_number": i + 1,
            "stamp_ns": record.get("stamp_ns"),
            "t_sec": round(float(t[i]), 6),
            "x_px": round(float(points[i, 0]), 3),
            "y_px": round(float(points[i, 1]), 3),
            "path_fraction": round(float(path_fraction[i]), 4),
            "incoming_displacement_px": None,
            "outgoing_displacement_px": None,
            "incoming_speed_px_s": None,
            "outgoing_speed_px_s": None,
            "incoming_heading_deg": None,
            "outgoing_heading_deg": None,
            "turn_deg": None,
            "turn_score": None,
            "central_speed_px_s": None,
            "adjacent_mean_speed_px_s": None,
            "speed_dip": None,
            "prominence_px": None,
        }
        if i < k or i >= n - k:
            rows.append(row)
            continue
        incoming = points[i] - points[i - k]
        outgoing = points[i + k] - points[i]
        incoming_disp = float(np.linalg.norm(incoming))
        outgoing_disp = float(np.linalg.norm(outgoing))
        incoming_heading = float(np.degrees(np.arctan2(incoming[1], incoming[0])))
        outgoing_heading = float(np.degrees(np.arctan2(outgoing[1], outgoing[0])))
        signed_turn = float((outgoing_heading - incoming_heading + 180.0) % 360.0 - 180.0)
        turn_score = (
            (1.0 - float(np.dot(incoming, outgoing)) / (incoming_disp * outgoing_disp)) / 2.0
            if incoming_disp and outgoing_disp else 0.0
        )
        incoming_segment_speed = (
            float(np.linalg.norm(points[i] - points[i - 1])) / float(t[i] - t[i - 1])
        )
        outgoing_segment_speed = (
            float(np.linalg.norm(points[i + 1] - points[i])) / float(t[i + 1] - t[i])
        )
        adjacent_mean_speed = (incoming_segment_speed + outgoing_segment_speed) / 2.0
        central_speed = (
            float(np.linalg.norm(points[i + 1] - points[i - 1]))
            / float(t[i + 1] - t[i - 1])
        )
        speed_dip = (
            float(np.clip(1.0 - central_speed / adjacent_mean_speed, 0.0, 1.0))
            if adjacent_mean_speed > 0 else 0.0
        )
        chord = points[i + k] - points[i - k]
        chord_length = float(np.linalg.norm(chord))
        prominence = (
            abs(float(np.cross(chord, points[i] - points[i - k]))) / chord_length
            if chord_length > 0 else min(incoming_disp, outgoing_disp)
        )
        row.update({
            "incoming_displacement_px": round(incoming_disp, 3),
            "outgoing_displacement_px": round(outgoing_disp, 3),
            "incoming_speed_px_s": round(incoming_disp / float(t[i] - t[i - k]), 3),
            "outgoing_speed_px_s": round(outgoing_disp / float(t[i + k] - t[i]), 3),
            "incoming_heading_deg": round(incoming_heading, 3),
            "outgoing_heading_deg": round(outgoing_heading, 3),
            "turn_deg": round(signed_turn, 3),
            "turn_score": round(turn_score, 4),
            "central_speed_px_s": round(central_speed, 3),
            "adjacent_mean_speed_px_s": round(adjacent_mean_speed, 3),
            "speed_dip": round(speed_dip, 4),
            "prominence_px": round(prominence, 3),
        })
        rows.append(row)
    return rows


def analyze_velocity_track(track_path: Path, direction_window: int = 2) -> dict:
    records, t, points = load_track(track_path)
    rows = calculate_point_metrics(records, t, points, direction_window)
    _, _, speed, _, _, turn = calculate_motion(t, points)
    return {
        "source": str(track_path),
        "params": {"direction_window": direction_window},
        "point_count": len(points),
        "duration_sec": round(float(t[-1] - t[0]), 6),
        "total_path_px": round(float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum()), 3),
        "mean_segment_speed_px_s": round(float(speed.mean()), 3),
        "max_segment_speed_px_s": round(float(speed.max()), 3),
        "mean_abs_adjacent_turn_deg": round(float(np.abs(turn).mean()), 3),
        "points": rows,
    }


def write_points_csv(report: dict, path: Path) -> None:
    rows = report["points"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_velocity_plot(report: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = report["points"]
    t = np.asarray([row["t_sec"] for row in rows])
    points = np.asarray([[row["x_px"], row["y_px"]] for row in rows])
    velocity_t, velocity, speed, heading, turn_t, turn = calculate_motion(t, points)
    fig = plt.figure(figsize=(12, 10), constrained_layout=True)
    grid = fig.add_gridspec(2, 2)
    track_ax = fig.add_subplot(grid[:, 0])
    speed_ax = fig.add_subplot(grid[0, 1])
    direction_ax = fig.add_subplot(grid[1, 1])
    turn_ax = direction_ax.twinx()
    track_ax.plot(points[:, 0], points[:, 1], color="#747474", linewidth=2.2)
    track_ax.scatter(points[:, 0], points[:, 1], s=58, c=t, cmap="plasma",
                     edgecolors="white", linewidths=1.2, zorder=3)
    arrows = track_ax.quiver(points[:-1, 0], points[:-1, 1], velocity[:, 0], velocity[:, 1],
                             speed, angles="xy", scale_units="xy", scale=8, cmap="viridis",
                             width=0.009)
    track_ax.invert_yaxis()
    track_ax.set_aspect("equal", adjustable="datalim")
    track_ax.set_title("Gaze trajectory and velocity direction")
    track_ax.set_xlabel("canvas x (px)")
    track_ax.set_ylabel("canvas y (px, downward)")
    fig.colorbar(arrows, ax=track_ax, label="speed (px/s)", shrink=0.7)
    speed_ax.plot(velocity_t, speed, "o-", color="#5b4ab0")
    speed_ax.set_title("Gaze speed")
    speed_ax.set_ylabel("speed (px/s)")
    direction_ax.plot(velocity_t, heading, "o-", color="#1878b4")
    turn_ax.bar(turn_t, turn, width=np.median(np.diff(velocity_t)) * 0.55,
                color="#e68a2e", alpha=0.35)
    direction_ax.set_title("Direction and signed direction change")
    direction_ax.set_ylabel("heading (deg)", color="#1878b4")
    turn_ax.set_ylabel("direction change (deg)", color="#b45f13")
    for ax in (track_ax, speed_ax, direction_ax):
        ax.grid(alpha=0.25)
        ax.set_xlabel("time (s)" if ax is not track_ax else "canvas x (px)")
    fig.suptitle(f"Gaze velocity-direction analysis — {Path(report['source']).parent}")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def find_tracks(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(path.rglob("stitched_gaze_track.json"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Track JSON or recordings directory")
    parser.add_argument("--direction-window", type=int, default=2)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    tracks = find_tracks(args.input)
    if not tracks:
        raise SystemExit(f"No stitched_gaze_track.json found under {args.input}")
    summary = []
    for track in tracks:
        try:
            report = analyze_velocity_track(track, args.direction_window)
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            print(f"SKIP {track}: {exc}")
            continue
        base = track.with_name(f"{track.stem}_velocity_direction")
        base.with_suffix(".json").write_text(json.dumps(report, indent=2))
        write_points_csv(report, base.with_suffix(".csv"))
        if not args.no_plot:
            save_velocity_plot(report, base.with_suffix(".png"))
        print(f"{track}: {report['point_count']} points, max speed={report['max_segment_speed_px_s']} px/s")
        summary.append({key: report[key] for key in (
            "source", "point_count", "duration_sec", "total_path_px",
            "mean_segment_speed_px_s", "max_segment_speed_px_s",
            "mean_abs_adjacent_turn_deg",
        )})
    if summary:
        summary_path = (args.input if args.input.is_dir() else args.input.parent) / "gaze_velocity_direction_summary.csv"
        with summary_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
            writer.writeheader()
            writer.writerows(summary)
        print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
