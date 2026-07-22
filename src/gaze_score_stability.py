"""Offline gaze-label score-stability analysis.

Reads the raw per-frame YOLO label scores logged in ``gaze_labels.json`` and
looks for *stable* moments — short spans where the detection scores barely move.

A sliding window of ``--window`` consecutive frames is swept over the recording.
Inside each window every label's scores are collected and its variance computed;
the window's stability metric is the *unweighted average* of those per-label
variances (a lower value = steadier scores = more stable).

Per-label occurrence rules inside a window:
  * seen >= 2 times  -> variance over the available scores (2 or more) is used;
  * seen exactly once -> the label has no spread, so it is dropped from the
    unweighted average entirely.
A window in which no label appears at least twice has no defined metric and is
skipped.

Selection of the reported windows:
  * ``--threshold`` first keeps only windows whose metric is *below* it;
  * ``--top`` then keeps the N smallest of those; when ``--top`` is omitted every
    window under the threshold is kept.

Outputs (unless ``--no-plot``), all written into the recording directory:
  * ``gaze_score_stability_variance.png`` — variance-vs-time point/line plot of
    every window, the threshold line, and the selected windows highlighted;
  * ``stitched_variance.png`` — the full gaze trajectory drawn on
    ``stitched_optimized.png`` with the selected windows' gaze points highlighted;
  * ``gaze_score_stability_frames.png`` — the selected windows' centre frames.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

# Allow `from src...` imports whether launched as a module or a script path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.gaze_rgb_config import (  # noqa: E402
    DEFAULT_GAZE_BOUNDARY_RADIUS,
    DEFAULT_GAZE_MIN_BOUNDARY_POINTS,
)
from src.gaze_track_boundary import boundary_region  # noqa: E402


def load_frames(recording_dir: Path) -> list[dict]:
    """Return the timestamp-ordered frame log from gaze_labels.json."""
    data = json.loads((recording_dir / "gaze_labels.json").read_text())
    frames = data.get("frames", [])
    return sorted(frames, key=lambda f: f["stamp_ns"])


def window_variance(frames: list[dict]) -> float | None:
    """Unweighted mean of per-label score variances within one window.

    Labels seen only once contribute no spread and are excluded. Returns
    ``None`` when no label occurs at least twice (metric undefined).
    """
    scores_by_label: dict[str, list[float]] = {}
    for frame in frames:
        for det in frame.get("detected", []):
            label, score = det.get("label"), det.get("score")
            if label is not None and score is not None:
                scores_by_label.setdefault(label, []).append(float(score))

    per_label_var = [
        float(np.var(scores))  # population variance (ddof=0)
        for scores in scores_by_label.values()
        if len(scores) >= 2
    ]
    if not per_label_var:
        return None
    return float(np.mean(per_label_var))


def analyze(frames: list[dict], window: int) -> list[dict]:
    """Slide the window over the frames and score every valid window.

    Results are returned in time order. Each carries the window's variance
    metric, the centre frame's timestamp (its representative), the member
    timestamps, and a per-label occurrence count.
    """
    results: list[dict] = []
    for start in range(len(frames) - window + 1):
        chunk = frames[start:start + window]
        metric = window_variance(chunk)
        if metric is None:
            continue
        centre = chunk[window // 2]
        results.append({
            "variance": metric,
            "center_stamp_ns": centre["stamp_ns"],
            "window_start_index": start,
            "member_stamps_ns": [f["stamp_ns"] for f in chunk],
            "label_counts": _label_counts(chunk),
        })
    return results


def _label_counts(frames: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for frame in frames:
        for det in frame.get("detected", []):
            label = det.get("label")
            if label is not None:
                counts[label] = counts.get(label, 0) + 1
    return counts


def _boundary_excluded(
    records: list[dict], radius: float, min_points: int, force_points: int
) -> set[int]:
    """Excluded start/end ``stamp_ns`` from stitched gaze-track records.

    Mirrors the endpoint filtering used for stitched_trajectory.png: the runs of
    gaze points that stay within ``radius`` px of the first/last point are the
    start/end fixations. When an endpoint's run does *not* reach ``min_points``
    (no genuine fixation cluster there), that endpoint is still force-excluded by
    dropping its outermost ``force_points`` points, so a non-settled endpoint can
    never leak into the candidate pool.
    """
    points_xy = np.asarray([r["canvas_xy"] for r in records], dtype=np.float64)
    n = len(records)
    start = boundary_region(points_xy, True, radius)
    if len(start) < min_points:
        start = set(range(min(force_points, n)))          # force first N points
    end = boundary_region(points_xy, False, radius)
    if len(end) < min_points:
        end = set(range(max(0, n - force_points), n))     # force last N points
    return {records[i]["stamp_ns"] for i in (start | end)}


def boundary_from_track(
    track_path: Path | None, radius: float, min_points: int, force_points: int
) -> tuple[set[int] | None, np.ndarray | None]:
    """Load a gaze track and return (excluded stamps, all track stamps).

    Returns ``(None, None)`` when exclusion is disabled (``radius <= 0``) or the
    track is missing/empty.
    """
    if track_path is None or radius <= 0 or not Path(track_path).exists():
        return None, None
    records = json.loads(Path(track_path).read_text()).get("points", [])
    if not records:
        return None, None
    excluded = _boundary_excluded(records, radius, min_points, force_points)
    track_stamps = np.array([r["stamp_ns"] for r in records])
    return excluded, track_stamps


def _is_boundary(center_stamp: int, excluded: set[int], track_stamps: np.ndarray) -> bool:
    """True if the window's centre maps to an excluded endpoint gaze sample.

    Window centres need not be track samples (only the stitched frames are), so
    the centre is matched to its nearest track sample by timestamp.
    """
    nearest = int(track_stamps[np.argmin(np.abs(track_stamps - center_stamp))])
    return nearest in excluded


def select_windows(
    results: list[dict], threshold: float | None, top: int | None,
    excluded: set[int] | None = None, track_stamps: np.ndarray | None = None,
) -> list[dict]:
    """Drop endpoint windows, keep those below ``threshold``, then the ``top`` smallest.

    Endpoint (start/end fixation) windows in ``excluded`` are removed *first*, so
    the threshold and ``top`` cut only ever see interior candidates. ``top`` of
    ``None`` keeps every remaining window that passes the threshold.
    """
    eligible = []
    for r in results:
        if excluded and _is_boundary(r["center_stamp_ns"], excluded, track_stamps):
            continue
        if threshold is not None and r["variance"] >= threshold:
            continue
        eligible.append(r)
    eligible.sort(key=lambda r: r["variance"])
    return eligible if top is None else eligible[:top]


def select_stable_windows(
    label_log: list[dict],
    track_path: Path | None = None,
    *,
    window: int = 3,
    threshold: float | None = None,
    top: int | None = 5,
    boundary_radius: float = DEFAULT_GAZE_BOUNDARY_RADIUS,
    min_boundary_points: int = DEFAULT_GAZE_MIN_BOUNDARY_POINTS,
    force_endpoint_points: int = 3,
) -> tuple[list[dict], set[int] | None]:
    """Shared core of the variance selector.

    Slides a variance window over the raw ``label_log`` scores, drops start/end
    fixation windows using ``track_path``, then keeps the lowest-variance interior
    windows (``threshold`` then ``top``). Returns ``(selected windows, excluded
    endpoint stamps)`` — callers derive whichever stamps they need (window centres
    for a one-per-window view, all member frames for score averaging).
    """
    frames = sorted(
        (f for f in label_log if f.get("stamp_ns") is not None),
        key=lambda f: f["stamp_ns"],
    )
    results = analyze(frames, window)
    if not results:
        return [], None
    excluded, track_stamps = boundary_from_track(
        track_path, boundary_radius, min_boundary_points, force_endpoint_points
    )
    selected = select_windows(results, threshold, top, excluded, track_stamps)
    return selected, excluded


def select_stable_stamps(
    label_log: list[dict],
    track_path: Path | None = None,
    *,
    window: int = 3,
    threshold: float | None = None,
    top: int | None = 5,
    boundary_radius: float = DEFAULT_GAZE_BOUNDARY_RADIUS,
    min_boundary_points: int = DEFAULT_GAZE_MIN_BOUNDARY_POINTS,
    force_endpoint_points: int = 3,
    visualize: bool = False,
) -> list[int]:
    """Runtime entry point returning the selected windows' centre ``stamp_ns``.

    A variance window only flags its centre timestamp as a point of interest, so
    the returned stamps are one per selected window (its centre) — the frames whose
    scores drive the low-spread YOLO average. When ``visualize`` is set and a track
    is given, also writes ``stitched_variance.png`` next to it.
    """
    selected, excluded = select_stable_windows(
        label_log, track_path, window=window, threshold=threshold, top=top,
        boundary_radius=boundary_radius, min_boundary_points=min_boundary_points,
        force_endpoint_points=force_endpoint_points,
    )
    if visualize and track_path is not None:
        plot_stitched_variance(
            Path(track_path).parent, selected,
            Path(track_path).parent / "stitched_variance.png", excluded=excluded,
        )
    return sorted({win["center_stamp_ns"] for win in selected})


def export_selected_frames(
    session_dir: Path,
    frames_dir: Path,
    label_log: list[dict],
    selected_stamps: list[int],
    subdir: str = "variance_selected",
) -> Path | None:
    """Copy the selected frames' images and dump their YOLO scores to a subfolder.

    Creates ``<session_dir>/<subdir>/`` holding one PNG per selected stamp (copied
    from ``frames_dir``) plus ``scores.json`` — each stamp's gaze pixel and its
    detected label/score list taken straight from the raw label log. Stamps with
    no saved image (e.g. a dropped RGB frame) are still recorded, with a null
    image. Returns the subfolder, or ``None`` when there is nothing to export.
    """
    if not selected_stamps:
        return None
    out_dir = session_dir / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    by_stamp = {f["stamp_ns"]: f for f in label_log if f.get("stamp_ns") is not None}

    records = []
    copied = 0
    for stamp in selected_stamps:
        entry = by_stamp.get(stamp, {})
        src = frames_dir / f"{stamp}.png"
        image_name = None
        if src.exists():
            image_name = src.name
            shutil.copy2(src, out_dir / src.name)
            copied += 1
        records.append({
            "stamp_ns": stamp,
            "image": image_name,
            "gaze_px": entry.get("gaze_px"),
            "detected": entry.get("detected", []),
        })

    (out_dir / "scores.json").write_text(json.dumps({
        "method": "variance",
        "count": len(records),
        "images_copied": copied,
        "frames": records,
    }, indent=2))
    print(f"Saved {copied}/{len(records)} variance-selected frame image(s) + "
          f"scores.json -> {out_dir}")
    return out_dir


def plot_variance_timeline(
    results: list[dict], selected: list[dict], threshold: float | None,
    t0_ns: int, out_path: Path, boundary_flags: np.ndarray | None = None,
) -> None:
    """Point/line plot of every window's variance over time.

    Selected windows are drawn in a distinct colour and endpoint (start/end
    fixation) windows are greyed out; the threshold (if any) is a horizontal
    reference line. A log y-axis keeps the wide variance range readable.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    times = np.array([(r["center_stamp_ns"] - t0_ns) / 1e9 for r in results])
    var = np.array([r["variance"] for r in results])
    # Identify selected windows by object identity, not center stamp: duplicate
    # frame timestamps (zedr async-stamp races) let different windows share a
    # centre stamp, so stamp membership would over-highlight non-selected ones.
    selected_ids = {id(r) for r in selected}
    is_sel = np.array([id(r) in selected_ids for r in results])
    is_bnd = (boundary_flags if boundary_flags is not None
              else np.zeros(len(results), dtype=bool))
    is_plain = ~is_sel & ~is_bnd

    # Floor zeros so the log axis can render them (variance can be exactly 0).
    positive = var[var > 0]
    floor = positive.min() * 0.1 if positive.size else 1e-12
    var_plot = np.clip(var, floor, None)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(times, var_plot, color="#9aa0a6", linewidth=1.2, zorder=1)
    ax.scatter(times[is_plain], var_plot[is_plain], s=28, color="#2a78d6",
               label="window", zorder=2)
    if is_bnd.any():
        ax.scatter(times[is_bnd], var_plot[is_bnd], s=40, color="#9aa0a6",
                   marker="x", linewidth=1.4, label="endpoint (excluded)", zorder=2)
    if is_sel.any():
        ax.scatter(times[is_sel], var_plot[is_sel], s=90, color="#e8590c",
                   edgecolor="black", linewidth=0.7, label="selected", zorder=3)
    if threshold is not None:
        ax.axhline(threshold, color="#d61f69", linestyle="--", linewidth=1.4,
                   label=f"threshold = {threshold:g}")

    ax.set_yscale("log")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("window label-score variance (log)")
    ax.set_title("Gaze label-score stability over time")
    ax.grid(True, which="both", color="#e4e3e0", linewidth=0.7)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Variance timeline ({len(results)} windows, "
          f"{len(selected)} selected) -> {out_path}")


def plot_stitched_variance(
    recording_dir: Path, selected: list[dict], out_path: Path,
    excluded: set[int] | None = None,
) -> None:
    """Draw the full gaze trajectory on stitched_optimized.png and highlight the
    selected windows' gaze points (matched to the nearest track sample). Endpoint
    (start/end fixation) samples in ``excluded`` are marked in cyan."""
    import cv2

    base_path = recording_dir / "stitched_optimized.png"
    track_path = recording_dir / "stitched_gaze_track.json"
    if not base_path.exists() or not track_path.exists():
        print(f"Stitched variance viz skipped: missing "
              f"{'stitched_optimized.png' if not base_path.exists() else ''}"
              f"{' and ' if not base_path.exists() and not track_path.exists() else ''}"
              f"{'stitched_gaze_track.json' if not track_path.exists() else ''}.")
        return

    canvas = cv2.imread(str(base_path))
    points = json.loads(track_path.read_text()).get("points", [])
    if canvas is None or not points:
        print("Stitched variance viz skipped: empty canvas or gaze track.")
        return

    pts_xy = [(int(round(p["canvas_xy"][0])), int(round(p["canvas_xy"][1])))
              for p in points]
    track_stamps = np.array([p["stamp_ns"] for p in points])

    # Full trajectory: thin red polyline + small red dots (as in stitched_trajectory).
    if len(pts_xy) >= 2:
        cv2.polylines(canvas, [np.array(pts_xy, dtype=np.int32)], False,
                      (0, 0, 255), 1, cv2.LINE_AA)
    for p in pts_xy:
        cv2.circle(canvas, p, 2, (0, 0, 255), -1, cv2.LINE_AA)

    # Mark excluded start/end fixation samples (cyan) so it is visible which
    # gaze points the endpoint filter removed from candidacy.
    _EXCL = (255, 255, 0)
    if excluded:
        for p, rec in zip(pts_xy, points):
            if rec["stamp_ns"] in excluded:
                cv2.circle(canvas, p, 5, _EXCL, 1, cv2.LINE_AA)

    # Highlight selected windows at their nearest track sample. Magenta avoids
    # clashing with the green YOLO circles baked into the source mosaic.
    _HL = (255, 0, 255)
    for rank, win in enumerate(selected, start=1):
        idx = int(np.argmin(np.abs(track_stamps - win["center_stamp_ns"])))
        cx, cy = pts_xy[idx]
        cv2.circle(canvas, (cx, cy), 8, _HL, 2, cv2.LINE_AA)
        cv2.circle(canvas, (cx, cy), 3, _HL, -1, cv2.LINE_AA)
        text = str(rank)
        cv2.putText(canvas, text, (cx + 6, cy - 6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, text, (cx + 6, cy - 6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, _HL, 1, cv2.LINE_AA)

    # Legend.
    y = 18
    cv2.circle(canvas, (12, y - 4), 3, (0, 0, 255), -1, cv2.LINE_AA)
    cv2.putText(canvas, "gaze trajectory", (24, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (0, 0, 255), 1, cv2.LINE_AA)
    y += 20
    cv2.circle(canvas, (12, y - 4), 6, _HL, 2, cv2.LINE_AA)
    cv2.putText(canvas, "selected (low-variance)", (24, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, _HL, 1, cv2.LINE_AA)
    if excluded:
        y += 20
        cv2.circle(canvas, (12, y - 4), 5, _EXCL, 1, cv2.LINE_AA)
        cv2.putText(canvas, "endpoint (excluded)", (24, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, _EXCL, 1, cv2.LINE_AA)

    cv2.imwrite(str(out_path), canvas)
    print(f"Stitched variance viz ({len(pts_xy)} gaze pts, "
          f"{len(selected)} highlighted) -> {out_path}")


def plot_frames_grid(
    recording_dir: Path, windows: list[dict], out_path: Path
) -> None:
    """Plot each selected window's centre frame in a grid and save to disk."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not windows:
        print("Frames grid skipped: no windows selected.")
        return

    frames_dir = recording_dir / "frames"
    n = len(windows)
    cols = min(n, 3) or 1
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)

    for ax in axes.ravel():
        ax.axis("off")

    for rank, (win, ax) in enumerate(zip(windows, axes.ravel()), start=1):
        stamp = win["center_stamp_ns"]
        img_path = frames_dir / f"{stamp}.png"
        title = f"#{rank}  var={win['variance']:.3e}\n{stamp}"
        if img_path.exists():
            ax.imshow(plt.imread(str(img_path)))
        else:
            ax.text(0.5, 0.5, f"missing\n{img_path.name}", ha="center", va="center")
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    fig.suptitle(
        f"Most stable windows (lowest label-score variance) — {recording_dir}",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Frames grid ({n} windows) -> {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("recording_dir", type=Path,
                        help="recording folder holding gaze_labels.json and frames/ "
                             "(e.g. recordings/test02/26)")
    parser.add_argument("-w", "--window", type=int, default=3,
                        help="sliding window length in frames (default: 3)")
    parser.add_argument("-n", "--top", type=int, default=None,
                        help="keep the N lowest-variance windows below the "
                             "threshold; omit to keep all that pass the threshold")
    parser.add_argument("-t", "--threshold", type=float, default=None,
                        help="only windows with variance below this value are "
                             "eligible (default: no threshold)")
    parser.add_argument("--boundary-radius", type=float,
                        default=DEFAULT_GAZE_BOUNDARY_RADIUS,
                        help="pixel radius on the stitched gaze track for the "
                             "start/end fixation regions excluded before "
                             f"selection (default: {DEFAULT_GAZE_BOUNDARY_RADIUS}; "
                             "0 disables endpoint exclusion)")
    parser.add_argument("--min-boundary-points", type=int,
                        default=DEFAULT_GAZE_MIN_BOUNDARY_POINTS,
                        help="minimum points for a start/end region to count as "
                             f"a fixation (default: {DEFAULT_GAZE_MIN_BOUNDARY_POINTS})")
    parser.add_argument("--force-endpoint-points", type=int, default=3,
                        help="when an endpoint has no fixation cluster, force-"
                             "exclude this many outermost points at that end "
                             "(default: 3)")
    parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=True,
                        help="write the timeline, stitched, and frame plots "
                             "(--no-plot to disable)")
    parser.add_argument("--export", action=argparse.BooleanOptionalAction, default=True,
                        help="copy the selected frames' images + YOLO scores into "
                             "<recording>/variance_selected/ (--no-export to disable)")
    return parser.parse_args()


def find_recordings(root: Path) -> list[Path]:
    """Resolve ``root`` to the recording folders to process.

    A folder holding ``gaze_labels.json`` is itself the single recording;
    otherwise every immediate sub-folder that holds one is returned (sorted),
    so an experiment path like ``recordings/test02`` batches all of its takes.
    """
    if (root / "gaze_labels.json").exists():
        return [root]
    return sorted(
        child for child in root.iterdir()
        if child.is_dir() and (child / "gaze_labels.json").exists()
    )


def process_recording(recording_dir: Path, args: argparse.Namespace) -> bool:
    """Analyze and (optionally) plot one recording. Returns True on success."""
    frames = load_frames(recording_dir)
    results = analyze(frames, args.window)
    if not results:
        print(f"Recording: {recording_dir}  -> skipped "
              f"(no window had a label seen at least twice).")
        return False

    excluded, track_stamps = boundary_from_track(
        recording_dir / "stitched_gaze_track.json", args.boundary_radius,
        args.min_boundary_points, args.force_endpoint_points,
    )
    boundary_flags = np.zeros(len(results), dtype=bool)
    if excluded:
        boundary_flags = np.array(
            [_is_boundary(r["center_stamp_ns"], excluded, track_stamps) for r in results]
        )

    selected = select_windows(
        results, args.threshold, args.top, excluded, track_stamps
    )
    t0_ns = frames[0]["stamp_ns"]

    if excluded is None and args.boundary_radius > 0:
        endpoint_note = "endpoint exclusion: unavailable (no stitched gaze track)"
    elif args.boundary_radius <= 0:
        endpoint_note = "endpoint exclusion: disabled"
    else:
        endpoint_note = (f"endpoint-excluded windows: {int(boundary_flags.sum())} "
                         f"(radius={args.boundary_radius:g}px)")

    print(f"Recording: {recording_dir}")
    print(f"Frames: {len(frames)}  window: {args.window}  valid windows: {len(results)}")
    print(f"Threshold: {args.threshold if args.threshold is not None else 'none'}  "
          f"top: {args.top if args.top is not None else 'all'}  "
          f"selected: {len(selected)}")
    print(endpoint_note)
    for rank, win in enumerate(selected, start=1):
        counts = ", ".join(f"{k}x{v}" for k, v in sorted(win["label_counts"].items()))
        print(f"#{rank}  variance={win['variance']:.6e}  "
              f"center_stamp_ns={win['center_stamp_ns']}")
        print(f"     window_start_index={win['window_start_index']}  "
              f"members={win['member_stamps_ns']}")
        print(f"     label_counts: {counts}")

    if args.plot:
        plot_variance_timeline(
            results, selected, args.threshold, t0_ns,
            recording_dir / "gaze_score_stability_variance.png",
            boundary_flags=boundary_flags,
        )
        plot_stitched_variance(
            recording_dir, selected,
            recording_dir / "stitched_variance.png",
            excluded=excluded,
        )
        plot_frames_grid(
            recording_dir, selected,
            recording_dir / "gaze_score_stability_frames.png",
        )

    if args.export:
        # One frame per selected window (its centre), matching the highlighted
        # points in stitched_variance.png.
        center_stamps = sorted({win["center_stamp_ns"] for win in selected})
        export_selected_frames(
            recording_dir, recording_dir / "frames", frames, center_stamps
        )
    return True


def main() -> None:
    args = parse_args()
    if args.window < 2:
        raise SystemExit("window must be >= 2 (variance needs at least 2 samples)")

    recordings = find_recordings(args.recording_dir)
    if not recordings:
        raise SystemExit(
            f"No gaze_labels.json found in {args.recording_dir} or its sub-folders."
        )

    ok = 0
    for i, recording_dir in enumerate(recordings):
        if len(recordings) > 1:
            print(f"\n===== [{i + 1}/{len(recordings)}] {recording_dir.name} =====")
        ok += process_recording(recording_dir, args)

    if len(recordings) > 1:
        print(f"\nDone: {ok}/{len(recordings)} recording(s) produced results.")


if __name__ == "__main__":
    main()
