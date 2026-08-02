"""Shared gaze-label score-stability selection and visualization.

Used by the online recording pipeline and ``src.offline_gaze_label``. This
module intentionally has no standalone CLI or alternate analysis modes: both
callers share the same variance, endpoint, fallback, plotting, and export code.

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

Selection keeps every interior window below the threshold. When none passes,
the ``top`` lowest-variance interior windows are kept instead. Endpoint windows
stay excluded either way.

Outputs written by the callers into the recording directory:
  * ``gaze_score_stability_variance.png`` — variance-vs-time point/line plot of
    every window, the threshold line, and the selected windows highlighted
    (``--hide-excluded`` drops the endpoint windows instead of greying them out);
  * ``stitched_variance.png`` — the full gaze trajectory drawn on
    ``stitched_optimized.png`` with the selected windows' gaze points highlighted;
  * ``gaze_score_stability_frames.png`` — the selected windows' centre frames.
"""

import json
import shutil
from pathlib import Path

import numpy as np

from src.detection_infill import frame_image_path
from src.gaze_rgb_config import (
    DEFAULT_GAZE_BOUNDARY_RADIUS,
    DEFAULT_GAZE_MIN_BOUNDARY_POINTS,
)
from src.gaze_track_boundary import boundary_region


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
    """Keep all sub-threshold interior windows, otherwise the ``top`` smallest.

    Endpoint (start/end fixation) windows in ``excluded`` are removed *first*, so
    the threshold and fallback ranking only ever see interior candidates.

    When the threshold leaves nothing, the ``top`` steadiest interior windows
    are returned regardless of the threshold,
    tagged with ``threshold_fallback``. ``top=None`` keeps every interior window
    in fallback. Endpoint windows are never resurrected this way: with no
    interior window at all the result stays empty.
    """
    interior = [
        r for r in results
        if not (excluded and _is_boundary(r["center_stamp_ns"], excluded, track_stamps))
    ]
    eligible = [
        r for r in interior
        if threshold is None or r["variance"] < threshold
    ]
    if not eligible and interior:
        fallback = sorted(interior, key=lambda r: r["variance"])
        if top is not None:
            fallback = fallback[:top]
        for result in fallback:
            result["threshold_fallback"] = True
        return fallback
    eligible.sort(key=lambda r: r["variance"])
    return eligible


def select_stable_windows(
    label_log: list[dict],
    track_path: Path | None = None,
    *,
    window: int = 3,
    threshold: float | None = None,
    top: int | None = 5,
    boundary_radius: float = DEFAULT_GAZE_BOUNDARY_RADIUS,
    force_endpoint_points: int = 3,
) -> tuple[list[dict], set[int] | None]:
    """Shared core of the variance selector.

    Slides a variance window over the raw ``label_log`` scores, drops start/end
    fixation windows using ``track_path``, then keeps every sub-threshold interior
    window; if the threshold rejects them all, the ``top`` steadiest interior
    windows are kept anyway. Returns
    ``(selected windows, excluded endpoint stamps)`` — callers derive whichever
    stamps they need (window centres for a one-per-window view, all member frames
    for score averaging).
    """
    frames = sorted(
        (f for f in label_log if f.get("stamp_ns") is not None),
        key=lambda f: f["stamp_ns"],
    )
    results = analyze(frames, window)
    if not results:
        return [], None
    excluded, track_stamps = boundary_from_track(
        track_path, boundary_radius, DEFAULT_GAZE_MIN_BOUNDARY_POINTS,
        force_endpoint_points,
    )
    selected = select_windows(results, threshold, top, excluded, track_stamps)
    return selected, excluded


def export_selected_frames(
    session_dir: Path,
    frames_dir: Path,
    label_log: list[dict],
    selected_stamps: list[int],
    subdir: str = "variance_selected",
) -> Path | None:
    """Copy the selected frames' images and dump their YOLO scores to a subfolder.

    Creates ``<session_dir>/<subdir>/`` holding one PNG per selected stamp plus
    ``scores.json`` — each stamp's gaze pixel and the detected label/score list
    from ``label_log``. The image is the in-fill redraw when one exists (so it
    shows the detections those scores were computed from), otherwise the frame
    from ``frames_dir``. Stamps with no saved image (e.g. a dropped RGB frame)
    are still recorded, with a null image. Returns the subfolder, or ``None``
    The directory is replaced on every run, including an empty selection, so
    stale timestamps/images can never survive a new variance calculation.
    """
    out_dir = session_dir / subdir
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    by_stamp = {f["stamp_ns"]: f for f in label_log if f.get("stamp_ns") is not None}

    records = []
    copied = 0
    for stamp in selected_stamps:
        entry = by_stamp.get(stamp, {})
        src = frame_image_path(session_dir, stamp, frames_subdir=frames_dir.name)
        image_name = None
        if src is not None:
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
        "selected_stamps_ns": selected_stamps,
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
    hide_excluded: bool = False, source_note: str | None = None,
) -> None:
    """Point/line plot of every window's variance over time.

    Selected windows are drawn in a distinct colour and endpoint (start/end
    fixation) windows are greyed out; the threshold (if any) is a horizontal
    reference line. A log y-axis keeps the wide variance range readable.

    ``hide_excluded`` drops the endpoint windows from the plot entirely instead
    of greying them out, leaving only the candidates selection actually ranked.
    The connecting line then spans the removed spans, so the time axis stays
    true but the polyline is no longer sample-contiguous.

    ``source_note`` goes under the title to record which detections the plotted
    variance was computed over — the figure should not be ambiguous about
    whether in-filled detections were included.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if hide_excluded and boundary_flags is not None and boundary_flags.any():
        keep = ~boundary_flags
        results = [r for r, k in zip(results, keep) if k]
        boundary_flags = np.zeros(len(results), dtype=bool)
        if not results:
            print("Variance timeline skipped: every window is an endpoint window.")
            return

    times = np.array([(r["center_stamp_ns"] - t0_ns) / 1e9 for r in results])
    var = np.array([r["variance"] for r in results])
    # Window start + members identify a window across a regenerated analysis.
    # Center stamp alone is insufficient because zedr async-stamp races can let
    # different windows share the same centre timestamp.
    def window_key(result: dict) -> tuple:
        return (
            result.get("window_start_index"),
            tuple(result.get("member_stamps_ns", [])),
        )

    selected_keys = {window_key(result) for result in selected}
    is_sel = np.array([window_key(result) in selected_keys for result in results])
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
    if source_note:
        ax.text(0.0, 1.005, f"detections: {source_note}", transform=ax.transAxes,
                fontsize=8.5, color="#52514e")
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
    """Plot each selected window's centre frame in a grid and save to disk.

    Images come from ``infilled_frames/`` when in-fill has run, so a window's
    picture shows the same detections its variance was computed from — a frame
    whose flicker was repaired is the redrawn ``_filled`` version, marked in its
    subplot title. Falls back to the raw ``frames/`` image otherwise.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not windows:
        print("Frames grid skipped: no windows selected.")
        return

    n = len(windows)
    cols = min(n, 3) or 1
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)

    for ax in axes.ravel():
        ax.axis("off")

    filled_shown = 0
    for rank, (win, ax) in enumerate(zip(windows, axes.ravel()), start=1):
        stamp = win["center_stamp_ns"]
        img_path = frame_image_path(recording_dir, stamp)
        title = f"#{rank}  var={win['variance']:.3e}\n{stamp}"
        if img_path is None:
            ax.text(0.5, 0.5, f"missing\n{stamp}.png", ha="center", va="center")
        else:
            ax.imshow(plt.imread(str(img_path)))
            if img_path.name.endswith("_filled.png"):
                title += "  (in-filled)"
                filled_shown += 1
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    fig.suptitle(
        f"Most stable windows (lowest label-score variance) — {recording_dir}",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Frames grid ({n} windows, {filled_shown} showing in-filled detections) "
          f"-> {out_path}")


def generate_stability_plots(
    recording_dir: Path,
    label_log: list[dict],
    selected: list[dict],
    *,
    window: int,
    threshold: float | None,
    boundary_radius: float = DEFAULT_GAZE_BOUNDARY_RADIUS,
    force_endpoint_points: int = 3,
    hide_excluded: bool = False,
    source_note: str | None = None,
) -> None:
    """Write the same three plots as the offline stability CLI.

    This is also used by the live recording flush path.  ``selected`` must be
    the windows already used for the /gaze_label calculation, so visualization
    can never silently select a different set of timestamps.
    """
    recording_dir = Path(recording_dir)
    output_paths = (
        recording_dir / "gaze_score_stability_variance.png",
        recording_dir / "stitched_variance.png",
        recording_dir / "gaze_score_stability_frames.png",
    )
    # These are replaceable derived artifacts. Remove all previous versions
    # first so a run with no valid/selected window cannot leave stale plots.
    for output_path in output_paths:
        if output_path.exists():
            output_path.unlink()

    frames = sorted(
        (frame for frame in label_log if frame.get("stamp_ns") is not None),
        key=lambda frame: frame["stamp_ns"],
    )
    results = analyze(frames, window)
    if not results:
        print("Stability plots skipped: no valid variance window.")
        return

    track_path = recording_dir / "stitched_gaze_track.json"
    excluded, track_stamps = boundary_from_track(
        track_path, boundary_radius, DEFAULT_GAZE_MIN_BOUNDARY_POINTS,
        force_endpoint_points,
    )
    boundary_flags = np.zeros(len(results), dtype=bool)
    if excluded:
        boundary_flags = np.array([
            _is_boundary(result["center_stamp_ns"], excluded, track_stamps)
            for result in results
        ])

    plot_variance_timeline(
        results, selected, threshold, frames[0]["stamp_ns"],
        output_paths[0],
        boundary_flags=boundary_flags,
        hide_excluded=hide_excluded,
        source_note=source_note,
    )
    plot_stitched_variance(
        recording_dir, selected, output_paths[1],
        excluded=excluded,
    )
    plot_frames_grid(
        recording_dir, selected,
        output_paths[2],
    )
