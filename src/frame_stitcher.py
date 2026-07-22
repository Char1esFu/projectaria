"""Label-anchored frame stitching for recorded gaze sessions.

Frames are aligned by their logged YOLO detection centers (same label = same
static object), not feature matching. Valid frames are chained in timestamp
order, then jointly optimized to minimize every label cluster's spread while
one maximum-label frame fixes the coordinate gauge; overlapping pixels are averaged.
"""

import json
import math
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# BGR colors cycled per YOLO label on the trajectory map (red is reserved for gaze).
_PALETTE = [
    (0, 255, 0), (255, 200, 0), (0, 200, 255), (255, 0, 255),
    (0, 128, 255), (255, 255, 0), (128, 255, 128), (255, 128, 128),
]


def _centers_by_stamp(label_log: list[dict]) -> dict[int, dict[str, np.ndarray]]:
    """Map stamp_ns -> {label: center_px}; first log entry per stamp wins."""
    out: dict[int, dict[str, np.ndarray]] = {}
    for entry in label_log:
        ts = entry.get("stamp_ns")
        if ts is None or ts in out:
            continue
        centers = {
            d["label"]: np.asarray(d["center_px"], dtype=np.float64)
            for d in entry.get("detected", [])
            if d.get("center_px") is not None
        }
        if centers:
            out[ts] = centers
    return out


def _center_crop(img: np.ndarray, ratio: float) -> tuple[np.ndarray, tuple[int, int]]:
    """Center-crop to ratio. Returns (view, crop origin in the original image)."""
    h, w = img.shape[:2]
    cw, ch = int(round(w * ratio)), int(round(h * ratio))
    x0, y0 = (w - cw) // 2, (h - ch) // 2
    return img[y0:y0 + ch, x0:x0 + cw], (x0, y0)


def _rigid_fit(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares rigid transform (R, t) mapping src -> dst, (N, 2) each."""
    sc, dc = src.mean(axis=0), dst.mean(axis=0)
    H = (src - sc).T @ (dst - dc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T)) or 1.0
    R = Vt.T @ np.diag([1.0, d]) @ U.T
    return R, dc - R @ sc


def _chain_poses(
    obs: list[dict[str, np.ndarray]], min_points: int = 3
) -> tuple[list[int], list[np.ndarray], list[np.ndarray], list[dict], list[dict]]:
    """Chain usable frames into rigid-pose segments and keep the largest.

    A frame is usable only when it has at least ``min_points`` detections.  It
    joins the best existing segment whose last placed frame shares at least that
    many labels with it — preferring the largest such segment, then the most
    recent one — and otherwise seeds a *new* segment instead of being discarded.

    Preferring the largest candidate keeps the established chain going and lets
    it bridge over a transient odd frame (which becomes a throwaway one-frame
    segment) exactly as before.  Seeding rather than dropping is what fixes the
    real failure: once a short early run's shared objects leave the view for
    good, later frames form their own segment instead of being sunk by a stuck
    reference.  The largest segment (most placed frames) wins.

    The returned R/t map full-frame pixels into the coordinate system of the
    chosen segment's first placed frame.  No joint/global optimization here.
    """
    frame_log = [
        {"frame_index": i, "label_count": len(points),
         "labels": sorted(points), "placed": False}
        for i, points in enumerate(obs)
    ]

    segments: list[dict] = []
    for i, points in enumerate(obs):
        if len(points) < min_points:
            frame_log[i]["skip_reason"] = "label_count_below_minimum"
            continue
        # Pick the segment this frame best continues: most placed frames first,
        # then the most recent tail (nearest reference => most local fit).
        best_key: Optional[tuple] = None
        chosen: Optional[dict] = None
        chosen_tail = -1
        chosen_shared: list[str] = []
        for seg in segments:
            tail = seg["placed"][-1]
            shared = sorted(obs[tail].keys() & obs[i].keys())
            if len(shared) < min_points:
                continue
            key = (len(seg["placed"]), tail)
            if best_key is None or key > best_key:
                best_key, chosen, chosen_tail, chosen_shared = key, seg, tail, shared

        if chosen is not None:
            # Current-frame pixels -> chosen tail pixels.
            src = np.stack([obs[i][label] for label in chosen_shared])
            dst = np.stack([obs[chosen_tail][label] for label in chosen_shared])
            relative_R, relative_t = _rigid_fit(src, dst)
            # Compose with tail -> segment-reference pose.
            prev_R, prev_t = chosen["Rs"][-1], chosen["ts"][-1]
            chosen["placed"].append(i)
            chosen["Rs"].append(prev_R @ relative_R)
            chosen["ts"].append(prev_R @ relative_t + prev_t)
            frame_log[i].update({
                "placed": True,
                "segment_index": chosen["index"],
                "previous_placed_frame_index": chosen_tail,
                "shared_label_count": len(chosen_shared),
                "shared_labels": chosen_shared,
                "relative_R": relative_R.tolist(),
                "relative_T": relative_t.tolist(),
            })
            continue

        # No active segment shares enough labels; seed a new one.
        seg = {"index": len(segments), "placed": [i],
               "Rs": [np.eye(2)], "ts": [np.zeros(2)]}
        segments.append(seg)
        frame_log[i].update({
            "placed": True, "segment_index": seg["index"],
            "previous_placed_frame_index": None, "shared_label_count": None,
            "relative_R": np.eye(2).tolist(), "relative_T": [0.0, 0.0],
        })

    if not segments:
        return [], [], [], frame_log, []

    best = max(segments, key=lambda s: len(s["placed"]))  # first max on ties
    for seg in segments:
        if seg is best:
            continue
        for idx in seg["placed"]:
            frame_log[idx].update({
                "placed": False, "skip_reason": "not_in_largest_segment",
            })

    placed = best["placed"]
    gaps = []
    for left, right in zip(placed, placed[1:]):
        skipped = list(range(left + 1, right))
        if skipped:
            gaps.append({
                "between_frame_indices": [left, right],
                "skipped_count": len(skipped),
                "skipped_frame_indices": skipped,
            })
    return placed, best["Rs"], best["ts"], frame_log, gaps


def _optimize_globally(
    obs: list[dict[str, np.ndarray]], placed: list[int],
    Rs: list[np.ndarray], ts: list[np.ndarray], max_nfev: int = 200,
) -> tuple[list[np.ndarray], list[np.ndarray], int, dict]:
    """Jointly minimize the within-label spread of all placed observations.

    Every non-anchor frame contributes one angle and one 2-D translation to a
    single nonlinear least-squares problem. The anchor pose is held fixed only
    to remove the global rigid-transform gauge freedom; its label positions are
    not used as fixed targets. Thus every observation contributes symmetrically
    to its label cluster center and frames can compromise with one another.
    """
    from scipy.optimize import least_squares

    anchor_pos = max(range(len(placed)), key=lambda j: len(obs[placed[j]]))
    anchor_idx = placed[anchor_pos]  # max() keeps the first maximum
    variable_positions = [i for i in range(len(placed)) if i != anchor_pos]
    variable_slot = {position: slot for slot, position in enumerate(variable_positions)}
    label_observations: dict[str, list[tuple[int, np.ndarray]]] = {}
    for position, frame_idx in enumerate(placed):
        for label, point in obs[frame_idx].items():
            label_observations.setdefault(label, []).append((position, point))
    # A label seen only once has no cluster spread and supplies no constraint.
    label_observations = {
        label: values for label, values in label_observations.items()
        if len(values) >= 2
    }

    x0 = np.empty(3 * len(variable_positions), dtype=np.float64)
    for slot, position in enumerate(variable_positions):
        x0[3 * slot] = math.atan2(Rs[position][1, 0], Rs[position][0, 0])
        x0[3 * slot + 1:3 * slot + 3] = ts[position]

    def poses_from_params(params: np.ndarray):
        current_Rs = [R.copy() for R in Rs]
        current_ts = [t.copy() for t in ts]
        for position, slot in variable_slot.items():
            theta, tx, ty = params[3 * slot:3 * slot + 3]
            c, s = math.cos(theta), math.sin(theta)
            current_Rs[position] = np.array([[c, -s], [s, c]])
            current_ts[position] = np.array([tx, ty])
        return current_Rs, current_ts

    def residuals(params: np.ndarray) -> np.ndarray:
        current_Rs, current_ts = poses_from_params(params)
        parts = []
        for values in label_observations.values():
            transformed = np.stack([
                current_Rs[position] @ point + current_ts[position]
                for position, point in values
            ])
            parts.append((transformed - transformed.mean(axis=0)).ravel())
        return np.concatenate(parts) if parts else np.empty(0, dtype=np.float64)

    initial_residuals = residuals(x0)
    result = least_squares(
        residuals, x0, method="trf", loss="linear", x_scale="jac",
        ftol=1e-10, xtol=1e-10, gtol=1e-10, max_nfev=max_nfev,
    )
    out_Rs, out_ts = poses_from_params(result.x)
    final_residuals = residuals(result.x)
    optimization_log = {
        "method": "joint_nonlinear_least_squares_label_cluster_spread",
        "anchor_frame_index": anchor_idx,
        "optimized_frame_count": len(variable_positions),
        "cluster_label_count": len(label_observations),
        "residual_count": int(len(final_residuals)),
        "max_function_evaluations": max_nfev,
        "function_evaluations": int(result.nfev),
        "jacobian_evaluations": int(result.njev) if result.njev is not None else None,
        "success": bool(result.success),
        "status": int(result.status),
        "message": result.message,
        "initial_sum_squared_error": float(initial_residuals @ initial_residuals),
        "final_sum_squared_error": float(final_residuals @ final_residuals),
        "initial_mean_squared_error": float(np.mean(initial_residuals ** 2)),
        "final_mean_squared_error": float(np.mean(final_residuals ** 2)),
    }
    return out_Rs, out_ts, anchor_idx, optimization_log


def _gaze_canvas_track(
    label_log: list[dict],
    stamp_affines: dict[int, np.ndarray],
) -> list[dict]:
    """All timestamped gaze samples as {stamp_ns, canvas_xy} in canvas pixels;
    entries whose stamp wasn't placed reuse the most recent placed affine."""
    first_affine = next(iter(stamp_affines.values()), None)
    if first_affine is None:
        return []
    affine = None
    track: list[dict] = []
    for entry in label_log:
        ts = entry.get("stamp_ns")
        if ts in stamp_affines:
            affine = stamp_affines[ts]
        g = entry.get("gaze_px")
        if ts is None or g is None:
            continue
        cur = affine if affine is not None else first_affine
        p = cur @ np.array([g[0], g[1], 1.0])
        track.append({"stamp_ns": ts, "canvas_xy": [float(p[0]), float(p[1])]})
    return track


def _detection_canvas_points(
    label_log: list[dict], stamp_affines: dict[int, np.ndarray]
) -> list[dict]:
    """Return logged YOLO centers transformed into stitched-canvas pixels."""
    first_affine = next(iter(stamp_affines.values()), None)
    if first_affine is None:
        return []
    points: list[dict] = []
    for entry in label_log:
        stamp_ns = entry.get("stamp_ns")
        current = stamp_affines.get(stamp_ns)
        if current is None:
            continue
        for detection in entry.get("detected", []):
            center = detection.get("center_px")
            if center is None:
                continue
            canvas_point = current @ np.array([center[0], center[1], 1.0])
            points.append({
                "stamp_ns": stamp_ns,
                "label": detection.get("label"),
                "canvas_xy": [float(canvas_point[0]), float(canvas_point[1])],
            })
    return points


def _save_trajectory_map(
    label_log: list[dict],
    stamp_affines: dict[int, np.ndarray],
    canvas: np.ndarray,
    out_path: Path,
    gaze_track: Optional[list[dict]] = None,
) -> None:
    """Draw all detection centers (colored per label) and the gaze trajectory
    (red polyline, start/end rings, legend) on a copy of the stitched mosaic."""
    canvas = canvas.copy()

    def to_canvas(px, A) -> tuple[int, int]:
        p = A @ np.array([px[0], px[1], 1.0])
        return int(round(p[0])), int(round(p[1]))

    labels = sorted({
        d["label"] for e in label_log for d in e.get("detected", [])
        if d.get("center_px") is not None
    })
    colors = {l: _PALETTE[i % len(_PALETTE)] for i, l in enumerate(labels)}

    first_affine = next(iter(stamp_affines.values()), None)
    if first_affine is None:
        return
    gaze_pts: list[tuple[int, int]] = []
    for entry in label_log:
        ts = entry.get("stamp_ns")
        cur = stamp_affines.get(ts)
        if cur is None:
            continue
        for d in entry.get("detected", []):
            c = d.get("center_px")
            if c is not None:
                cv2.circle(canvas, to_canvas(c, cur), 4, colors[d["label"]], -1, cv2.LINE_AA)
    if gaze_track is None:
        for entry in label_log:
            cur = stamp_affines.get(entry.get("stamp_ns"))
            g = entry.get("gaze_px")
            if cur is not None and g is not None:
                gaze_pts.append(to_canvas(g, cur))
    else:
        gaze_pts = [
            (int(round(rec["canvas_xy"][0])), int(round(rec["canvas_xy"][1])))
            for rec in gaze_track
        ]

    if len(gaze_pts) >= 2:
        cv2.polylines(canvas, [np.array(gaze_pts, dtype=np.int32)], False,
                      (0, 0, 255), 1, cv2.LINE_AA)
    for p in gaze_pts:
        cv2.circle(canvas, p, 2, (0, 0, 255), -1, cv2.LINE_AA)
    # Keep these labels aligned with the zero-based indices stored in
    # stitched_gaze_track.json and stitched_gaze_track_peak.json.
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.35
    font_thickness = 1
    canvas_h, canvas_w = canvas.shape[:2]
    for index, (px, py) in enumerate(gaze_pts):
        text = str(index)
        (text_w, text_h), baseline = cv2.getTextSize(
            text, font, font_scale, font_thickness
        )
        text_x = min(max(px + 4, 0), max(0, canvas_w - text_w - 1))
        text_y = min(max(py - 4, text_h + 1), canvas_h - baseline - 1)
        # A dark outline keeps the index readable on both bright and dark
        # regions of the stitched RGB image.
        cv2.putText(canvas, text, (text_x, text_y), font, font_scale,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, text, (text_x, text_y), font, font_scale,
                    (0, 255, 255), font_thickness, cv2.LINE_AA)
    _START_COLOR, _END_COLOR = (0, 255, 0), (255, 0, 255)
    if gaze_pts:
        cv2.circle(canvas, gaze_pts[0], 8, _START_COLOR, 2, cv2.LINE_AA)
        cv2.circle(canvas, gaze_pts[-1], 8, _END_COLOR, 2, cv2.LINE_AA)

    y = 18
    cv2.circle(canvas, (12, y - 4), 4, (0, 0, 255), -1, cv2.LINE_AA)
    cv2.putText(canvas, "gaze", (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (0, 0, 255), 1, cv2.LINE_AA)
    y += 18
    cv2.putText(canvas, "numbers: gaze index (0-based)", (12, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1,
                cv2.LINE_AA)
    for text, col in (("gaze start", _START_COLOR), ("gaze end", _END_COLOR)):
        y += 18
        cv2.circle(canvas, (12, y - 4), 5, col, 2, cv2.LINE_AA)
        cv2.putText(canvas, text, (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    col, 1, cv2.LINE_AA)
    for l in labels:
        y += 18
        cv2.circle(canvas, (12, y - 4), 4, colors[l], -1, cv2.LINE_AA)
        cv2.putText(canvas, l, (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    colors[l], 1, cv2.LINE_AA)

    cv2.imwrite(str(out_path), canvas)
    print(f"Trajectory map ({len(gaze_pts)} gaze pts, {len(labels)} labels) → {out_path}")


def _central_speed(t: np.ndarray, P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Central-difference speed magnitude; endpoints excluded. Returns
    (t_mid, speed)."""
    dt = t[2:] - t[:-2]
    ok = dt > 0
    vel = (P[2:] - P[:-2])[ok] / dt[ok, None]
    return t[1:-1][ok], np.linalg.norm(vel, axis=1)


def _smooth_series(y: np.ndarray, window: int = 9) -> np.ndarray:
    """Gaussian moving average with reflect padding; window clamped to an odd
    size <= len(y)."""
    n = len(y)
    window = min(window, n if n % 2 else n - 1)
    if window < 3:
        return y.copy()
    if window % 2 == 0:
        window -= 1
    half = window // 2
    idx = np.arange(window) - half
    kernel = np.exp(-0.5 * (idx / (window / 4.0)) ** 2)
    kernel /= kernel.sum()
    return np.convolve(np.pad(y, half, mode="reflect"), kernel, mode="valid")


def _save_gaze_kinematics(
    label_log: list[dict],
    stamp_affines: dict[int, np.ndarray],
    out_path: Path,
    smooth_window: int = 13,
    speed_smooth_window: int = 3,
) -> None:
    """Plot gaze canvas x, y, and speed over time (stacked panels). Writes the
    raw track to out_path plus <stem>_smoothed.png and <stem>_smoothed_speed.png
    with Gaussian-smoothed variants."""
    # One sample per unique stamp (first wins) so dt > 0 for the differences.
    seen: set[int] = set()
    ts_ns: list[int] = []
    pts: list[list[float]] = []
    for rec in _gaze_canvas_track(label_log, stamp_affines):
        if rec["stamp_ns"] in seen:
            continue
        seen.add(rec["stamp_ns"])
        ts_ns.append(rec["stamp_ns"])
        pts.append(rec["canvas_xy"])

    if len(pts) < 3:
        print(f"Gaze kinematics skipped: only {len(pts)} timestamped gaze sample(s).")
        return

    P = np.stack(pts)
    t = (np.asarray(ts_ns, dtype=np.float64) - ts_ns[0]) / 1e9

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Gaze kinematics skipped: matplotlib not available.")
        return

    surface, ink, muted = "#fcfcfb", "#0b0b0b", "#52514e"

    def render(track: np.ndarray, title: str, path: Path,
               smooth_speed: bool = False) -> int:
        t_mid, speed = _central_speed(t, track)
        if smooth_speed:
            speed = _smooth_series(speed, speed_smooth_window)
        panels = [
            (t, track[:, 0], "canvas x (px)", "#2a78d6"),
            (t, track[:, 1], "canvas y (px)", "#1baf7a"),
            (t_mid, speed, "speed (px/s)", "#4a3aa7"),
        ]
        fig, axes = plt.subplots(3, 1, figsize=(10, 7.5), sharex=True)
        fig.patch.set_facecolor(surface)
        for ax, (tx, y, ylabel, color) in zip(axes, panels):
            ax.set_facecolor(surface)
            ax.plot(tx, y, color=color, linewidth=2, marker="o", markersize=5,
                    markeredgecolor=surface, markeredgewidth=0.8)
            ax.set_ylabel(ylabel, color=ink, fontsize=10)
            ax.grid(True, color="#e4e3e0", linewidth=0.8)
            ax.tick_params(colors=muted, labelsize=9)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color("#d0cfcb")
        axes[0].set_title(title, color=ink, fontsize=12, loc="left")
        axes[-1].set_xlabel("time (s)", color=ink, fontsize=10)
        fig.tight_layout()
        fig.savefig(path, dpi=120, facecolor=surface)
        plt.close(fig)
        return len(speed)

    n_v = render(P, "Gaze on stitched canvas over time", out_path)
    print(f"Gaze kinematics ({len(pts)} samples, {n_v} velocity pts) → {out_path}")

    P_smooth = np.column_stack([
        _smooth_series(P[:, 0], smooth_window),
        _smooth_series(P[:, 1], smooth_window),
    ])
    smooth_path = out_path.with_name(out_path.stem + "_smoothed.png")
    render(P_smooth,
           f"Gaze on stitched canvas over time (smoothed track, window={smooth_window})",
           smooth_path)
    print(f"Smoothed gaze kinematics → {smooth_path}")

    smooth_speed_path = out_path.with_name(out_path.stem + "_smoothed_speed.png")
    render(P_smooth,
           f"Gaze on stitched canvas over time "
           f"(smoothed track w={smooth_window} + speed w={speed_smooth_window})",
           smooth_speed_path, smooth_speed=True)
    print(f"Smoothed-speed gaze kinematics → {smooth_speed_path}")


def stitch_recording(
    frames: list[tuple[int, np.ndarray]],
    label_log: list[dict],
    out_path: Path,
    max_canvas_dim: int = 12000,
    save_placements: bool = True,
    crop_ratio: float = 0.65,
) -> Optional[Path]:
    """Chain adjacent valid frames, then tighten them against one anchor."""
    centers_map = _centers_by_stamp(label_log)
    seen: set[int] = set()
    timeline: list[tuple[int, np.ndarray, tuple[int, int]]] = []
    for stamp_ns, image in frames:
        if stamp_ns in seen:
            continue
        seen.add(stamp_ns)
        cropped, crop_xy = _center_crop(image, crop_ratio)
        timeline.append((stamp_ns, cropped, crop_xy))
    obs = [centers_map.get(stamp_ns, {}) for stamp_ns, _, _ in timeline]
    placed, chained_Rs, chained_ts, frame_log, gaps = _chain_poses(obs)
    for rec, (stamp_ns, _, _) in zip(frame_log, timeline):
        rec["stamp_ns"] = stamp_ns
    for gap in gaps:
        left, right = gap["between_frame_indices"]
        gap["between_stamp_ns"] = [timeline[left][0], timeline[right][0]]
        gap["skipped_stamp_ns"] = [
            timeline[i][0] for i in gap["skipped_frame_indices"]
        ]
    if len(placed) < 2:
        print(f"Stitching skipped: only {len(placed)} frame(s) could be chained.")
        return None

    optimized_Rs, optimized_ts, anchor_idx, optimization_log = _optimize_globally(
        obs, placed, chained_Rs, chained_ts
    )

    def compose(Rs: list[np.ndarray], translations: list[np.ndarray]):
        transformed_corners = []
        for frame_idx, R, t in zip(placed, Rs, translations):
            _, image, crop_xy = timeline[frame_idx]
            h, w = image.shape[:2]
            full_corners = np.array([
                crop_xy, (crop_xy[0] + w, crop_xy[1]),
                (crop_xy[0] + w, crop_xy[1] + h),
                (crop_xy[0], crop_xy[1] + h),
            ], dtype=np.float64)
            transformed_corners.append(full_corners @ R.T + t)
        bounds = np.concatenate(transformed_corners)
        min_xy, max_xy = bounds.min(axis=0), bounds.max(axis=0)
        unscaled_size = np.maximum(np.ceil(max_xy - min_xy).astype(int), 1)
        scale = min(1.0, max_canvas_dim / float(max(unscaled_size)))
        canvas_w, canvas_h = np.maximum(np.ceil(unscaled_size * scale).astype(int), 1)
        acc = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
        count = np.zeros((canvas_h, canvas_w), dtype=np.float32)
        placements = []
        for frame_idx, R, t in zip(placed, Rs, translations):
            stamp_ns, image, crop_xy = timeline[frame_idx]
            A_full = np.hstack([
                scale * R, (scale * (t - min_xy))[:, None]
            ])
            A_crop = A_full.copy()
            A_crop[:, 2] += A_full[:, :2] @ np.asarray(crop_xy, dtype=np.float64)
            h, w = image.shape[:2]
            crop_corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64)
            canvas_corners = crop_corners @ A_crop[:, :2].T + A_crop[:, 2]
            x0 = max(int(np.floor(canvas_corners[:, 0].min())), 0)
            y0 = max(int(np.floor(canvas_corners[:, 1].min())), 0)
            x1 = min(int(np.ceil(canvas_corners[:, 0].max())), canvas_w)
            y1 = min(int(np.ceil(canvas_corners[:, 1].max())), canvas_h)
            if x1 <= x0 or y1 <= y0:
                continue
            A_roi = A_crop.copy()
            A_roi[:, 2] -= (x0, y0)
            warped = cv2.warpAffine(image, A_roi, (x1 - x0, y1 - y0))
            mask = cv2.warpAffine(
                np.full((h, w), 255, dtype=np.uint8), A_roi,
                (x1 - x0, y1 - y0), flags=cv2.INTER_NEAREST,
            ) > 0
            acc[y0:y1, x0:x1][mask] += warped[mask]
            count[y0:y1, x0:x1][mask] += 1
            placements.append({
                "frame_index": frame_idx, "stamp_ns": stamp_ns,
                "theta_deg": math.degrees(math.atan2(R[1, 0], R[0, 0])),
                "R": R.tolist(), "T": t.tolist(),
                "affine_full_to_canvas": A_full.tolist(),
            })
        canvas = (acc / np.maximum(count, 1)[..., None]).astype(np.uint8)
        return canvas, placements, scale

    raw_canvas, raw_placements, raw_scale = compose(chained_Rs, chained_ts)
    optimized_canvas, optimized_placements, optimized_scale = compose(
        optimized_Rs, optimized_ts
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), raw_canvas)
    optimized_path = out_path.with_name(out_path.stem + "_optimized.png")
    cv2.imwrite(str(optimized_path), optimized_canvas)

    stamp_affines = {
        rec["stamp_ns"]: np.asarray(rec["affine_full_to_canvas"])
        for rec in optimized_placements
    }
    gaze_by_stamp = {
        entry["stamp_ns"]: entry.get("gaze_px") for entry in label_log
        if entry.get("stamp_ns") is not None and entry.get("gaze_px") is not None
    }
    gaze_track: list[dict] = []
    for segment, (left_idx, right_idx) in enumerate(zip(placed, placed[1:])):
        left_stamp, right_stamp = timeline[left_idx][0], timeline[right_idx][0]
        left_gaze, right_gaze = gaze_by_stamp.get(left_stamp), gaze_by_stamp.get(right_stamp)
        if left_gaze is None or right_gaze is None:
            continue
        left_xy = stamp_affines[left_stamp] @ np.array([*left_gaze, 1.0])
        right_xy = stamp_affines[right_stamp] @ np.array([*right_gaze, 1.0])
        if segment == 0:
            gaze_track.append({"stamp_ns": left_stamp,
                               "canvas_xy": left_xy.tolist(), "interpolated": False})
        skipped_indices = list(range(left_idx + 1, right_idx))
        for offset, skipped_idx in enumerate(skipped_indices, start=1):
            alpha = offset / (len(skipped_indices) + 1)
            xy = (1.0 - alpha) * left_xy + alpha * right_xy
            gaze_track.append({
                "stamp_ns": timeline[skipped_idx][0], "canvas_xy": xy.tolist(),
                "interpolated": True, "between_stamp_ns": [left_stamp, right_stamp],
            })
        gaze_track.append({"stamp_ns": right_stamp,
                           "canvas_xy": right_xy.tolist(), "interpolated": False})
    # A placed frame can occur after a segment whose endpoint gaze is missing.
    unique_track = {rec["stamp_ns"]: rec for rec in gaze_track}
    gaze_track = [unique_track[stamp] for stamp in sorted(unique_track)]
    if gaze_track:
        t0 = gaze_track[0]["stamp_ns"]
        for rec in gaze_track:
            rec["t_sec"] = round((rec["stamp_ns"] - t0) / 1e9, 6)

    trajectory_path = out_path.with_name(out_path.stem + "_trajectory.png")
    _save_trajectory_map(
        label_log, stamp_affines, optimized_canvas, trajectory_path, gaze_track
    )
    if gaze_track:
        track_path = out_path.with_name(out_path.stem + "_gaze_track.json")
        track_path.write_text(json.dumps({
            "coordinate_frame": "optimized stitched canvas pixels",
            "t0_stamp_ns": gaze_track[0]["stamp_ns"], "count": len(gaze_track),
            "points": gaze_track,
            "detection_points": _detection_canvas_points(label_log, stamp_affines),
        }, indent=2))

    segment_indices = sorted({
        rec["segment_index"] for rec in frame_log
        if rec.get("segment_index") is not None
    })
    chosen_segment_index = frame_log[placed[0]]["segment_index"]

    if save_placements:
        out_path.with_name(out_path.stem + "_placements.json").write_text(json.dumps({
            "method": "sequential_rigid_then_joint_global_optimization",
            "minimum_detection_and_shared_label_count": 3,
            "crop_ratio": crop_ratio,
            "segment_count": len(segment_indices),
            "chosen_segment_index": chosen_segment_index,
            "initial_reference_frame_index": placed[0],
            "initial_reference_stamp_ns": timeline[placed[0]][0],
            "optimization_anchor_frame_index": anchor_idx,
            "optimization_anchor_stamp_ns": timeline[anchor_idx][0],
            "raw_canvas_scale": raw_scale, "optimized_canvas_scale": optimized_scale,
            "frames": frame_log, "skipped_gaps": gaps,
            "optimization": optimization_log,
            "raw_placements": raw_placements,
            "optimized_placements": optimized_placements,
        }, indent=2))
    if len(segment_indices) > 1:
        print(
            f"Note: {len(segment_indices)} disjoint segments found "
            f"(too few shared labels to link); kept the largest "
            f"({len(placed)} frames)."
        )
    print(f"Sequential stitch: {len(placed)}/{len(timeline)} frames → {out_path}")
    print(
        f"Joint global optimization: {optimization_log['function_evaluations']} "
        f"function evaluations, MSE "
        f"{optimization_log['initial_mean_squared_error']:.6g} → "
        f"{optimization_log['final_mean_squared_error']:.6g}"
    )
    print(f"Globally optimized mosaic → {optimized_path}")
    return out_path
