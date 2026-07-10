"""Label-anchored frame stitching for recorded gaze sessions.

Frames are aligned by their logged YOLO detection centers (same label = same
static object), not feature matching. Per-frame 2D rigid poses are solved
jointly via alternating least squares; overlapping pixels are averaged.
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


def _pick_ref_idx(counts: list[int]) -> int:
    """First frame with >= 3 labels whose successor sees the same count;
    fallback: the frame with the most labels."""
    for i in range(len(counts) - 1):
        if counts[i] >= 3 and counts[i + 1] == counts[i]:
            return i
    return int(np.argmax(counts))


def _solve_poses(
    obs: list[dict[str, np.ndarray]],
    ref_idx: int,
    iters: int = 100,
    tol: float = 1e-4,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Jointly solve one rigid pose per frame by alternating least squares
    (world point = mean of transformed observations, then each pose refit).
    The reference frame ends up with the identity pose."""
    n = len(obs)
    Rs = [np.eye(2) for _ in range(n)]
    ts = [np.zeros(2) for _ in range(n)]

    # Init translations by chaining median deltas of shared labels.
    for i in range(1, n):
        shared = obs[i - 1].keys() & obs[i].keys()
        if shared:
            deltas = np.stack([
                (Rs[i - 1] @ obs[i - 1][l] + ts[i - 1]) - obs[i][l] for l in shared
            ])
            ts[i] = np.median(deltas, axis=0)
        else:
            ts[i] = ts[i - 1].copy()

    prev_world: dict[str, np.ndarray] = {}
    for _ in range(iters):
        sums: dict[str, np.ndarray] = {}
        cnts: dict[str, int] = {}
        for i in range(n):
            for l, p in obs[i].items():
                q = Rs[i] @ p + ts[i]
                sums[l] = sums.get(l, 0.0) + q
                cnts[l] = cnts.get(l, 0) + 1
        world = {l: sums[l] / cnts[l] for l in sums}

        for i in range(n):
            labels = list(obs[i].keys())
            src = np.stack([obs[i][l] for l in labels])
            dst = np.stack([world[l] for l in labels])
            if len(labels) == 1:
                ts[i] = dst[0] - Rs[i] @ src[0]
            else:
                Rs[i], ts[i] = _rigid_fit(src, dst)

        if prev_world and all(
            np.linalg.norm(world[l] - prev_world[l]) < tol for l in world
        ):
            break
        prev_world = world

    # Gauge fix: express every pose relative to the reference frame.
    R0T = Rs[ref_idx].T
    t0 = ts[ref_idx]
    for i in range(n):
        Rs[i] = R0T @ Rs[i]
        ts[i] = R0T @ (ts[i] - t0)
    return Rs, ts


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


def _save_trajectory_map(
    label_log: list[dict],
    stamp_affines: dict[int, np.ndarray],
    canvas: np.ndarray,
    out_path: Path,
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
    affine = None
    gaze_pts: list[tuple[int, int]] = []
    for entry in label_log:
        ts = entry.get("stamp_ns")
        if ts in stamp_affines:
            affine = stamp_affines[ts]
        cur = affine if affine is not None else first_affine
        for d in entry.get("detected", []):
            c = d.get("center_px")
            if c is not None:
                cv2.circle(canvas, to_canvas(c, cur), 4, colors[d["label"]], -1, cv2.LINE_AA)
        g = entry.get("gaze_px")
        if g is not None:
            gaze_pts.append(to_canvas(g, cur))

    if len(gaze_pts) >= 2:
        cv2.polylines(canvas, [np.array(gaze_pts, dtype=np.int32)], False,
                      (0, 0, 255), 1, cv2.LINE_AA)
    for p in gaze_pts:
        cv2.circle(canvas, p, 2, (0, 0, 255), -1, cv2.LINE_AA)
    _START_COLOR, _END_COLOR = (0, 255, 0), (255, 0, 255)
    if gaze_pts:
        cv2.circle(canvas, gaze_pts[0], 8, _START_COLOR, 2, cv2.LINE_AA)
        cv2.circle(canvas, gaze_pts[-1], 8, _END_COLOR, 2, cv2.LINE_AA)

    y = 18
    cv2.circle(canvas, (12, y - 4), 4, (0, 0, 255), -1, cv2.LINE_AA)
    cv2.putText(canvas, "gaze", (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (0, 0, 255), 1, cv2.LINE_AA)
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
    """Stitch recorded frames into one mosaic anchored on YOLO detection centers.

    out_path is only the naming base (<stem>_trajectory.png, _gaze_track.json,
    _placements.json, ...); the bare mosaic itself is not written — the
    trajectory map shows it under its overlays. Frames are center-cropped to
    crop_ratio before pasting to trim the fisheye vignette.

    Returns out_path on success, None if fewer than two frames could be placed.
    """
    centers_map = _centers_by_stamp(label_log)

    # First frame per unique stamp with anchor detections, center-cropped.
    seen: set[int] = set()
    keyed: list[tuple[int, np.ndarray]] = []
    crop_xy = (0, 0)
    for ts, img in frames:
        if ts in seen or ts not in centers_map:
            continue
        seen.add(ts)
        cropped, crop_xy = _center_crop(img, crop_ratio)
        keyed.append((ts, cropped))

    if len(keyed) < 2:
        print(f"Stitching skipped: only {len(keyed)} frame(s) with anchor detections.")
        return None

    obs = [centers_map[ts] for ts, _ in keyed]
    ref_idx = _pick_ref_idx([len(o) for o in obs])
    Rs, ts_ = _solve_poses(obs, ref_idx)
    print(f"Poses solved for {len(keyed)} frames (ref frame index {ref_idx}).")

    # Canvas bounds from the transformed corners of every cropped frame.
    ch, cw = keyed[0][1].shape[:2]
    crop_off = np.asarray(crop_xy, dtype=np.float64)
    corners_q = np.array([[0, 0], [cw, 0], [cw, ch], [0, ch]], dtype=np.float64)
    all_pts = np.concatenate([
        (corners_q + crop_off) @ R.T + t for R, t in zip(Rs, ts_)
    ])
    min_xy = all_pts.min(axis=0)
    max_xy = all_pts.max(axis=0)
    canvas_w = int(np.ceil(max_xy[0] - min_xy[0]))
    canvas_h = int(np.ceil(max_xy[1] - min_xy[1]))

    scale = 1.0
    if max(canvas_w, canvas_h) > max_canvas_dim:
        scale = max_canvas_dim / max(canvas_w, canvas_h)
        canvas_w = int(np.ceil(canvas_w * scale))
        canvas_h = int(np.ceil(canvas_h * scale))
        print(f"Stitch canvas capped: scaling by {scale:.3f}")

    # Average compositing: float accumulator + coverage count, per-frame ROI.
    acc = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
    cnt = np.zeros((canvas_h, canvas_w), dtype=np.float32)
    placement_log: list[dict] = []
    for (ts, img), R, t in zip(keyed, Rs, ts_):
        A_full = np.hstack([scale * R, (scale * (t - min_xy))[:, None]])  # full-frame → canvas
        A_crop = A_full.copy()  # cropped → canvas
        A_crop[:, 2] += A_full[:, :2] @ crop_off

        pts = (corners_q @ A_crop[:, :2].T) + A_crop[:, 2]
        x0 = max(int(np.floor(pts[:, 0].min())), 0)
        y0 = max(int(np.floor(pts[:, 1].min())), 0)
        x1 = min(int(np.ceil(pts[:, 0].max())), canvas_w)
        y1 = min(int(np.ceil(pts[:, 1].max())), canvas_h)
        if x1 <= x0 or y1 <= y0:
            continue
        A_roi = A_crop.copy()
        A_roi[:, 2] -= (x0, y0)

        warped = cv2.warpAffine(img, A_roi, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR)
        mask = cv2.warpAffine(
            np.full(img.shape[:2], 255, dtype=np.uint8), A_roi,
            (x1 - x0, y1 - y0), flags=cv2.INTER_NEAREST,
        ) > 0
        acc[y0:y1, x0:x1][mask] += warped[mask]
        cnt[y0:y1, x0:x1][mask] += 1.0

        placement_log.append({
            "stamp_ns": ts,
            "theta_deg": math.degrees(math.atan2(R[1, 0], R[0, 0])),
            "affine_full_to_canvas": A_full.tolist(),
        })

    canvas = (acc / np.maximum(cnt, 1.0)[..., None]).astype(np.uint8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Stitched {len(placement_log)} frames (avg-blended, {canvas_w}x{canvas_h})")

    # The mosaic is only saved with detection centers + gaze trajectory on top.
    stamp_affines = {
        p["stamp_ns"]: np.asarray(p["affine_full_to_canvas"]) for p in placement_log
    }
    _save_trajectory_map(
        label_log, stamp_affines, canvas,
        out_path.with_name(out_path.stem + "_trajectory.png"),
    )
    _save_gaze_kinematics(
        label_log, stamp_affines,
        out_path.with_name(out_path.stem + "_gaze_kinematics.png"),
    )

    # Standalone JSON of the gaze track in canvas pixel coords.
    gaze_track = _gaze_canvas_track(label_log, stamp_affines)
    if gaze_track:
        t0 = gaze_track[0]["stamp_ns"]
        for rec in gaze_track:
            rec["t_sec"] = round((rec["stamp_ns"] - t0) / 1e9, 6)
        track_path = out_path.with_name(out_path.stem + "_gaze_track.json")
        track_path.write_text(json.dumps({
            "coordinate_frame": "stitched canvas pixels (matches stitched_trajectory.png)",
            "t0_stamp_ns": t0,
            "count": len(gaze_track),
            "points": gaze_track,
        }, indent=2))
        print(f"Gaze track ({len(gaze_track)} pts) → {track_path}")

    if save_placements:
        placements_path = out_path.with_name(out_path.stem + "_placements.json")
        placements_path.write_text(json.dumps({
            "scale": scale,
            "frame_size": [cw, ch],
            "crop_ratio": crop_ratio,
            "crop_offset_xy": [crop_xy[0], crop_xy[1]],
            "ref_stamp_ns": keyed[ref_idx][0],
            # Per frame: canvas_px = affine_full_to_canvas @ [x_fullframe, y_fullframe, 1]
            "placements": placement_log,
        }, indent=2))

    return out_path
