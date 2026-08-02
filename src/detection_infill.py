"""Offline in-fill of YOLO detections that flickered out during a recording.

YOLO runs per frame on a small gaze-centred crop, so a static object often drops
out for a frame or two even though it never left that crop. Those dropouts bias
the /gaze_label average and inflate the label-score variance the 'variance'
selector measures.

This runs once the recording is over, right after stitching, and reuses the
rigid poses the stitcher already solved: because every stitched label is a
*static* object, a frame's pose is the scene-to-frame rigid transform, so a
detection observed in frame A can be reprojected into frame B's pixels through
the stitched canvas. For a missing (frame, label) pair the nearest earlier and
the nearest later observation of that label are both reprojected and averaged —
offline we can look forwards as well as backwards.

The one thing in-fill must not do is invent something the wearer could not see.
YOLO only ever sees the CROP_SIZE box around that frame's gaze point, so a
predicted centre landing outside that box is dropped instead of filled: the goal
is to repair unstable *detection*, not to hallucinate what is out of view.

Filled entries carry ``"infilled": true`` and are otherwise indistinguishable
from real ones, so they feed the score average, the variance selector and the
per-frame exports exactly like logged detections.
"""

import argparse
import bisect
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.frame_stitcher import _rigid_fit  # noqa: E402
from src.gaze_rgb_config import CROP_SIZE, RESIZE_SIZE  # noqa: E402
from utils.encode_video import encode_frame_video  # noqa: E402
from src.yolo_overlay import compute_score  # noqa: E402

# A pose fitted from an unplaced frame's own detections is only trusted when the
# fit lands its points this close (canvas px) to their canvas cluster centres;
# a 2-point fit on a short baseline can otherwise be wildly off.
MAX_FIT_RMS_PX = 30.0


def _apply(affine: np.ndarray, point) -> np.ndarray:
    return affine[:, :2] @ np.asarray(point, dtype=np.float64) + affine[:, 2]


def _invert(affine: np.ndarray) -> np.ndarray:
    linear_inv = np.linalg.inv(affine[:, :2])
    return np.hstack([linear_inv, (-linear_inv @ affine[:, 2])[:, None]])


def _angle_of(affine: np.ndarray) -> float:
    return math.atan2(affine[1, 0], affine[0, 0])


def _affine_from(theta: float, scale: float, translation: np.ndarray) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    linear = scale * np.array([[c, -s], [s, c]], dtype=np.float64)
    return np.hstack([linear, np.asarray(translation, dtype=np.float64)[:, None]])


def _load_placements(placements_path: Path) -> tuple[dict[int, np.ndarray], float]:
    """Return (stamp_ns -> full-frame->canvas affine, canvas scale) from the
    globally optimized stitch placements."""
    data = json.loads(Path(placements_path).read_text())
    affines = {
        rec["stamp_ns"]: np.asarray(rec["affine_full_to_canvas"], dtype=np.float64)
        for rec in data.get("optimized_placements", [])
    }
    scale = float(data.get("optimized_canvas_scale") or 1.0)
    return affines, scale


def _canvas_label_map(
    entries: list[dict], affines: dict[int, np.ndarray]
) -> dict[str, np.ndarray]:
    """Mean canvas position per label over the placed frames' observations.

    This is the same cluster centre the stitcher's joint optimization minimizes
    spread around, so it is the scene map an unplaced frame can be fitted to.
    """
    canvas_points: dict[str, list[np.ndarray]] = {}
    for entry in entries:
        affine = affines.get(entry["stamp_ns"])
        if affine is None:
            continue
        for det in entry.get("detected", []):
            if det.get("center_px") is None:
                continue
            canvas_points.setdefault(det["label"], []).append(
                _apply(affine, det["center_px"])
            )
    return {
        label: np.mean(np.stack(points), axis=0)
        for label, points in canvas_points.items()
    }


def _frame_affines(
    entries: list[dict],
    affines: dict[int, np.ndarray],
    canvas_labels: dict[str, np.ndarray],
    canvas_scale: float,
) -> tuple[list[Optional[np.ndarray]], list[str]]:
    """Best available full-frame -> canvas affine for every logged frame.

    Three tiers, best evidence first:
      * ``stitched``     — the frame was placed, use its optimized pose;
      * ``detection_fit``— unplaced (fewer than 3 detections, so the stitcher
        dropped it) but at least 2 of its detections are known scene labels:
        fit its own points to their canvas cluster centres at the known canvas
        scale, and keep the fit only if it is tight;
      * ``interpolated`` — nothing to fit: interpolate angle and translation
        between the nearest bracketing frames that do have a pose, the same
        approximation the stitched gaze track already uses for skipped frames.
    """
    poses: list[Optional[np.ndarray]] = [None] * len(entries)
    sources: list[str] = ["none"] * len(entries)

    for i, entry in enumerate(entries):
        affine = affines.get(entry["stamp_ns"])
        if affine is not None:
            poses[i], sources[i] = affine, "stitched"

    for i, entry in enumerate(entries):
        if poses[i] is not None:
            continue
        pairs = [
            (det["center_px"], canvas_labels[det["label"]])
            for det in entry.get("detected", [])
            if det.get("center_px") is not None and det["label"] in canvas_labels
        ]
        if len(pairs) < 2:
            continue
        src = np.stack([np.asarray(p, dtype=np.float64) for p, _ in pairs]) * canvas_scale
        dst = np.stack([q for _, q in pairs])
        R, t = _rigid_fit(src, dst)
        rms = float(np.sqrt(np.mean(np.sum((src @ R.T + t - dst) ** 2, axis=1))))
        if rms > MAX_FIT_RMS_PX:
            continue
        poses[i] = np.hstack([canvas_scale * R, t[:, None]])
        sources[i] = "detection_fit"

    known = [i for i, pose in enumerate(poses) if pose is not None]
    if len(known) >= 2:
        stamps = [entry["stamp_ns"] for entry in entries]
        for i, pose in enumerate(poses):
            if pose is not None:
                continue
            slot = bisect.bisect_left(known, i)
            if slot == 0 or slot >= len(known):
                continue  # outside the placed span: nothing to interpolate between
            left, right = known[slot - 1], known[slot]
            span = stamps[right] - stamps[left]
            alpha = (stamps[i] - stamps[left]) / span if span else 0.5
            theta_l, theta_r = _angle_of(poses[left]), _angle_of(poses[right])
            delta = math.atan2(math.sin(theta_r - theta_l), math.cos(theta_r - theta_l))
            translation = (
                (1.0 - alpha) * poses[left][:, 2] + alpha * poses[right][:, 2]
            )
            poses[i] = _affine_from(theta_l + alpha * delta, canvas_scale, translation)
            sources[i] = "interpolated"

    return poses, sources


def _crop_box(gaze_px, frame_size: tuple[int, int]) -> tuple[int, int, int]:
    """The YOLO crop for a frame: (x_start, y_start, side), mirroring the live
    crop in GazeOverlay.draw()."""
    width, height = frame_size
    side = min(CROP_SIZE, width, height)
    x_start = max(0, min(int(gaze_px[0]) - side // 2, width - side))
    y_start = max(0, min(int(gaze_px[1]) - side // 2, height - side))
    return x_start, y_start, side


def infill_missing_detections(
    label_log: list[dict],
    placements_path: Path,
    frame_size: tuple[int, int],
    *,
    s_min: float = 0.0,
    dist_threshold: float = 1080.0,
    std_dist: float = 200.0,
    min_label_observations: int = 2,
    report_path: Optional[Path] = None,
) -> dict:
    """Fill flicker gaps in ``label_log`` in place; returns a summary dict.

    A (frame, label) pair is filled when the label is missing from that frame,
    was observed at least ``min_label_observations`` times overall, both the
    frame and its reference frame have a pose, and the reprojected centre lands
    inside the frame's YOLO crop. The score is recomputed from the filled centre
    exactly as the live path does (Gaussian on the gaze distance in resized-crop
    pixels) and, like the live path, entries below ``s_min`` are not logged.

    ``label_log`` is read-only: the filled points come back in the summary's
    ``infilled_frames``, in the same shape as the logged frames, and only get
    combined with the raw records by :func:`merge_infilled`. Keeping the two
    apart on disk is what makes a score traceable to its origin.
    """
    # Own shallow copies: nothing here may reach back into the caller's log.
    # Detections tagged ``infilled`` come from an older build that wrote fills
    # inline; they are dropped so a re-run always derives from raw evidence.
    entries = [
        {**e, "detected": [
            det for det in e.get("detected", []) if not det.get("infilled")
        ]}
        for e in sorted(
            (
                e for e in label_log
                if e.get("stamp_ns") is not None and e.get("gaze_px") is not None
            ),
            key=lambda e: e["stamp_ns"],
        )
    ]
    if len(entries) < 2:
        return {"filled_count": 0, "reason": "fewer than 2 usable frames"}

    affines, canvas_scale = _load_placements(placements_path)
    if not affines:
        return {"filled_count": 0, "reason": "no stitched placements"}
    canvas_labels = _canvas_label_map(entries, affines)
    poses, pose_sources = _frame_affines(entries, affines, canvas_labels, canvas_scale)
    inverse_poses = [None if p is None else _invert(p) for p in poses]

    observations: dict[str, list[int]] = {}
    for i, entry in enumerate(entries):
        for det in entry["detected"]:
            if det.get("center_px") is not None:
                observations.setdefault(det["label"], []).append(i)

    fills: list[dict] = []
    # stamp_ns -> the detections in-fill adds to that frame, kept apart from the
    # logged ones and only merged in by merge_infilled().
    infilled_by_stamp: dict[int, list[dict]] = {}
    # Candidates that were computed but deliberately not written, kept so the
    # frame export can show what the crop rule and s_min actually threw away.
    rejections: list[dict] = []
    skipped = {"no_pose": 0, "no_reference": 0, "outside_crop": 0, "below_s_min": 0}
    per_label: dict[str, dict] = {}

    for label, seen in observations.items():
        stats = per_label.setdefault(
            label, {"observed": len(seen), "filled": 0, "outside_crop": 0}
        )
        if len(seen) < min_label_observations:
            stats["skipped_label"] = "below min_label_observations"
            continue
        centers = {
            i: np.asarray(
                next(
                    det["center_px"] for det in entries[i]["detected"]
                    if det["label"] == label and det.get("center_px") is not None
                ),
                dtype=np.float64,
            )
            for i in seen
        }
        present = set(seen)
        for i, entry in enumerate(entries):
            if i in present:
                continue
            if inverse_poses[i] is None:
                skipped["no_pose"] += 1
                continue
            slot = bisect.bisect_left(seen, i)
            references = []
            if slot > 0:
                references.append(seen[slot - 1])
            if slot < len(seen):
                references.append(seen[slot])
            predictions = [
                _apply(inverse_poses[i], _apply(poses[j], centers[j]))
                for j in references if poses[j] is not None
            ]
            if not predictions:
                skipped["no_reference"] += 1
                continue
            point = np.mean(np.stack(predictions), axis=0)

            gaze_px = entry["gaze_px"]
            x_start, y_start, side = _crop_box(gaze_px, frame_size)
            candidate = {
                "stamp_ns": entry["stamp_ns"],
                "label": label,
                "center_px": [int(round(point[0])), int(round(point[1]))],
                "pose_source": pose_sources[i],
                "from_stamps_ns": [entries[j]["stamp_ns"] for j in references],
                "gaze_px": [int(gaze_px[0]), int(gaze_px[1])],
                "crop": [x_start, y_start, side],
            }
            if not (x_start <= point[0] <= x_start + side - 1
                    and y_start <= point[1] <= y_start + side - 1):
                # Outside the crop YOLO actually looked at — not a missed
                # detection, just something the wearer was not looking at.
                skipped["outside_crop"] += 1
                stats["outside_crop"] += 1
                rejections.append({**candidate, "reason": "outside_crop"})
                continue

            # Live scoring works in resized-crop pixels, so scale the full-frame
            # gaze distance by the same RESIZE_SIZE / crop_side factor.
            distance = float(np.hypot(point[0] - gaze_px[0], point[1] - gaze_px[1]))
            score = compute_score(
                distance * (RESIZE_SIZE / side), dist_threshold, std_dist
            )
            if score < s_min:
                skipped["below_s_min"] += 1
                rejections.append(
                    {**candidate, "reason": "below_s_min", "score": round(score, 4)}
                )
                continue

            # Same three keys a live-logged detection carries, so the two
            # sources merge without any format juggling.
            infilled_by_stamp.setdefault(entry["stamp_ns"], []).append({
                "label": label,
                "score": round(score, 4),
                "center_px": [int(round(point[0])), int(round(point[1]))],
            })
            stats["filled"] += 1
            fills.append({
                **candidate,
                "score": round(score, 4),
                "reference_count": len(predictions),
            })

    gaze_by_stamp = {entry["stamp_ns"]: entry["gaze_px"] for entry in entries}
    infilled_frames = [
        {
            "stamp_ns": stamp_ns,
            "gaze_px": gaze_by_stamp[stamp_ns],
            "detected": sorted(detected, key=lambda det: -det["score"]),
        }
        for stamp_ns, detected in sorted(infilled_by_stamp.items())
    ]

    pose_source_counts: dict[str, int] = {}
    for source in pose_sources:
        pose_source_counts[source] = pose_source_counts.get(source, 0) + 1
    # How many *filled points* rest on each pose tier: fills made through a
    # fitted or interpolated pose are the ones worth double-checking when a
    # recording's /gaze_label result looks off.
    fill_pose_source_counts: dict[str, int] = {}
    for fill in fills:
        source = fill["pose_source"]
        fill_pose_source_counts[source] = fill_pose_source_counts.get(source, 0) + 1

    summary = {
        "method": "stitched_pose_reprojection",
        "frame_count": len(entries),
        "filled_count": len(fills),
        "min_label_observations": min_label_observations,
        "crop_size": CROP_SIZE,
        "resize_size": RESIZE_SIZE,
        "frame_size": [frame_size[0], frame_size[1]],
        "score_params": {
            "dist_threshold": dist_threshold, "std_dist": std_dist, "s_min": s_min,
        },
        "pose_source_counts": pose_source_counts,
        "fill_pose_source_counts": fill_pose_source_counts,
        "skipped": skipped,
        "per_label": per_label,
        "infilled_frame_count": len(infilled_frames),
        "infilled_frames": infilled_frames,
        "fills": fills,
        "rejections": rejections,
    }
    print(
        f"Detection in-fill: added {len(fills)} detection(s) across "
        f"{len(entries)} frame(s) "
        f"(from poses: {fill_pose_source_counts.get('stitched', 0)} stitched, "
        f"{fill_pose_source_counts.get('detection_fit', 0)} fitted, "
        f"{fill_pose_source_counts.get('interpolated', 0)} interpolated); "
        f"skipped {skipped['outside_crop']} outside the YOLO crop, "
        f"{skipped['below_s_min']} below s_min, {skipped['no_pose']} without a pose "
        f"({pose_source_counts.get('none', 0)}/{len(entries)} frames have no pose)."
    )
    if report_path is not None:
        Path(report_path).write_text(json.dumps(summary, indent=2))
        print(f"Detection in-fill report -> {report_path}")
    return summary


def summary_brief(summary: dict) -> dict:
    """The summary without its long per-point lists, for embedding in
    gaze_labels.json (detection_infill.json keeps the full version)."""
    dropped = ("fills", "rejections", "infilled_frames")
    return {k: v for k, v in summary.items() if k not in dropped}


def merge_infilled(
    frames: list[dict], infilled_frames: Optional[list[dict]]
) -> list[dict]:
    """Frames with the in-filled detections folded in, by timestamp.

    gaze_labels.json keeps the two apart — ``frames`` is exactly what the live
    run logged, ``infilled_frames`` is what in-fill added afterwards — so every
    score stays traceable to its origin. Analysis wants them as one timeline,
    which is what this returns: for each stamp the logged detections plus that
    stamp's in-filled ones, re-sorted by score. Neither input is modified, and
    the merged copies carry ``"infilled": True`` on the added entries so a
    consumer can still tell them apart in memory.
    """
    if not infilled_frames:
        return list(frames)
    added_by_stamp: dict[int, list[dict]] = {}
    for entry in infilled_frames:
        stamp_ns = entry.get("stamp_ns")
        if stamp_ns is not None:
            added_by_stamp.setdefault(stamp_ns, []).extend(entry.get("detected", []))

    merged: list[dict] = []
    for entry in frames:
        added = added_by_stamp.pop(entry.get("stamp_ns"), None)
        if not added:
            merged.append(entry)
            continue
        detected = list(entry.get("detected", []))
        detected += [{**det, "infilled": True} for det in added]
        detected.sort(key=lambda det: -det.get("score", 0.0))
        merged.append({**entry, "detected": detected})

    # An in-filled stamp with no logged frame cannot happen (fills are derived
    # from the log), but dropping one silently would corrupt the timeline.
    if added_by_stamp:
        by_stamp = {entry["stamp_ns"]: entry for entry in infilled_frames}
        for stamp_ns, added in added_by_stamp.items():
            merged.append({
                **by_stamp[stamp_ns],
                "detected": [{**det, "infilled": True} for det in added],
            })
        merged.sort(key=lambda entry: entry.get("stamp_ns", 0))
    return merged


def frame_image_path(
    recording_dir: Path,
    stamp_ns: int,
    *,
    infilled_subdir: str = "infilled_frames",
    frames_subdir: str = "frames",
) -> Optional[Path]:
    """The image to show for a frame, preferring what in-fill produced.

    ``infilled_frames/`` holds the whole recording after in-fill — redrawn
    ``<stamp>_filled.png`` where detections were added, a plain copy otherwise —
    so anything visualising a frame should read from there and only fall back to
    the raw ``frames/`` when in-fill has not been run. Returns ``None`` when no
    image exists for the stamp.
    """
    infilled_dir = Path(recording_dir) / infilled_subdir
    candidates = (
        infilled_dir / f"{stamp_ns}_filled.png",
        infilled_dir / f"{stamp_ns}.png",
        Path(recording_dir) / frames_subdir / f"{stamp_ns}.png",
    )
    return next((path for path in candidates if path.exists()), None)


def load_label_frames(
    recording_dir: Path, *, use_infilled: bool = True
) -> list[dict]:
    """Timestamp-ordered frames from a recording's gaze_labels.json.

    With ``use_infilled`` (the default) the ``infilled_frames`` section is
    merged in, so analysis sees every detection that survived in-fill; pass
    ``False`` to look at the raw logged detections only.
    """
    data = json.loads((Path(recording_dir) / "gaze_labels.json").read_text())
    frames = sorted(data.get("frames", []), key=lambda f: f["stamp_ns"])
    if not use_infilled:
        return frames
    return merge_infilled(frames, data.get("infilled_frames"))


_FILLED_COLOR = (255, 0, 255)      # magenta — added by in-fill
_REJECTED_COLOR = (0, 165, 255)    # orange — computed but refused
_CROP_COLOR = (0, 255, 255)        # yellow — the box YOLO actually saw
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _put_label(image: np.ndarray, text: str, origin: tuple[int, int],
               color: tuple[int, int, int], scale: float = 0.45) -> None:
    """Text with a dark outline so it stays readable over any frame content."""
    cv2.putText(image, text, origin, _FONT, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, text, origin, _FONT, scale, color, 1, cv2.LINE_AA)


def export_infilled_frames(
    recording_dir: Path,
    summary: dict,
    *,
    frames_subdir: str = "frames",
    out_subdir: str = "infilled_frames",
) -> Optional[Path]:
    """Write a reviewable copy of the whole recording next to ``frames/``.

    Every frame image is carried over into ``<recording_dir>/<out_subdir>/``:
    untouched frames are copied byte-for-byte under their original name, and a
    frame that gained detections is redrawn and saved as ``<stamp>_filled.png``
    — so the file name alone says what in-fill did to each frame.

    On a ``_filled`` frame the drawing shows the whole decision: the yellow box
    is the gaze crop YOLO actually ran on, magenta is what was filled in, and
    orange crosses are candidates that were computed and then refused (mostly
    for landing outside that box). The green circles are the live YOLO
    detections already baked into the recorded frame.
    """
    frames_dir = recording_dir / frames_subdir
    if not frames_dir.is_dir():
        print(f"Frame export skipped: {frames_dir} does not exist.")
        return None
    if not summary.get("fills"):
        # Nothing was filled, so the folder would be a byte-for-byte duplicate
        # of frames/ — not worth the disk.
        print("Frame export skipped: no detection was in-filled.")
        return None

    fills_by_stamp: dict[int, list[dict]] = {}
    for fill in summary.get("fills", []):
        fills_by_stamp.setdefault(fill["stamp_ns"], []).append(fill)
    rejections_by_stamp: dict[int, list[dict]] = {}
    for rejection in summary.get("rejections", []):
        rejections_by_stamp.setdefault(rejection["stamp_ns"], []).append(rejection)

    out_dir = recording_dir / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("*.png"):
        stale.unlink()

    manifest: list[dict] = []
    copied = filled_count = 0
    for image_path in sorted(frames_dir.glob("*.png")):
        try:
            stamp = int(image_path.stem)
        except ValueError:
            continue
        fills = fills_by_stamp.get(stamp, [])
        if not fills:
            shutil.copy2(image_path, out_dir / image_path.name)
            copied += 1
            manifest.append({
                "stamp_ns": stamp, "file": image_path.name, "filled": [],
                "rejected": rejections_by_stamp.get(stamp, []),
            })
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Frame export: unreadable {image_path.name}, skipped.")
            continue
        height, width = image.shape[:2]

        x_start, y_start, side = fills[0]["crop"]
        cv2.rectangle(image, (x_start, y_start),
                      (x_start + side - 1, y_start + side - 1), _CROP_COLOR, 1)
        _put_label(image, "YOLO crop", (x_start + 3, max(y_start - 5, 11)),
                   _CROP_COLOR, 0.4)

        # Markers carry only an index; the legend at the bottom spells each one
        # out, so several fills landing on top of each other stay readable.
        for rank, fill in enumerate(fills, start=1):
            px, py = fill["center_px"]
            cv2.circle(image, (px, py), 13, _FILLED_COLOR, 2, cv2.LINE_AA)
            cv2.circle(image, (px, py), 3, _FILLED_COLOR, -1, cv2.LINE_AA)
            cv2.line(image, (px, py), tuple(fill["gaze_px"]), _FILLED_COLOR,
                     1, cv2.LINE_AA)
            _put_label(image, f"F{rank}", (px + 15, py - 8), _FILLED_COLOR)

        # Refusals go on top: they are the decisions worth double-checking, and
        # a rejected point often sits right next to the fills it was crowded by.
        rejected = rejections_by_stamp.get(stamp, [])
        for rank, rejection in enumerate(rejected, start=1):
            px, py = rejection["center_px"]
            if not (0 <= px < width and 0 <= py < height):
                continue  # off-frame entirely; the legend still lists it
            cv2.drawMarker(image, (px, py), _REJECTED_COLOR,
                           cv2.MARKER_TILTED_CROSS, 14, 2, cv2.LINE_AA)
            _put_label(image, f"R{rank}", (px + 9, py - 7), _REJECTED_COLOR, 0.4)

        lines = [
            (f"F{rank}  {fill['label']}  score={fill['score']:.3f}  "
             f"pose={fill['pose_source']}  "
             f"from {len(fill['from_stamps_ns'])} neighbour frame(s)", _FILLED_COLOR)
            for rank, fill in enumerate(fills, start=1)
        ] + [
            (f"R{rank}  {rejection['label']}  NOT filled: {rejection['reason']}",
             _REJECTED_COLOR)
            for rank, rejection in enumerate(rejected, start=1)
        ]
        y = height - 12 - 18 * len(lines)
        _put_label(image, f"in-filled {len(fills)} detection(s), "
                          f"refused {len(rejected)}", (10, y - 20),
                   _FILLED_COLOR, 0.5)
        for text, color in lines:
            _put_label(image, text, (10, y), color, 0.42)
            y += 18

        out_name = f"{stamp}_filled.png"
        cv2.imwrite(str(out_dir / out_name), image)
        filled_count += 1
        manifest.append({
            "stamp_ns": stamp, "file": out_name, "filled": fills,
            "rejected": rejections_by_stamp.get(stamp, []),
        })

    (out_dir / "infill_frames.json").write_text(json.dumps({
        "source_frames": str(frames_dir),
        "frame_count": len(manifest),
        "filled_frames": filled_count,
        "copied_frames": copied,
        "legend": {
            "magenta circle": "detection added by in-fill (line points at the gaze)",
            "orange cross": "candidate computed then refused (see reason)",
            "yellow box": "the gaze crop YOLO ran on; nothing outside it is filled",
            "green circle": "live YOLO detection, already drawn in the recorded frame",
        },
        "frames": manifest,
    }, indent=2))
    print(f"Frame export: {filled_count} redrawn as *_filled.png, {copied} copied "
          f"unchanged -> {out_dir}")
    return out_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("recording_dir", type=Path,
                        help="recording folder holding gaze_labels.json, frames/ "
                             "and stitched_placements.json (e.g. recordings/test02/26)")
    parser.add_argument("--s-min", type=float, default=0.0, dest="s_min",
                        help="drop filled entries scoring below this (default: 0.0)")
    parser.add_argument("--dist-threshold", type=float, default=1080.0,
                        help="score cut-off distance in resized-crop px (default: 1080)")
    parser.add_argument("--std-dist", type=float, default=200.0, dest="std_dist",
                        help="Gaussian std in resized-crop px (default: 200)")
    parser.add_argument("--min-label-observations", type=int, default=2,
                        help="only in-fill labels observed at least this many "
                             "times in the recording (default: 2)")
    parser.add_argument("--write", action="store_true",
                        help="rewrite gaze_labels.json with the filled entries "
                             "(default: report only)")
    parser.add_argument("--export-frames", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="write a reviewable copy of every frame into "
                             "<recording>/<--out-subdir>/: untouched frames "
                             "copied as-is, in-filled frames redrawn as "
                             "<stamp>_filled.png (default: on)")
    parser.add_argument("--out-subdir", type=str, default="infilled_frames",
                        help="sub-folder of the recording for the exported "
                             "frames (default: infilled_frames)")
    parser.add_argument("--video", action=argparse.BooleanOptionalAction, default=True,
                        help="encode the exported frames into "
                             "<recording>/gaze_overlay_infilled.mp4 (default: on)")
    return parser.parse_args()


def process_recording(recording_dir: Path, args: argparse.Namespace) -> Optional[dict]:
    """Run in-fill on one recording folder and write its outputs."""
    labels_path = recording_dir / "gaze_labels.json"
    payload = json.loads(labels_path.read_text())
    label_log = payload.get("frames", [])

    sample = next(iter(sorted((recording_dir / "frames").glob("*.png"))), None)
    if sample is None:
        print(f"{recording_dir}: no frames/*.png to read the frame size from, skipped.")
        return None
    height, width = cv2.imread(str(sample)).shape[:2]

    summary = infill_missing_detections(
        label_log,
        recording_dir / "stitched_placements.json",
        (width, height),
        s_min=args.s_min,
        dist_threshold=args.dist_threshold,
        std_dist=args.std_dist,
        min_label_observations=args.min_label_observations,
        report_path=recording_dir / "detection_infill.json",
    )
    if args.write:
        # frames stays exactly as the live run logged it; in-fill only ever adds
        # the separate infilled_frames section below it.
        payload["frames"] = [
            {**entry, "detected": [
                det for det in entry.get("detected", []) if not det.get("infilled")
            ]}
            for entry in label_log
        ]
        payload["infilled_frames"] = summary.get("infilled_frames", [])
        payload["detection_infill"] = summary_brief(summary)
        labels_path.write_text(json.dumps(payload, indent=2))
        print(f"Rewrote {labels_path}: {summary['filled_count']} detection(s) in "
              f"infilled_frames across {summary.get('infilled_frame_count', 0)} "
              f"frame(s); frames left untouched.")
    if args.export_frames:
        out_dir = export_infilled_frames(
            recording_dir, summary, out_subdir=args.out_subdir
        )
        if out_dir is not None and args.video:
            encode_frame_video(recording_dir, out_dir, "gaze_overlay_infilled.mp4")
    return summary


def find_recordings(root: Path) -> list[Path]:
    """``root`` itself when it holds gaze_labels.json, else every sub-folder
    that does — so an experiment path like recordings/test02 batches its takes."""
    if (root / "gaze_labels.json").exists():
        return [root]
    return sorted(
        child for child in root.iterdir()
        if child.is_dir() and (child / "gaze_labels.json").exists()
    )


def main() -> None:
    args = parse_args()
    recordings = find_recordings(args.recording_dir)
    if not recordings:
        raise SystemExit(
            f"No gaze_labels.json found in {args.recording_dir} or its sub-folders."
        )
    for i, recording_dir in enumerate(recordings):
        if len(recordings) > 1:
            print(f"\n===== [{i + 1}/{len(recordings)}] {recording_dir} =====")
        process_recording(recording_dir, args)


if __name__ == "__main__":
    main()
