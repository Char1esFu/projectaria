import json
import math
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from src.frame_stitcher import stitch_recording
from src.gaze_track_peak import find_gaze_center_stamp, visualize_analyzed_peak


def normalize_entries(entries: list[dict]) -> list[dict]:
    """Normalize scores so they sum to 1 over labels."""
    total = sum(it["score"] for it in entries)
    if total <= 0:
        return entries
    return [
        {"label": it["label"], "score": round(it["score"] / total, 4)}
        for it in entries
    ]


def compute_weighted_entries(
    label_log: list[dict], center_stamp_ns: Optional[int]
) -> list[dict]:
    """Peak-center-weighted average of per-frame detection scores."""
    frames = [f for f in label_log if f.get("stamp_ns") is not None]
    if not frames:
        return []
    if center_stamp_ns is None:
        weights = [1.0] * len(frames)
    else:
        weights = [
            math.exp(-1e-8 * abs(f["stamp_ns"] - center_stamp_ns))
            for f in frames
        ]
    total_w = sum(weights)
    if total_w <= 0:
        return []
    scores: dict[str, float] = {}
    for frame, weight in zip(frames, weights):
        w = weight / total_w
        for item in frame.get("detected", []):
            lbl = item["label"]
            scores[lbl] = scores.get(lbl, 0.0) + w * item["score"]
    weighted = [{"label": lbl, "score": round(s, 4)} for lbl, s in scores.items()]
    weighted.sort(key=lambda x: -x["score"])
    return weighted


def validate_peak_center(
    track_path: Path,
    center_stamp_ns: Optional[int],
    recording_stamps_ns: list[int],
    max_yolo_distance_px: float,
) -> tuple[Optional[int], Optional[str]]:
    """Reject peaks near recording edges or far from every canvas YOLO center."""
    if center_stamp_ns is None:
        return None, "peak analysis did not choose a point"

    stamps = [stamp for stamp in recording_stamps_ns if stamp is not None]
    if len(stamps) < 2:
        return None, "recording has fewer than two valid timestamps"
    start_stamp_ns = min(stamps)
    end_stamp_ns = max(stamps)
    duration_ns = end_stamp_ns - start_stamp_ns
    if duration_ns <= 0:
        return None, "recording duration is not positive"

    edge_threshold_ns = duration_ns / 8.0
    start_gap_ns = center_stamp_ns - start_stamp_ns
    end_gap_ns = end_stamp_ns - center_stamp_ns
    if start_gap_ns < edge_threshold_ns or end_gap_ns < edge_threshold_ns:
        return None, (
            "peak is too close to a recording edge "
            f"(start gap={start_gap_ns / 1e9:.3f}s, "
            f"end gap={end_gap_ns / 1e9:.3f}s, "
            f"minimum={edge_threshold_ns / 1e9:.3f}s)"
        )

    track = json.loads(track_path.read_text())
    peak_record = next(
        (point for point in track.get("points", [])
         if point.get("stamp_ns") == center_stamp_ns),
        None,
    )
    if peak_record is None or peak_record.get("canvas_xy") is None:
        return None, "peak canvas coordinate is unavailable"

    detection_points = [
        detection["canvas_xy"]
        for detection in track.get("detection_points", [])
        if detection.get("canvas_xy") is not None
    ]
    if not detection_points:
        return None, "no YOLO centers are available on the stitched canvas"

    peak_xy = np.asarray(peak_record["canvas_xy"], dtype=np.float64)
    yolo_xy = np.asarray(detection_points, dtype=np.float64)
    nearest_distance_px = float(np.linalg.norm(yolo_xy - peak_xy, axis=1).min())
    if nearest_distance_px > max_yolo_distance_px:
        return None, (
            f"nearest canvas YOLO center is {nearest_distance_px:.1f}px away "
            f"(maximum={max_yolo_distance_px:.1f}px)"
        )

    print(
        "Gaze peak validation passed: "
        f"nearest YOLO center={nearest_distance_px:.1f}px, "
        f"recording-edge minimum={edge_threshold_ns / 1e9:.3f}s."
    )
    return center_stamp_ns, None


class GazeRecord:
    """Owns all recording / gaze-label state: frame buffers, the per-frame
    label log, the on-screen score display, and the final /gaze_label result.

    No ROS inside: incoming events are delivered through the public methods
    (note_manip_stamp, on_recording_start, on_label_start, stop, ...) and the
    outgoing /gaze_label publish goes through the plain callable injected via
    set_label_publisher."""

    def __init__(
        self,
        participant: str,
        s_min: float,
        label_tail_duration: float,
        gaze_peak_window: int,
        gaze_peak_radius: float,
        gaze_peak_yolo_max_distance: float = 200.0,
    ) -> None:
        self._participant = participant
        self._s_min = s_min
        self._label_tail_duration = label_tail_duration
        self._gaze_peak_window = gaze_peak_window
        self._gaze_peak_radius = gaze_peak_radius
        if gaze_peak_yolo_max_distance <= 0:
            raise ValueError("gaze_peak_yolo_max_distance must be > 0")
        self._gaze_peak_yolo_max_distance = gaze_peak_yolo_max_distance

        # Latest camera_info stamp from the manipulation workstation (ns).
        self.latest_manip_stamp_ns: Optional[int] = None
        self._publish_label_msg: Optional[Callable[[str], None]] = None

        self._writer_lock = threading.Lock()
        self._rec_active: bool = False
        self._rec_dir: Optional[Path] = None
        self._rec_frames: list[tuple[int, np.ndarray]] = []
        # Pristine copies (post-warp, pre-annotation) for stitching — one per
        # unique manip stamp, staged via stage_clean_frame().
        self._rec_clean_frames: list[tuple[int, np.ndarray]] = []
        self._clean_frame: Optional[np.ndarray] = None

        self._label_active: bool = False
        # Top-left overlay content: live per-frame scores while accumulating,
        # the final weighted result during the post-release re-publish tail.
        self._gaze_label_display: list[dict] = []
        # Per-frame label log captured during recording; written to
        # <session_dir>/gaze_labels.json when recording stops.
        self._label_log: list[dict] = []

    def set_label_publisher(self, publish: Callable[[str], None]) -> None:
        """Inject the /gaze_label publish function (JSON string -> None)."""
        self._publish_label_msg = publish

    def note_manip_stamp(self, stamp_ns: int) -> None:
        self.latest_manip_stamp_ns = stamp_ns

    @property
    def label_display(self) -> list[dict]:
        """Entries for the top-left score overlay (may be empty)."""
        return self._gaze_label_display

    def on_recording_start(self) -> None:
        if not self._participant:
            return
        base = Path("recordings") / self._participant
        base.mkdir(parents=True, exist_ok=True)
        existing = sorted([int(p.name) for p in base.iterdir() if p.is_dir() and p.name.isdigit()])
        if not existing:
            return
        session_dir = base / f"{existing[-1]:02d}"
        if self.latest_manip_stamp_ns is not None:
            meta = {"rosbag_start_time_ns": self.latest_manip_stamp_ns}
            (session_dir / "sync.json").write_text(json.dumps(meta, indent=2))
        with self._writer_lock:
            self._rec_dir = session_dir / "frames"
            self._rec_active = True
            self._rec_frames = []
            self._rec_clean_frames = []
            self._clean_frame = None
            self._gaze_label_display = []
            print(f"Image recording started -> {self._rec_dir}")

    def on_label_start(self) -> None:
        """Begin gaze_label accumulation after the calibration beep ends."""
        with self._writer_lock:
            self._label_active = True
            self._label_log = []
            self._gaze_label_display = []
        print("gaze_label accumulation started.")

    def stage_clean_frame(self, frame: np.ndarray) -> None:
        """Keep a pristine copy of the display frame (post-warp, before any
        annotation) for stitching; record_frame() consumes it."""
        if self._rec_active:
            self._clean_frame = frame.copy()

    def record_frame(self, frame: np.ndarray) -> None:
        with self._writer_lock:
            if not self._rec_active or self._rec_dir is None:
                return
            ts = self.latest_manip_stamp_ns if self.latest_manip_stamp_ns is not None else 0
            self._rec_frames.append((ts, frame))
            if self._clean_frame is not None:
                if not self._rec_clean_frames or self._rec_clean_frames[-1][0] != ts:
                    self._rec_clean_frames.append((ts, self._clean_frame))
                self._clean_frame = None

    def stop(self) -> None:
        with self._writer_lock:
            if not self._rec_active:
                return
            self._rec_active = False
            self._label_active = False
            rec_dir = self._rec_dir
            frames = self._rec_frames
            clean_frames = self._rec_clean_frames
            label_log = self._label_log
            self._rec_dir = None
            self._rec_frames = []
            self._rec_clean_frames = []
            self._clean_frame = None
            self._label_log = []

        if rec_dir is None or not frames:
            return
        threading.Thread(
            target=self._flush_and_encode,
            args=(rec_dir, frames, label_log, clean_frames), daemon=True,
        ).start()

    def log_labels(
        self,
        det_summary: list[tuple[str, float, float, tuple[int, int]]],
        gaze_px: Optional[tuple[int, int]] = None,
    ) -> None:
        """Log per-frame label scores while accumulation is active and update
        the on-screen display. (/gaze_label_raw publishing is disabled to save
        LAN bandwidth, so nothing is published per frame.)"""
        entries = [
            {"label": label, "score": round(score, 4)}
            for label, _, score, _ in det_summary
            if score >= self._s_min
        ]
        entries.sort(key=lambda x: -x["score"])

        centers_px: dict[str, list[int]] = {
            label: [int(center_px[0]), int(center_px[1])]
            for label, _, score, center_px in det_summary
            if score >= self._s_min
        }

        if self._label_active:
            detected_log = [
                {**e, "center_px": centers_px.get(e["label"])} for e in entries
            ]
            with self._writer_lock:
                self._label_log.append({
                    "stamp_ns": self.latest_manip_stamp_ns,
                    "gaze_px": [int(gaze_px[0]), int(gaze_px[1])] if gaze_px is not None else None,
                    "detected": detected_log,
                })
            self._gaze_label_display = entries

    def _flush_and_encode(
        self,
        frames_dir: Path,
        frames: list[tuple[int, np.ndarray]],
        label_log: Optional[list[dict]] = None,
        clean_frames: Optional[list[tuple[int, np.ndarray]]] = None,
    ) -> None:
        session_dir = frames_dir.parent

        center_stamp_ns: Optional[int] = None
        fallback_reason: Optional[str] = None
        if label_log:
            try:
                stitched_path = stitch_recording(
                    clean_frames if clean_frames else frames,
                    label_log,
                    session_dir / "stitched.png",
                )
                if stitched_path is not None:
                    track_path = stitched_path.with_name(
                        f"{stitched_path.stem}_gaze_track.json"
                    )
                    if track_path.exists():
                        center_stamp_ns = find_gaze_center_stamp(
                            track_path,
                            self._gaze_peak_window,
                            self._gaze_peak_radius,
                            visualize=False,
                        )
                        center_stamp_ns, fallback_reason = validate_peak_center(
                            track_path,
                            center_stamp_ns,
                            [stamp_ns for stamp_ns, _ in frames],
                            self._gaze_peak_yolo_max_distance,
                        )
                        if center_stamp_ns is not None:
                            visualize_analyzed_peak(
                                track_path, self._gaze_peak_radius
                            )
                    else:
                        print(f"Gaze center analysis skipped: {track_path} not found.")
            except Exception as exc:
                fallback_reason = f"peak processing failed: {exc}"
                print(f"Frame stitching or gaze center analysis failed: {exc}")

        if center_stamp_ns is None:
            print(
                "Gaze peak rejected"
                + (f": {fallback_reason}" if fallback_reason else "")
                + "; "
                "/gaze_label falls back to the unweighted average."
            )
        elif label_log:
            stamped = [f for f in label_log if f.get("stamp_ns") is not None]
            if stamped:
                best_dt = min(abs(f["stamp_ns"] - center_stamp_ns) for f in stamped)
                for f in stamped:
                    if abs(f["stamp_ns"] - center_stamp_ns) != best_dt:
                        continue
                    dets = normalize_entries([
                        {"label": d["label"], "score": d["score"]}
                        for d in f.get("detected", [])
                    ])
                    print(
                        f"Gaze-center frame stamp_ns={f['stamp_ns']} "
                        f"(|dt|={best_dt / 1e6:.1f} ms): {json.dumps(dets) if dets else '(no detections)'}"
                    )

        weighted = compute_weighted_entries(label_log or [], center_stamp_ns)
        print()
        published = self._publish_final_labels(weighted)

        frames_dir.mkdir(parents=True, exist_ok=True)
        timestamps: list[int] = []
        for ts, frame in frames:
            cv2.imwrite(str(frames_dir / f"{ts}.png"), frame)
            timestamps.append(ts)
        print(f"Saved {len(timestamps)} frames to {frames_dir}")

        if label_log:
            payload = {
                "gaze_center_method": (
                    "peak" if center_stamp_ns is not None else "unweighted_average"
                ),
                "gaze_center_stamp_ns": center_stamp_ns,
                "gaze_peak_fallback_reason": fallback_reason,
                "published": published,
                "frames": label_log,
            }
            labels_path = session_dir / "gaze_labels.json"
            labels_path.write_text(json.dumps(payload, indent=2))
            print(f"Saved {len(label_log)} label records to {labels_path}")

        self._encode_video(frames_dir, timestamps)

    def _publish_final_labels(self, entries: list[dict]) -> list[dict]:
        """Publish the weighted /gaze_label result, with a short repeat tail."""
        out_entries = normalize_entries(entries)
        with self._writer_lock:
            self._gaze_label_display = out_entries.copy()
        if self._publish_label_msg is None:
            print("/gaze_label publisher not set; skipping publish.")
            return out_entries
        data = json.dumps(out_entries) if out_entries else ""
        print(f"/gaze_label publishing: {data or '(empty)'}")
        deadline = time.monotonic() + self._label_tail_duration
        while True:
            self._publish_label_msg(data)
            if time.monotonic() >= deadline:
                break
            time.sleep(0.1)
            if self._rec_active or self._label_active:
                return out_entries
        with self._writer_lock:
            if not self._rec_active and not self._label_active:
                self._gaze_label_display = []
        return out_entries

    def _encode_video(self, frames_dir: Path, timestamps: list[int]) -> None:
        session_dir = frames_dir.parent
        concat_path = session_dir / "ffconcat.txt"

        lines = ["ffconcat version 1.0"]
        for i, ts in enumerate(timestamps):
            dur = (timestamps[i + 1] - ts) / 1e9 if i + 1 < len(timestamps) else (timestamps[-1] - timestamps[-2]) / 1e9
            lines.append(f"file 'frames/{ts}.png'")
            lines.append(f"duration {dur:.9f}")
        concat_path.write_text("\n".join(lines) + "\n")

        out_path = session_dir / "gaze_overlay.mp4"
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_path),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(out_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"Video encoded -> {out_path}")
        else:
            print(f"ffmpeg encoding failed:\n{result.stderr[-800:]}")
