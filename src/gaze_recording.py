import json
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from src.frame_stitcher import stitch_recording
from src.gaze_track_peak import find_stable_gaze_stamps, visualize_analyzed_peak


def normalize_entries(entries: list[dict]) -> list[dict]:
    """Normalize scores so they sum to 1 over labels."""
    total = sum(it["score"] for it in entries)
    if total <= 0:
        return entries
    return [
        {"label": it["label"], "score": round(it["score"] / total, 4)}
        for it in entries
    ]


def compute_average_entries(
    label_log: list[dict], selected_stamps_ns: Optional[set[int]]
) -> list[dict]:
    """Unweighted average over selected frames, or all frames for fallback."""
    frames = [f for f in label_log if f.get("stamp_ns") is not None]
    if selected_stamps_ns is not None:
        frames = [f for f in frames if f["stamp_ns"] in selected_stamps_ns]
    if not frames:
        return []
    scores: dict[str, float] = {}
    frame_weight = 1.0 / len(frames)
    for frame in frames:
        for item in frame.get("detected", []):
            lbl = item["label"]
            scores[lbl] = scores.get(lbl, 0.0) + frame_weight * item["score"]
    weighted = [{"label": lbl, "score": round(s, 4)} for lbl, s in scores.items()]
    weighted.sort(key=lambda x: -x["score"])
    return weighted


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
    ) -> None:
        self._participant = participant
        self._s_min = s_min
        self._label_tail_duration = label_tail_duration
        self._gaze_peak_window = gaze_peak_window
        self._gaze_peak_radius = gaze_peak_radius

        # Latest camera_info stamp from the manipulation workstation (ns).
        self.latest_manip_stamp_ns: Optional[int] = None
        # Timestamp assigned to the RGB frame currently passing through the
        # overlay/recording pipeline. It comes from ZED or Aria hardware,
        # depending on the visualizer's selected timestamp source.
        self.latest_rgb_stamp_ns: Optional[int] = None
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
        # the final averaged result during the post-release re-publish tail.
        self._gaze_label_display: list[dict] = []
        # Per-frame label log captured during recording; written to
        # <session_dir>/gaze_labels.json when recording stops.
        self._label_log: list[dict] = []

    def set_label_publisher(self, publish: Callable[[str], None]) -> None:
        """Inject the /gaze_label publish function (JSON string -> None)."""
        self._publish_label_msg = publish

    def note_manip_stamp(self, stamp_ns: int) -> None:
        self.latest_manip_stamp_ns = stamp_ns
        self.latest_rgb_stamp_ns = stamp_ns

    def note_hardware_rgb_stamp(self, stamp_ns: int) -> None:
        self.latest_rgb_stamp_ns = stamp_ns

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
            ts = self.latest_rgb_stamp_ns if self.latest_rgb_stamp_ns is not None else 0
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
                    "stamp_ns": self.latest_rgb_stamp_ns,
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

        selected_stamps_ns: list[int] = []
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
                        selected_stamps_ns = find_stable_gaze_stamps(
                            track_path,
                            self._gaze_peak_window,
                            self._gaze_peak_radius,
                            visualize=False,
                        )
                        if selected_stamps_ns:
                            visualize_analyzed_peak(
                                track_path, self._gaze_peak_radius
                            )
                        else:
                            fallback_reason = "no non-boundary gaze point is below MSD threshold"
                    else:
                        print(f"Gaze center analysis skipped: {track_path} not found.")
            except Exception as exc:
                fallback_reason = f"peak processing failed: {exc}"
                print(f"Frame stitching or gaze center analysis failed: {exc}")

        selected_stamp_set = set(selected_stamps_ns) if selected_stamps_ns else None
        averaged = compute_average_entries(label_log or [], selected_stamp_set)
        used_low_msd_frames = bool(selected_stamps_ns and averaged)
        stamped_frames = [
            frame for frame in (label_log or []) if frame.get("stamp_ns") is not None
        ]

        if selected_stamps_ns and not averaged:
            fallback_reason = (
                "low-MSD gaze frames have no corresponding YOLO detections"
            )
            averaged = compute_average_entries(label_log or [], None)
            print(
                "Selected low-MSD gaze frames contain no YOLO detections; "
                "/gaze_label falls back to the whole-recording unweighted average."
            )
        elif not selected_stamps_ns:
            print(
                "No low-MSD gaze frames selected"
                + (f": {fallback_reason}" if fallback_reason else "")
                + "; "
                "/gaze_label falls back to the unweighted average."
            )
        elif label_log:
            print(
                f"Selected {len(selected_stamps_ns)} low-MSD gaze timestamp(s) "
                "for unweighted YOLO score averaging."
            )

        if used_low_msd_frames:
            averaging_frames = [
                frame for frame in stamped_frames
                if frame["stamp_ns"] in selected_stamp_set
            ]
            averaging_source = "LOW_MSD_FRAMES"
        else:
            averaging_frames = stamped_frames
            averaging_source = "WHOLE_RECORDING_FALLBACK"
        detection_frame_count = sum(
            bool(frame.get("detected")) for frame in averaging_frames
        )
        print(
            "YOLO score averaging source: "
            f"{averaging_source}; unweighted average over "
            f"{len(averaging_frames)} frame(s), "
            f"{detection_frame_count} with YOLO detections"
            + (f"; reason: {fallback_reason}" if not used_low_msd_frames and fallback_reason else "")
        )

        print()
        published = self._publish_final_labels(averaged)

        frames_dir.mkdir(parents=True, exist_ok=True)
        timestamps: list[int] = []
        for ts, frame in frames:
            cv2.imwrite(str(frames_dir / f"{ts}.png"), frame)
            timestamps.append(ts)
        print(f"Saved {len(timestamps)} frames to {frames_dir}")

        if label_log:
            payload = {
                "gaze_center_method": (
                    "low_msd_frames" if used_low_msd_frames else "unweighted_average"
                ),
                "gaze_selected_stamps_ns": selected_stamps_ns,
                "gaze_peak_fallback_reason": fallback_reason,
                "published": published,
                "frames": label_log,
            }
            labels_path = session_dir / "gaze_labels.json"
            labels_path.write_text(json.dumps(payload, indent=2))
            print(f"Saved {len(label_log)} label records to {labels_path}")

        self._encode_video(frames_dir, timestamps)

    def _publish_final_labels(self, entries: list[dict]) -> list[dict]:
        """Publish the averaged /gaze_label result, with a short repeat tail."""
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
