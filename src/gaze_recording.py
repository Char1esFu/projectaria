import json
import threading
import time
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from src.detection_infill import (
    export_infilled_frames,
    infill_missing_detections,
    merge_infilled,
    summary_brief,
)
from src.frame_stitcher import stitch_recording
from utils.encode_video import encode_frame_video
from src.gaze_score_stability import (
    export_selected_frames,
    generate_stability_plots,
    select_stable_windows,
)


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


def save_published_result(recording_dir: Path, entries: list[dict]) -> Path:
    """Replace the standalone normalized /gaze_label result file."""
    path = Path(recording_dir) / "published.json"
    path.write_text(json.dumps(entries, indent=2))
    return path


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
        boundary_radius: float,
        gaze_var_window: int = 3,
        gaze_var_threshold: Optional[float] = None,
        gaze_var_top: Optional[int] = 1,
        gaze_var_force_endpoint_points: int = 1,
        hide_excluded: bool = False,
        dist_threshold: float = 1080.0,
        std_dist: float = 200.0,
        detection_infill: bool = True,
        infill_min_observations: int = 2,
    ) -> None:
        self._participant = participant
        self._s_min = s_min
        self._label_tail_duration = label_tail_duration
        self._boundary_radius = boundary_radius
        # Each selected variance window contributes only its centre frame.
        self._gaze_var_window = gaze_var_window
        self._gaze_var_threshold = gaze_var_threshold
        self._gaze_var_top = gaze_var_top
        self._gaze_var_force_endpoint_points = gaze_var_force_endpoint_points
        self._hide_excluded = hide_excluded
        # Scoring parameters, mirrored from the live path so in-filled
        # detections are scored exactly like logged ones.
        self._dist_threshold = dist_threshold
        self._std_dist = std_dist
        # Offline repair of YOLO flicker: reproject a label observed in nearby
        # frames into the frames that dropped it, using the stitching poses.
        self._detection_infill = detection_infill
        self._infill_min_observations = infill_min_observations

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

    def _infill_detections(
        self,
        stitched_path: Path,
        session_dir: Path,
        label_log: list[dict],
        frames: list[tuple[int, np.ndarray]],
    ) -> Optional[dict]:
        """Fill YOLO flicker gaps in label_log using the stitching poses.

        Failures here must never cost us the /gaze_label publish, so anything
        unexpected degrades to the raw log."""
        try:
            height, width = frames[0][1].shape[:2]
            return infill_missing_detections(
                label_log,
                stitched_path.with_name(f"{stitched_path.stem}_placements.json"),
                (width, height),
                s_min=self._s_min,
                dist_threshold=self._dist_threshold,
                std_dist=self._std_dist,
                min_label_observations=self._infill_min_observations,
                report_path=session_dir / "detection_infill.json",
            )
        except Exception as exc:
            print(f"Detection in-fill skipped: {exc}")
            return None

    def _flush_and_encode(
        self,
        frames_dir: Path,
        frames: list[tuple[int, np.ndarray]],
        label_log: Optional[list[dict]] = None,
        clean_frames: Optional[list[tuple[int, np.ndarray]]] = None,
    ) -> None:
        session_dir = frames_dir.parent

        selected_stamps_ns: list[int] = []
        selected_windows: list[dict] = []
        fallback_reason: Optional[str] = None
        infill_summary: Optional[dict] = None
        # Everything downstream of in-fill analyses this merged timeline; the
        # raw label_log is what gets saved, unchanged, as gaze_labels.json's
        # "frames".
        analysis_log: list[dict] = label_log or []
        if label_log:
            try:
                stitched_path = stitch_recording(
                    clean_frames if clean_frames else frames,
                    label_log,
                    session_dir / "stitched.png",
                )
                if stitched_path is not None:
                    # Repair YOLO flicker before anything reads the scores, so
                    # the variance windows and the average both see a label that
                    # stayed inside the gaze crop as present in every frame.
                    if self._detection_infill:
                        infill_summary = self._infill_detections(
                            stitched_path, session_dir, label_log,
                            clean_frames if clean_frames else frames,
                        )
                        if infill_summary:
                            analysis_log = merge_infilled(
                                label_log, infill_summary.get("infilled_frames")
                            )
                    track_path = stitched_path.with_name(
                        f"{stitched_path.stem}_gaze_track.json"
                    )
                    if track_path.exists():
                        selected_windows, excluded = select_stable_windows(
                            analysis_log,
                            track_path,
                            window=self._gaze_var_window,
                            threshold=self._gaze_var_threshold,
                            top=self._gaze_var_top,
                            boundary_radius=self._boundary_radius,
                            force_endpoint_points=self._gaze_var_force_endpoint_points,
                        )
                        # A variance window only flags its centre timestamp;
                        # only those centre frames feed the label average.
                        selected_stamps_ns = sorted(
                            {w["center_stamp_ns"] for w in selected_windows}
                        )
                        if selected_windows:
                            if any(w.get("threshold_fallback")
                                   for w in selected_windows):
                                print(
                                    "No interior window is below the variance "
                                    f"threshold ({self._gaze_var_threshold}); "
                                    f"using the {len(selected_windows)} lowest-"
                                    "variance interior window(s) for /gaze_label."
                                )
                        else:
                            fallback_reason = (
                                "no interior gaze window exists (every window "
                                "centre falls in a start/end fixation region)"
                            )
                    else:
                        print(f"Variance selection skipped: {track_path} not found.")
            except Exception as exc:
                fallback_reason = f"variance processing failed: {exc}"
                print(f"Frame stitching or variance selection failed: {exc}")

        selected_stamp_set = set(selected_stamps_ns) if selected_stamps_ns else None
        averaged = compute_average_entries(analysis_log, selected_stamp_set)
        used_selected_frames = bool(selected_stamps_ns and averaged)
        stamped_frames = [
            frame for frame in analysis_log if frame.get("stamp_ns") is not None
        ]

        if selected_stamps_ns and not averaged:
            fallback_reason = (
                "variance-selected frames have no corresponding YOLO detections"
            )
            averaged = compute_average_entries(analysis_log, None)
            print(
                "Variance-selected frames contain no YOLO detections; "
                "/gaze_label falls back to the whole-recording unweighted average."
            )
        elif not selected_stamps_ns:
            print(
                "No variance frame selected"
                + (f": {fallback_reason}" if fallback_reason else "")
                + "; "
                "/gaze_label falls back to the unweighted average."
            )
        elif label_log:
            print(
                f"Selected {len(selected_stamps_ns)} variance timestamp(s) "
                "for unweighted YOLO score averaging."
            )

        if used_selected_frames:
            averaging_frames = [
                frame for frame in stamped_frames
                if frame["stamp_ns"] in selected_stamp_set
            ]
            averaging_source = "VARIANCE_SELECTED_FRAMES"
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
            + (f"; reason: {fallback_reason}" if not used_selected_frames and fallback_reason else "")
        )

        print()
        published = self._publish_final_labels(averaged)

        frames_dir.mkdir(parents=True, exist_ok=True)
        timestamps: list[int] = []
        for ts, frame in frames:
            cv2.imwrite(str(frames_dir / f"{ts}.png"), frame)
            timestamps.append(ts)
        print(f"Saved {len(timestamps)} frames to {frames_dir}")

        # Redraw the in-filled frames first: the selected-frame export below
        # picks its images from that folder, so a chosen frame is shown with the
        # detections its scores actually came from.
        self._export_infilled(session_dir, infill_summary)

        # Generate all shared stability plots using the exact windows that fed
        # /gaze_label (no second selection pass).
        if label_log:
            infilled_count = sum(
                len(frame.get("detected", []))
                for frame in (infill_summary or {}).get("infilled_frames", [])
            )
            source_note = (
                f"logged + in-filled ({infilled_count} merged detection(s))"
                if infilled_count else "logged only"
            )
            try:
                generate_stability_plots(
                    session_dir,
                    analysis_log,
                    selected_windows,
                    window=self._gaze_var_window,
                    threshold=self._gaze_var_threshold,
                    boundary_radius=self._boundary_radius,
                    force_endpoint_points=self._gaze_var_force_endpoint_points,
                    hide_excluded=self._hide_excluded,
                    source_note=source_note,
                )
            except Exception as exc:
                print(f"Gaze score stability plots skipped: {exc}")

        # For the variance selector, dump the chosen frames' images + YOLO scores
        # into a dedicated subfolder of the session for offline inspection. These
        # are the same centre frames used for the average (one per selected
        # window), matching stitched_variance.png.
        if label_log:
            export_selected_frames(
                session_dir, frames_dir, analysis_log, selected_stamps_ns
            )

        if label_log:
            payload = {
                # The per-point fill/rejection lists stay in
                # detection_infill.json; here we only keep the counts.
                "detection_infill": (
                    summary_brief(infill_summary) if infill_summary else None
                ),
                # Exactly what the live run logged, never edited by in-fill...
                "frames": label_log,
                # ...and, separately, what in-fill added on top: same record
                # shape, so analysis merges the two by stamp_ns while the origin
                # of every score stays traceable.
                "infilled_frames": (
                    infill_summary.get("infilled_frames", []) if infill_summary else []
                ),
            }
            labels_path = session_dir / "gaze_labels.json"
            labels_path.write_text(json.dumps(payload, indent=2))
            print(
                f"Saved {len(label_log)} label records "
                f"(+{len(payload['infilled_frames'])} in-filled frame record(s)) "
                f"to {labels_path}"
            )
            published_path = save_published_result(session_dir, published)
            print(f"Saved /gaze_label result to {published_path}")

        encode_frame_video(session_dir, frames_dir, "gaze_overlay.mp4", "ffconcat.txt")

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

    def _export_infilled(self, session_dir: Path, summary: Optional[dict]) -> None:
        """Write the reviewable in-fill frame folder and its video.

        Runs only once frames/ is on disk, since the export copies/redraws those
        images. Same outputs the offline tool produces, so a live recording can
        be checked without re-running anything."""
        if not summary or not summary.get("fills"):
            return
        try:
            out_dir = export_infilled_frames(session_dir, summary)
            if out_dir is not None:
                encode_frame_video(session_dir, out_dir, "gaze_overlay_infilled.mp4")
        except Exception as exc:
            print(f"In-fill frame export skipped: {exc}")
