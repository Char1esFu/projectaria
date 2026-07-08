import argparse
import json
import math
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO

from src.frame_stitcher import stitch_recording
from src.gaze_track_peak import GAZE_CENTER_METHODS, find_gaze_center_stamp
from utils.aria_rgb_stream import AriaRgbStream

MODEL_PATH = Path(__file__).parent.parent / "yolo_model" / "last_aria_4.pt"
CROP_SIZE = 200
RESIZE_SIZE = 1080
DEFAULT_GAZE_CLUSTER_WINDOW = 9
DEFAULT_GAZE_CLUSTER_RADIUS = 20.0
DEFAULT_GAZE_CENTER_METHOD = "peak"


class GazeOverlay:
    """Subscribes to /aria/gaze_euler and draws a crosshair on the display image."""

    def __init__(
        self,
        homography_path: Optional[Path] = None,
        enable_yolo: bool = False,
        draw_gaze: bool = False,
        enable_capture: bool = False,
        model_path: Optional[Path] = None,
        conf_threshold: float = 0.25,
        device: Optional[str] = None,
        filter_labels: Optional[list[str]] = None,
        capture_interval: float = 0.0,
        dist_threshold: float = 1080.0,
        std_dist: float = 200.0,
        s_min: float = 0.3,
        participant: str = "",
        gaze_cluster_window: int = DEFAULT_GAZE_CLUSTER_WINDOW,
        gaze_cluster_radius: float = DEFAULT_GAZE_CLUSTER_RADIUS,
        gaze_center_method: str = DEFAULT_GAZE_CENTER_METHOD,
    ) -> None:
        self.gaze_pitch: float = 0.0
        self.gaze_yaw: float = 0.0
        self._rclpy = None
        self._ros_node = None
        self._ros_thread = None

        # Image sequence recording state
        self._participant = participant
        self._writer_lock = threading.Lock()
        self._rec_active: bool = False
        self._rec_dir: Optional[Path] = None
        self._rec_frames: list[tuple[int, np.ndarray]] = []
        # Pristine copies (post-warp, pre-annotation) for stitching — one per
        # unique manip stamp, handed from draw() to record_frame().
        self._rec_clean_frames: list[tuple[int, np.ndarray]] = []
        self._clean_frame: Optional[np.ndarray] = None

        self._label_active: bool = False
        self._label_tail_duration: float = 2.0
        # Top-left overlay content: live per-frame scores while accumulating, the final weighted result during the post-release re-publish tail.
        self._gaze_label_display: list[dict] = []
        # Per-frame label log captured during recording: one entry per published
        # frame with the raw detections and the averaged (/gaze_label) result.
        # Written to <session_dir>/gaze_labels.json when recording stops.
        self._label_log: list[dict] = []

        # Latest camera_info stamp from manipulation workstation (nanoseconds, rosbag time)
        self._latest_manip_stamp_ns: Optional[int] = None

        self._setup_ros_subscriber()

        # Optional homography to shift RGB view to the symmetric center
        self.H: Optional[np.ndarray] = None
        if homography_path is not None:
            self.H = np.loadtxt(homography_path)
            print(f"Loaded homography from {homography_path}")

        # YOLO model (always loaded if model_path given)
        self.model = None
        self.enable_yolo = enable_yolo
        self.conf_threshold = conf_threshold
        self.yolo_device = device
        if self.enable_yolo and model_path is not None:
            import torch
            if device is None:
                self.yolo_device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Loading YOLO model from {model_path} on device={self.yolo_device}")
            self.model = YOLO(str(model_path))

        self.draw_gaze = draw_gaze

        # Capture settings
        self.enable_capture = enable_capture
        self.capture_active = False
        self.capture_dir = Path("saved_images")
        self.capture_index = 0
        self.capture_interval = capture_interval
        self._last_capture_time: float = 0.0
        if self.enable_capture:
            self.capture_dir.mkdir(parents=True, exist_ok=True)
            print(f"Capture enabled. Press S to start/stop saving to {self.capture_dir}/")

        # Labels to filter out from YOLO results
        self.filter_labels: set[str] = set(l.lower() for l in (filter_labels or []))

        # Score = 0 when gaze distance from bbox center >= dist_threshold
        self.dist_threshold = dist_threshold
        self.std_dist: float = std_dist
        self.s_min: float = s_min
        self._last_det_summary: list[tuple[str, float, float]] = []
        self._gaze_cluster_window = gaze_cluster_window
        self._gaze_cluster_radius = gaze_cluster_radius
        if gaze_center_method not in GAZE_CENTER_METHODS:
            raise ValueError(
                f"gaze_center_method must be one of {GAZE_CENTER_METHODS}, "
                f"got {gaze_center_method!r}"
            )
        self._gaze_center_method = gaze_center_method

    def _setup_ros_subscriber(self) -> None:
        try:
            import rclpy
            from geometry_msgs.msg import Vector3
            from sensor_msgs.msg import CameraInfo
            from std_msgs.msg import Empty, String

            if not rclpy.ok():
                rclpy.init(args=None)
            self._rclpy = rclpy
            self._ros_node = rclpy.create_node("gaze_rgb_visualizer")

            def _gaze_callback(msg: Vector3) -> None:
                self.gaze_pitch = float(msg.x)
                self.gaze_yaw = float(msg.y)

            def _camera_info_callback(msg: CameraInfo) -> None:
                self._latest_manip_stamp_ns = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec

            self._ros_node.create_subscription(
                Vector3, "/aria/gaze_euler", _gaze_callback, 10
            )
            self._ros_node.create_subscription(
                Empty, "/recording/start", lambda _: self._on_recording_start(), 10
            )
            # gaze_label accumulation starts here (published once the beep ends),
            # decoupled from video recording start above.
            self._ros_node.create_subscription(
                Empty, "/gaze_label_recording_start", lambda _: self._on_gaze_label_start(), 10
            )
            self._ros_node.create_subscription(
                Empty, "/key/b/release", lambda _: self._stop_recording(), 10
            )
            self._ros_node.create_subscription(
                CameraInfo,
                "/zedr/zed_node/rgb/camera_info",
                _camera_info_callback,
                10,
            )
            # /gaze_label_raw: unconditional backup stream (always published).
            self._gaze_label_raw_pub = self._ros_node.create_publisher(String, "/gaze_label_raw", 10)
            # /gaze_label: same format, but only carries content while recording (+2s tail).
            self._gaze_label_pub = self._ros_node.create_publisher(String, "/gaze_label", 10)
            # Dedicated executor so multiple modules can spin their own nodes
            # in parallel under main_entry.py without contending for the
            # rclpy global executor.
            from rclpy.executors import SingleThreadedExecutor
            self._ros_executor = SingleThreadedExecutor()
            self._ros_executor.add_node(self._ros_node)
            self._ros_thread = threading.Thread(
                target=self._ros_executor.spin, daemon=True
            )
            self._ros_thread.start()
            print("ROS2 publishers started: /gaze_label_raw, /gaze_label")
        except Exception as exc:
            raise RuntimeError(f"ROS2 subscriber unavailable: {exc}") from exc

    def _on_recording_start(self) -> None:
        if not self._participant:
            return
        base = Path("recordings") / self._participant
        base.mkdir(parents=True, exist_ok=True)
        existing = sorted([int(p.name) for p in base.iterdir() if p.is_dir() and p.name.isdigit()])
        if not existing:
            return
        session_dir = base / f"{existing[-1]:02d}"
        if self._latest_manip_stamp_ns is not None:
            meta = {"rosbag_start_time_ns": self._latest_manip_stamp_ns}
            (session_dir / "sync.json").write_text(json.dumps(meta, indent=2))
        with self._writer_lock:
            # Defer mkdir to _stop_recording — during recording we touch the
            # disk for nothing, only buffering frames in memory.
            self._rec_dir = session_dir / "frames"
            self._rec_active = True
            self._rec_frames = []
            self._rec_clean_frames = []
            self._clean_frame = None
            # Clear the previous session's weighted result so the recorded frames
            # between now and the beep end don't carry a stale overlay.
            self._gaze_label_display = []
            print(f"Image recording started → {self._rec_dir}")

    def _on_gaze_label_start(self) -> None:
        """Begin gaze_label accumulation. Fired by /gaze_label_recording_start once the
        calibration beep ends — this, not video recording start, marks t=0 for the
        weighted /gaze_label score."""
        with self._writer_lock:
            self._label_active = True
            self._label_log = []
            self._gaze_label_display = []
        print("gaze_label accumulation started.")

    def record_frame(self, frame: np.ndarray) -> None:
        with self._writer_lock:
            if not self._rec_active or self._rec_dir is None:
                return
            ts = self._latest_manip_stamp_ns if self._latest_manip_stamp_ns is not None else 0
            # Buffer only — no disk I/O while recording. `frame` (the loop's
            # `display`) is a fresh array each iteration, so we can keep the
            # reference without copying.
            self._rec_frames.append((ts, frame))
            # Clean copy stashed by draw() this iteration (same thread, so it
            # matches `frame`). The stitcher uses one frame per unique stamp,
            # so buffer only the first clean frame per stamp.
            if self._clean_frame is not None:
                if not self._rec_clean_frames or self._rec_clean_frames[-1][0] != ts:
                    self._rec_clean_frames.append((ts, self._clean_frame))
                self._clean_frame = None

    def _stop_recording(self) -> None:
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
        # Flush the whole batch to disk and encode off the main thread; lag here
        # is fine since recording has already stopped.
        threading.Thread(
            target=self._flush_and_encode,
            args=(rec_dir, frames, label_log, clean_frames), daemon=True,
        ).start()

    def _flush_and_encode(
        self, frames_dir: Path, frames: list[tuple[int, np.ndarray]],
        label_log: Optional[list[dict]] = None,
        clean_frames: Optional[list[tuple[int, np.ndarray]]] = None,
    ) -> None:
        session_dir = frames_dir.parent

        # Priority path first: stitch → cluster → weighted /gaze_label publish.
        # This must run BEFORE the PNG flush and video encode (both take tens of
        # seconds) — audio_record holds /transcription until /gaze_label is out,
        # and its wait times out, so any slow work ahead of the publish risks the
        # receiver pairing the transcription with the previous query's label.
        # stitch_recording works on the in-memory frames, so no disk flush is
        # needed yet.
        center_stamp_ns: Optional[int] = None
        if label_log:
            # Stitch the recorded frames into one large mosaic, anchored on the
            # logged YOLO detection centers (no feature-point matching). Uses the
            # pristine (pre-annotation) copies; falls back to the display frames
            # if none were captured. A stitch failure must not block the final
            # /gaze_label publish or video encoding.
            try:
                stitched_path = stitch_recording(
                    clean_frames if clean_frames else frames,
                    label_log, session_dir / "stitched.png",
                )
                if stitched_path is not None:
                    track_path = stitched_path.with_name(
                        f"{stitched_path.stem}_gaze_track.json"
                    )
                    if track_path.exists():
                        center_stamp_ns = find_gaze_center_stamp(
                            track_path,
                            self._gaze_center_method,
                            self._gaze_cluster_window,
                            self._gaze_cluster_radius,
                        )
                    else:
                        print(f"Gaze center analysis skipped: {track_path} not found.")
            except Exception as exc:
                print(f"Frame stitching or gaze center analysis failed: {exc}")

        # Final /gaze_label result: cluster-center-weighted average over the whole
        # recording, published exactly once here (plus the 2s re-publish tail
        # driven by the draw loop).
        if center_stamp_ns is None:
            print(
                f"No gaze center found ({self._gaze_center_method}); "
                "/gaze_label falls back to the unweighted average."
            )
        elif label_log:
            # Console reference: the raw detections of the highest-weight frame(s)
            # (stamp closest to the cluster center), to eyeball against the
            # published weighted result printed below.
            stamped = [f for f in label_log if f.get("stamp_ns") is not None]
            if stamped:
                best_dt = min(abs(f["stamp_ns"] - center_stamp_ns) for f in stamped)
                for f in stamped:
                    if abs(f["stamp_ns"] - center_stamp_ns) != best_dt:
                        continue
                    # Normalized like the published result, so the two lines compare
                    # score-for-score.
                    dets = self._normalize_entries([
                        {"label": d["label"], "score": d["score"]}
                        for d in f.get("detected", [])
                    ])
                    print(
                        f"Gaze-center frame stamp_ns={f['stamp_ns']} "
                        f"(|dt|={best_dt / 1e6:.1f} ms): {json.dumps(dets) if dets else '(no detections)'}"
                    )
        weighted = self._compute_weighted_entries(label_log or [], center_stamp_ns)
        print()
        published = self._publish_final_labels(weighted)

        # Slow path: flush the frame PNGs, persist the label log, encode the video.
        frames_dir.mkdir(parents=True, exist_ok=True)
        timestamps: list[int] = []
        for ts, frame in frames:
            cv2.imwrite(str(frames_dir / f"{ts}.png"), frame)
            timestamps.append(ts)
        print(f"Saved {len(timestamps)} frames to {frames_dir}")

        if label_log:
            payload = {
                "gaze_center_method": self._gaze_center_method,
                "gaze_center_stamp_ns": center_stamp_ns,
                "published": published,
                "frames": label_log,
            }
            labels_path = session_dir / "gaze_labels.json"
            labels_path.write_text(json.dumps(payload, indent=2))
            print(f"Saved {len(label_log)} label records to {labels_path}")

        self._encode_video(frames_dir, timestamps)

    def _encode_video(self, frames_dir: Path, timestamps: list[int]) -> None:
        import subprocess
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
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(concat_path),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(out_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"Video encoded → {out_path}")
        else:
            print(f"ffmpeg encoding failed:\n{result.stderr[-800:]}")

    def draw(self, display_image: np.ndarray, camera_matrix: Optional[np.ndarray], key: int = -1) -> None:
        if self.enable_capture and key in (ord("s"), ord("S")):
            self.capture_active = not self.capture_active
            state = "started" if self.capture_active else "stopped"
            print(f"Capture {state}.")

        # Warp RGB to symmetric center if homography is available
        if self.H is not None:
            h, w = display_image.shape[:2]
            warped = cv2.warpPerspective(display_image, self.H, (w, h))
            display_image[:] = warped

        # Pristine copy for stitching: post-warp, before any annotation (or the
        # capture-mode zoom) touches display_image. record_frame() consumes it
        # on the same loop iteration/thread.
        if self._rec_active:
            self._clean_frame = display_image.copy()

        if camera_matrix is None:
            return

        fx = camera_matrix[0, 0]
        fy = camera_matrix[1, 1]
        cx = camera_matrix[0, 2]
        cy = camera_matrix[1, 2]

        # After rot90(-1): raw col → display row (pitch), raw row → display col (yaw).
        display_col = cx + fx * np.tan(self.gaze_yaw)
        display_row = cy - fy * np.tan(self.gaze_pitch)
        gx, gy = int(round(display_col)), int(round(display_row))

        h, w = display_image.shape[:2]
        color = (0, 0, 255)
        cross_size = 40
        gaze_valid = 0 <= gx < w and 0 <= gy < h

        # Always crop around gaze for YOLO; CROP_SIZE and RESIZE_SIZE are module constants.
        gaze_pt: Optional[tuple] = None
        crop_img: Optional[np.ndarray] = None
        resized_crop: Optional[np.ndarray] = None
        crop_vis: Optional[np.ndarray] = None
        x_start = y_start = cs = 0
        if gaze_valid:
            cs = min(CROP_SIZE, w, h)
            x_start = max(0, min(gx - cs // 2, w - cs))
            y_start = max(0, min(gy - cs // 2, h - cs))
            crop_img = display_image[y_start:y_start + cs, x_start:x_start + cs].copy()
            gaze_pt = (gx, gy)

        if self.enable_capture and crop_img is not None:
            resized_crop = cv2.resize(crop_img, (RESIZE_SIZE, RESIZE_SIZE), interpolation=cv2.INTER_LINEAR)
            resized_crop = cv2.bilateralFilter(resized_crop, d=5, sigmaColor=15, sigmaSpace=15)
            blurred = cv2.GaussianBlur(resized_crop, (0, 0), sigmaX=5)
            resized_crop = cv2.addWeighted(resized_crop, 5, blurred, -4, 0)

        # YOLO inference on the cropped+resized region
        if self.model is not None and gaze_valid and crop_img is not None:
            if resized_crop is None:
                resized_crop = cv2.resize(crop_img, (RESIZE_SIZE, RESIZE_SIZE), interpolation=cv2.INTER_LINEAR)
                resized_crop = cv2.bilateralFilter(resized_crop, d=5, sigmaColor=15, sigmaSpace=15)
                blurred = cv2.GaussianBlur(resized_crop, (0, 0), sigmaX=5)
                resized_crop = cv2.addWeighted(resized_crop, 5, blurred, -4, 0)
                resized_crop = cv2.convertScaleAbs(resized_crop, alpha=1.2, beta=30)
                


            results = self.model(resized_crop, conf=self.conf_threshold, device=self.yolo_device, verbose=False)
            self._filter_results(results[0])

            scale = RESIZE_SIZE / cs
            gaze_in_crop = ((gx - x_start) * scale, (gy - y_start) * scale)

            if self.enable_capture:
                annotated = self._draw_circles(resized_crop, results[0])
                display_image[:] = cv2.resize(annotated, (w, h), interpolation=cv2.INTER_LINEAR)
                gaze_pt = (int((gx - x_start) * w / cs), int((gy - y_start) * h / cs))
            else:
                annotated = self._draw_circles(
                    display_image, results[0],
                    crop_transform=(x_start, y_start, scale),
                )
                np.copyto(display_image, annotated)

            det_summary = []
            if results[0].boxes is not None:
                names_map = results[0].names
                for xyxy, cid, conf in zip(
                    results[0].boxes.xyxy.tolist(),
                    results[0].boxes.cls.int().tolist(),
                    results[0].boxes.conf.tolist(),
                ):
                    x1, y1, x2, y2 = xyxy
                    ocx = (x1 + x2) / 2.0
                    ocy = (y1 + y2) / 2.0
                    d = math.sqrt((gaze_in_crop[0] - ocx) ** 2 + (gaze_in_crop[1] - ocy) ** 2)
                    score = self._compute_score(d)
                    # Map the detection center from resized-crop space back to full
                    # display-image pixels, so the logged coordinate matches the saved frame.
                    center_px = (
                        int(round(x_start + ocx / scale)),
                        int(round(y_start + ocy / scale)),
                    )
                    det_summary.append((names_map.get(cid, str(cid)), conf, score, center_px))

            self._last_det_summary = sorted(det_summary, key=lambda x: x[2], reverse=True)
            crop_vis = self._draw_circles(resized_crop, results[0])
            vis_y = 18
            for det_label, _, det_score, _ in self._last_det_summary:
                cv2.putText(crop_vis, f"{det_label}  score={det_score:.2f}",
                            (6, vis_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
                vis_y += 18
            self._publish_labels(det_summary, gaze_px=(gx, gy))
            results[0].boxes = None
        elif self.enable_capture and resized_crop is not None:
            display_image[:] = cv2.resize(resized_crop, (w, h), interpolation=cv2.INTER_LINEAR)
            gaze_pt = (int((gx - x_start) * w / cs), int((gy - y_start) * h / cs))

        # Keep a clean frame for saving before drawing viewer-only overlays.
        capture_frame = None
        if self.enable_capture and self.capture_active:
            capture_frame = display_image.copy()

        # Draw gaze info overlay
        cv2.putText(
            display_image,
            f"pitch={np.degrees(self.gaze_pitch):.1f}  yaw={np.degrees(self.gaze_yaw):.1f} deg",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )
        # Top-left detection text: live per-frame scores (/gaze_label_raw content)
        # while accumulating, the final weighted /gaze_label result during the 2s
        # post-release tail, empty otherwise.
        if self._gaze_label_display:
            det_y = 60
            for entry in self._gaze_label_display:
                cv2.putText(
                    display_image,
                    f"{entry['label']}  score={entry['score']:.2f}",
                    (10, det_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
                det_y += 22

        # if crop_vis is not None:
        #     cv2.imshow("YOLO Crop", crop_vis)

        if self.draw_gaze and gaze_pt is not None:
            cv2.circle(display_image, gaze_pt, cross_size, color, 2)
            cv2.circle(display_image, gaze_pt, 4, color, -1)

        if self.enable_capture:
            capture_text = "CAPTURE: ON (press S to stop)" if self.capture_active else "CAPTURE: OFF (press S to start)"
            capture_color = (0, 255, 0) if self.capture_active else (0, 255, 255)
            cap_y = 60 + 22 * len(self._gaze_label_display)
            cv2.putText(
                display_image,
                capture_text,
                (10, cap_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                capture_color,
                2,
                cv2.LINE_AA,
            )

            if self.capture_active and capture_frame is not None:
                now = time.monotonic()
                if now - self._last_capture_time >= self.capture_interval:
                    timestamp_ms = int(cv2.getTickCount() * 1000 / cv2.getTickFrequency())
                    filename = self.capture_dir / f"frame_{timestamp_ms}_{self.capture_index:06d}.png"
                    cv2.imwrite(str(filename), capture_frame)
                    self.capture_index += 1
                    self._last_capture_time = now

    # ------------------------------------------------------------------
    # Score and publish helpers
    # ------------------------------------------------------------------

    def _compute_score(self, d: float) -> float:
        """Gaussian score based on distance from bbox center; 0.0 when d >= dist_threshold."""
        if d >= self.dist_threshold:
            return 0.0
        return math.exp(-(d ** 2) / (2 * self.std_dist ** 2))

    @staticmethod
    def _normalize_entries(entries: list[dict]) -> list[dict]:
        """Divide each score by the sum of scores in the result, so the published
        scores form a relative distribution summing to 1 over its labels."""
        total = sum(it["score"] for it in entries)
        if total <= 0:
            return entries
        return [
            {"label": it["label"], "score": round(it["score"] / total, 4)}
            for it in entries
        ]

    @staticmethod
    def _compute_weighted_entries(
        label_log: list[dict], center_stamp_ns: Optional[int]
    ) -> list[dict]:
        """Cluster-center-weighted average of the per-frame scores in label_log.

        Each logged frame gets weight exp(-1e-8 * |stamp_ns - center_stamp_ns|):
        highest at the kept gaze cluster's center, dropping off sharply along the
        time axis (the 1e-8 factor folds in the ns unit conversion). Weights are
        normalized to sum 1 across frames, then each label's score is the weighted
        sum of its per-frame scores — frames where the label was not detected
        contribute 0. Without a cluster center all weights are equal, which
        degrades to a plain average over the whole recording."""
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

    def _publish_final_labels(self, entries: list[dict]) -> list[dict]:
        """Publish the weighted /gaze_label result, re-publishing at 10 Hz for
        _label_tail_duration seconds so a late or lossy subscriber still gets it.

        Runs on the flush thread right after clustering and before the PNG/video
        flush, so /gaze_label is guaranteed to go out before /transcription
        (audio_record holds the transcription until it sees this message). Always
        publishes — empty data when there is no result — so the audio side never
        stalls. The tail aborts early if a new recording starts.
        Returns the normalized entries that were published."""
        from std_msgs.msg import String

        out_entries = self._normalize_entries(entries)
        with self._writer_lock:
            self._gaze_label_display = out_entries.copy()
        msg = String()
        msg.data = json.dumps(out_entries) if out_entries else ""
        print(f"/gaze_label publishing: {msg.data or '(empty)'}")
        deadline = time.monotonic() + self._label_tail_duration
        while True:
            self._gaze_label_pub.publish(msg)
            if time.monotonic() >= deadline:
                break
            time.sleep(0.1)
            if self._rec_active or self._label_active:
                # A new recording started; stop repeating the old result.
                return out_entries
        # Tail finished: drop the live-view overlay.
        with self._writer_lock:
            if not self._rec_active and not self._label_active:
                self._gaze_label_display = []
        return out_entries

    def _publish_labels(
        self,
        det_summary: list[tuple[str, float, float, tuple[int, int]]],
        gaze_px: Optional[tuple[int, int]] = None,
    ) -> None:
        """Publish {"label", "score"} for detections with score >= s_min, sorted descending.

        /gaze_label_raw always carries the current frame's detections (backup stream).
        /gaze_label is never published here: the cluster-weighted result is computed
        and published (with its re-publish tail) by _flush_and_encode after
        /key/b/release."""
        from std_msgs.msg import String

        entries = [
            {"label": label, "score": round(score, 4)}
            for label, _, score, _ in det_summary
            if score >= self.s_min
        ]
        entries.sort(key=lambda x: -x["score"])

        # Detection-center pixel coords (full display-image space) keyed by label, for
        # the JSON log only — kept out of the published messages to avoid changing the
        # /gaze_label(_raw) wire format. Labels are unique here (_filter_results keeps
        # one detection per class).
        centers_px: dict[str, list[int]] = {
            label: [int(center_px[0]), int(center_px[1])]
            for label, _, score, center_px in det_summary
            if score >= self.s_min
        }

        raw_data = json.dumps(entries) if entries else ""
        raw_msg = String()
        raw_msg.data = raw_data
        self._gaze_label_raw_pub.publish(raw_msg)

        if self._label_active:
            # Log this frame's raw detections, keyed by the same manip stamp the
            # frame PNGs use. Each detected entry carries its center pixel coord;
            # the frame carries the gaze pixel coord — both in full display-image
            # space, matching the saved PNGs. This log feeds both the stitcher and
            # the final cluster-weighted /gaze_label result.
            detected_log = [
                {**e, "center_px": centers_px.get(e["label"])} for e in entries
            ]
            with self._writer_lock:
                self._label_log.append({
                    "stamp_ns": self._latest_manip_stamp_ns,
                    "gaze_px": [int(gaze_px[0]), int(gaze_px[1])] if gaze_px is not None else None,
                    "detected": detected_log,
                })
            # Recorded frames show the live per-frame scores (same content as
            # /gaze_label_raw); the weighted result doesn't exist yet.
            self._gaze_label_display = entries

    def _draw_circles(
        self, img: np.ndarray, result,
        crop_transform: Optional[tuple] = None,
    ) -> np.ndarray:
        """Draw a circle per detection on img.

        crop_transform=(x_start, y_start, scale): when provided, box coordinates are in
        resized-crop space and are mapped back to display-image space before drawing."""
        out = img.copy()
        if result.boxes is None or len(result.boxes) == 0:
            return out
        names = result.names
        for xyxy, cid, conf in zip(
            result.boxes.xyxy.tolist(),
            result.boxes.cls.int().tolist(),
            result.boxes.conf.tolist(),
        ):
            x1, y1, x2, y2 = xyxy
            if crop_transform is not None:
                xs, ys, sc = crop_transform
                cx = int(xs + (x1 + x2) / 2 / sc)
                cy = int(ys + (y1 + y2) / 2 / sc)
                radius = int(math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) / 2 / sc)
            else:
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                radius = int(math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) / 2)
            label = names.get(cid, str(cid))
            cv2.circle(out, (cx, cy), radius, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(out, label, (cx - radius, max(cy - radius - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
        return out

    def _filter_results(self, result) -> None:
        """Remove detections whose label is in self.filter_labels and keep only
        the highest-confidence detection per class (in-place)."""
        if result.boxes is None or len(result.boxes) == 0:
            return
        names = result.names
        cls_ids = result.boxes.cls.int().tolist()
        confs = result.boxes.conf.tolist()

        if self.filter_labels:
            keep = [i for i, cid in enumerate(cls_ids) if names.get(cid, "").lower() not in self.filter_labels]
            result.boxes = result.boxes[keep]
            if len(result.boxes) == 0:
                return
            cls_ids = result.boxes.cls.int().tolist()
            confs = result.boxes.conf.tolist()

        best: dict[int, tuple[int, float]] = {}
        for i, (cid, conf) in enumerate(zip(cls_ids, confs)):
            if cid not in best or conf > best[cid][1]:
                best[cid] = (i, conf)
        keep = [idx for idx, _ in best.values()]
        result.boxes = result.boxes[keep]

    def shutdown(self) -> None:
        self._stop_recording()
        try:
            self._ros_executor.shutdown()
        except Exception:
            pass
        try:
            self._ros_node.destroy_node()
            self._rclpy.shutdown()
        except Exception:
            pass


def run_gaze_rgb_visualizer(
    device_ip: Optional[str] = None,
    update_iptables_rules: bool = False,
    homography_path: Optional[Path] = None,
    enable_yolo: bool = False,
    draw_gaze: bool = False,
    enable_capture: bool = False,
    model_path: Optional[Path] = None,
    conf_threshold: float = 0.25,
    device: Optional[str] = None,
    filter_labels: Optional[list[str]] = None,
    capture_interval: float = 0.0,
    dist_threshold: float = 1080.0,
    std_dist: float = 200.0,
    s_min: float = 0.3,
    participant: str = "",
    gaze_cluster_window: int = DEFAULT_GAZE_CLUSTER_WINDOW,
    gaze_cluster_radius: float = DEFAULT_GAZE_CLUSTER_RADIUS,
    gaze_center_method: str = DEFAULT_GAZE_CENTER_METHOD,
) -> None:
    overlay = GazeOverlay(
        homography_path=homography_path,
        enable_yolo=enable_yolo,
        draw_gaze=draw_gaze,
        enable_capture=enable_capture,
        model_path=model_path,
        conf_threshold=conf_threshold,
        device=device,
        filter_labels=filter_labels,
        capture_interval=capture_interval,
        dist_threshold=dist_threshold,
        std_dist=std_dist,
        s_min=s_min,
        participant=participant,
        gaze_cluster_window=gaze_cluster_window,
        gaze_cluster_radius=gaze_cluster_radius,
        gaze_center_method=gaze_center_method,
    )
    stream = AriaRgbStream(
        device_ip=device_ip,
        update_iptables_rules=update_iptables_rules,
        window_name="Aria RGB Gaze",
    )
    stream.add_overlay(overlay)
    stream.set_frame_callback(overlay.record_frame)
    try:
        stream.run()
    finally:
        overlay.shutdown()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize gaze direction on Aria RGB stream with optional YOLO detection."
    )
    parser.add_argument(
        "--device-ip", 
        default="192.168.8.117",
        help="IP address of the Aria device"
    )
    parser.add_argument(
        "--update_iptables",
        default=False,
        action="store_true",
        help="Update iptables for DDS UDP stream (Linux only).",
    )
    parser.add_argument(
        "--homography", type=Path, default="test_homography/homography.txt",
        help="Path to a homography matrix file (e.g. test_homography/homography.txt). "
             "If provided, the RGB image is warped before gaze overlay.",
    )
    parser.add_argument(
        "--yolo",
        default=False,
        action="store_true",
        help="Enable YOLO detection. If not set, only gaze overlay is shown.",
    )
    parser.add_argument(
        "--draw-gaze",
        default=False,
        action="store_true",
        help="Draw gaze marker (red circle and red dot) on RGB view.",
    )
    parser.add_argument(
        "--capture",
        default=False,
        action="store_true",
        help="Enable runtime capture toggle. Press S to start/stop continuous saving to saved_images/.",
    )
    parser.add_argument("--model", type=str, default=str(MODEL_PATH), help="YOLO model path")
    parser.add_argument("--yolo-conf", type=float, default=0.8, help="YOLO confidence threshold")
    parser.add_argument("--device", type=str, default=None, help="Inference device: cuda / cpu / mps (default: auto)")
    parser.add_argument(
        "--filter-label", type=str, nargs="+", default=["beer bottle", "mayonnaise bottle", "oil bottle", "water bottle"],
        help="Labels to exclude from YOLO results (e.g. --filter-label 'beer bottle' 'water bottle').",
    )
    parser.add_argument(
        "--capture-interval", type=float, default=1.0,
        help="Minimum time interval (seconds) between saved frames. 0 = save every frame.",
    )
    parser.add_argument(
        "--dist-threshold", type=float, default=1080.0,
        help="Max distance (pixels) from bbox center to gaze point. Score = 0 when d >= dist_threshold. Default: 200.",
    )
    parser.add_argument(
        "--std-dist", type=float, default=200.0, dest="std_dist",
        help="Gaussian std (pixels) for score falloff within dist_threshold. Default: 80.",
    )
    parser.add_argument(
        "--s-min", type=float, default=0.0, dest="s_min",
        help="Minimum score threshold for publishing /gaze_label entries (default: 0.2).",
    )
    parser.add_argument(
        "--participant", type=str, default="",
        help="Participant ID (e.g. AB12). Required for video recording to recordings/<participant>/NN/.",
    )
    parser.add_argument(
        "--gaze-cluster-window", type=int, default=DEFAULT_GAZE_CLUSTER_WINDOW,
        help="Odd temporal window for automatic stitched gaze clustering. "
             "Default: 7.",
    )
    parser.add_argument(
        "--gaze-cluster-radius", type=float, default=DEFAULT_GAZE_CLUSTER_RADIUS,
        help="Pixel radius for automatic stitched gaze clustering. Default: 30.",
    )
    parser.add_argument(
        "--gaze-center-method", choices=GAZE_CENTER_METHODS,
        default=DEFAULT_GAZE_CENTER_METHOD,
        help="How the gaze-center stamp for score weighting is found: 'cluster' "
             "(temporal window/radius clustering, middle cluster kept) or 'peak' "
             "(densest track point with boundary-connectivity check). "
             f"Default: {DEFAULT_GAZE_CENTER_METHOD}.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_gaze_rgb_visualizer(
        device_ip=args.device_ip,
        update_iptables_rules=args.update_iptables,
        homography_path=args.homography,
        enable_yolo=args.yolo,
        draw_gaze=args.draw_gaze,
        enable_capture=args.capture,
        model_path=Path(args.model) if args.yolo else None,
        conf_threshold=args.yolo_conf,
        device=args.device,
        filter_labels=args.filter_label,
        capture_interval=args.capture_interval,
        dist_threshold=args.dist_threshold,
        std_dist=args.std_dist,
        s_min=args.s_min,
        participant=args.participant,
        gaze_cluster_window=args.gaze_cluster_window,
        gaze_cluster_radius=args.gaze_cluster_radius,
        gaze_center_method=args.gaze_center_method,
    )


if __name__ == "__main__":
    main()
