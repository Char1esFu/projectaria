import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.gaze_recording import GazeRecord
from src.gaze_rgb_config import (
    CROP_SIZE,
    GAZE_LABEL_TAIL_SEC,
    RESIZE_SIZE,
)
from src.gaze_ros import GazeRos
from src.yolo_overlay import draw_circles, enhance_crop, filter_results, summarize_detections


class GazeOverlay:
    """Draws gaze + detections on the display image. Composes GazeRecord
    (recording/label state) and GazeRos (node, subscriptions, publisher);
    all cross-component wiring is explicit in __init__."""

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
        gaze_cluster_window: int = 7,
        gaze_cluster_radius: float = 20.0,
        gaze_center_method: str = "peak",
    ) -> None:
        self.gaze_pitch: float = 0.0
        self.gaze_yaw: float = 0.0

        # Recording/label state lives in GazeRecord; ROS wiring in GazeRos.
        # Every cross-component dependency is spelled out right here.
        self.recorder = GazeRecord(
            participant=participant,
            s_min=s_min,
            label_tail_duration=GAZE_LABEL_TAIL_SEC,
            gaze_center_method=gaze_center_method,
            gaze_cluster_window=gaze_cluster_window,
            gaze_cluster_radius=gaze_cluster_radius,
        )
        self.ros = GazeRos(
            on_gaze=self._set_gaze,
            on_manip_stamp=self.recorder.note_manip_stamp,
            on_recording_start=self.recorder.on_recording_start,
            on_label_start=self.recorder.on_label_start,
            on_recording_stop=self.recorder.stop,
        )
        self.recorder.set_label_publisher(self.ros.publish_gaze_label)

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
            from ultralytics import YOLO

            if device is None:
                self.yolo_device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Loading YOLO model from {model_path} on device={self.yolo_device}")
            self.model = YOLO(str(model_path))

        self.draw_gaze = draw_gaze

        # Clean capture settings, for yolo training
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
        self._last_det_summary: list[tuple[str, float, float]] = []

    def _set_gaze(self, pitch: float, yaw: float) -> None:
        self.gaze_pitch = pitch
        self.gaze_yaw = yaw

    def record_frame(self, frame: np.ndarray) -> None:
        self.recorder.record_frame(frame)

    def shutdown(self) -> None:
        self.recorder.stop()
        self.ros.shutdown()

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
        self.recorder.stage_clean_frame(display_image)

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
        x_start = y_start = cs = 0
        if gaze_valid:
            cs = min(CROP_SIZE, w, h)
            x_start = max(0, min(gx - cs // 2, w - cs))
            y_start = max(0, min(gy - cs // 2, h - cs))
            crop_img = display_image[y_start:y_start + cs, x_start:x_start + cs].copy()
            gaze_pt = (gx, gy)

        if self.enable_capture and crop_img is not None:
            resized_crop = enhance_crop(crop_img)

        # YOLO inference on the cropped+resized region
        if self.model is not None and gaze_valid and crop_img is not None:
            if resized_crop is None:
                resized_crop = enhance_crop(crop_img)
                resized_crop = cv2.convertScaleAbs(resized_crop, alpha=1.2, beta=30)

            results = self.model(resized_crop, conf=self.conf_threshold, device=self.yolo_device, verbose=False)
            filter_results(results[0], self.filter_labels)

            scale = RESIZE_SIZE / cs
            gaze_in_crop = ((gx - x_start) * scale, (gy - y_start) * scale)

            if self.enable_capture:
                annotated = draw_circles(resized_crop, results[0])
                display_image[:] = cv2.resize(annotated, (w, h), interpolation=cv2.INTER_LINEAR)
                gaze_pt = (int((gx - x_start) * w / cs), int((gy - y_start) * h / cs))
            else:
                annotated = draw_circles(
                    display_image, results[0],
                    crop_transform=(x_start, y_start, scale),
                )
                np.copyto(display_image, annotated)

            det_summary = summarize_detections(
                results[0],
                gaze_in_crop,
                x_start,
                y_start,
                scale,
                self.dist_threshold,
                self.std_dist,
            )

            self._last_det_summary = sorted(det_summary, key=lambda x: x[2], reverse=True)
            crop_vis = draw_circles(resized_crop, results[0])
            vis_y = 18
            for det_label, _, det_score, _ in self._last_det_summary:
                cv2.putText(crop_vis, f"{det_label}  score={det_score:.2f}",
                            (6, vis_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
                vis_y += 18
            self.recorder.log_labels(det_summary, gaze_px=(gx, gy))
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
        # Top-left detection text: live per-frame scores while accumulating,
        # the final weighted /gaze_label result during the post-release tail,
        # empty otherwise.
        label_display = self.recorder.label_display
        if label_display:
            det_y = 60
            for entry in label_display:
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

        if self.draw_gaze and gaze_pt is not None:
            cv2.circle(display_image, gaze_pt, cross_size, color, 2)
            cv2.circle(display_image, gaze_pt, 4, color, -1)

        if self.enable_capture:
            capture_text = "CAPTURE: ON (press S to stop)" if self.capture_active else "CAPTURE: OFF (press S to start)"
            capture_color = (0, 255, 0) if self.capture_active else (0, 255, 255)
            cap_y = 60 + 22 * len(label_display)
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
