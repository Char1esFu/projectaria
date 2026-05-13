import argparse
import json
import math
import threading
import time
from pathlib import Path
from typing import Optional

FPS_WARMUP_FRAMES = 10  # frames used to estimate actual stream fps before opening VideoWriter

import cv2
import numpy as np
from ultralytics import YOLO

from utils.aria_rgb_stream import AriaRgbStream

MODEL_PATH = Path(__file__).parent.parent / "yolo_model" / "best_aria.pt"
CROP_SIZE = 200
RESIZE_SIZE = 1080


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
    ) -> None:
        self.gaze_pitch: float = 0.0
        self.gaze_yaw: float = 0.0
        self._rclpy = None
        self._ros_node = None
        self._ros_thread = None

        # Video recording state
        self._participant = participant
        self._video_writer: Optional[cv2.VideoWriter] = None
        self._writer_lock = threading.Lock()
        self._pending_record = False
        self._pending_path: Optional[Path] = None
        self._warmup_buf: Optional[list] = None  # list of (frame, timestamp) during fps estimation

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

    def _setup_ros_subscriber(self) -> None:
        try:
            import rclpy
            from geometry_msgs.msg import Vector3
            from std_msgs.msg import Empty, String

            if not rclpy.ok():
                rclpy.init(args=None)
            self._rclpy = rclpy
            self._ros_node = rclpy.create_node("gaze_rgb_visualizer")

            def _gaze_callback(msg: Vector3) -> None:
                self.gaze_pitch = float(msg.x)
                self.gaze_yaw = float(msg.y)

            self._ros_node.create_subscription(
                Vector3, "/aria/gaze_euler", _gaze_callback, 10
            )
            self._ros_node.create_subscription(
                Empty, "/recording/start", lambda _: self._on_recording_start(), 10
            )
            self._ros_node.create_subscription(
                String, "/transcription", lambda _: self._stop_recording(), 10
            )
            self._gaze_label_pub = self._ros_node.create_publisher(String, "/gaze_label", 10)
            self._ros_thread = threading.Thread(
                target=rclpy.spin, args=(self._ros_node,), daemon=True
            )
            self._ros_thread.start()
            print("ROS2 publisher started: /gaze_label")
        except Exception as exc:
            raise RuntimeError(f"ROS2 subscriber unavailable: {exc}") from exc

    def _on_recording_start(self) -> None:
        if not self._participant:
            return
        base = Path("recordings") / self._participant
        base.mkdir(parents=True, exist_ok=True)
        existing = sorted([int(p.name) for p in base.iterdir() if p.is_dir() and p.name.isdigit()])
        if existing:
            session_dir = base / f"{existing[-1]:02d}"
        else:
            session_dir = base / "01"
            session_dir.mkdir()
        with self._writer_lock:
            self._pending_path = session_dir / "gaze_overlay.mp4"
            self._pending_record = True

    def record_frame(self, frame: np.ndarray) -> None:
        now = time.monotonic()

        with self._writer_lock:
            # START signal received: reset state and begin fps warmup
            if self._pending_record:
                self._pending_record = False
                if self._video_writer is not None:
                    self._video_writer.release()
                    self._video_writer = None
                self._warmup_buf = [(frame, now)]
                return

            # Warmup: collect frames until we have enough to estimate fps
            if self._warmup_buf is not None:
                self._warmup_buf.append((frame, now))
                if len(self._warmup_buf) < FPS_WARMUP_FRAMES:
                    return
                # Estimate fps from the warmup window and open VideoWriter
                t0, t1 = self._warmup_buf[0][1], self._warmup_buf[-1][1]
                fps = (len(self._warmup_buf) - 1) / (t1 - t0)
                h, w = frame.shape[:2]
                # Codecs require even dimensions; trim 1px if odd (visually imperceptible)
                self._rec_h = h - h % 2
                self._rec_w = w - w % 2
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                self._video_writer = cv2.VideoWriter(
                    str(self._pending_path), fourcc, fps, (self._rec_w, self._rec_h)
                )
                for f, _ in self._warmup_buf:
                    self._video_writer.write(f[:self._rec_h, :self._rec_w])
                print(f"Video recording started (fps≈{fps:.1f}) → {self._pending_path}")
                self._warmup_buf = None
                self._pending_path = None
                return

            if self._video_writer is not None:
                self._video_writer.write(frame[:self._rec_h, :self._rec_w])

    def _stop_recording(self) -> None:
        with self._writer_lock:
            self._warmup_buf = None
            self._pending_path = None
            if self._video_writer is None:
                return
            self._video_writer.release()
            self._video_writer = None
            print("Video recording stopped.")

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
                    det_summary.append((names_map.get(cid, str(cid)), conf, score))

            self._last_det_summary = sorted(det_summary, key=lambda x: x[2], reverse=True)
            crop_vis = self._draw_circles(resized_crop, results[0])
            vis_y = 18
            for det_label, _, det_score in self._last_det_summary:
                cv2.putText(crop_vis, f"{det_label}  score={det_score:.2f}",
                            (6, vis_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
                vis_y += 18
            self._publish_labels(det_summary)
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
        if self._last_det_summary:
            det_y = 60
            for det_label, _, det_score in self._last_det_summary:
                cv2.putText(
                    display_image,
                    f"{det_label}  score={det_score:.2f}",
                    (10, det_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
                det_y += 22

        if crop_vis is not None:
            cv2.imshow("YOLO Crop", crop_vis)

        if self.draw_gaze and gaze_pt is not None:
            cv2.circle(display_image, gaze_pt, cross_size, color, 2)
            cv2.circle(display_image, gaze_pt, 4, color, -1)

        if self.enable_capture:
            capture_text = "CAPTURE: ON (press S to stop)" if self.capture_active else "CAPTURE: OFF (press S to start)"
            capture_color = (0, 255, 0) if self.capture_active else (0, 255, 255)
            cap_y = 60 + 22 * len(self._last_det_summary)
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

    def _publish_labels(self, det_summary: list[tuple[str, float, float]]) -> None:
        """Publish {"label", "score"} for detections with score >= s_min, sorted descending."""
        from std_msgs.msg import String
        entries = [
            {"label": label, "score": round(score, 4)}
            for label, _, score in det_summary
            if score >= self.s_min
        ]
        entries.sort(key=lambda x: -x["score"])
        msg = String()
        msg.data = json.dumps(entries) if entries else ""
        self._gaze_label_pub.publish(msg)

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
    parser.add_argument("--device-ip", help="IP address of the Aria device")
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
    )


if __name__ == "__main__":
    main()
