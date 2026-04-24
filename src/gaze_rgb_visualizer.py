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

from utils.aria_rgb_stream import AriaRgbStream

MODEL_PATH = Path(__file__).parent.parent / "yolo_model" / "mixed.pt"
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
        dist_threshold: float = 200.0,
        show_confidence: bool = False,
        saccade_threshold: float = 5.0,
        s_min: float = 0.3,
        gain: float = 3.0,
    ) -> None:
        self.gaze_pitch: float = 0.0
        self.gaze_yaw: float = 0.0
        self._rclpy = None
        self._ros_node = None
        self._ros_thread = None
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

        # Gaze proximity threshold (pixels): boxes beyond this radius are ignored
        self.dist_threshold = dist_threshold

        # Sticky-Glance confidence field (paper Algorithm 1)
        self.prev_gaze_raw: Optional[tuple[int, int]] = None  # (gx, gy) in display-image coords
        self._confidence: dict[int, float] = {}       # class_id -> c(t,i) in [0, 1]
        self._prev_obj_dist: dict[int, float] = {}    # class_id -> d(t-1) in detection coords
        self._last_frame_time: float = time.monotonic()
        self.show_confidence: bool = show_confidence
        self.saccade_threshold: float = saccade_threshold
        self.s_min: float = s_min
        self.gain: float = gain

    def _setup_ros_subscriber(self) -> None:
        try:
            import rclpy
            from geometry_msgs.msg import Vector3
            from std_msgs.msg import String

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
            self._gaze_label_pub = self._ros_node.create_publisher(String, "/gaze_label", 10)
            # daemon=True so the thread exits automatically when main exits.
            self._ros_thread = threading.Thread(
                target=rclpy.spin, args=(self._ros_node,), daemon=True
            )
            self._ros_thread.start()
            # print("ROS2 subscriber started: /aria/gaze_euler")
            print("ROS2 publisher started: /gaze_label")
        except Exception as exc:
            raise RuntimeError(f"ROS2 subscriber unavailable: {exc}") from exc

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

        now = time.monotonic()
        delta_t = min(now - self._last_frame_time, 0.1)  # cap at 100 ms
        self._last_frame_time = now

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

            results = self.model(resized_crop, conf=self.conf_threshold, device=self.yolo_device, verbose=False)
            self._filter_results(results[0])

            scale = RESIZE_SIZE / cs
            curr_in_resized = ((gx - x_start) * scale, (gy - y_start) * scale)
            prev_in_resized: Optional[tuple] = None
            if self.prev_gaze_raw is not None:
                prev_in_resized = (
                    (self.prev_gaze_raw[0] - x_start) * scale,
                    (self.prev_gaze_raw[1] - y_start) * scale,
                )
                self._update_confidence(results[0], prev_in_resized, curr_in_resized, delta_t)

            self._publish_closest_labels(results[0])

            if self.enable_capture:
                # Capture mode: show the enhanced crop so the user sees what is saved.
                annotated = self._draw_circles(resized_crop, results[0], prev_in_resized, curr_in_resized)
                display_image[:] = cv2.resize(annotated, (w, h), interpolation=cv2.INTER_LINEAR)
                gaze_pt = (int((gx - x_start) * w / cs), int((gy - y_start) * h / cs))
            else:
                # Normal mode: map detections back to original display image.
                annotated = self._draw_circles(
                    display_image, results[0],
                    self.prev_gaze_raw, (gx, gy),
                    crop_transform=(x_start, y_start, scale),
                )
                np.copyto(display_image, annotated)
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
        if self.draw_gaze and gaze_pt is not None:
            cv2.circle(display_image, gaze_pt, cross_size, color, 2)
            cv2.circle(display_image, gaze_pt, 4, color, -1)

        if self.enable_capture:
            capture_text = "CAPTURE: ON (press S to stop)" if self.capture_active else "CAPTURE: OFF (press S to start)"
            capture_color = (0, 255, 0) if self.capture_active else (0, 255, 255)
            cv2.putText(
                display_image,
                capture_text,
                (10, 60),
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

        # Persist gaze position for next-frame affinity computation.
        if gaze_valid:
            self.prev_gaze_raw = (gx, gy)

    # ------------------------------------------------------------------
    # Gaze-trajectory affinity helpers
    # ------------------------------------------------------------------

    def _tangent_info(
        self, px: float, py: float, cx: float, cy: float, r: float
    ) -> Optional[tuple]:
        """Return (t1_vec, t2_vec, touch1, touch2) for tangent lines from point P=(px,py)
        to circle (cx,cy,r), or None when P is inside the circle.
        t1/t2 are unit direction vectors; touch points are pixel coords on the circle."""
        dx, dy = cx - px, cy - py
        d = math.sqrt(dx * dx + dy * dy)
        if d <= r:
            return None
        tan_len = math.sqrt(d * d - r * r)
        ux, uy = dx / d, dy / d          # unit vec P → C
        sin_a, cos_a = r / d, tan_len / d
        # Rotate ux,uy by +alpha and -alpha
        t1 = (ux * cos_a - uy * sin_a, ux * sin_a + uy * cos_a)
        t2 = (ux * cos_a + uy * sin_a, -ux * sin_a + uy * cos_a)
        touch1 = (int(round(px + t1[0] * tan_len)), int(round(py + t1[1] * tan_len)))
        touch2 = (int(round(px + t2[0] * tan_len)), int(round(py + t2[1] * tan_len)))
        return t1, t2, touch1, touch2

    def _intersections_with_circle(
        self, p: tuple, q: tuple, cx: float, cy: float, r: float
    ) -> int:
        """Count intersections of segment PQ with circle (cx,cy,r). Returns 0, 1, or 2."""
        dx, dy = q[0] - p[0], q[1] - p[1]
        fx, fy = p[0] - cx, p[1] - cy
        a = dx * dx + dy * dy
        if a < 1e-9:
            return 0
        b = 2.0 * (fx * dx + fy * dy)
        c = fx * fx + fy * fy - r * r
        disc = b * b - 4.0 * a * c
        if disc < 0:
            return 0
        sq = math.sqrt(disc)
        return sum(1 for t in ((-b - sq) / (2 * a), (-b + sq) / (2 * a)) if 0.0 <= t <= 1.0)

    def _update_confidence(
        self,
        result,
        prev_pt: tuple[float, float],
        curr_pt: tuple[float, float],
        delta_t: float,
    ) -> None:
        """Sticky-Glance Algorithm 1: update per-object confidence c(t,i) in [0,1].

        Inside bounding circle: Gaussian e_dist (σ=r/2) instead of flat 1,
        so overlapping circles don't give equal evidence from both objects.
        Outside: paper Eq.1 (r/d)*σ_trend. Direction evidence: paper Eq.2."""
        vx = curr_pt[0] - prev_pt[0]
        vy = curr_pt[1] - prev_pt[1]
        delta_g = math.sqrt(vx * vx + vy * vy)

        detected_cids: set[int] = set()

        if result.boxes is not None and len(result.boxes) > 0:
            for xyxy, cid in zip(result.boxes.xyxy.tolist(), result.boxes.cls.int().tolist()):
                x1, y1, x2, y2 = xyxy
                ocx, ocy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                r = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) / 2.0
                detected_cids.add(cid)

                d_curr = math.sqrt((curr_pt[0] - ocx) ** 2 + (curr_pt[1] - ocy) ** 2)
                d_prev = self._prev_obj_dist.get(cid, d_curr)
                sigma_trend = 1.0 if d_prev > d_curr else (-1.0 if d_prev < d_curr else 1.0)

                # --- Distance evidence (e_dist) ---
                if d_curr <= r:
                    # Gaussian with σ=r/3: value at edge ≈ exp(-4.5) ≈ 0.011,
                    # at 0.7r ≈ 0.11 — tighter than r/2 to reduce overlap bleed
                    sigma_g = r / 3.0
                    e_dist = math.exp(-d_curr ** 2 / (2.0 * sigma_g ** 2))
                else:
                    e_dist = (r / d_curr) * sigma_trend  # Eq.1, in [-1, 1]

                # --- Directional evidence (e_dir) ---
                e_dir = 0.0
                if d_prev > r and delta_g > self.saccade_threshold:
                    # Case 1: prev outside, saccade → Eq.2
                    n = self._intersections_with_circle(prev_pt, curr_pt, ocx, ocy, r)
                    if n == 2:
                        e_dir = -1.0  # gaze already passed through object
                    else:
                        cos_theta = math.sqrt(max(0.0, 1.0 - (r / d_prev) ** 2))
                        if delta_g > 0 and d_curr > 0:
                            obj_dx, obj_dy = ocx - curr_pt[0], ocy - curr_pt[1]
                            cos_phi = (vx * obj_dx + vy * obj_dy) / (delta_g * d_curr)
                            cos_phi = max(-1.0, min(1.0, cos_phi))
                        else:
                            cos_phi = 0.0
                        delta_dir = 1.0 if cos_phi >= cos_theta else -1.0
                        denom = 1.0 - delta_dir * cos_theta
                        if abs(denom) > 1e-6:
                            e_dir = max(-1.0, min(1.0, (cos_phi - cos_theta) / denom))
                elif d_prev <= r and d_curr > r:
                    # Case 2b: gaze just exited circle → strong repulsion
                    e_dir = -1.0
                # Case 2a (fixation inside): e_dir = 0, Gaussian e_dist carries the signal

                e_i = e_dist + e_dir
                # Gate: only allow positive evidence within r + dist_threshold of the object
                # center, so each object's zone scales with its own bounding circle size.
                if d_curr > r + self.dist_threshold:
                    e_i = min(e_i, 0.0)
                c_prev = self._confidence.get(cid, 0.0)
                self._confidence[cid] = max(0.0, min(1.0, c_prev + self.gain * delta_t * e_i))
                self._prev_obj_dist[cid] = d_curr

        # Decay confidence for objects not detected this frame
        for cid in list(self._confidence.keys()):
            if cid not in detected_cids:
                self._confidence[cid] = max(0.0, self._confidence[cid] - 0.05 * delta_t)

    def _draw_circles(
        self, img: np.ndarray, result,
        prev_gaze_pt: Optional[tuple] = None,
        gaze_pt: Optional[tuple] = None,
        crop_transform: Optional[tuple] = None,
    ) -> np.ndarray:
        """Draw a circle per detection on img.

        crop_transform=(x_start, y_start, scale): when provided, box coordinates are in
        resized-crop space and are mapped back to display-image space before drawing.
        prev_gaze_pt and gaze_pt must already be in the same space as img (display coords)."""
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
            score = self._compute_score(cid)
            label = f"{names.get(cid, str(cid))} s={score:.2f}"
            cv2.circle(out, (cx, cy), radius, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(out, label, (cx - radius, max(cy - radius - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
            if self.show_confidence and prev_gaze_pt is not None:
                pg = (int(round(prev_gaze_pt[0])), int(round(prev_gaze_pt[1])))
                tg = self._tangent_info(pg[0], pg[1], cx, cy, radius)
                if tg is not None:
                    _, _, touch1, touch2 = tg
                    cv2.line(out, pg, touch1, (0, 255, 255), 1, cv2.LINE_AA)
                    cv2.line(out, pg, touch2, (0, 255, 255), 1, cv2.LINE_AA)
        return out

    def _filter_results(self, result) -> None:
        """Remove detections whose label is in self.filter_labels and keep only
        the highest-confidence detection per class (in-place)."""
        if result.boxes is None or len(result.boxes) == 0:
            return
        names = result.names  # {class_id: label_str}
        cls_ids = result.boxes.cls.int().tolist()
        confs = result.boxes.conf.tolist()

        # 1. Remove detections whose label is in filter_labels
        if self.filter_labels:
            keep = [i for i, cid in enumerate(cls_ids) if names.get(cid, "").lower() not in self.filter_labels]
            result.boxes = result.boxes[keep]
            if len(result.boxes) == 0:
                return
            cls_ids = result.boxes.cls.int().tolist()
            confs = result.boxes.conf.tolist()

        # 2. Keep only the highest-confidence detection per class
        best: dict[int, tuple[int, float]] = {}  # class_id -> (index, conf)
        for i, (cid, conf) in enumerate(zip(cls_ids, confs)):
            if cid not in best or conf > best[cid][1]:
                best[cid] = (i, conf)
        keep = [idx for idx, _ in best.values()]
        result.boxes = result.boxes[keep]

    def _compute_score(self, cid: int) -> float:
        return self._confidence.get(cid, 0.0)

    def _publish_closest_labels(self, result) -> None:
        """Publish {"label", "score"} for every detection sorted by confidence descending."""
        from std_msgs.msg import String
        if result.boxes is None or len(result.boxes) == 0:
            self._gaze_label_pub.publish(String(data=""))
            return

        names = result.names
        entries: list[dict] = []

        for cid in result.boxes.cls.int().tolist():
            score = self._compute_score(cid)
            if score < self.s_min:
                continue
            entries.append({"label": names.get(cid, str(cid)), "score": round(score, 4)})

        if not entries:
            self._gaze_label_pub.publish(String(data=""))
            return

        entries.sort(key=lambda x: -x["score"])
        msg = String()
        msg.data = json.dumps(entries)
        self._gaze_label_pub.publish(msg)

    def shutdown(self) -> None:
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
    dist_threshold: float = 200.0,
    show_confidence: bool = False,
    saccade_threshold: float = 5.0,
    s_min: float = 0.3,
    gain: float = 3.0,
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
        show_confidence=show_confidence,
        saccade_threshold=saccade_threshold,
        s_min=s_min,
        gain=gain,
    )
    stream = AriaRgbStream(
        device_ip=device_ip,
        update_iptables_rules=update_iptables_rules,
        window_name="Aria RGB Gaze",
    )
    stream.add_overlay(overlay)
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
    parser.add_argument("--yolo-conf", type=float, default=0.75, help="YOLO confidence threshold")
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
        "--dist-threshold", type=float, default=100.0,
        help="Extra pixel margin beyond each object's bounding circle radius within which "
             "positive evidence is allowed (gate = r + dist_threshold). Default: 100.",
    )
    parser.add_argument(
        "--show-confidence",
        default=False,
        action="store_true",
        help="Visualize gaze-trajectory confidence: draw line from g_{t-1} to each detection center.",
    )
    parser.add_argument(
        "--saccade-threshold", type=float, default=20.0,
        help="Min gaze displacement (pixels) to treat as a saccade and compute e_dir (default: 5).",
    )
    parser.add_argument(
        "--s-min", "--c-min", type=float, default=0.2, dest="s_min",
        help="Minimum score threshold for publishing /gaze_label entries (default: 0.2).",
    )
    parser.add_argument(
        "--gain", type=float, default=2.0,
        help="Evidence gain multiplier. Higher values make confidence respond more strongly to each frame's evidence (default: 2.0).",
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
        show_confidence=args.show_confidence,
        saccade_threshold=args.saccade_threshold,
        s_min=args.s_min,
        gain=args.gain,
    )


if __name__ == "__main__":
    main()
