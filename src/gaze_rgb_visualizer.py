import argparse
import threading
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO

from utils.aria_rgb_stream import AriaRgbStream

MODEL_PATH = Path(__file__).parent.parent / "yolo_model" / "best.pt"


class GazeOverlay:
    """Subscribes to /aria/gaze_euler and draws a crosshair on the display image."""

    def __init__(
        self,
        homography_path: Optional[Path] = None,
        model_path: Optional[Path] = None,
        conf_threshold: float = 0.25,
        device: Optional[str] = None,
        crop: bool = False,
        crop_size: int = 480,
        resize_size: int = 1920,
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
        self.conf_threshold = conf_threshold
        self.yolo_device = device
        if model_path is not None:
            import torch
            if device is None:
                self.yolo_device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Loading YOLO model from {model_path} on device={self.yolo_device}")
            self.model = YOLO(str(model_path))

        # Crop settings
        self.crop = crop
        self.crop_size = crop_size
        self.resize_size = resize_size

    def _setup_ros_subscriber(self) -> None:
        try:
            import rclpy
            from geometry_msgs.msg import Vector3

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
            # daemon=True so the thread exits automatically when main exits.
            self._ros_thread = threading.Thread(
                target=rclpy.spin, args=(self._ros_node,), daemon=True
            )
            self._ros_thread.start()
            print("ROS2 subscriber started: /aria/gaze_euler")
        except Exception as exc:
            raise RuntimeError(f"ROS2 subscriber unavailable: {exc}") from exc

    def draw(self, display_image: np.ndarray, camera_matrix: Optional[np.ndarray], key: int = -1) -> None:
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

        # YOLO inference (prerequisite: valid gaze point)
        gaze_pt = None
        if self.model is not None and gaze_valid:
            if self.crop:
                # Crop square around gaze point, clamped to image bounds
                cs = min(self.crop_size, w, h)
                x_start = max(0, min(gx - cs // 2, w - cs))
                y_start = max(0, min(gy - cs // 2, h - cs))
                crop_img = display_image[y_start:y_start + cs, x_start:x_start + cs].copy()

                # Resize for YOLO
                resized = cv2.resize(crop_img, (self.resize_size, self.resize_size), interpolation=cv2.INTER_LINEAR)

                # Denoise: bilateral filter (preserve edges)
                resized = cv2.bilateralFilter(resized, d=5, sigmaColor=15, sigmaSpace=15)
                # Sharpen: unsharp mask
                blurred = cv2.GaussianBlur(resized, (0, 0), sigmaX=5)
                resized = cv2.addWeighted(resized, 5, blurred, -4, 0)

                # YOLO inference on enhanced crop
                results = self.model(resized, conf=self.conf_threshold, device=self.yolo_device, verbose=False)
                annotated = results[0].plot()

                # Resize annotated back to display image size
                final = cv2.resize(annotated, (w, h), interpolation=cv2.INTER_LINEAR)
                display_image[:] = final

                # Gaze crosshair in crop-relative coords, scaled to display
                gaze_pt = (
                    int(round((gx - x_start) * w / cs)),
                    int(round((gy - y_start) * h / cs)),
                )
            else:
                # No crop: YOLO on full image
                results = self.model(display_image, conf=self.conf_threshold, device=self.yolo_device, verbose=False)
                np.copyto(display_image, results[0].plot())
                gaze_pt = (gx, gy)
        elif gaze_valid:
            gaze_pt = (gx, gy)

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
        if gaze_pt is not None:
            cv2.circle(display_image, gaze_pt, cross_size, color, 2)
            cv2.circle(display_image, gaze_pt, 4, color, -1)

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
    model_path: Optional[Path] = None,
    conf_threshold: float = 0.25,
    device: Optional[str] = None,
    crop: bool = False,
    crop_size: int = 480,
    resize_size: int = 1920,
) -> None:
    overlay = GazeOverlay(
        homography_path=homography_path,
        model_path=model_path,
        conf_threshold=conf_threshold,
        device=device,
        crop=crop,
        crop_size=crop_size,
        resize_size=resize_size,
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
    parser.add_argument("--model", type=str, default=str(MODEL_PATH), help="YOLO model path")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument("--device", type=str, default=None, help="Inference device: cuda / cpu / mps (default: auto)")
    parser.add_argument(
        "--crop", default=False, action="store_true",
        help="Enable gaze-centered crop+resize before YOLO inference.",
    )
    parser.add_argument("--crop-size", type=int, default=200, help="Side length of square crop around gaze point (pixels)")
    parser.add_argument("--resize-size", type=int, default=1920, help="Resize crop to this size before YOLO inference")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_gaze_rgb_visualizer(
        device_ip=args.device_ip,
        update_iptables_rules=args.update_iptables,
        homography_path=args.homography,
        model_path=Path(args.model),
        conf_threshold=args.conf,
        device=args.device,
        crop=args.crop,
        crop_size=args.crop_size,
        resize_size=args.resize_size,
    )


if __name__ == "__main__":
    main()
