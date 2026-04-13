import argparse
import threading
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from utils.aria_rgb_stream import AriaRgbStream


class GazeOverlay:
    """Subscribes to /aria/gaze_euler and draws a crosshair on the display image."""

    def __init__(self, homography_path: Optional[Path] = None) -> None:
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
        # Display image has been cropped so we use its actual dimensions for bounds check.
        display_col = cx + fx * np.tan(self.gaze_yaw)
        display_row = cy - fy * np.tan(self.gaze_pitch)
        gaze_pt = (int(round(display_col)), int(round(display_row)))

        h, w = display_image.shape[:2]
        cross_size = 40
        color = (0, 0, 255)

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

        if 0 <= gaze_pt[0] < w and 0 <= gaze_pt[1] < h:
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
) -> None:
    overlay = GazeOverlay(homography_path=homography_path)
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
        description="Visualize gaze direction on Aria RGB stream."
    )
    parser.add_argument("--device-ip", help="IP address of the Aria device")
    parser.add_argument(
        "--update_iptables",
        default=False,
        action="store_true",
        help="Update iptables for DDS UDP stream (Linux only).",
    )
    parser.add_argument(
        "--homography", type=Path, default=None,
        help="Path to a homography matrix file (e.g. test_homography/homography.txt). "
             "If provided, the RGB image is warped before gaze overlay.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_gaze_rgb_visualizer(
        device_ip=args.device_ip,
        update_iptables_rules=args.update_iptables,
        homography_path=args.homography,
    )


if __name__ == "__main__":
    main()
