import argparse
import threading
from typing import Optional

import cv2
import numpy as np

from utils.aria_rgb_stream import AriaRgbStream


class GazeOverlay:
    """Subscribes to /aria/gaze_euler and draws a crosshair on the display image."""

    GAZE_ORIGIN_X_OFFSET: float = 0.05

    def __init__(self) -> None:
        self.gaze_pitch: float = 0.0
        self.gaze_yaw: float = 0.0
        self._rclpy = None
        self._ros_node = None
        self._ros_thread = None
        self._setup_ros_subscriber()

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
            self._ros_thread = threading.Thread(
                target=rclpy.spin, args=(self._ros_node,), daemon=True
            )
            self._ros_thread.start()
            print("ROS2 subscriber started: /aria/gaze_euler")
        except Exception as exc:
            raise RuntimeError(f"ROS2 subscriber unavailable: {exc}") from exc

    def draw(self, display_image: np.ndarray, camera_matrix: Optional[np.ndarray]) -> None:
        if camera_matrix is None:
            return

        fx = camera_matrix[0, 0]
        fy = camera_matrix[1, 1]
        cx = camera_matrix[0, 2]
        cy = camera_matrix[1, 2]

        # After rot90(-1): raw col → display row (pitch), raw row → display col (yaw).
        # Display image has been cropped so we use its actual dimensions for bounds check.
        display_col = cx + fx * np.tan(self.gaze_yaw) + fx * self.GAZE_ORIGIN_X_OFFSET
        display_row = cy - fy * np.tan(self.gaze_pitch)
        gaze_pt = (int(round(display_col)), int(round(display_row)))

        h, w = display_image.shape[:2]
        cross_size = max(20, min(w, h) // 30)
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
            cv2.drawMarker(
                display_image,
                gaze_pt,
                color,
                markerType=cv2.MARKER_CROSS,
                markerSize=cross_size * 2,
                thickness=2,
            )
            cv2.circle(display_image, gaze_pt, cross_size, color, 2)

    def shutdown(self) -> None:
        try:
            self._ros_node.destroy_node()
            self._rclpy.shutdown()
        except Exception:
            pass


def run_gaze_rgb_visualizer(
    device_ip: Optional[str] = None,
    update_iptables_rules: bool = False,
    undistort_width: int = 1408,
    undistort_height: int = 1408,
) -> None:
    overlay = GazeOverlay()
    stream = AriaRgbStream(
        device_ip=device_ip,
        update_iptables_rules=update_iptables_rules,
        undistort_width=undistort_width,
        undistort_height=undistort_height,
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
    parser.add_argument("--undistort-width", type=int, default=1408)
    parser.add_argument("--undistort-height", type=int, default=1408)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_gaze_rgb_visualizer(
        device_ip=args.device_ip,
        update_iptables_rules=args.update_iptables,
        undistort_width=args.undistort_width,
        undistort_height=args.undistort_height,
    )


if __name__ == "__main__":
    main()
