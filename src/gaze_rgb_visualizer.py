import argparse
import sys
import time
from typing import Optional

import aria.sdk as aria
import cv2
import numpy as np

from utils.common import quit_keypress, update_iptables
from projectaria_tools.core.calibration import (
    device_calibration_from_json_string,
    distort_by_calibration,
    get_linear_camera_calibration,
)
from projectaria_tools.core.sensor_data import ImageDataRecord


class GazeRgbVisualizer:
    def __init__(
        self,
        device_ip: Optional[str],
        update_iptables_rules: bool,
        undistort_width: int,
        undistort_height: int,
        undistort_focal_length: float,
    ) -> None:
        self.device_ip = device_ip
        self.update_iptables_rules = update_iptables_rules
        self.undistort_width = undistort_width
        self.undistort_height = undistort_height
        self.undistort_focal_length = undistort_focal_length

        self.rgb_calib = None
        self.dst_calib = None
        self.camera_matrix = None

        # Latest gaze from ROS2 (pitch, yaw in radians)
        self.gaze_pitch: float = 0.0
        self.gaze_yaw: float = 0.0

        self.streaming_client = None
        self.observer = None

    def run(self) -> None:
        if self.update_iptables_rules and sys.platform.startswith("linux"):
            update_iptables()

        self._load_rgb_calibration()
        self._setup_dst_calib()

        aria.set_log_level(aria.Level.Info)

        self._setup_ros_subscriber()
        self._setup_streaming()
        self._run_loop()

        print("Stop listening to RGB data")
        self.streaming_client.unsubscribe()
        self._shutdown_ros()

    def _run_loop(self) -> None:
        window_name = "Aria RGB Gaze"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1024, 1024)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
        cv2.moveWindow(window_name, 50, 50)

        while not quit_keypress():
            if self.observer.rgb_image is None:
                time.sleep(0.001)
                continue

            rgb_image = self._prepare_rgb_image(self.observer.rgb_image)
            self.observer.rgb_image = None

            # Rotate first, then draw — so text and crosshair are in display coordinates
            # np.ascontiguousarray is required because rot90 returns a non-contiguous view
            # and OpenCV draw functions require contiguous memory layout
            display_image = np.ascontiguousarray(np.rot90(rgb_image, -1))
            self._draw_gaze(display_image)

            # Crop to the inscribed square of the fisheye circle to remove black corners.
            # For a square image of side S, the inscribed square has side = S / sqrt(2).
            # The crop is symmetric so cx/cy remain at the center of the cropped image.
            h, w = display_image.shape[:2]
            crop_size = int(min(w, h) / 1.4143)  # ≈ 1/sqrt(2), slightly conservative
            ox = (w - crop_size) // 2
            oy = (h - crop_size) // 2
            display_image = display_image[oy:oy + crop_size, ox:ox + crop_size]

            cv2.putText(
                display_image,
                f"pitch={np.degrees(self.gaze_pitch):.1f}  yaw={np.degrees(self.gaze_yaw):.1f} deg",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(window_name, display_image)

    def _prepare_rgb_image(self, bgr_image: np.ndarray) -> np.ndarray:
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        if self.rgb_calib is not None and self.dst_calib is not None:
            rgb_image = distort_by_calibration(rgb_image, self.dst_calib, self.rgb_calib)
        return rgb_image

    # Gaze ray origin not the optical center.
    # Camera X maps to display row (after rot90), so the projection shifts by
    GAZE_ORIGIN_X_OFFSET: float = 0.05

    def _draw_gaze(self, display_image: np.ndarray) -> None:
        """Project gaze (pitch, yaw) onto the display image (already rot90'd) and draw crosshair.

        The Aria RGB camera is physically rotated 90°. After np.rot90(img, -1):
          raw col (u) → display row   →  vertical axis  → pitch
          raw row (v) → display col   →  horizontal axis → yaw

        The gaze ray origin is offset GAZE_ORIGIN_X_OFFSET in camera X (→ display row):
          display_col = cx + fx * tan(yaw)
          display_row = cy - fy * tan(pitch) + fy * GAZE_ORIGIN_X_OFFSET
        """
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]

        display_col = cx + fx * np.tan(self.gaze_yaw) + fx * self.GAZE_ORIGIN_X_OFFSET
        display_row = cy - fy * np.tan(self.gaze_pitch) 

        gaze_pt = (int(round(display_col)), int(round(display_row)))

        h, w = display_image.shape[:2]
        cross_size = max(20, min(w, h) // 30)
        thickness = 2
        color = (0, 0, 255)  # red when displayed by cv2.imshow (BGR interpretation of RGB array)

        # Only draw if within image bounds
        if 0 <= gaze_pt[0] < w and 0 <= gaze_pt[1] < h:
            cv2.drawMarker(
                display_image,
                gaze_pt,
                color,
                markerType=cv2.MARKER_CROSS,
                markerSize=cross_size * 2,
                thickness=thickness,
            )
            cv2.circle(display_image, gaze_pt, cross_size, color, thickness)


    def _load_rgb_calibration(self) -> None:
        device_client = aria.DeviceClient()
        client_config = aria.DeviceClientConfig()
        if self.device_ip:
            client_config.ip_v4_address = self.device_ip
        device_client.set_client_config(client_config)
        device = None
        try:
            device = device_client.connect()
            sensors_calib_json = device.streaming_manager.sensors_calibration()
            sensors_calib = device_calibration_from_json_string(sensors_calib_json)
            self.rgb_calib = sensors_calib.get_camera_calib("camera-rgb")
            print("RGB calibration loaded from device.")
        except Exception as exc:
            print(
                f"Warning: could not load RGB calibration from device ({exc}). "
                "Undistortion will be skipped; raw fisheye image will be used."
            )
            self.rgb_calib = None
        finally:
            if device is not None:
                try:
                    device_client.disconnect(device)
                except Exception:
                    pass

    def _setup_dst_calib(self) -> None:
        if self.rgb_calib is not None:
            # Scale focal length from device calibration to output resolution.
            # cx/cy must stay at image center (output_width/2, output_height/2) because
            # dst_calib defines where the optical axis lands in the OUTPUT image — always center.
            # The fisheye principal point offset is a src_calib property and is handled
            # internally by distort_by_calibration; it must NOT be propagated to dst_calib.
            src_w, src_h = self.rgb_calib.get_image_size()
            if self.undistort_width != self.undistort_height or src_w != src_h:
                raise ValueError(
                    f"Non-square images are not supported (src={src_w}x{src_h}, "
                    f"dst={self.undistort_width}x{self.undistort_height}). "
                    "get_linear_camera_calibration requires fx==fy."
                )
            src_focal = self.rgb_calib.get_focal_lengths()[0]
            focal_length = src_focal * self.undistort_width / src_w
            print(f"dst_calib: focal_length={focal_length:.2f} px ({src_focal:.2f} * {self.undistort_width}/{src_w})")
        else:
            focal_length = self.undistort_focal_length
            print(f"dst_calib: focal_length={focal_length:.2f} px (fallback)")

        self.dst_calib = get_linear_camera_calibration(
            self.undistort_width,
            self.undistort_height,
            focal_length,
            "camera-rgb",
        )
        fx, fy = self.dst_calib.get_focal_lengths()
        cx, cy = self.dst_calib.get_principal_point()
        self.camera_matrix = np.array(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64
        )

    def _setup_ros_subscriber(self) -> None:
        try:
            import rclpy
            from geometry_msgs.msg import Vector3

            rclpy.init(args=None)
            self._rclpy = rclpy
            self._ros_node = rclpy.create_node("gaze_rgb_visualizer")

            def _gaze_callback(msg: Vector3) -> None:
                self.gaze_pitch = float(msg.x)
                self.gaze_yaw = float(msg.y)

            self._ros_node.create_subscription(
                Vector3, "/aria/gaze_euler", _gaze_callback, 10
            )

            import threading
            self._ros_thread = threading.Thread(
                target=rclpy.spin, args=(self._ros_node,), daemon=True
            )
            self._ros_thread.start()
            print("ROS2 subscriber started: /aria/gaze_euler")
        except Exception as exc:
            raise RuntimeError(f"ROS2 subscriber unavailable: {exc}") from exc

    def _setup_streaming(self) -> None:
        self.streaming_client = aria.StreamingClient()
        config = self.streaming_client.subscription_config
        config.subscriber_data_type = aria.StreamingDataType.Rgb
        config.message_queue_size[aria.StreamingDataType.Rgb] = 1
        options = aria.StreamingSecurityOptions()
        options.use_ephemeral_certs = True
        config.security_options = options
        self.streaming_client.subscription_config = config

        class StreamingClientObserver:
            def __init__(self):
                self.rgb_image = None

            def on_image_received(self, image: np.ndarray, record: ImageDataRecord):
                if record.camera_id != aria.CameraId.Rgb:
                    return
                self.rgb_image = image

        self.observer = StreamingClientObserver()
        self.streaming_client.set_streaming_client_observer(self.observer)
        print("Start listening to RGB data")
        self.streaming_client.subscribe()

    def _shutdown_ros(self) -> None:
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
    undistort_focal_length: float = 450.0,
) -> None:
    visualizer = GazeRgbVisualizer(
        device_ip=device_ip,
        update_iptables_rules=update_iptables_rules,
        undistort_width=undistort_width,
        undistort_height=undistort_height,
        undistort_focal_length=undistort_focal_length,
    )
    visualizer.run()


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
        "--undistort-width", type=int, default=1408,
        help="Width of undistorted RGB output image.",
    )
    parser.add_argument(
        "--undistort-height", type=int, default=1408,
        help="Height of undistorted RGB output image.",
    )
    parser.add_argument(
        "--undistort-focal-length", type=float, default=450.0,
        help="Focal length for the undistorted linear camera calibration.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_gaze_rgb_visualizer(
        device_ip=args.device_ip,
        update_iptables_rules=args.update_iptables,
        undistort_width=args.undistort_width,
        undistort_height=args.undistort_height,
        undistort_focal_length=args.undistort_focal_length,
    )


if __name__ == "__main__":
    main()
