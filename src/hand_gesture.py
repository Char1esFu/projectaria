"""
Hand gesture detection on Aria RGB stream using MediaPipe.
Estimates each hand's 6-DOF pose (position + orientation) relative to the camera
by combining hand_world_landmarks (3D metric) with hand_landmarks (2D image) via solvePnP.

Usage:
    python3 -m src.hand_gesture --device-ip 192.168.8.117
"""

import argparse
import sys
import time
from typing import Optional

import aria.sdk as aria
import cv2
import mediapipe as mp
import numpy as np

from projectaria_tools.core.calibration import (
    device_calibration_from_json_string,
    distort_by_calibration,
    get_linear_camera_calibration,
)
from projectaria_tools.core.sensor_data import ImageDataRecord
from utils.common import quit_keypress, update_iptables


class HandGestureDetector:
    def __init__(
        self,
        device_ip: Optional[str],
        update_iptables_rules: bool,
        undistort_width: int,
        undistort_height: int,
        max_hands: int,
        detection_confidence: float,
        tracking_confidence: float,
    ) -> None:
        self.device_ip = device_ip
        self.update_iptables_rules = update_iptables_rules
        self.undistort_width = undistort_width
        self.undistort_height = undistort_height

        self.rgb_calib = None
        self.dst_calib = None
        self.camera_matrix = None
        self.dist_coeffs = np.zeros((5, 1), dtype=np.float64)  # image already undistorted

        self.streaming_client = None
        self.observer = None

        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )

    def run(self) -> None:
        if self.update_iptables_rules and sys.platform.startswith("linux"):
            update_iptables()

        self._load_rgb_calibration()
        self._setup_dst_calib()
        aria.set_log_level(aria.Level.Info)
        self._setup_streaming()
        self._run_loop()

        print("Stop listening to RGB data")
        self.streaming_client.unsubscribe()
        self.hands.close()

    def _run_loop(self) -> None:
        window_name = "Aria RGB Hand Gesture"
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

            self._draw_hands(rgb_image)

            # Rotate CW 90°; ascontiguousarray required for OpenCV on rot90 views
            display = np.ascontiguousarray(np.rot90(rgb_image, -1))

            # Crop inscribed square to remove fisheye black border (same as gaze_rgb_visualizer.py)
            h, w = display.shape[:2]
            crop_size = int(min(w, h) / 1.4143)
            ox = (w - crop_size) // 2
            oy = (h - crop_size) // 2
            display = display[oy:oy + crop_size, ox:ox + crop_size]

            cv2.imshow(window_name, display)

    def _prepare_rgb_image(self, bgr_image: np.ndarray) -> np.ndarray:
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        if self.rgb_calib is not None and self.dst_calib is not None:
            rgb_image = distort_by_calibration(rgb_image, self.dst_calib, self.rgb_calib)
        return rgb_image

    def _draw_hands(self, rgb_image: np.ndarray) -> None:
        """Detect hands, draw skeleton, and estimate 6-DOF pose via solvePnP."""
        results = self.hands.process(rgb_image)
        if not results.multi_hand_landmarks:
            return

        h, w = rgb_image.shape[:2]

        for hand_landmarks, hand_world_landmarks, handedness in zip(
            results.multi_hand_landmarks,
            results.multi_hand_world_landmarks,
            results.multi_handedness,
        ):
            # Draw skeleton
            self.mp_drawing.draw_landmarks(
                rgb_image,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style(),
            )

            # 2D pixel coords from normalized hand_landmarks (in unrotated image space,
            # matching the camera_matrix defined for the undistorted output image).
            pts_2d = np.array(
                [[lm.x * w, lm.y * h] for lm in hand_landmarks.landmark],
                dtype=np.float64,
            )

            # 3D metric coords from hand_world_landmarks (origin = wrist, unit = meters).
            pts_3d = np.array(
                [[lm.x, lm.y, lm.z] for lm in hand_world_landmarks.landmark],
                dtype=np.float64,
            )
            print(pts_3d)
            # solvePnP: find rvec/tvec that maps hand_world frame -> camera frame.
            # tvec = wrist position in camera coordinates (meters).
            success, rvec, tvec = cv2.solvePnP(
                pts_3d,
                pts_2d,
                self.camera_matrix,
                self.dist_coeffs,
                flags=cv2.SOLVEPNP_SQPNP,
            )
            if not success:
                continue

            # Draw 3D coordinate axes at the wrist (5 cm arms)
            cv2.drawFrameAxes(
                rgb_image, self.camera_matrix, self.dist_coeffs, rvec, tvec, 0.05
            )

            # Label: hand side + distance from camera
            label = handedness.classification[0].label
            dist_m = float(np.linalg.norm(tvec))
            wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
            px, py = int(wrist.x * w), int(wrist.y * h)
            cv2.putText(
                rgb_image,
                f"{label}  {dist_m:.2f} m",
                (px, py - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )

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
            raise RuntimeError(f"Failed to load RGB calibration: {exc}") from exc
        finally:
            if device is not None:
                try:
                    device_client.disconnect(device)
                except Exception:
                    pass

    def _setup_dst_calib(self) -> None:
        # Scale focal length from device calibration to the chosen output resolution,
        # matching the approach in gaze_rgb_visualizer.py.
        if self.rgb_calib is not None:
            src_w, _ = self.rgb_calib.get_image_size()
            src_focal = self.rgb_calib.get_focal_lengths()[0]
            focal_length = src_focal * self.undistort_width / src_w
            print(
                f"dst_calib: focal_length={focal_length:.2f} px "
                f"({src_focal:.2f} * {self.undistort_width}/{src_w})"
            )
        else:
            raise RuntimeError(
                "Unable to compute destination calibration: RGB device calibration is unavailable, "
                "so focal_length could not be read from the device."
            )

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
        print(f"camera_matrix: fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}")

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MediaPipe hand gesture on Aria RGB stream")
    parser.add_argument("--device-ip", help="IP address of the Aria device")
    parser.add_argument(
        "--update_iptables",
        default=True,
        action="store_true",
        help="Update iptables for DDS UDP stream (Linux only).",
    )
    parser.add_argument(
        "--undistort-width", type=int, default=1408,
        help="Width of the undistorted RGB output image.",
    )
    parser.add_argument(
        "--undistort-height", type=int, default=1408,
        help="Height of the undistorted RGB output image.",
    )
    parser.add_argument(
        "--max-hands", type=int, default=2,
        help="Maximum number of hands to detect.",
    )
    parser.add_argument(
        "--detection-confidence", type=float, default=0.5,
        help="Minimum hand detection confidence.",
    )
    parser.add_argument(
        "--tracking-confidence", type=float, default=0.5,
        help="Minimum hand tracking confidence.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.update_iptables and sys.platform.startswith("linux"):
        update_iptables()
    detector = HandGestureDetector(
        device_ip=args.device_ip,
        update_iptables_rules=False,
        undistort_width=args.undistort_width,
        undistort_height=args.undistort_height,
        max_hands=args.max_hands,
        detection_confidence=args.detection_confidence,
        tracking_confidence=args.tracking_confidence,
    )
    detector.run()


if __name__ == "__main__":
    main()
