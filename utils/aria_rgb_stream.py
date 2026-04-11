"""
Shared Aria RGB streaming infrastructure.

AriaRgbStream handles device connection, calibration loading, undistortion,
and the display loop. Overlays are objects with a draw(display_image, camera_matrix)
method and are registered via add_overlay().

Overlays implement draw(display_image, camera_matrix), which is called after
rot90 + crop. camera_matrix has cx/cy adjusted to the cropped image origin.

Example usage (single feature):
    stream = AriaRgbStream(device_ip="192.168.x.x")
    stream.add_overlay(YoloOverlay(...))
    stream.run()

Example usage (combined):
    stream = AriaRgbStream(device_ip="192.168.x.x")
    stream.add_overlay(YoloOverlay(...))
    stream.add_overlay(GazeOverlay(...))
    stream.run()
"""

import sys
import time
from typing import Optional

import aria.sdk as aria
import cv2
import numpy as np

from utils.common import update_iptables
from projectaria_tools.core.calibration import (
    device_calibration_from_json_string,
    distort_by_calibration,
    get_linear_camera_calibration,
)
from projectaria_tools.core.sensor_data import ImageDataRecord


class AriaRgbStream:
    """Single RGB stream + display loop shared across all overlays."""

    def __init__(
        self,
        device_ip: Optional[str] = None,
        update_iptables_rules: bool = False,
        window_name: str = "Aria RGB",
        window_size: int = 1024,
    ) -> None:
        self.device_ip = device_ip
        self.update_iptables_rules = update_iptables_rules
        self.window_name = window_name
        self.window_size = window_size

        self.rgb_calib = None
        self.dst_calib = None
        self.camera_matrix: Optional[np.ndarray] = None

        self.streaming_client = None
        self.observer = None

        self._overlays: list = []

    def add_overlay(self, overlay) -> None:
        self._overlays.append(overlay)

    def run(self) -> None:
        if self.update_iptables_rules and sys.platform.startswith("linux"):
            update_iptables()

        self._load_rgb_calibration()

        aria.set_log_level(aria.Level.Info)
        self._setup_streaming()
        self._run_loop()

        print("Stop listening to RGB data")
        self.streaming_client.unsubscribe()

    def _run_loop(self) -> None:
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.window_size, self.window_size)
        cv2.setWindowProperty(self.window_name, cv2.WND_PROP_TOPMOST, 1)
        cv2.moveWindow(self.window_name, 50, 50)

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break

            if self.observer.rgb_image is None:
                time.sleep(0.001)
                continue

            if self.dst_calib is None:
                # Initialise calibration on first frame using actual streaming resolution.
                frame_h, frame_w = self.observer.rgb_image.shape[:2]
                self._setup_dst_calib(frame_w, frame_h)

            rgb_image = self._prepare_rgb_image(self.observer.rgb_image)
            self.observer.rgb_image = None

            # Aria RGB sensor is mounted rotated; rot90(-1) puts the image upright.
            display = np.ascontiguousarray(np.rot90(rgb_image, -1))
            h, w = display.shape[:2]
            # The undistorted image is square but rotated, so the inscribed axis-aligned
            # square has side = original_side / sqrt(2) ≈ original_side / 1.4143.
            crop_size = int(min(w, h) / 1.4143)
            ox = (w - crop_size) // 2
            oy = (h - crop_size) // 2
            display = np.ascontiguousarray(display[oy:oy + crop_size, ox:ox + crop_size])

            # Shift cx/cy so overlays can use camera_matrix directly in cropped image coordinates.
            display_matrix = self.camera_matrix.copy()
            display_matrix[0, 2] -= ox
            display_matrix[1, 2] -= oy

            for overlay in self._overlays:
                overlay.draw(display, display_matrix, key)

            cv2.imshow(self.window_name, display)

    def _prepare_rgb_image(self, bgr_image: np.ndarray) -> np.ndarray:
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        return distort_by_calibration(rgb_image, self.dst_calib, self.rgb_calib)

    def _load_rgb_calibration(self) -> None:
        # Connect only to read calibration JSON, then disconnect immediately.
        # Streaming is set up separately via StreamingClient.
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
            raise RuntimeError(f"Failed to load RGB calibration from device: {exc}") from exc
        finally:
            if device is not None:
                try:
                    device_client.disconnect(device)
                except Exception:
                    pass

    def _setup_dst_calib(self, stream_w: int, stream_h: int) -> None:
        calib_w, calib_h = self.rgb_calib.get_image_size()
        calib_focal = self.rgb_calib.get_focal_lengths()[0]
        # Scale focal length proportionally from calibration resolution to actual stream resolution.
        scale = stream_w / calib_w
        dst_focal = calib_focal * scale
        print(f"dst_calib: calib_res={calib_w}x{calib_h}, stream_res={stream_w}x{stream_h}, scale={scale:.4f}, focal={dst_focal:.2f} px")

        self.dst_calib = get_linear_camera_calibration(
            stream_w,
            stream_h,
            dst_focal,
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
