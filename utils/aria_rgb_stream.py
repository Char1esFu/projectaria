"""
Shared Aria RGB streaming infrastructure.

AriaRgbStream handles device connection, calibration loading, undistortion,
and the display loop. Feature-specific logic is implemented as RgbOverlay
subclasses and registered via add_overlay().

Overlay draw hooks:
  draw(rgb_image, camera_matrix)         -- called on the pre-rotation RGB image
  draw_display(display_image, camera_matrix) -- called after rot90 + crop

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

from utils.common import quit_keypress, update_iptables
from projectaria_tools.core.calibration import (
    device_calibration_from_json_string,
    distort_by_calibration,
    get_linear_camera_calibration,
)
from projectaria_tools.core.sensor_data import ImageDataRecord


class RgbOverlay:
    """Base class for overlays drawn on the Aria RGB stream.

    Subclasses override draw() and/or draw_display().
    Set requires_calibration = True to make AriaRgbStream raise at startup
    if calibration could not be loaded from the device.
    """

    requires_calibration: bool = False

    def draw(self, rgb_image: np.ndarray, camera_matrix: Optional[np.ndarray]) -> None:
        """Draw on the pre-rotation undistorted RGB image (camera coordinate space)."""

    def draw_display(self, display_image: np.ndarray, camera_matrix: Optional[np.ndarray]) -> None:
        """Draw on the post-rotation cropped display image (display coordinate space)."""


class AriaRgbStream:
    """Single RGB stream + display loop shared across all overlays."""

    def __init__(
        self,
        device_ip: Optional[str] = None,
        update_iptables_rules: bool = False,
        undistort_width: int = 1408,
        undistort_height: int = 1408,
        undistort_focal_length: float = 450.0,
        window_name: str = "Aria RGB",
        window_size: int = 1024,
    ) -> None:
        self.device_ip = device_ip
        self.update_iptables_rules = update_iptables_rules
        self.undistort_width = undistort_width
        self.undistort_height = undistort_height
        self.undistort_focal_length = undistort_focal_length
        self.window_name = window_name
        self.window_size = window_size

        self.rgb_calib = None
        self.dst_calib = None
        self.camera_matrix: Optional[np.ndarray] = None

        self.streaming_client = None
        self.observer = None

        self._overlays: list[RgbOverlay] = []

    def add_overlay(self, overlay: RgbOverlay) -> None:
        self._overlays.append(overlay)

    def run(self) -> None:
        if self.update_iptables_rules and sys.platform.startswith("linux"):
            update_iptables()

        self._load_rgb_calibration()
        self._setup_dst_calib()

        if any(o.requires_calibration for o in self._overlays) and self.camera_matrix is None:
            raise RuntimeError(
                "One or more overlays require device calibration, "
                "but calibration could not be loaded from the device."
            )

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

        while not quit_keypress():
            if self.observer.rgb_image is None:
                time.sleep(0.001)
                continue

            rgb_image = self._prepare_rgb_image(self.observer.rgb_image)
            self.observer.rgb_image = None

            for overlay in self._overlays:
                overlay.draw(rgb_image, self.camera_matrix)

            display = np.ascontiguousarray(np.rot90(rgb_image, -1))
            h, w = display.shape[:2]
            crop_size = int(min(w, h) / 1.4143)
            ox = (w - crop_size) // 2
            oy = (h - crop_size) // 2
            display = display[oy:oy + crop_size, ox:ox + crop_size]

            for overlay in self._overlays:
                overlay.draw_display(display, self.camera_matrix)

            cv2.imshow(self.window_name, display)

    def _prepare_rgb_image(self, bgr_image: np.ndarray) -> np.ndarray:
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        if self.rgb_calib is not None and self.dst_calib is not None:
            rgb_image = distort_by_calibration(rgb_image, self.dst_calib, self.rgb_calib)
        return rgb_image

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
            src_w, src_h = self.rgb_calib.get_image_size()
            if self.undistort_width != self.undistort_height or src_w != src_h:
                raise ValueError(
                    f"Non-square images are not supported (src={src_w}x{src_h}, "
                    f"dst={self.undistort_width}x{self.undistort_height})."
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
