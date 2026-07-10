"""Shared Aria RGB/audio streaming helpers.

RGB overlays receive undistorted, rotated, cropped frames plus a matching
camera matrix. Audio handlers are called from the SDK audio thread.
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
from projectaria_tools.core.sensor_data import AudioData, AudioDataRecord, ImageDataRecord


def crop_fisheye_img(image: np.ndarray):
    """Crop the centered display square and return its offset."""
    h, w = image.shape[:2]
    crop_size = int(min(w, h) / 1.5)
    crop_size += crop_size % 2  # ensure even for codec compatibility
    ox = (w - crop_size) // 2
    oy = (h - crop_size) // 2
    cropped = np.ascontiguousarray(image[oy:oy + crop_size, ox:ox + crop_size])
    return cropped, ox, oy


class AriaStream:
    """Single StreamingClient for RGB overlays and audio handlers."""

    def __init__(
        self,
        device_ip: Optional[str] = None,
        update_iptables_rules: bool = False,
        window_name: str = "Aria RGB",
        window_size: int = 1024,
        data_types: int = aria.StreamingDataType.Rgb,
    ) -> None:
        self.device_ip = device_ip
        self.update_iptables_rules = update_iptables_rules
        self.window_name = window_name
        self.window_size = window_size
        self.data_types = data_types

        self.rgb_calib = None
        self.dst_calib = None
        self.camera_matrix: Optional[np.ndarray] = None
        # One-time undistort+rot90+crop LUT (built on first frame).
        self._map_x: Optional[np.ndarray] = None
        self._map_y: Optional[np.ndarray] = None
        self._crop_offset: tuple[int, int] = (0, 0)

        self.streaming_client = None
        self.observer = None

        self._overlays: list = []
        self._audio_handlers: list = []
        self._frame_callback = None

    @property
    def _rgb_enabled(self) -> bool:
        # aria.sdk.StreamingDataType supports | (returns a flag set) but not &
        # against individual flags, so cast to int for the membership test.
        return bool(int(self.data_types) & int(aria.StreamingDataType.Rgb))

    @property
    def _audio_enabled(self) -> bool:
        return bool(int(self.data_types) & int(aria.StreamingDataType.Audio))

    def add_overlay(self, overlay) -> None:
        self._overlays.append(overlay)

    def add_audio_handler(self, handler) -> None:
        """Register an audio-frame handler."""
        self._audio_handlers.append(handler)

    def set_frame_callback(self, fn) -> None:
        self._frame_callback = fn

    def run(self) -> None:
        if self.update_iptables_rules and sys.platform.startswith("linux"):
            update_iptables()

        if self._rgb_enabled:
            self._load_rgb_calibration()

        aria.set_log_level(aria.Level.Info)
        self._setup_streaming()
        try:
            if self._rgb_enabled:
                self._run_loop()
            else:
                self._run_idle_loop()
        finally:
            print("Stop listening to Aria data")
            self.streaming_client.unsubscribe()

    def _run_idle_loop(self) -> None:
        """Keep non-RGB streaming alive until Ctrl+C."""
        print("Streaming without RGB display. Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(0.2)
        except KeyboardInterrupt:
            pass

    def _run_loop(self) -> None:
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.window_size, self.window_size)
        cv2.setWindowProperty(self.window_name, cv2.WND_PROP_TOPMOST, 1)
        cv2.moveWindow(self.window_name, 50, 50)

        pending_key = 255  # buffer key presses between frames (waitKey polls at 1ms, frames arrive ~33ms)

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break

            if key != 255:
                pending_key = key

            if self.observer.rgb_image is None:
                time.sleep(0.001)
                continue

            if self.dst_calib is None:
                # Initialise calibration on first frame using actual streaming resolution.
                frame_h, frame_w = self.observer.rgb_image.shape[:2]
                self._setup_dst_calib(frame_w, frame_h)
                self._build_undistort_maps(frame_w, frame_h)

            rgb_image = cv2.cvtColor(self.observer.rgb_image, cv2.COLOR_BGR2RGB)
            self.observer.rgb_image = None

            # Undistort + rot90 (sensor is mounted rotated) + fisheye crop,
            # all in one remap through the precomputed LUT.
            display = cv2.remap(rgb_image, self._map_x, self._map_y, cv2.INTER_LINEAR)
            ox, oy = self._crop_offset

            # Shift cx/cy so overlays can use camera_matrix directly in cropped image coordinates.
            display_matrix = self.camera_matrix.copy()
            display_matrix[0, 2] -= ox
            display_matrix[1, 2] -= oy

            frame_key = pending_key
            pending_key = 255

            for overlay in self._overlays:
                overlay.draw(display, display_matrix, frame_key)

            if self._frame_callback is not None:
                self._frame_callback(display)

            cv2.imshow(self.window_name, display)

    def _build_undistort_maps(self, w: int, h: int) -> None:
        """Precompute the undistort, rotate, and crop remap."""
        xs = np.tile(np.arange(1, w + 1, dtype=np.float32)[None, :], (h, 1))
        ys = np.tile(np.arange(1, h + 1, dtype=np.float32)[:, None], (1, w))
        map_x = distort_by_calibration(xs, self.dst_calib, self.rgb_calib)
        map_y = distort_by_calibration(ys, self.dst_calib, self.rgb_calib)
        invalid = (map_x == 0.0) | (map_y == 0.0)
        map_x -= 1.0
        map_y -= 1.0
        map_x[invalid] = -1.0
        map_y[invalid] = -1.0

        # Fold display rotation and crop into the remap.
        map_x = np.rot90(map_x, -1)
        map_y = np.rot90(map_y, -1)
        rh, rw = map_x.shape
        crop_size = int(min(rw, rh) / 1.5)
        crop_size += crop_size % 2  # even, matching crop_fisheye_img
        ox = (rw - crop_size) // 2
        oy = (rh - crop_size) // 2
        self._map_x = np.ascontiguousarray(map_x[oy:oy + crop_size, ox:ox + crop_size])
        self._map_y = np.ascontiguousarray(map_y[oy:oy + crop_size, ox:ox + crop_size])
        self._crop_offset = (ox, oy)

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
        config.subscriber_data_type = self.data_types
        if self._rgb_enabled:
            # Display only ever cares about the newest frame.
            config.message_queue_size[aria.StreamingDataType.Rgb] = 1
        if self._audio_enabled:
            # Accumulate audio frames so handlers don't drop samples while busy.
            config.message_queue_size[aria.StreamingDataType.Audio] = 100
        options = aria.StreamingSecurityOptions()
        options.use_ephemeral_certs = True
        config.security_options = options
        self.streaming_client.subscription_config = config

        audio_handlers = self._audio_handlers

        class StreamingClientObserver:
            def __init__(self):
                self.rgb_image = None

            def on_image_received(self, image: np.ndarray, record: ImageDataRecord):
                if record.camera_id != aria.CameraId.Rgb:
                    return
                self.rgb_image = image

            def on_audio_received(self, audio_data: AudioData, record: AudioDataRecord):
                for handler in audio_handlers:
                    handler.on_audio_received(audio_data, record)

        self.observer = StreamingClientObserver()
        self.streaming_client.set_streaming_client_observer(self.observer)
        active = []
        if self._rgb_enabled:
            active.append("RGB")
        if self._audio_enabled:
            active.append("Audio")
        print(f"Start listening to Aria data: {', '.join(active) or 'none'}")
        self.streaming_client.subscribe()


# Backward-compatible alias. Existing callers (aruco_localization, save_rgb_frames,
# gaze_rgb_visualizer) construct AriaRgbStream(...) for RGB-only use.
AriaRgbStream = AriaStream
