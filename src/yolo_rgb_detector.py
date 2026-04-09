import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import aria.sdk as aria
import cv2
import numpy as np
from ultralytics import YOLO

from utils.common import quit_keypress, update_iptables
from projectaria_tools.core.calibration import (
    device_calibration_from_json_string,
    distort_by_calibration,
    get_linear_camera_calibration,
)
from projectaria_tools.core.sensor_data import ImageDataRecord

MODEL_PATH = Path(__file__).parent.parent / "yolo_model" / "best.pt"


class YoloRgbDetector:
    def __init__(
        self,
        device_ip: Optional[str],
        update_iptables_rules: bool,
        model_path: Path,
        conf_threshold: float,
        undistort_width: int,
        undistort_height: int,
        undistort_focal_length: float,
    ) -> None:
        self.device_ip = device_ip
        self.update_iptables_rules = update_iptables_rules
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.undistort_width = undistort_width
        self.undistort_height = undistort_height
        self.undistort_focal_length = undistort_focal_length

        self.rgb_calib = None
        self.dst_calib = None

        self.model = None
        self.streaming_client = None
        self.observer = None

    def run(self) -> None:
        if self.update_iptables_rules and sys.platform.startswith("linux"):
            update_iptables()

        self._load_rgb_calibration()
        self._setup_dst_calib()

        print(f"Loading YOLO model from {self.model_path}")
        self.model = YOLO(str(self.model_path))

        aria.set_log_level(aria.Level.Info)
        self._setup_streaming()
        self._run_loop()

        print("Stop listening to RGB data")
        self.streaming_client.unsubscribe()

    def _run_loop(self) -> None:
        window_name = "Aria RGB YOLO"
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

            # YOLO expects BGR; rgb_image is RGB after _prepare_rgb_image
            bgr_for_yolo = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            results = self.model(bgr_for_yolo, conf=self.conf_threshold, verbose=False)
            annotated = results[0].plot()  # returns BGR

            display = np.ascontiguousarray(np.rot90(annotated, -1))
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
    parser = argparse.ArgumentParser(description="Run YOLO detection on Aria RGB stream.")
    parser.add_argument("--device-ip", help="IP address of the Aria device")
    parser.add_argument(
        "--update_iptables",
        default=False,
        action="store_true",
        help="Update iptables for DDS UDP stream (Linux only).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=str(MODEL_PATH),
        help="Path to YOLO model weights.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Detection confidence threshold.",
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
    detector = YoloRgbDetector(
        device_ip=args.device_ip,
        update_iptables_rules=args.update_iptables,
        model_path=Path(args.model),
        conf_threshold=args.conf,
        undistort_width=args.undistort_width,
        undistort_height=args.undistort_height,
        undistort_focal_length=args.undistort_focal_length,
    )
    detector.run()


if __name__ == "__main__":
    main()
