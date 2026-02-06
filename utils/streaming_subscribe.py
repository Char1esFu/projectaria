# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import sys
import time

import aria.sdk as aria

import cv2
import numpy as np
from utils.common import quit_keypress, update_iptables

from projectaria_tools.core.calibration import (
    device_calibration_from_json_string,
    distort_by_calibration,
    get_linear_camera_calibration,
)
from projectaria_tools.core.sensor_data import (
    AudioData,
    AudioDataRecord,
    BarometerData,
    ImageDataRecord,
    MotionData,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--update_iptables",
        default=False,
        action="store_true",
        help="Update iptables to enable receiving the data stream, only for Linux.",
    )
    parser.add_argument(
        "--device-ip", help="IP address to connect to the device over wifi"
    )
    parser.add_argument(
        "--no-undistort",
        action="store_true",
        help="Disable RGB undistortion (enabled by default).",
    )
    parser.add_argument(
        "--undistort-width",
        type=int,
        default=1408,
        help="Width of the undistorted RGB output image.",
    )
    parser.add_argument(
        "--undistort-height",
        type=int,
        default=1408,
        help="Height of the undistorted RGB output image.",
    )
    parser.add_argument(
        "--undistort-focal-length",
        type=float,
        default=450.0,
        help="Focal length for the undistorted RGB output calibration.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.update_iptables and sys.platform.startswith("linux"):
        update_iptables()

    undistort_enabled = not args.no_undistort
    rgb_calib = None
    dst_calib = None
    if undistort_enabled:
        device_client = aria.DeviceClient()
        client_config = aria.DeviceClientConfig()
        if args.device_ip:
            client_config.ip_v4_address = args.device_ip
        device_client.set_client_config(client_config)
        device = None
        try:
            device = device_client.connect()
            sensors_calib_json = device.streaming_manager.sensors_calibration()
            sensors_calib = device_calibration_from_json_string(sensors_calib_json)
            rgb_calib = sensors_calib.get_camera_calib("camera-rgb")
            dst_calib = get_linear_camera_calibration(
                args.undistort_width,
                args.undistort_height,
                args.undistort_focal_length,
                "camera-rgb",
            )
        except Exception as exc:
            undistort_enabled = False
            print(f"Warning: failed to load RGB calibration for undistortion: {exc}")
        finally:
            if device is not None:
                device_client.disconnect(device)

    #  Optional: Set SDK's log level to Trace or Debug for more verbose logs. Defaults to Info
    aria.set_log_level(aria.Level.Info)

    # 1. Create StreamingClient instance
    streaming_client = aria.StreamingClient()

    #  2. Configure subscription to listen to Aria's streams.
    # @see StreamingDataType for the other data types
    config = streaming_client.subscription_config
    config.subscriber_data_type = (
        aria.StreamingDataType.Rgb
        | aria.StreamingDataType.Slam
        | aria.StreamingDataType.EyeTrack
        | aria.StreamingDataType.Audio
        | aria.StreamingDataType.Imu
        | aria.StreamingDataType.Magneto
        | aria.StreamingDataType.Baro
    )

    # A shorter queue size may be useful if the processing callback is always slow and you wish to process more recent data
    # For visualizing the images, we only need the most recent frame so set the queue size to 1
    config.message_queue_size[aria.StreamingDataType.Rgb] = 1
    config.message_queue_size[aria.StreamingDataType.Slam] = 1
    config.message_queue_size[aria.StreamingDataType.EyeTrack] = 1
    config.message_queue_size[aria.StreamingDataType.Audio] = 1
    config.message_queue_size[aria.StreamingDataType.Imu] = 1
    config.message_queue_size[aria.StreamingDataType.Magneto] = 1
    config.message_queue_size[aria.StreamingDataType.Baro] = 1

    # Set the security options
    # @note we need to specify the use of ephemeral certs as this sample app assumes
    # aria-cli was started using the --use-ephemeral-certs flag
    options = aria.StreamingSecurityOptions()
    options.use_ephemeral_certs = True
    config.security_options = options
    streaming_client.subscription_config = config

    # 3. Create and attach observer
    class StreamingClientObserver:
        def __init__(self):
            self.images = {}
            self.last_print_time = {}
            self.sample_counts = {}
            self.latest_imu = {}
            self.latest_magneto = None
            self.latest_baro = None
            self.latest_audio = None

        def _tick(self, key: str, count: int = 1, extra: str = ""):
            now = time.time()
            if key not in self.last_print_time:
                self.last_print_time[key] = now
                self.sample_counts[key] = 0
            self.sample_counts[key] += count
            elapsed = now - self.last_print_time[key]
            if elapsed >= 1.0:
                rate = self.sample_counts[key] / elapsed
                suffix = f", {extra}" if extra else ""
                print(f"{key}: {rate:.2f} Hz{suffix}")
                self.last_print_time[key] = now
                self.sample_counts[key] = 0

        def on_image_received(self, image: np.array, record: ImageDataRecord):
            self.images[record.camera_id] = image

        def on_audio_received(self, audio_data: AudioData, record: AudioDataRecord):
            self.latest_audio = (len(audio_data.data), len(record.capture_timestamps_ns))
            self._tick(
                "Audio",
                count=len(record.capture_timestamps_ns) or 1,
                extra=f"samples={len(audio_data.data)}",
            )

        def on_imu_received(self, samples: list[MotionData], imu_idx: int):
            if not samples:
                return
            self.latest_imu[imu_idx] = samples[-1]
            self._tick(f"IMU{imu_idx}", count=len(samples))

        def on_magneto_received(self, sample: MotionData):
            self.latest_magneto = sample
            self._tick("Magneto")

        def on_baro_received(self, sample: BarometerData):
            self.latest_baro = sample
            self._tick("Baro")

    observer = StreamingClientObserver()
    streaming_client.set_streaming_client_observer(observer)

    # 4. Start listening
    print("Start listening to image data")
    streaming_client.subscribe()

    # 5. Visualize the streaming data until we close the window
    rgb_window = "Aria RGB"
    slam_window = "Aria SLAM"
    eyetrack_window = "Aria EyeTrack"

    cv2.namedWindow(rgb_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(rgb_window, 1024, 1024)
    cv2.setWindowProperty(rgb_window, cv2.WND_PROP_TOPMOST, 1)
    cv2.moveWindow(rgb_window, 50, 50)

    cv2.namedWindow(slam_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(slam_window, 480 * 2, 640)
    cv2.setWindowProperty(slam_window, cv2.WND_PROP_TOPMOST, 1)
    cv2.moveWindow(slam_window, 1100, 50)

    cv2.namedWindow(eyetrack_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(eyetrack_window, 640, 240)
    cv2.setWindowProperty(eyetrack_window, cv2.WND_PROP_TOPMOST, 1)
    cv2.moveWindow(eyetrack_window, 50, 800)
    
    while not quit_keypress():
        # Render the RGB image
        if aria.CameraId.Rgb in observer.images:
            rgb_image = cv2.cvtColor(
                observer.images[aria.CameraId.Rgb], cv2.COLOR_BGR2RGB
            )
            if undistort_enabled and rgb_calib is not None and dst_calib is not None:
                rgb_image = distort_by_calibration(rgb_image, dst_calib, rgb_calib)
            rgb_image = np.rot90(rgb_image, -1)
            cv2.imshow(rgb_window, rgb_image)
            del observer.images[aria.CameraId.Rgb]

        # Stack and display the SLAM images
        if (
            aria.CameraId.Slam1 in observer.images
            and aria.CameraId.Slam2 in observer.images
        ):
            slam1_image = np.rot90(observer.images[aria.CameraId.Slam1], -1)
            slam2_image = np.rot90(observer.images[aria.CameraId.Slam2], -1)
            cv2.imshow(slam_window, np.hstack((slam1_image, slam2_image)))
            del observer.images[aria.CameraId.Slam1]
            del observer.images[aria.CameraId.Slam2]

        # Display the EyeTrack image
        if aria.CameraId.EyeTrack in observer.images:
            eyetrack_image = observer.images[aria.CameraId.EyeTrack]
            cv2.imshow(eyetrack_window, eyetrack_image)
            del observer.images[aria.CameraId.EyeTrack]

    # 6. Unsubscribe to clean up resources
    print("Stop listening to image data")
    streaming_client.unsubscribe()


if __name__ == "__main__":
    main()
