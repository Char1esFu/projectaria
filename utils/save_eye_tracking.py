import argparse
import os
import sys
import time
from typing import Sequence
import socket

import aria.sdk as aria
import cv2
import numpy as np

from common import update_iptables
from projectaria_tools.core.sensor_data import (
    BarometerData,
    ImageDataRecord,
    MotionData,
)


class EyeTrackingSaverObserver:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.frame_count = 0
        print(f"Saving Eye Tracking images to {os.path.abspath(output_dir)}")

    def on_image_received(self, image: np.array, record: ImageDataRecord) -> None:
        if record.camera_id == aria.CameraId.EyeTrack:
            # Rotate image to match upright orientation (same as visualizer)
            image = np.rot90(image, 2)
            
            # Use timestamp for filename
            timestamp = record.capture_timestamp_ns
            filename = f"et_{timestamp}.png"
            filepath = os.path.join(self.output_dir, filename)
            
            # Save image
            cv2.imwrite(filepath, image)
            
            self.frame_count += 1
            if self.frame_count % 100 == 0:
                print(f"Saved {self.frame_count} frames", end='\r')

    def on_imu_received(self, samples: Sequence[MotionData], imu_idx: int) -> None:
        pass

    def on_magneto_received(self, sample: MotionData) -> None:
        pass

    def on_baro_received(self, sample: BarometerData) -> None:
        pass

    def on_streaming_client_failure(self, reason: aria.ErrorCode, message: str) -> None:
        print(f"Streaming Client Failure: {reason}: {message}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--interface",
        dest="streaming_interface",
        type=str,
        required=True,
        help="Type of interface to use for streaming. Options are usb or wifi.",
        choices=["usb", "wifi"],
    )
    parser.add_argument(
        "--update_iptables",
        default=False,
        action="store_true",
        help="Update iptables to enable receiving the data stream, only for Linux.",
    )
    parser.add_argument(
        "--profile",
        dest="profile_name",
        type=str,
        default="profile18",
        required=False,
        help="Profile to be used for streaming.",
    )
    parser.add_argument(
        "--device-ip", help="IP address to connect to the device over wifi"
    )
    parser.add_argument(
        "--output-dir",
        default="eyetrack_images",
        help="Directory to save eye tracking images",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.update_iptables and sys.platform.startswith("linux"):
        update_iptables()

    aria.set_log_level(aria.Level.Info)

    # 1. Create DeviceClient instance
    device_client = aria.DeviceClient()

    client_config = aria.DeviceClientConfig()
    if args.device_ip:
        client_config.ip_v4_address = args.device_ip
    device_client.set_client_config(client_config)

    # 2. Connect to the device
    print("Connecting to device...")
    try:
        device = device_client.connect()
    except Exception as e:
        print(f"Failed to connect to device: {e}")
        return

    # 3. Retrieve streaming manager
    streaming_manager = device.streaming_manager
    streaming_client = streaming_manager.streaming_client

    # 4. Configure streaming
    streaming_config = aria.StreamingConfig()
    streaming_config.profile_name = args.profile_name

    if args.streaming_interface == "usb":
        streaming_config.streaming_interface = aria.StreamingInterface.Usb
    
    streaming_config.security_options.use_ephemeral_certs = True
    streaming_manager.streaming_config = streaming_config

    # 5. Start streaming
    print("Starting streaming...")
    streaming_manager.start_streaming()
    print(f"Streaming state: {streaming_manager.streaming_state}")

    # 6. Set observer
    observer = EyeTrackingSaverObserver(args.output_dir)
    streaming_client.set_streaming_client_observer(observer)
    streaming_client.subscribe()

    print("Listening for eye tracking frames. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        print("Stopping streaming...")
        streaming_client.unsubscribe()
        streaming_manager.stop_streaming()
        device_client.disconnect(device)
        print("Disconnected.")

if __name__ == "__main__":
    main()
