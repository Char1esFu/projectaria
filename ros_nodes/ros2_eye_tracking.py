import argparse
import sys
import time
from typing import Sequence

import aria.sdk as aria
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

from common import update_iptables
from projectaria_tools.core.sensor_data import (
    BarometerData,
    ImageDataRecord,
    MotionData,
)

class AriaEyeTrackingNode(Node):
    def __init__(self):
        super().__init__('aria_eye_tracking_publisher')
        self.publisher_ = self.create_publisher(Image, 'aria/eye_tracking', 10)
        self.get_logger().info('Eye Tracking Publisher Node Started')
        self.frame_count = 0

    def publish_eye_tracking(self, image, timestamp_ns):
        try:
            msg = Image()
            msg.header.stamp.sec = int(timestamp_ns // 1_000_000_000)
            msg.header.stamp.nanosec = int(timestamp_ns % 1_000_000_000)
            msg.header.frame_id = "aria_et_frame"
            
            # Assuming mono8 for grayscale eye tracking image
            height, width = image.shape
            msg.height = height
            msg.width = width
            msg.encoding = "mono8"
            msg.is_bigendian = 0
            msg.step = width
            msg.data = image.tobytes()
            
            self.publisher_.publish(msg)
            
            self.frame_count += 1
            if self.frame_count % 100 == 0:
                self.get_logger().info(f'Published {self.frame_count} frames')
                
        except Exception as e:
            self.get_logger().error(f'Error publishing image: {e}')


class EyeTrackingRosObserver:
    def __init__(self, node: AriaEyeTrackingNode):
        self.node = node

    def on_image_received(self, image: np.array, record: ImageDataRecord) -> None:
        if record.camera_id == aria.CameraId.EyeTrack:
            # Rotate image to match upright orientation
            image = np.rot90(image, 2)
            self.node.publish_eye_tracking(image, record.capture_timestamp_ns)

    def on_imu_received(self, samples: Sequence[MotionData], imu_idx: int) -> None:
        pass

    def on_magneto_received(self, sample: MotionData) -> None:
        pass

    def on_baro_received(self, sample: BarometerData) -> None:
        pass

    def on_streaming_client_failure(self, reason: aria.ErrorCode, message: str) -> None:
        self.node.get_logger().error(f"Streaming Client Failure: {reason}: {message}")


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
    return parser.parse_args()


def main():
    args = parse_args()
    if args.update_iptables and sys.platform.startswith("linux"):
        update_iptables()

    # Initialize ROS 2
    rclpy.init()
    node = AriaEyeTrackingNode()

    # Initialize Aria
    aria.set_log_level(aria.Level.Info)

    # 1. Create DeviceClient instance
    device_client = aria.DeviceClient()

    client_config = aria.DeviceClientConfig()
    if args.device_ip:
        client_config.ip_v4_address = args.device_ip
    device_client.set_client_config(client_config)

    # 2. Connect to the device
    node.get_logger().info("Connecting to device...")
    try:
        device = device_client.connect()
    except Exception as e:
        node.get_logger().error(f"Failed to connect to device: {e}")
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
    node.get_logger().info("Starting streaming...")
    streaming_manager.start_streaming()
    node.get_logger().info(f"Streaming state: {streaming_manager.streaming_state}")

    # 6. Set observer
    observer = EyeTrackingRosObserver(node)
    streaming_client.set_streaming_client_observer(observer)
    streaming_client.subscribe()

    node.get_logger().info("Streaming started.")

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard Interrupt...")
    finally:
        node.get_logger().info("Stopping streaming...")
        streaming_client.unsubscribe()
        streaming_manager.stop_streaming()
        device_client.disconnect(device)
        node.get_logger().info("Disconnected.")
        
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
