#!/usr/bin/env python3
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
import numpy as np

import aria.sdk as aria
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image

from common import update_iptables
from projectaria_tools.core.sensor_data import ImageDataRecord

class AriaRgbPublisher(Node):
    def __init__(self):
        super().__init__('aria_rgb_publisher')
        # QoS 与 RViz 默认兼容：可靠、KeepLast、深度10
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.rgb_pub = self.create_publisher(Image, 'aria/rgb', sensor_qos)
        self.frame_count = 0
        self.get_logger().info('Aria RGB Publisher Node Started')

    def publish_image(self, image: np.array, record: ImageDataRecord):
        try:
            # We only care about RGB camera for this node, but check just in case
            if record.camera_id != aria.CameraId.Rgb:
                return

            timestamp_ns = record.capture_timestamp_ns
            
            msg = Image()
            msg.header.stamp.sec = int(timestamp_ns // 1_000_000_000)
            msg.header.stamp.nanosec = int(timestamp_ns % 1_000_000_000)
            msg.header.frame_id = "aria_rgb_frame"

            # Optimization: Publish raw image without rotation for performance.
            # Rotating in Python (np.rot90) makes the array non-contiguous and 
            # tobytes() becomes very slow (memory copy).
            # Users should handle rotation via TF (static_transform_publisher).
            
            if len(image.shape) == 3:
                height, width, channels = image.shape
                msg.height = height
                msg.width = width
                msg.encoding = "rgb8"
                msg.step = width * 3
            else:
                self.get_logger().warn("Received non-RGB image on RGB channel")
                return

            msg.is_bigendian = 0
            # 确保为连续内存，避免隐式慢速复制
            contiguous = np.ascontiguousarray(image)
            msg.data = contiguous.tobytes()
            
            self.rgb_pub.publish(msg)
            
            self.frame_count += 1
            if self.frame_count % 10 == 0:
                self.get_logger().info(f'Published {self.frame_count} RGB images')

        except Exception as e:
            self.get_logger().error(f'Error publishing image: {e}')


class AriaStreamingObserver:
    def __init__(self, node: AriaRgbPublisher):
        self.node = node

    def on_image_received(self, image: np.array, record: ImageDataRecord) -> None:
        self.node.publish_image(image, record)

    # We need to implement other methods of the observer interface even if empty
    def on_imu_received(self, samples, imu_idx) -> None: pass
    def on_magneto_received(self, sample) -> None: pass
    def on_baro_received(self, sample) -> None: pass

    def on_streaming_client_failure(self, reason: aria.ErrorCode, message: str) -> None:
        self.node.get_logger().error(f"Streaming Client Failure: {reason}: {message}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--interface",
        dest="streaming_interface",
        type=str,
        default="usb",
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

    rclpy.init()
    node = AriaRgbPublisher()

    aria.set_log_level(aria.Level.Info)

    # 1. Create DeviceClient
    device_client = aria.DeviceClient()
    client_config = aria.DeviceClientConfig()
    if args.device_ip:
        client_config.ip_v4_address = args.device_ip
    device_client.set_client_config(client_config)

    # 2. Connect
    node.get_logger().info("Connecting to Aria device...")
    try:
        device = device_client.connect()
    except Exception as e:
        node.get_logger().error(f"Failed to connect: {e}")
        return

    streaming_manager = device.streaming_manager
    streaming_client = streaming_manager.streaming_client

    # 3. Configure Streaming Manager (Device side)
    streaming_config = aria.StreamingConfig()
    streaming_config.profile_name = args.profile_name
    if args.streaming_interface == "usb":
        streaming_config.streaming_interface = aria.StreamingInterface.Usb
    streaming_config.security_options.use_ephemeral_certs = True
    streaming_manager.streaming_config = streaming_config

    # 4. Configure Subscription (Client side)
    # 仅订阅 RGB，并将队列调大以减少丢帧
    subscription_config = streaming_client.subscription_config
    subscription_config.subscriber_data_type = aria.StreamingDataType.Rgb
    subscription_config.message_queue_size[aria.StreamingDataType.Rgb] = 50
    streaming_client.subscription_config = subscription_config

    # 5. Start Streaming
    node.get_logger().info("Starting streaming...")
    streaming_manager.start_streaming()

    # 6. Subscribe with Observer
    observer = AriaStreamingObserver(node)
    streaming_client.set_streaming_client_observer(observer)
    streaming_client.subscribe()
    
    node.get_logger().info("Subscribed to RGB stream.")

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info("Shutting down...")
        streaming_client.unsubscribe()
        streaming_manager.stop_streaming()
        device_client.disconnect(device)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
