import argparse
import sys
import time
from typing import Sequence

import aria.sdk as aria
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, Imu, MagneticField, FluidPressure

from common import update_iptables
from projectaria_tools.core.sensor_data import (
    BarometerData,
    ImageDataRecord,
    MotionData,
)

class AriaSensorPublisherNode(Node):
    def __init__(self):
        super().__init__('aria_sensor_publisher')
        
        # Image Publishers (10 Hz)
        self.rgb_pub = self.create_publisher(Image, 'aria/rgb', 10)
        self.slam_left_pub = self.create_publisher(Image, 'aria/slam_left', 10)
        self.slam_right_pub = self.create_publisher(Image, 'aria/slam_right', 10)
        self.et_pub = self.create_publisher(Image, 'aria/eye_tracking', 10)
        
        # IMU Publishers (1000 Hz and 800 Hz)
        self.imu0_pub = self.create_publisher(Imu, 'aria/imu0', 1000)
        self.imu1_pub = self.create_publisher(Imu, 'aria/imu1', 800)
        
        # Other Sensor Publishers (10 Hz and 50 Hz)
        self.mag_pub = self.create_publisher(MagneticField, 'aria/magnetometer', 10)
        self.baro_pub = self.create_publisher(FluidPressure, 'aria/barometer', 50)
        
        self.get_logger().info('Aria Sensor Publisher Node Started')
        self.frame_count = 0

    def publish_image(self, image: np.array, record: ImageDataRecord):
        try:
            timestamp_ns = record.capture_timestamp_ns
            camera_id = record.camera_id
            
            msg = Image()
            msg.header.stamp.sec = int(timestamp_ns // 1_000_000_000)
            msg.header.stamp.nanosec = int(timestamp_ns % 1_000_000_000)
            
            # Set frame_id based on camera
            if camera_id == aria.CameraId.Rgb:
                msg.header.frame_id = "aria_rgb_frame"
                publisher = self.rgb_pub
            elif camera_id == aria.CameraId.Slam1:
                msg.header.frame_id = "aria_slam_left_frame"
                publisher = self.slam_left_pub
            elif camera_id == aria.CameraId.Slam2:
                msg.header.frame_id = "aria_slam_right_frame"
                publisher = self.slam_right_pub
            elif camera_id == aria.CameraId.EyeTrack:
                msg.header.frame_id = "aria_et_frame"
                publisher = self.et_pub
            else:
                return

            if len(image.shape) == 3:
                height, width, channels = image.shape
                msg.height = height
                msg.width = width
                msg.encoding = "rgb8"
                msg.step = width * 3
            else:
                height, width = image.shape
                msg.height = height
                msg.width = width
                msg.encoding = "mono8"
                msg.step = width

            msg.is_bigendian = 0
            msg.data = image.tobytes()
            
            publisher.publish(msg)
            
            # Log occasionally (only counting total frames for simplicity)
            self.frame_count += 1
            if self.frame_count % 100 == 0:
                self.get_logger().info(f'Published {self.frame_count} images total')

        except Exception as e:
            self.get_logger().error(f'Error publishing image: {e}')

    def publish_imu(self, samples: Sequence[MotionData], imu_idx: int):
        publisher = self.imu0_pub if imu_idx == 0 else self.imu1_pub
        frame_id = f"aria_imu{imu_idx}_frame"
        
        for sample in samples:
            msg = Imu()
            timestamp_ns = sample.capture_timestamp_ns
            msg.header.stamp.sec = int(timestamp_ns // 1_000_000_000)
            msg.header.stamp.nanosec = int(timestamp_ns % 1_000_000_000)
            msg.header.frame_id = frame_id
            
            # Linear acceleration (m/s^2)
            msg.linear_acceleration.x = float(sample.accel_msec2[0])
            msg.linear_acceleration.y = float(sample.accel_msec2[1])
            msg.linear_acceleration.z = float(sample.accel_msec2[2])
            
            # Angular velocity (rad/s)
            msg.angular_velocity.x = float(sample.gyro_radsec[0])
            msg.angular_velocity.y = float(sample.gyro_radsec[1])
            msg.angular_velocity.z = float(sample.gyro_radsec[2])
            
            # Orientation not provided per sample in raw data
            msg.orientation.w = 1.0 # Identity quaternion
            
            publisher.publish(msg)

    def publish_magnetometer(self, sample: MotionData):
        msg = MagneticField()
        timestamp_ns = sample.capture_timestamp_ns
        msg.header.stamp.sec = int(timestamp_ns // 1_000_000_000)
        msg.header.stamp.nanosec = int(timestamp_ns % 1_000_000_000)
        msg.header.frame_id = "aria_mag_frame"
        
        msg.magnetic_field.x = float(sample.mag_tesla[0])
        msg.magnetic_field.y = float(sample.mag_tesla[1])
        msg.magnetic_field.z = float(sample.mag_tesla[2])
        
        self.mag_pub.publish(msg)

    def publish_barometer(self, sample: BarometerData):
        msg = FluidPressure()
        timestamp_ns = sample.capture_timestamp_ns
        msg.header.stamp.sec = int(timestamp_ns // 1_000_000_000)
        msg.header.stamp.nanosec = int(timestamp_ns % 1_000_000_000)
        msg.header.frame_id = "aria_baro_frame"
        
        msg.fluid_pressure = float(sample.pressure) # Pascal
        # FluidPressure message doesn't have temperature field, but data has it.
        
        self.baro_pub.publish(msg)

class AriaRosObserver:
    def __init__(self, node: AriaSensorPublisherNode):
        self.node = node

    def on_image_received(self, image: np.array, record: ImageDataRecord) -> None:
        # Rotate images to match display orientation
        
        if record.camera_id != aria.CameraId.EyeTrack:
            image = np.rot90(image, 3)
             
        self.node.publish_image(image, record)

    def on_imu_received(self, samples: Sequence[MotionData], imu_idx: int) -> None:
        self.node.publish_imu(samples, imu_idx)

    def on_magneto_received(self, sample: MotionData) -> None:
        self.node.publish_magnetometer(sample)

    def on_baro_received(self, sample: BarometerData) -> None:
        self.node.publish_barometer(sample)

    def on_streaming_client_failure(self, reason: aria.ErrorCode, message: str) -> None:
        self.node.get_logger().error(f"Streaming Client Failure: {reason}: {message}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--interface",
        dest="streaming_interface",
        type=str,
        default="usb",
        required=False,
        help="Type of interface to use for streaming. Options are usb or wifi.",
        choices=["usb", "wifi"],
    )
    parser.add_argument(
        "--update_iptables",
        default=True,
        action="store_true",
        help="Update iptables to enable receiving the data stream, only for Linux.",
    )
    parser.add_argument(
        "--profile",
        dest="profile_name",
        type=str,
        default="profile18",
        required=True,
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
    node = AriaSensorPublisherNode()

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
    observer = AriaRosObserver(node)
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
