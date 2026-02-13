import argparse
import math
from typing import Optional

import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped, TransformStamped, Vector3
from rclpy.node import Node
from scipy.spatial.transform import Rotation
from tf2_ros import Buffer, StaticTransformBroadcaster, TransformListener
from visualization_msgs.msg import Marker


def _yaw_pitch_to_unit_vector(yaw: float, pitch: float) -> np.ndarray:
    cos_pitch = math.cos(pitch)
    return np.array(
        [
            cos_pitch * math.sin(yaw),
            -math.sin(pitch),
            cos_pitch * math.cos(yaw),
        ],
        dtype=np.float64,
    )

class GazeIntersectionNode(Node):
    def __init__(
        self,
        gaze_x_offset: float,
        publish_static: bool,
        visualize: bool,
    ) -> None:
        super().__init__("gaze_xy_intersection")
        self._base_frame = "base_link"
        self._camera_frame = "aria_camera_rgb"
        self._gaze_frame = "aria_gaze"
        self._gaze_x_offset = gaze_x_offset
        self._visualize = visualize

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._static_broadcaster: Optional[StaticTransformBroadcaster] = None
        if publish_static:
            self._static_broadcaster = StaticTransformBroadcaster(self)
            self._publish_static_gaze_frame()

        self._publisher = self.create_publisher(
            PointStamped, "/aria/gaze_xy_intersect", 10
        )
        self._marker_pub = None
        if self._visualize:
            self._marker_pub = self.create_publisher(Marker, "/aria/gaze_markers", 10)
        self._subscription = self.create_subscription(
            Vector3, "/aria/gaze_euler", self._on_gaze, 10
        )
        self.get_logger().info(
            "Listening on /aria/gaze_euler, publishing /aria/gaze_xy_intersect"
        )
        self.get_logger().info("ROS2 publishing enabled: /aria/gaze_xy_intersect")
        if self._visualize:
            self.get_logger().info("ROS2 publishing enabled: /aria/gaze_markers")
        if self._static_broadcaster is not None:
            self.get_logger().info("ROS2 publishing enabled: /tf_static (aria_camera_rgb -> aria_gaze)")

    def _publish_static_gaze_frame(self) -> None:
        if self._static_broadcaster is None:
            return
        tf_msg = TransformStamped()
        tf_msg.header.stamp = self.get_clock().now().to_msg()
        tf_msg.header.frame_id = self._camera_frame
        tf_msg.child_frame_id = self._gaze_frame
        tf_msg.transform.translation.x = float(self._gaze_x_offset)
        tf_msg.transform.translation.y = 0.0
        tf_msg.transform.translation.z = 0.0
        tf_msg.transform.rotation.x = 0.0
        tf_msg.transform.rotation.y = 0.0
        tf_msg.transform.rotation.z = 0.0
        tf_msg.transform.rotation.w = 1.0
        self._static_broadcaster.sendTransform(tf_msg)

    def _on_gaze(self, msg: Vector3) -> None:
        pitch = float(msg.x)
        yaw = float(msg.y)
        dir_gaze = _yaw_pitch_to_unit_vector(yaw, pitch)
        if self._visualize:
            self._publish_gaze_ray(dir_gaze)

        try:
            transform = self._tf_buffer.lookup_transform(
                self._base_frame, self._gaze_frame, rclpy.time.Time()
            )
        except Exception as exc:
            self.get_logger().warn(
                f"TF lookup failed {self._base_frame}<-{self._gaze_frame}: {exc}"
            )
            return

        origin = np.array(
            [
                float(transform.transform.translation.x),
                float(transform.transform.translation.y),
                float(transform.transform.translation.z),
            ],
            dtype=np.float64,
        )
        q = transform.transform.rotation
        rot = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        dir_base = rot @ dir_gaze

        if abs(dir_base[2]) < 1e-6:
            return
        t = -origin[2] / dir_base[2]
        if t <= 0.0:
            return
        hit = origin + t * dir_base

        point_msg = PointStamped()
        point_msg.header.stamp = self.get_clock().now().to_msg()
        point_msg.header.frame_id = self._base_frame
        point_msg.point.x = float(hit[0])
        point_msg.point.y = float(hit[1])
        point_msg.point.z = 0.0
        self._publisher.publish(point_msg)
        if self._visualize:
            self._publish_hit_marker(hit)

    def _publish_gaze_ray(self, dir_gaze: np.ndarray) -> None:
        if self._marker_pub is None:
            return
        ray_length = 1.0
        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = self._gaze_frame
        marker.ns = "aria_gaze"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.01
        marker.color.r = 0.1
        marker.color.g = 0.8
        marker.color.b = 0.1
        marker.color.a = 1.0
        marker.points = []
        marker.points.append(PointStamped().point)
        end_point = PointStamped().point
        end_point.x = float(dir_gaze[0] * ray_length)
        end_point.y = float(dir_gaze[1] * ray_length)
        end_point.z = float(dir_gaze[2] * ray_length)
        marker.points.append(end_point)
        marker.lifetime.sec = 0
        marker.lifetime.nanosec = 200_000_000
        self._marker_pub.publish(marker)

    def _publish_hit_marker(self, hit: np.ndarray) -> None:
        if self._marker_pub is None:
            return
        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = self._base_frame
        marker.ns = "aria_gaze"
        marker.id = 1
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.scale.x = 0.03
        marker.scale.y = 0.03
        marker.scale.z = 0.03
        marker.color.r = 1.0
        marker.color.g = 0.2
        marker.color.b = 0.2
        marker.color.a = 1.0
        marker.pose.position.x = float(hit[0])
        marker.pose.position.y = float(hit[1])
        marker.pose.position.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.lifetime.sec = 0
        marker.lifetime.nanosec = 200_000_000
        self._marker_pub.publish(marker)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gaze-x-offset",
        type=float,
        default=0.05,
        help="X offset from aria_camera_rgb to aria_gaze (meters).",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Publish visualization markers for gaze ray and XY intersection.",
    )
    parser.add_argument(
        "--no-static-gaze-frame",
        action="store_true",
        help="Do not publish the static aria_camera_rgb -> aria_gaze transform.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rclpy.init(args=None)
    node = GazeIntersectionNode(
        gaze_x_offset=args.gaze_x_offset,
        publish_static=not args.no_static_gaze_frame,
        visualize=args.visualize,
    )
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
