import json
import os
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation


class RosPosePublisher:
    def __init__(
        self,
        topic: str,
        marker_frame_prefix: str,
        camera_frame: str,
        static_tf_config_path: str,
        camera_frame_correction_q_xyzw: Optional[np.ndarray] = None,
    ) -> None:
        self.topic = topic
        self.marker_frame_prefix = marker_frame_prefix
        self.camera_frame = camera_frame
        self.static_tf_config_path = static_tf_config_path
        self.camera_frame_correction_q_xyzw = camera_frame_correction_q_xyzw

        self.static_marker_transforms = {}
        self._warned_missing_static = set()

        self.rclpy = None
        self.PoseStamped = None
        self.TransformStamped = None
        self.ros_node = None
        self.ros_publisher = None
        self.ros_clock = None
        self.tf_broadcaster = None
        self.static_tf_broadcaster = None

    def setup(self) -> None:
        try:
            import rclpy
            from geometry_msgs.msg import PoseStamped
            from geometry_msgs.msg import TransformStamped
            from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

            if not rclpy.ok():
                rclpy.init(args=None)
            self.rclpy = rclpy
            self.PoseStamped = PoseStamped
            self.TransformStamped = TransformStamped
            self.ros_node = rclpy.create_node("aria_camera_pose_publisher")
            self.ros_publisher = self.ros_node.create_publisher(PoseStamped, self.topic, 10)
            self.ros_clock = self.ros_node.get_clock()
            self.tf_broadcaster = TransformBroadcaster(self.ros_node)
            self.static_tf_broadcaster = StaticTransformBroadcaster(self.ros_node)
            print(f"ROS2 publishing enabled: {self.topic}")
            print("ROS2 publishing enabled: /tf (dynamic transforms)")
            print("ROS2 publishing enabled: /tf_static (static marker transforms)")
        except Exception as exc:
            raise RuntimeError(f"ROS2 publisher unavailable: {exc}") from exc

    def publish_static_marker_tf(self) -> None:
        if not os.path.isfile(self.static_tf_config_path):
            return
        try:
            with open(self.static_tf_config_path, "r", encoding="utf-8") as config_file:
                config_data = json.load(config_file)
            static_transforms = []
            for entry in config_data.get("static_transforms", []):
                marker_id = int(entry.get("marker_id"))
                parent_frame = str(entry.get("parent_frame", "world"))
                child_frame = f"{self.marker_frame_prefix}{marker_id}"
                translation = entry.get("translation", [0.0, 0.0, 0.0])
                rotation = entry.get("rotation_xyzw", [0.0, 0.0, 0.0, 1.0])

                self.static_marker_transforms[marker_id] = (
                    parent_frame,
                    np.array([float(translation[0]), float(translation[1]), float(translation[2])], dtype=np.float64),
                    np.array([float(rotation[0]), float(rotation[1]), float(rotation[2]), float(rotation[3])], dtype=np.float64),
                )

                tf_msg = self.TransformStamped()
                tf_msg.header.stamp = (
                    self.ros_clock.now().to_msg() if self.ros_clock is not None else self.rclpy.time.Time().to_msg()
                )
                tf_msg.header.frame_id = parent_frame
                tf_msg.child_frame_id = child_frame
                tf_msg.transform.translation.x = float(translation[0])
                tf_msg.transform.translation.y = float(translation[1])
                tf_msg.transform.translation.z = float(translation[2])
                tf_msg.transform.rotation.x = float(rotation[0])
                tf_msg.transform.rotation.y = float(rotation[1])
                tf_msg.transform.rotation.z = float(rotation[2])
                tf_msg.transform.rotation.w = float(rotation[3])
                static_transforms.append(tf_msg)

            if static_transforms:
                self.static_tf_broadcaster.sendTransform(static_transforms)
                print(f"Published {len(static_transforms)} static TF(s) from {self.static_tf_config_path}")
        except Exception as exc:
            print(f"Failed to load static TF config {self.static_tf_config_path}: {exc}")

    def get_static_entry(self, marker_id: int):
        """Returns (parent_frame, marker_t, marker_q) for the given marker_id, or None if not found."""
        static_entry = self.static_marker_transforms.get(marker_id)
        if static_entry is None:
            if marker_id not in self._warned_missing_static:
                print(f"No static TF for marker {marker_id}; cannot publish world cam pose.")
                self._warned_missing_static.add(marker_id)
            return None
        return static_entry

    def publish_cam_pose(self, parent_frame: str, world_t: np.ndarray, world_q_xyzw: np.ndarray) -> None:
        if self.camera_frame_correction_q_xyzw is not None:
            world_q_xyzw = (
                Rotation.from_quat(world_q_xyzw) * Rotation.from_quat(self.camera_frame_correction_q_xyzw)
            ).as_quat()
        stamp = self.ros_clock.now().to_msg() if self.ros_clock is not None else self.rclpy.time.Time().to_msg()

        pose_msg = self.PoseStamped()
        pose_msg.header.stamp = stamp
        pose_msg.header.frame_id = parent_frame
        pose_msg.pose.position.x = float(world_t[0])
        pose_msg.pose.position.y = float(world_t[1])
        pose_msg.pose.position.z = float(world_t[2])
        pose_msg.pose.orientation.x = float(world_q_xyzw[0])
        pose_msg.pose.orientation.y = float(world_q_xyzw[1])
        pose_msg.pose.orientation.z = float(world_q_xyzw[2])
        pose_msg.pose.orientation.w = float(world_q_xyzw[3])
        self.ros_publisher.publish(pose_msg)

        tf_msg = self.TransformStamped()
        tf_msg.header.stamp = stamp
        tf_msg.header.frame_id = parent_frame
        tf_msg.child_frame_id = self.camera_frame
        tf_msg.transform.translation.x = float(world_t[0])
        tf_msg.transform.translation.y = float(world_t[1])
        tf_msg.transform.translation.z = float(world_t[2])
        tf_msg.transform.rotation.x = float(world_q_xyzw[0])
        tf_msg.transform.rotation.y = float(world_q_xyzw[1])
        tf_msg.transform.rotation.z = float(world_q_xyzw[2])
        tf_msg.transform.rotation.w = float(world_q_xyzw[3])
        self.tf_broadcaster.sendTransform(tf_msg)

    def shutdown(self) -> None:
        if self.ros_node is None:
            return
        try:
            self.ros_node.destroy_node()
            self.rclpy.shutdown()
        except Exception:
            pass
