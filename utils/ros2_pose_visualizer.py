import copy
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import open3d as o3d
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from scipy.spatial.transform import Rotation


@dataclass
class PoseData:
    position: np.ndarray
    quaternion_wxyz: np.ndarray


class PoseStampedViewer(Node):
    def __init__(self, topic: str) -> None:
        super().__init__("pose_stamped_viewer")
        self._pose: Optional[PoseData] = None
        self._pose_updated = False
        self.create_subscription(PoseStamped, topic, self._on_pose, 10)

    def _on_pose(self, msg: PoseStamped) -> None:
        position = np.array(
            [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z],
            dtype=np.float64,
        )
        quaternion_wxyz = np.array(
            [
                msg.pose.orientation.w,
                msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z,
            ],
            dtype=np.float64,
        )
        self._pose = PoseData(position=position, quaternion_wxyz=quaternion_wxyz)
        self._pose_updated = True

    def consume_pose(self) -> Optional[PoseData]:
        if not self._pose_updated:
            return None
        self._pose_updated = False
        return self._pose


def _pose_to_transform(pose: PoseData) -> np.ndarray:
    quat_xyzw = np.array(
        [pose.quaternion_wxyz[1], pose.quaternion_wxyz[2], pose.quaternion_wxyz[3], pose.quaternion_wxyz[0]],
        dtype=np.float64,
    )
    rotation = Rotation.from_quat(quat_xyzw).as_matrix()
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = pose.position
    return transform


def _create_grid(size_m: float = 1.0, step_m: float = 0.1) -> o3d.geometry.LineSet:
    half = size_m / 2.0
    points = []
    lines = []
    color = [0.6, 0.6, 0.6]
    idx = 0
    num_steps = int(size_m / step_m)
    for i in range(num_steps + 1):
        offset = -half + i * step_m
        points.append([offset, -half, 0.0])
        points.append([offset, half, 0.0])
        lines.append([idx, idx + 1])
        idx += 2
        points.append([-half, offset, 0.0])
        points.append([half, offset, 0.0])
        lines.append([idx, idx + 1])
        idx += 2
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(np.array(points, dtype=np.float64))
    line_set.lines = o3d.utility.Vector2iVector(np.array(lines, dtype=np.int32))
    line_set.colors = o3d.utility.Vector3dVector([color for _ in range(len(lines))])
    return line_set


def main() -> None:
    rclpy.init(args=None)
    node = PoseStampedViewer("/aruco/camera_pose")

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="PoseStamped Viewer", width=960, height=720)

    marker_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.04)
    camera_frame_base = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.03)
    camera_frame = copy.deepcopy(camera_frame_base)
    grid = _create_grid(size_m=1.0, step_m=0.1)

    line_points = o3d.utility.Vector3dVector(
        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float64)
    )
    line_set = o3d.geometry.LineSet()
    line_set.points = line_points
    line_set.lines = o3d.utility.Vector2iVector(np.array([[0, 1]], dtype=np.int32))
    line_set.colors = o3d.utility.Vector3dVector(np.array([[1.0, 0.2, 0.2]], dtype=np.float64))

    last_print_time = 0.0

    vis.add_geometry(grid)
    vis.add_geometry(marker_frame)
    vis.add_geometry(camera_frame)
    vis.add_geometry(line_set)

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.0)
            pose = node.consume_pose()
            if pose is not None:
                transform = _pose_to_transform(pose)
                vis.remove_geometry(camera_frame, reset_bounding_box=False)
                camera_frame = copy.deepcopy(camera_frame_base)
                camera_frame.transform(transform)
                vis.add_geometry(camera_frame, reset_bounding_box=False)

                line_set.points = o3d.utility.Vector3dVector(
                    np.array([[0.0, 0.0, 0.0], pose.position], dtype=np.float64)
                )
                vis.update_geometry(line_set)

                now = time.time()
                if now - last_print_time >= 1.0:
                    distance = float(np.linalg.norm(pose.position))
                    print(f"Camera distance from marker origin: {distance:.4f} m")
                    last_print_time = now

            if not vis.poll_events():
                break
            vis.update_renderer()
    finally:
        vis.destroy_window()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
