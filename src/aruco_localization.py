import os
import argparse
import json
from typing import Optional

import cv2
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from utils.aria_rgb_stream import AriaRgbStream


def average_quaternions(quats_xyzw: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    ref = quats_xyzw[0]
    accum = np.zeros(4, dtype=np.float64)
    for quat, weight in zip(quats_xyzw, weights):
        if np.dot(ref, quat) < 0.0:
            accum -= weight * quat
        else:
            accum += weight * quat
    norm = np.linalg.norm(accum)
    if norm < 1e-12:
        return ref
    return accum / norm


class ArucoOverlay:
    """Detects ArUco markers, draws them, and publishes poses via ROS2."""

    def __init__(
        self,
        marker_length_m: float,
        dictionary_name: str,
        allowed_marker_ids: Optional[list[int]],
        use_ema: bool,
        ema_alpha: float,
        ros: "RosPosePublisher",
    ) -> None:
        self.marker_length_m = marker_length_m
        self.allowed_marker_ids = allowed_marker_ids
        self.use_ema = bool(use_ema)
        self.ema_alpha = float(ema_alpha)
        self.ros = ros
        self.dist_coeffs = np.zeros((5, 1), dtype=np.float64)
        self._ema_state = {}

        self.aruco = cv2.aruco
        dict_id = getattr(self.aruco, dictionary_name, None)
        if dict_id is None:
            raise ValueError(f"Unknown ArUco dictionary: {dictionary_name}")
        dictionary = self.aruco.getPredefinedDictionary(dict_id)
        detector_params = self.aruco.DetectorParameters()
        if hasattr(self.aruco, "CORNER_REFINE_SUBPIX"):
            detector_params.cornerRefinementMethod = self.aruco.CORNER_REFINE_SUBPIX
        if hasattr(self.aruco, "ArucoDetector"):
            self.aruco_detector = self.aruco.ArucoDetector(dictionary, detector_params)
        else:
            self.aruco_detector = None
        self._dictionary = dictionary
        self._detector_params = detector_params

        self._allowed_id_set = set(allowed_marker_ids) if allowed_marker_ids else None

    def draw(self, display_image: np.ndarray, camera_matrix: Optional[np.ndarray]) -> None:
        self._detect_markers(display_image, camera_matrix)

    def _detect_markers(self, display_image: np.ndarray, camera_matrix: Optional[np.ndarray]) -> None:
        gray = cv2.cvtColor(display_image, cv2.COLOR_RGB2GRAY)
        if self.aruco_detector is not None:
            corners, ids, _ = self.aruco_detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self._dictionary, parameters=self._detector_params)

        if ids is None or len(ids) == 0:
            return

        half = self.marker_length_m / 2.0
        obj_pts = np.array([
            [-half,  half, 0],
            [ half,  half, 0],
            [ half, -half, 0],
            [-half, -half, 0],
        ], dtype=np.float32)
        rvecs, tvecs = [], []
        for corner in corners:
            _, rvec, tvec = cv2.solvePnP(obj_pts, corner[0], camera_matrix, self.dist_coeffs)
            rvecs.append(rvec)
            tvecs.append(tvec)

        pose_entries = []
        for marker_id, corner_set, rvec, tvec in zip(ids.flatten(), corners, rvecs, tvecs):
            if self._allowed_id_set is not None and int(marker_id) not in self._allowed_id_set:
                continue
            entry = self._compute_world_pose(int(marker_id), rvec, tvec)
            if entry is not None:
                pose_entries.append(entry)
            self._draw_marker(display_image, int(marker_id), corner_set, rvec, tvec, camera_matrix)

        if not pose_entries:
            return

        parent_frame = pose_entries[0][0]
        world_ts = [entry[3] for entry in pose_entries]
        world_quats = [entry[4] for entry in pose_entries]
        cam_dists = np.array([entry[6] for entry in pose_entries], dtype=np.float64)
        weights = 1.0 / (cam_dists + 1e-6)
        weights_sum = np.sum(weights)
        weights = weights / weights_sum if weights_sum > 0.0 else np.ones_like(weights) / float(len(weights))
        avg_t = np.sum(np.stack(world_ts, axis=0) * weights[:, None], axis=0)
        avg_q = average_quaternions(world_quats, weights)
        avg_t, avg_q = self._apply_ema(parent_frame, avg_t, avg_q)
        stamp = self.ros.publish_world_pose(parent_frame, avg_t, avg_q)
        self.ros.publish_world_camera_tf(parent_frame, avg_t, avg_q, stamp)

    def _compute_world_pose(
        self, marker_id: int, rvec: np.ndarray, tvec: np.ndarray
    ) -> Optional[tuple]:
        rotation_marker_in_cam, _ = cv2.Rodrigues(rvec)
        rotation_cam_in_marker = rotation_marker_in_cam.T
        tvec_cam_in_marker = -rotation_cam_in_marker @ tvec.reshape(3, 1)
        quat_xyzw = Rotation.from_matrix(rotation_cam_in_marker).as_quat()
        quat_cam_in_marker = np.array(
            [float(quat_xyzw[3]), float(quat_xyzw[0]), float(quat_xyzw[1]), float(quat_xyzw[2])],
            dtype=np.float64,
        )

        static_entry = self.ros.get_static_entry(marker_id)
        if static_entry is None:
            return None

        parent_frame, marker_t, marker_q = static_entry
        cam_t = np.array(
            [float(tvec_cam_in_marker[0][0]), float(tvec_cam_in_marker[1][0]), float(tvec_cam_in_marker[2][0])],
            dtype=np.float64,
        )
        cam_q = np.array(
            [float(quat_cam_in_marker[1]), float(quat_cam_in_marker[2]),
             float(quat_cam_in_marker[3]), float(quat_cam_in_marker[0])],
            dtype=np.float64,
        )
        marker_rot = Rotation.from_quat(marker_q)
        world_t = marker_t + marker_rot.apply(cam_t)
        world_q_xyzw = (marker_rot * Rotation.from_quat(cam_q)).as_quat()
        cam_dist = float(np.linalg.norm(cam_t))
        return parent_frame, marker_t, marker_q, world_t, world_q_xyzw, marker_id, cam_dist

    def _apply_ema(self, parent_frame: str, t: np.ndarray, q_xyzw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if (not self.use_ema) or self.ema_alpha <= 0.0:
            return t, q_xyzw
        prev = self._ema_state.get(parent_frame)
        if prev is None:
            self._ema_state[parent_frame] = (t, q_xyzw)
            return t, q_xyzw
        prev_t, prev_q = prev
        alpha = self.ema_alpha
        new_t = (1.0 - alpha) * prev_t + alpha * t
        rotations = Rotation.from_quat([prev_q, q_xyzw])
        slerp = Slerp([0.0, 1.0], rotations)
        new_q = slerp(alpha).as_quat()
        self._ema_state[parent_frame] = (new_t, new_q)
        return new_t, new_q

    def _draw_marker(
        self,
        rgb_image: np.ndarray,
        marker_id: int,
        corner_set,
        rvec: np.ndarray,
        tvec: np.ndarray,
        camera_matrix: np.ndarray,
    ) -> None:
        pts = corner_set.reshape(-1, 2).astype(np.int32)
        cv2.polylines(rgb_image, [pts], True, (0, 255, 0), 2)
        text_pos = (int(pts[0][0]), int(pts[0][1]) - 6)
        cv2.putText(
            rgb_image, f"id:{marker_id}", text_pos,
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA,
        )
        cv2.drawFrameAxes(rgb_image, camera_matrix, self.dist_coeffs, rvec, tvec, 0.03)


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
            self.ros_node = rclpy.create_node("aruco_camera_pose_publisher")
            self.ros_publisher = self.ros_node.create_publisher(PoseStamped, self.topic, 10)
            self.ros_clock = self.ros_node.get_clock()
            self.tf_broadcaster = TransformBroadcaster(self.ros_node)
            self.static_tf_broadcaster = StaticTransformBroadcaster(self.ros_node)
            print(f"ROS2 publishing enabled: {self.topic}")
            print("ROS2 publishing enabled: /tf (dynamic transforms)")
            print("ROS2 publishing enabled: /tf_static (static marker transforms)")
        except Exception as exc:
            raise RuntimeError(f"ROS2 publisher unavailable: {exc}") from exc

    def publish_static_tf(self) -> None:
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
        static_entry = self.static_marker_transforms.get(marker_id)
        if static_entry is None:
            if marker_id not in self._warned_missing_static:
                print(f"No static TF for marker {marker_id}; cannot publish world cam pose.")
                self._warned_missing_static.add(marker_id)
            return None
        return static_entry

    def publish_world_pose(self, parent_frame: str, world_t: np.ndarray, world_q_xyzw: np.ndarray):
        if self.camera_frame_correction_q_xyzw is not None:
            world_q_xyzw = (
                Rotation.from_quat(world_q_xyzw) * Rotation.from_quat(self.camera_frame_correction_q_xyzw)
            ).as_quat()
        pose_msg = self.PoseStamped()
        if self.ros_clock is not None:
            pose_msg.header.stamp = self.ros_clock.now().to_msg()
        pose_msg.header.frame_id = parent_frame
        pose_msg.pose.position.x = float(world_t[0])
        pose_msg.pose.position.y = float(world_t[1])
        pose_msg.pose.position.z = float(world_t[2])
        pose_msg.pose.orientation.x = float(world_q_xyzw[0])
        pose_msg.pose.orientation.y = float(world_q_xyzw[1])
        pose_msg.pose.orientation.z = float(world_q_xyzw[2])
        pose_msg.pose.orientation.w = float(world_q_xyzw[3])
        self.ros_publisher.publish(pose_msg)
        return pose_msg.header.stamp

    def publish_camera_tf(self, marker_frame: str, cam_t: np.ndarray, cam_q: np.ndarray, stamp):
        if self.camera_frame_correction_q_xyzw is not None:
            cam_q = (Rotation.from_quat(cam_q) * Rotation.from_quat(self.camera_frame_correction_q_xyzw)).as_quat()
        tf_msg = self.TransformStamped()
        tf_msg.header.stamp = stamp
        tf_msg.header.frame_id = marker_frame
        tf_msg.child_frame_id = self.camera_frame
        tf_msg.transform.translation.x = float(cam_t[0])
        tf_msg.transform.translation.y = float(cam_t[1])
        tf_msg.transform.translation.z = float(cam_t[2])
        tf_msg.transform.rotation.w = float(cam_q[3])
        tf_msg.transform.rotation.x = float(cam_q[0])
        tf_msg.transform.rotation.y = float(cam_q[1])
        tf_msg.transform.rotation.z = float(cam_q[2])
        self.tf_broadcaster.sendTransform(tf_msg)

    def publish_world_camera_tf(self, parent_frame: str, world_t: np.ndarray, world_q: np.ndarray, stamp):
        if self.camera_frame_correction_q_xyzw is not None:
            world_q = (
                Rotation.from_quat(world_q) * Rotation.from_quat(self.camera_frame_correction_q_xyzw)
            ).as_quat()
        tf_msg = self.TransformStamped()
        tf_msg.header.stamp = stamp
        tf_msg.header.frame_id = parent_frame
        tf_msg.child_frame_id = self.camera_frame
        tf_msg.transform.translation.x = float(world_t[0])
        tf_msg.transform.translation.y = float(world_t[1])
        tf_msg.transform.translation.z = float(world_t[2])
        tf_msg.transform.rotation.x = float(world_q[0])
        tf_msg.transform.rotation.y = float(world_q[1])
        tf_msg.transform.rotation.z = float(world_q[2])
        tf_msg.transform.rotation.w = float(world_q[3])
        self.tf_broadcaster.sendTransform(tf_msg)

    def shutdown(self) -> None:
        if self.ros_node is None:
            return
        try:
            self.ros_node.destroy_node()
            self.rclpy.shutdown()
        except Exception:
            pass


def run_rgb_aruco_localization(
    device_ip: Optional[str] = None,
    marker_length_m: float = 0.04,
    dictionary_name: str = "DICT_4X4_50",
    update_iptables_rules: bool = False,
    allowed_marker_ids: Optional[list[int]] = None,
    use_ema: bool = True,
    ema_alpha: float = 0.95,
) -> None:
    static_tf_config_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "config", "aruco_tf.json")
    )
    camera_frame_correction_q_xyzw = Rotation.from_euler("z", -90.0, degrees=True).as_quat()

    ros = RosPosePublisher(
        topic="/aria/cam_pose",
        marker_frame_prefix="aruco_marker_",
        camera_frame="aria_camera_rgb",
        static_tf_config_path=static_tf_config_path,
        camera_frame_correction_q_xyzw=camera_frame_correction_q_xyzw,
    )
    ros.setup()
    ros.publish_static_tf()

    overlay = ArucoOverlay(
        marker_length_m=marker_length_m,
        dictionary_name=dictionary_name,
        allowed_marker_ids=allowed_marker_ids,
        use_ema=use_ema,
        ema_alpha=ema_alpha,
        ros=ros,
    )

    stream = AriaRgbStream(
        device_ip=device_ip,
        update_iptables_rules=update_iptables_rules,
        window_name="Aria RGB ArUco",
    )
    stream.add_overlay(overlay)
    try:
        stream.run()
    finally:
        ros.shutdown()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-ip", help="IP address to connect to the device")
    parser.add_argument("--marker-length-m", type=float, default=0.2)
    parser.add_argument("--dictionary", type=str, default="DICT_4X4_50")
    parser.add_argument("--update_iptables", default=True, action="store_true")
    parser.add_argument("--marker-ids", type=int, nargs="+", default=None)
    parser.add_argument("--ema-alpha", type=float, default=0.2)
    parser.add_argument("--disable-ema", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_rgb_aruco_localization(
        device_ip=args.device_ip,
        marker_length_m=args.marker_length_m,
        dictionary_name=args.dictionary,
        update_iptables_rules=args.update_iptables,
        allowed_marker_ids=args.marker_ids,
        use_ema=not args.disable_ema,
        ema_alpha=args.ema_alpha,
    )


if __name__ == "__main__":
    main()
