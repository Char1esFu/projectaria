import os
import argparse
import json
import time
from typing import Optional

import cv2
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from utils.aria_rgb_stream import AriaRgbStream


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
        self._prev_time = time.time()

        dict_id = getattr(cv2.aruco, dictionary_name, None)
        if dict_id is None:
            raise ValueError(f"Unknown ArUco dictionary: {dictionary_name}")
        dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
        detector_params = cv2.aruco.DetectorParameters()
        detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.aruco_detector = cv2.aruco.ArucoDetector(dictionary, detector_params)

        self._allowed_id_set = set(allowed_marker_ids) if allowed_marker_ids else None

        half = marker_length_m / 2.0
        # ArUco corner order: top-left, top-right, bottom-right, bottom-left
        # in marker-local frame (x right, y up, z out of marker).
        self._local_corners = np.array([
            [-half,  half, 0],
            [ half,  half, 0],
            [ half, -half, 0],
            [-half, -half, 0],
        ], dtype=np.float64)

    def draw(self, display_image: np.ndarray, camera_matrix: Optional[np.ndarray], key: int = -1) -> None:
        # camera_matrix is provided by AriaRgbStream from device calibration on first frame.
        if camera_matrix is None:
            return
        curr_time = time.time()
        fps = 1.0 / (curr_time - self._prev_time) if (curr_time - self._prev_time) > 0 else 0.0
        self._prev_time = curr_time
        cv2.putText(display_image, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        gray = cv2.cvtColor(display_image, cv2.COLOR_RGB2GRAY)
        corners, ids, _ = self.aruco_detector.detectMarkers(gray)

        if ids is None or len(ids) == 0:
            return

        # Accumulate multi-marker PnP correspondences in world frame.
        all_obj_pts = []   # 3D corners in world frame
        all_img_pts = []   # corresponding 2D image corners
        valid_markers = [] # (marker_id, corner_set, parent_frame, marker_t, marker_q)

        for marker_id, corner_set in zip(ids.flatten(), corners):
            if self._allowed_id_set is not None and int(marker_id) not in self._allowed_id_set:
                continue
            static_entry = self.ros.get_static_entry(int(marker_id))
            if static_entry is None:
                continue
            parent_frame, marker_t, marker_q = static_entry
            # Transform marker-local corners into world frame.
            world_corners = marker_t + Rotation.from_quat(marker_q).apply(self._local_corners)
            all_obj_pts.append(world_corners)
            all_img_pts.append(corner_set[0])
            valid_markers.append((int(marker_id), corner_set, parent_frame, marker_t, marker_q))

        if not valid_markers:
            return

        obj_pts = np.vstack(all_obj_pts).astype(np.float32)  # (N*4, 3)
        img_pts = np.vstack(all_img_pts).astype(np.float32)  # (N*4, 2)

        # Joint PnP: solve for the world-to-camera transform over all visible markers.
        # solvePnP returns rvec/tvec such that p_cam = R * p_world + t  (world-in-camera).
        success, rvec, tvec = cv2.solvePnP(
            obj_pts, img_pts, camera_matrix, self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            return

        R_wc, _ = cv2.Rodrigues(rvec)  # world-in-camera rotation
        # Invert to get camera-in-world pose.
        R_cw = R_wc.T
        world_t = (-R_cw @ tvec.reshape(3, 1)).flatten()
        world_q_xyzw = Rotation.from_matrix(R_cw).as_quat()

        # Draw each detected marker using its pose derived from the joint solution.
        for marker_id, corner_set, _, marker_t_m, marker_q_m in valid_markers:
            R_mw = Rotation.from_quat(marker_q_m).as_matrix()
            # marker-in-camera: R_mc = R_wc @ R_mw,  t_mc = R_wc @ marker_t_m + tvec
            R_mc = R_wc @ R_mw
            t_mc = R_wc @ marker_t_m.reshape(3, 1) + tvec.reshape(3, 1)
            rvec_m, _ = cv2.Rodrigues(R_mc)
            self._draw_marker(display_image, marker_id, corner_set, rvec_m, t_mc, camera_matrix)

        parent_frame = valid_markers[0][2]
        world_t, world_q_xyzw = self._apply_ema(parent_frame, world_t, world_q_xyzw)
        self.ros.publish_cam_pose(parent_frame, world_t, world_q_xyzw)

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
        # Slerp for rotation (linear interpolation would denormalize the quaternion).
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
        '''
        Returns (parent_frame, marker_t, marker_q) for the given marker_id, or None if not found.
        '''
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


def run_rgb_aruco_localization(
    device_ip: Optional[str] = None,
    marker_length_m: float = 0.13,
    dictionary_name: str = "DICT_4X4_50",
    update_iptables_rules: bool = False,
    allowed_marker_ids: Optional[list[int]] = None,
    use_ema: bool = True,
    ema_alpha: float = 0.95,
) -> None:
    static_tf_config_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "config", "aruco_tf.json")
    )

    ros = RosPosePublisher(
        topic="/aria/cam_pose",
        marker_frame_prefix="aruco_marker_",
        camera_frame="aria_camera_rgb",
        static_tf_config_path=static_tf_config_path,
        camera_frame_correction_q_xyzw=None,
    )
    ros.setup()
    ros.publish_static_marker_tf()

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
    parser.add_argument("--marker-length-m", type=float, default=0.13)
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
