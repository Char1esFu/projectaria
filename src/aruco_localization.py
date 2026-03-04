import os
import sys

import argparse
import json
import time
from typing import Optional
import aria.sdk as aria
import cv2
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from utils.common import quit_keypress, update_iptables
from projectaria_tools.core.calibration import (
    device_calibration_from_json_string,
    distort_by_calibration,
    get_linear_camera_calibration,
)
from projectaria_tools.core.sensor_data import ImageDataRecord



def _camera_matrix_from_calib(calib) -> np.ndarray:
    fx, fy = calib.get_focal_lengths()
    cx, cy = calib.get_principal_point()
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)

def _get_aruco_dictionary(name: str) -> cv2.aruco_Dictionary:
    aruco = cv2.aruco
    dict_id = getattr(aruco, name, None)
    if dict_id is None:
        raise ValueError(f"Unknown ArUco dictionary: {name}")
    return aruco.getPredefinedDictionary(dict_id)

def _invert_pose(rvec: np.ndarray, tvec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rotation_matrix, _ = cv2.Rodrigues(rvec)
    rotation_inv = rotation_matrix.T
    tvec_inv = -rotation_inv @ tvec.reshape(3, 1)
    rvec_inv, _ = cv2.Rodrigues(rotation_inv)
    return rvec_inv.reshape(3, 1), tvec_inv.reshape(3, 1)

def _compose_pose(
    parent_t: np.ndarray,
    parent_q_xyzw: np.ndarray,
    child_t: np.ndarray,
    child_q_xyzw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    parent_rot = Rotation.from_quat(parent_q_xyzw)
    child_rot = Rotation.from_quat(child_q_xyzw)
    composed_rot = parent_rot * child_rot
    composed_t = parent_t + parent_rot.apply(child_t)
    return composed_t, composed_rot.as_quat()

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

class ArucoLocalizer:
    def __init__(
        self,
        device_ip: Optional[str],
        marker_length_m: float,
        dictionary_name: str,
        update_iptables_rules: bool,
        undistort_width: int,
        undistort_height: int,
        undistort_focal_length: float,
        allowed_marker_ids: Optional[list[int]],
        use_ema: bool,
        ema_alpha: float,
    ) -> None:
        self.device_ip = device_ip
        self.marker_length_m = marker_length_m
        self.dictionary_name = dictionary_name
        self.update_iptables_rules = update_iptables_rules
        self.undistort_width = undistort_width
        self.undistort_height = undistort_height
        self.undistort_focal_length = undistort_focal_length
        self.allowed_marker_ids = allowed_marker_ids
        self.use_ema = bool(use_ema)
        self.ema_alpha = float(ema_alpha)

        self.ros2_topic = "/aria/cam_pose"
        self.ros2_marker_frame_prefix = "aruco_marker_"
        self.ros2_camera_frame = "aria_camera_rgb"
        
        # rgb camera frame and image are rotated 90 degrees. Compensate to match convention of camera facing forward and x right, y down, z forward.
        self.camera_frame_correction_q_xyzw = Rotation.from_euler("z", -90.0, degrees=True).as_quat()
        
        self.static_tf_config_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "config", "aruco_tf.json")
        )

        self.rgb_calib = None
        self.dst_calib = None
        self.camera_matrix = None
        self.dist_coeffs = np.zeros((5, 1), dtype=np.float64)

        self.aruco = None
        self.dictionary = None
        self.detector_params = None
        self.aruco_detector = None

        self.ros = None

        self.streaming_client = None
        self.observer = None

        self._ema_state = {}

    def run(self) -> None:
        if self.update_iptables_rules and sys.platform.startswith("linux"):
            update_iptables()

        self._load_rgb_calibration()
        self._setup_dst_calib()
        self._setup_aruco_detector()

        aria.set_log_level(aria.Level.Info)

        self._setup_ros()
        self.ros.publish_static_tf()
        self._setup_streaming()

        self._run_loop()

        print("Stop listening to RGB data")
        self.streaming_client.unsubscribe()
        self._shutdown_ros()

    def _run_loop(self) -> None:
        window_name = "Aria RGB ArUco"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1024, 1024)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
        cv2.moveWindow(window_name, 50, 50)

        allowed_id_set = set(self.allowed_marker_ids) if self.allowed_marker_ids else None

        while not quit_keypress():
            if self.observer.rgb_image is None:
                time.sleep(0.001)
                continue

            rgb_image = self._prepare_rgb_image(self.observer.rgb_image)
            self._detect_markers(rgb_image, allowed_id_set)

            cv2.imshow(window_name, np.rot90(rgb_image, -1))
            self.observer.rgb_image = None

    def _prepare_rgb_image(self, bgr_image: np.ndarray) -> np.ndarray:
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        if self.rgb_calib is not None and self.dst_calib is not None:
            rgb_image = distort_by_calibration(rgb_image, self.dst_calib, self.rgb_calib)
        return rgb_image

    def _detect_markers(self, rgb_image: np.ndarray, allowed_id_set: Optional[set[int]]):
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        corners, ids, _ = self.aruco_detector.detectMarkers(gray)

        if ids is None or len(ids) == 0:
            return

        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, self.marker_length_m, self.camera_matrix, self.dist_coeffs
        )

        pose_entries = []
        for marker_id, corner_set, rvec, tvec in zip(ids.flatten(), corners, rvecs, tvecs):
            if allowed_id_set is not None and int(marker_id) not in allowed_id_set:
                continue
            entry = self._compute_world_pose(int(marker_id), rvec, tvec)
            if entry is not None:
                pose_entries.append(entry)
            self._draw_marker(rgb_image, int(marker_id), corner_set, rvec, tvec)

        if not pose_entries:
            return

        if len(pose_entries) == 1:
            parent_frame, _, _, world_t, world_q_xyzw, _, _ = pose_entries[0]
            world_t, world_q_xyzw = self._apply_ema(parent_frame, world_t, world_q_xyzw)
            stamp = self.ros.publish_world_pose(parent_frame, world_t, world_q_xyzw)
            self.ros.publish_world_camera_tf(parent_frame, world_t, world_q_xyzw, stamp)
            return

        parent_frame = pose_entries[0][0]
        filtered_entries = [entry for entry in pose_entries if entry[0] == parent_frame]
        if len(filtered_entries) != len(pose_entries):
            print("Mixed parent frames detected; averaging only matching parent frame poses.")

        if len(filtered_entries) == 1:
            parent_frame, _, _, world_t, world_q_xyzw, _, _ = filtered_entries[0]
            world_t, world_q_xyzw = self._apply_ema(parent_frame, world_t, world_q_xyzw)
            stamp = self.ros.publish_world_pose(parent_frame, world_t, world_q_xyzw)
            self.ros.publish_world_camera_tf(parent_frame, world_t, world_q_xyzw, stamp)
            return

        world_ts = [entry[3] for entry in filtered_entries]
        world_quats = [entry[4] for entry in filtered_entries]
        cam_dists = np.array([entry[6] for entry in filtered_entries], dtype=np.float64)
        weights = 1.0 / (cam_dists + 1e-6)
        weights_sum = np.sum(weights)
        if weights_sum > 0.0:
            weights = weights / weights_sum
        else:
            weights = np.ones_like(weights) / float(len(weights))
        avg_t = np.sum(np.stack(world_ts, axis=0) * weights[:, None], axis=0)
        avg_q = average_quaternions(world_quats, weights)
        avg_t, avg_q = self._apply_ema(parent_frame, avg_t, avg_q)
        stamp = self.ros.publish_world_pose(parent_frame, avg_t, avg_q)
        self.ros.publish_world_camera_tf(parent_frame, avg_t, avg_q, stamp)

    def _compute_world_pose(
        self, marker_id: int, rvec: np.ndarray, tvec: np.ndarray
    ) -> Optional[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, float]]:
        rvec_cam_in_marker, tvec_cam_in_marker = _invert_pose(rvec, tvec)
        rotation_cam_in_marker, _ = cv2.Rodrigues(rvec_cam_in_marker)
        quat_xyzw = Rotation.from_matrix(rotation_cam_in_marker).as_quat()
        quat_cam_in_marker = np.array(
            [
                float(quat_xyzw[3]),
                float(quat_xyzw[0]),
                float(quat_xyzw[1]),
                float(quat_xyzw[2]),
            ],
            dtype=np.float64,
        )

        static_entry = self.ros.get_static_entry(marker_id)
        if static_entry is None:
            return None

        parent_frame, marker_t, marker_q = static_entry
        cam_t = np.array(
            [
                float(tvec_cam_in_marker[0][0]),
                float(tvec_cam_in_marker[1][0]),
                float(tvec_cam_in_marker[2][0]),
            ],
            dtype=np.float64,
        )
        cam_q = np.array(
            [
                float(quat_cam_in_marker[1]),
                float(quat_cam_in_marker[2]),
                float(quat_cam_in_marker[3]),
                float(quat_cam_in_marker[0]),
            ],
            dtype=np.float64,
        )
        world_t, world_q_xyzw = _compose_pose(marker_t, marker_q, cam_t, cam_q)
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
    ) -> None:
        pts = corner_set.reshape(-1, 2).astype(np.int32)
        cv2.polylines(rgb_image, [pts], True, (0, 255, 0), 2)
        text_pos = (int(pts[0][0]), int(pts[0][1]) - 6)
        cv2.putText(
            rgb_image,
            f"id:{marker_id}",
            text_pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
        cv2.drawFrameAxes(rgb_image, self.camera_matrix, self.dist_coeffs, rvec, tvec, 0.03)

    def _load_rgb_calibration(self) -> None:
        device_client = aria.DeviceClient()
        client_config = aria.DeviceClientConfig()
        if self.device_ip:
            client_config.ip_v4_address = self.device_ip
        device_client.set_client_config(client_config)
        device = None
        try:
            device = device_client.connect()
            sensors_calib_json = device.streaming_manager.sensors_calibration()
            sensors_calib = device_calibration_from_json_string(sensors_calib_json)
            self.rgb_calib = sensors_calib.get_camera_calib("camera-rgb")
        except Exception as exc:
            raise RuntimeError(f"Failed to load RGB calibration: {exc}") from exc
        finally:
            if device is not None:
                device_client.disconnect(device)

    def _setup_dst_calib(self) -> None:
        self.dst_calib = get_linear_camera_calibration(
            self.undistort_width,
            self.undistort_height,
            self.undistort_focal_length,
            "camera-rgb",
        )
        self.camera_matrix = _camera_matrix_from_calib(self.dst_calib)

    def _setup_aruco_detector(self) -> None:
        self.dictionary = _get_aruco_dictionary(self.dictionary_name)
        self.aruco = cv2.aruco
        self.detector_params = self.aruco.DetectorParameters()
        if hasattr(self.aruco, "CORNER_REFINE_SUBPIX"):
            self.detector_params.cornerRefinementMethod = self.aruco.CORNER_REFINE_SUBPIX
        if hasattr(self.aruco, "ArucoDetector"):
            self.aruco_detector = self.aruco.ArucoDetector(self.dictionary, self.detector_params)
        else:
            self.aruco_detector = None

    def _setup_ros(self) -> None:
        self.ros = RosPosePublisher(
            topic=self.ros2_topic,
            marker_frame_prefix=self.ros2_marker_frame_prefix,
            camera_frame=self.ros2_camera_frame,
            static_tf_config_path=self.static_tf_config_path,
            camera_frame_correction_q_xyzw=self.camera_frame_correction_q_xyzw,
        )
        self.ros.setup()
        if self.ros is None:
            raise RuntimeError("ROS2 publisher setup failed: ros instance is None.")

    def _setup_streaming(self) -> None:
        self.streaming_client = aria.StreamingClient()
        config = self.streaming_client.subscription_config
        config.subscriber_data_type = aria.StreamingDataType.Rgb
        config.message_queue_size[aria.StreamingDataType.Rgb] = 1
        options = aria.StreamingSecurityOptions()
        options.use_ephemeral_certs = True
        config.security_options = options
        self.streaming_client.subscription_config = config

        class StreamingClientObserver:
            def __init__(self):
                self.rgb_image = None
                self.last_print_time = {}
                self.sample_counts = {}

            def _tick(self, key: str, count: int = 1):
                now = time.time()
                if key not in self.last_print_time:
                    self.last_print_time[key] = now
                    self.sample_counts[key] = 0
                self.sample_counts[key] += count
                elapsed = now - self.last_print_time[key]
                if elapsed >= 1.0:
                    rate = self.sample_counts[key] / elapsed
                    # print(f"{key}: {rate:.2f} Hz")
                    self.last_print_time[key] = now
                    self.sample_counts[key] = 0

            def on_image_received(self, image: np.ndarray, record: ImageDataRecord):
                if record.camera_id != aria.CameraId.Rgb:
                    return
                self.rgb_image = image
                self._tick("RGB")

        self.observer = StreamingClientObserver()
        self.streaming_client.set_streaming_client_observer(self.observer)

        print("Start listening to RGB data")
        self.streaming_client.subscribe()

    def _shutdown_ros(self) -> None:
        if self.ros is not None:
            self.ros.shutdown()


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
                    np.array(
                        [float(translation[0]), float(translation[1]), float(translation[2])],
                        dtype=np.float64,
                    ),
                    np.array(
                        [float(rotation[0]), float(rotation[1]), float(rotation[2]), float(rotation[3])],
                        dtype=np.float64,
                    ),
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
    undistort_width: int = 1408,
    undistort_height: int = 1408,
    undistort_focal_length: float = 450.0,
    allowed_marker_ids: Optional[list[int]] = None,
    use_ema: bool = True,
    ema_alpha: float = 0.95,
) -> None:
    localizer = ArucoLocalizer(
        device_ip=device_ip,
        marker_length_m=marker_length_m,
        dictionary_name=dictionary_name,
        update_iptables_rules=update_iptables_rules,
        undistort_width=undistort_width,
        undistort_height=undistort_height,
        undistort_focal_length=undistort_focal_length,
        allowed_marker_ids=allowed_marker_ids,
        use_ema=use_ema,
        ema_alpha=ema_alpha,
    )
    localizer.run()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-ip", help="IP address to connect to the device")
    parser.add_argument(
        "--marker-length-m",
        type=float,
        default=0.047,
        help="Marker side length in meters.",
    )
    parser.add_argument(
        "--dictionary",
        type=str,
        default="DICT_4X4_50",
        help="OpenCV ArUco dictionary name (e.g., DICT_5X5_100).",
    )
    parser.add_argument(
        "--update_iptables",
        default=False,
        action="store_true",
        help="Update iptables to enable receiving the data stream, only for Linux.",
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
    parser.add_argument(
        "--marker-ids",
        type=int,
        nargs="+",
        default=None,
        help="Only detect/publish these marker IDs (space-separated).",
    )
    parser.add_argument(
        "--ema-alpha",
        type=float,
        default=0.2,
        help="EMA smoothing factor for pose (0 disables).",
    )
    parser.add_argument(
        "--disable-ema",
        action="store_true",
        help="Disable EMA smoothing regardless of --ema-alpha.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_rgb_aruco_localization(
        device_ip=args.device_ip,
        marker_length_m=args.marker_length_m,
        dictionary_name=args.dictionary,
        update_iptables_rules=args.update_iptables,
        undistort_width=args.undistort_width,
        undistort_height=args.undistort_height,
        undistort_focal_length=args.undistort_focal_length,
        allowed_marker_ids=args.marker_ids,
        use_ema=not args.disable_ema,
        ema_alpha=args.ema_alpha,
    )


if __name__ == "__main__":
    main()
