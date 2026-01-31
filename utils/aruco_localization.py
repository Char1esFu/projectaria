import argparse
import sys
import time
from typing import Callable, Optional

import aria.sdk as aria
import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from common import quit_keypress, update_iptables
from projectaria_tools.core.calibration import (
    device_calibration_from_json_string,
    distort_by_calibration,
    get_linear_camera_calibration,
)
from projectaria_tools.core.sensor_data import ImageDataRecord



def _get_focal_lengths(calib) -> tuple[float, float]:
    if hasattr(calib, "get_focal_lengths"):
        fx, fy = calib.get_focal_lengths()
        return float(fx), float(fy)
    if hasattr(calib, "get_focal_length"):
        f = float(calib.get_focal_length())
        return f, f
    if hasattr(calib, "get_intrinsics"):
        intr = calib.get_intrinsics()
        if len(intr) >= 2:
            return float(intr[0]), float(intr[1])
    raise ValueError("Unable to get focal lengths from calibration")

def _get_principal_point(calib) -> tuple[float, float]:
    if hasattr(calib, "get_principal_point"):
        cx, cy = calib.get_principal_point()
        return float(cx), float(cy)
    if hasattr(calib, "get_intrinsics"):
        intr = calib.get_intrinsics()
        if len(intr) >= 4:
            return float(intr[2]), float(intr[3])
    raise ValueError("Unable to get principal point from calibration")

def _camera_matrix_from_calib(calib) -> np.ndarray:
    fx, fy = _get_focal_lengths(calib)
    cx, cy = _get_principal_point(calib)
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)

def _get_aruco_dictionary(name: str) -> cv2.aruco_Dictionary:
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("OpenCV ArUco module not available. Install opencv-contrib-python.")
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

def run_rgb_aruco_localization(
    device_ip: Optional[str] = None,
    marker_length_m: float = 0.04,
    dictionary_name: str = "DICT_4X4_50",
    update_iptables_rules: bool = False,
    undistort: bool = True,
    undistort_width: int = 1408,
    undistort_height: int = 1408,
    undistort_focal_length: float = 450.0,
    ros2_publish: bool = False,
    ros2_topic: str = "/aruco/camera_pose",
    on_pose: Optional[Callable[[int, np.ndarray, np.ndarray], None]] = None,
) -> None:
    if update_iptables_rules and sys.platform.startswith("linux"):
        update_iptables()

    rgb_calib = None
    dst_calib = None
    camera_matrix = None
    dist_coeffs = np.zeros((5, 1), dtype=np.float64)

    device_client = aria.DeviceClient()
    client_config = aria.DeviceClientConfig()
    if device_ip:
        client_config.ip_v4_address = device_ip
    device_client.set_client_config(client_config)
    device = None
    try:
        device = device_client.connect()
        sensors_calib_json = device.streaming_manager.sensors_calibration()
        sensors_calib = device_calibration_from_json_string(sensors_calib_json)
        rgb_calib = sensors_calib.get_camera_calib("camera-rgb")
    except Exception as exc:
        raise RuntimeError(f"Failed to load RGB calibration: {exc}") from exc
    finally:
        if device is not None:
            device_client.disconnect(device)

    if undistort:
        dst_calib = get_linear_camera_calibration(
            undistort_width,
            undistort_height,
            undistort_focal_length,
            "camera-rgb",
        )
        camera_matrix = _camera_matrix_from_calib(dst_calib)
    else:
        camera_matrix = _camera_matrix_from_calib(rgb_calib)

    dictionary = _get_aruco_dictionary(dictionary_name)
    aruco = cv2.aruco
    detector_params = aruco.DetectorParameters()
    if hasattr(aruco, "CORNER_REFINE_SUBPIX"):
        detector_params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    if hasattr(aruco, "ArucoDetector"):
        aruco_detector = aruco.ArucoDetector(dictionary, detector_params)
    else:
        aruco_detector = None

    aria.set_log_level(aria.Level.Info)

    ros_node = None
    ros_publisher = None
    ros_clock = None
    if ros2_publish:
        try:
            import rclpy
            from geometry_msgs.msg import PoseStamped

            rclpy.init(args=None)
            ros_node = rclpy.create_node("aruco_camera_pose_publisher")
            ros_publisher = ros_node.create_publisher(PoseStamped, ros2_topic, 10)
            ros_clock = ros_node.get_clock()
            print(f"ROS2 publishing enabled: {ros2_topic}")
        except Exception as exc:
            print(f"ROS2 publisher unavailable: {exc}")
            ros_node = None
            ros_publisher = None
            ros_clock = None

    streaming_client = aria.StreamingClient()
    config = streaming_client.subscription_config
    config.subscriber_data_type = aria.StreamingDataType.Rgb
    config.message_queue_size[aria.StreamingDataType.Rgb] = 1
    options = aria.StreamingSecurityOptions()
    options.use_ephemeral_certs = True
    config.security_options = options
    streaming_client.subscription_config = config

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
                print(f"{key}: {rate:.2f} Hz")
                self.last_print_time[key] = now
                self.sample_counts[key] = 0

        def on_image_received(self, image: np.ndarray, record: ImageDataRecord):
            if record.camera_id != aria.CameraId.Rgb:
                return
            self.rgb_image = image
            self._tick("RGB")

    observer = StreamingClientObserver()
    streaming_client.set_streaming_client_observer(observer)

    print("Start listening to RGB data")
    streaming_client.subscribe()

    window_name = "Aria RGB ArUco"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1024, 1024)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
    cv2.moveWindow(window_name, 50, 50)

    last_print = 0.0
    with_poses = 0

    while not quit_keypress():
        if observer.rgb_image is None:
            time.sleep(0.001)
            continue

        rgb_image = cv2.cvtColor(observer.rgb_image, cv2.COLOR_BGR2RGB)
        if undistort and rgb_calib is not None and dst_calib is not None:
            rgb_image = distort_by_calibration(rgb_image, dst_calib, rgb_calib)

        gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        if aruco_detector is not None:
            corners, ids, _ = aruco_detector.detectMarkers(gray)
        else:
            corners, ids, _ = aruco.detectMarkers(
                gray, dictionary, parameters=detector_params
            )

        if ids is not None and len(ids) > 0:
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, marker_length_m, camera_matrix, dist_coeffs
            )
            for marker_id, corner_set, rvec, tvec in zip(
                ids.flatten(), corners, rvecs, tvecs
            ):
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

                print(
                    "Cam->Marker id=%d quat[w,x,y,z]=[%.6f, %.6f, %.6f, %.6f]"
                    % (
                        int(marker_id),
                        float(quat_cam_in_marker[0]),
                        float(quat_cam_in_marker[1]),
                        float(quat_cam_in_marker[2]),
                        float(quat_cam_in_marker[3]),
                    )
                )
                print(
                    "Cam->Marker id=%d tvec=[%.6f, %.6f, %.6f]"
                    % (
                        int(marker_id),
                        float(tvec_cam_in_marker[0][0]),
                        float(tvec_cam_in_marker[1][0]),
                        float(tvec_cam_in_marker[2][0]),
                    )
                )
                if ros_publisher is not None and ros_node is not None:
                    pose_msg = PoseStamped()
                    if ros_clock is not None:
                        pose_msg.header.stamp = ros_clock.now().to_msg()
                    pose_msg.header.frame_id = f"aruco_marker_{int(marker_id)}"
                    pose_msg.pose.position.x = float(tvec_cam_in_marker[0][0])
                    pose_msg.pose.position.y = float(tvec_cam_in_marker[1][0])
                    pose_msg.pose.position.z = float(tvec_cam_in_marker[2][0])
                    pose_msg.pose.orientation.w = float(quat_cam_in_marker[0])
                    pose_msg.pose.orientation.x = float(quat_cam_in_marker[1])
                    pose_msg.pose.orientation.y = float(quat_cam_in_marker[2])
                    pose_msg.pose.orientation.z = float(quat_cam_in_marker[3])
                    ros_publisher.publish(pose_msg)
                pts = corner_set.reshape(-1, 2).astype(np.int32)
                cv2.polylines(rgb_image, [pts], True, (0, 255, 0), 2)
                text_pos = (int(pts[0][0]), int(pts[0][1]) - 6)
                cv2.putText(
                    rgb_image,
                    f"id:{int(marker_id)}",
                    text_pos,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
                cv2.drawFrameAxes(rgb_image, camera_matrix, dist_coeffs, rvec, tvec, 0.03)
                if on_pose is not None:
                    on_pose(int(marker_id), rvec, tvec)
            with_poses += len(ids)

        now = time.time()
        if now - last_print >= 1.0:
            print(f"Detected markers: {with_poses} poses/s")
            with_poses = 0
            last_print = now

        cv2.imshow(window_name, np.rot90(rgb_image, -1))
        observer.rgb_image = None

    print("Stop listening to RGB data")
    streaming_client.unsubscribe()
    if ros_node is not None:
        try:
            ros_node.destroy_node()
            import rclpy

            rclpy.shutdown()
        except Exception:
            pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-ip", help="IP address to connect to the device")
    parser.add_argument(
        "--marker-length-m",
        type=float,
        default=0.053,
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
        "--no-undistort",
        action="store_true",
        help="Disable RGB undistortion (enabled by default).",
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
        "--ros2-publish",
        action="store_true",
        help="Publish camera pose as ROS2 PoseStamped.",
    )
    parser.add_argument(
        "--ros2-topic",
        type=str,
        default="/aruco/camera_pose",
        help="ROS2 topic for PoseStamped.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_rgb_aruco_localization(
        device_ip=args.device_ip,
        marker_length_m=args.marker_length_m,
        dictionary_name=args.dictionary,
        update_iptables_rules=args.update_iptables,
        undistort=not args.no_undistort,
        undistort_width=args.undistort_width,
        undistort_height=args.undistort_height,
        undistort_focal_length=args.undistort_focal_length,
        ros2_publish=args.ros2_publish,
        ros2_topic=args.ros2_topic,
    )


if __name__ == "__main__":
    main()
