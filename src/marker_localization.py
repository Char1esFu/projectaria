import os
import argparse
import time
from typing import Optional

import cv2
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from utils.aria_rgb_stream import AriaRgbStream
from utils.ros_pose_publisher import RosPosePublisher

_MARKER_CONFIGS = {
    "aruco":    {"frame_prefix": "aruco_marker_",  "config": "aruco_tf.json",    "window": "Aria RGB ArUco",    "color": (0, 255, 0)},
    "apriltag": {"frame_prefix": "apriltag_",       "config": "apriltag_tf.json", "window": "Aria RGB AprilTag", "color": (0, 200, 255)},
}


class MarkerOverlay:
    """Detects ArUco or AprilTag markers and publishes camera pose via ROS2.

    Multi-marker joint PnP: all visible markers with a known world pose from
    the static TF config contribute corner correspondences to a single solvePnP
    call (SOLVEPNP_ITERATIVE / Levenberg-Marquardt), giving one camera-in-world
    estimate directly.

    Select marker type with --marker-type aruco|apriltag.
    """

    def __init__(
        self,
        marker_size_m: float,
        marker_type: str,
        allowed_ids: Optional[list[int]],
        use_ema: bool,
        ema_alpha: float,
        ros: "RosPosePublisher",
        dictionary_name: str = "DICT_4X4_50",
        tag_family: str = "tag36h11",
    ) -> None:
        if marker_type not in _MARKER_CONFIGS:
            raise ValueError(f"Unknown marker_type '{marker_type}'. Choose: {list(_MARKER_CONFIGS)}")

        self.marker_type = marker_type
        self.allowed_ids = allowed_ids
        self.use_ema = bool(use_ema)
        self.ema_alpha = float(ema_alpha)
        self.ros = ros
        self.dist_coeffs = np.zeros((5, 1), dtype=np.float64)
        self._draw_color = _MARKER_CONFIGS[marker_type]["color"]
        self._ema_state = {}
        self._prev_time = time.time()
        self._allowed_id_set = set(allowed_ids) if allowed_ids else None

        half = marker_size_m / 2.0

        if marker_type == "aruco":
            dict_id = getattr(cv2.aruco, dictionary_name, None)
            if dict_id is None:
                raise ValueError(f"Unknown ArUco dictionary: {dictionary_name}")
            dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
            params = cv2.aruco.DetectorParameters()
            params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
            self._detector = cv2.aruco.ArucoDetector(dictionary, params)
            # corner order: top-left, top-right, bottom-right, bottom-left
            # tag frame: x right, y up, z out of marker
            self._local_corners = np.array([
                [-half,  half, 0],
                [ half,  half, 0],
                [ half, -half, 0],
                [-half, -half, 0],
            ], dtype=np.float64)

        else:  # apriltag
            from pupil_apriltags import Detector
            self._detector = Detector(
                families=tag_family,
                nthreads=2,
                quad_decimate=1.0,
                quad_sigma=0.0,
                refine_edges=1,
                decode_sharpening=0.25,
            )
            # corner order: lower-left, lower-right, upper-right, upper-left
            # tag frame: x right, y up, z out of tag
            self._local_corners = np.array([
                [-half, -half, 0],
                [ half, -half, 0],
                [ half,  half, 0],
                [-half,  half, 0],
            ], dtype=np.float64)

    def _detect(self, gray: np.ndarray) -> list[tuple[int, np.ndarray]]:
        """Return list of (marker_id, corners_2d (4,2)) for all detected markers."""
        if self.marker_type == "aruco":
            corners, ids, _ = self._detector.detectMarkers(gray)
            if ids is None or len(ids) == 0:
                return []
            return [(int(id_), c[0]) for id_, c in zip(ids.flatten(), corners)]
        else:
            return [(det.tag_id, det.corners) for det in self._detector.detect(gray)]

    def draw(self, display_image: np.ndarray, camera_matrix: Optional[np.ndarray], key: int = -1) -> None:
        if camera_matrix is None:
            return
        curr_time = time.time()
        fps = 1.0 / (curr_time - self._prev_time) if (curr_time - self._prev_time) > 0 else 0.0
        self._prev_time = curr_time
        cv2.putText(display_image, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        gray = cv2.cvtColor(display_image, cv2.COLOR_RGB2GRAY)
        detections = self._detect(gray)

        # Accumulate multi-marker PnP correspondences in world frame.
        all_obj_pts = []  # 3D corners in world frame
        all_img_pts = []  # corresponding 2D image corners
        valid = []        # (marker_id, corners_2d, parent_frame, marker_t, marker_q)

        for marker_id, corners_2d in detections:
            if self._allowed_id_set is not None and marker_id not in self._allowed_id_set:
                continue
            static_entry = self.ros.get_static_entry(marker_id)
            if static_entry is None:
                continue
            parent_frame, marker_t, marker_q = static_entry
            world_corners = marker_t + Rotation.from_quat(marker_q).apply(self._local_corners)
            all_obj_pts.append(world_corners)
            all_img_pts.append(corners_2d)
            valid.append((marker_id, corners_2d, parent_frame, marker_t, marker_q))

        if not valid:
            return

        obj_pts = np.vstack(all_obj_pts).astype(np.float32)  # (N*4, 3)
        img_pts = np.vstack(all_img_pts).astype(np.float32)  # (N*4, 2)

        # Joint PnP: solve for world-to-camera transform over all visible markers.
        # solvePnP returns rvec/tvec such that p_cam = R * p_world + t  (world-in-camera).
        success, rvec, tvec = cv2.solvePnP(
            obj_pts, img_pts, camera_matrix, self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            return

        R_wc, _ = cv2.Rodrigues(rvec)
        R_cw = R_wc.T
        world_t = (-R_cw @ tvec.reshape(3, 1)).flatten()
        world_q_xyzw = Rotation.from_matrix(R_cw).as_quat()

        # Draw each marker using its pose derived from the joint solution.
        for marker_id, corners_2d, _, marker_t_m, marker_q_m in valid:
            R_mw = Rotation.from_quat(marker_q_m).as_matrix()
            R_mc = R_wc @ R_mw
            t_mc = R_wc @ marker_t_m.reshape(3, 1) + tvec.reshape(3, 1)
            rvec_m, _ = cv2.Rodrigues(R_mc)
            self._draw_marker(display_image, marker_id, corners_2d, rvec_m, t_mc, camera_matrix)

        parent_frame = valid[0][2]
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
        rotations = Rotation.from_quat([prev_q, q_xyzw])
        new_q = Slerp([0.0, 1.0], rotations)(alpha).as_quat()
        self._ema_state[parent_frame] = (new_t, new_q)
        return new_t, new_q

    def _draw_marker(
        self,
        rgb_image: np.ndarray,
        marker_id: int,
        corners_2d: np.ndarray,
        rvec: np.ndarray,
        tvec: np.ndarray,
        camera_matrix: np.ndarray,
    ) -> None:
        pts = corners_2d.reshape(-1, 2).astype(np.int32)
        cv2.polylines(rgb_image, [pts], True, self._draw_color, 2)
        cv2.putText(
            rgb_image, f"id:{marker_id}",
            (int(pts[0][0]), int(pts[0][1]) - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, self._draw_color, 1, cv2.LINE_AA,
        )
        cv2.drawFrameAxes(rgb_image, camera_matrix, self.dist_coeffs, rvec, tvec, 0.03)


def run_marker_localization(
    device_ip: Optional[str] = None,
    marker_size_m: float = 0.13,
    marker_type: str = "aruco",
    update_iptables_rules: bool = False,
    allowed_ids: Optional[list[int]] = None,
    use_ema: bool = True,
    ema_alpha: float = 0.2,
    dictionary_name: str = "DICT_4X4_50",
    tag_family: str = "tag36h11",
) -> None:
    cfg = _MARKER_CONFIGS[marker_type]
    static_tf_config_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "config", cfg["config"])
    )

    ros = RosPosePublisher(
        topic="/aria/cam_pose",
        marker_frame_prefix=cfg["frame_prefix"],
        camera_frame="aria_camera_rgb",
        static_tf_config_path=static_tf_config_path,
    )
    ros.setup()
    ros.publish_static_marker_tf()

    overlay = MarkerOverlay(
        marker_size_m=marker_size_m,
        marker_type=marker_type,
        allowed_ids=allowed_ids,
        use_ema=use_ema,
        ema_alpha=ema_alpha,
        ros=ros,
        dictionary_name=dictionary_name,
        tag_family=tag_family,
    )

    stream = AriaRgbStream(
        device_ip=device_ip,
        update_iptables_rules=update_iptables_rules,
        window_name=cfg["window"],
    )
    stream.add_overlay(overlay)
    try:
        stream.run()
    finally:
        ros.shutdown()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aria RGB marker-based camera localization")
    parser.add_argument("--device-ip", help="IP address to connect to the device")
    parser.add_argument("--marker-type", choices=["aruco", "apriltag"], default="aruco",
                        help="Marker detection backend (default: aruco)")
    parser.add_argument("--marker-size-m", type=float, default=0.13,
                        help="Physical marker side length in metres")
    parser.add_argument("--marker-ids", type=int, nargs="+", default=None,
                        help="Whitelist of marker IDs to use (default: all in config)")
    parser.add_argument("--ema-alpha", type=float, default=0.2)
    parser.add_argument("--disable-ema", action="store_true")
    parser.add_argument("--update_iptables", default=True, action="store_true")
    # ArUco-specific
    parser.add_argument("--dictionary", type=str, default="DICT_4X4_50",
                        help="ArUco dictionary name (aruco only)")
    # AprilTag-specific
    parser.add_argument("--tag-family", type=str, default="tag36h11",
                        help="AprilTag family: tag36h11, tag25h9, tag16h5, ... (apriltag only)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_marker_localization(
        device_ip=args.device_ip,
        marker_size_m=args.marker_size_m,
        marker_type=args.marker_type,
        update_iptables_rules=args.update_iptables,
        allowed_ids=args.marker_ids,
        use_ema=not args.disable_ema,
        ema_alpha=args.ema_alpha,
        dictionary_name=args.dictionary,
        tag_family=args.tag_family,
    )


if __name__ == "__main__":
    main()
