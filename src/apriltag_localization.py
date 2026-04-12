import os
import argparse
import time
from typing import Optional

import cv2
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from utils.aria_rgb_stream import AriaRgbStream
from utils.ros_pose_publisher import RosPosePublisher


class AprilTagOverlay:
    """Detects AprilTag markers, draws them, and publishes poses via ROS2.

    Multi-marker joint PnP: all visible tags whose world poses are known from the
    static TF config contribute corner correspondences to a single solvePnP call,
    giving one camera-in-world estimate directly — no per-tag inversion or
    weighted averaging needed.

    AprilTag corner convention (pupil-apriltags / C apriltag library):
        corners[0] = lower-left,  corners[1] = lower-right,
        corners[2] = upper-right, corners[3] = upper-left
    Tag-local frame: x right, y up, z toward viewer (same as ArUco).
    """

    def __init__(
        self,
        tag_size_m: float,
        tag_family: str,
        allowed_tag_ids: Optional[list[int]],
        use_ema: bool,
        ema_alpha: float,
        ros: "RosPosePublisher",
    ) -> None:
        self.tag_size_m = tag_size_m
        self.allowed_tag_ids = allowed_tag_ids
        self.use_ema = bool(use_ema)
        self.ema_alpha = float(ema_alpha)
        self.ros = ros
        self.dist_coeffs = np.zeros((5, 1), dtype=np.float64)
        self._ema_state = {}
        self._prev_time = time.time()

        from pupil_apriltags import Detector
        self.detector = Detector(
            families=tag_family,
            nthreads=2,
            quad_decimate=1.0,
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.25,
        )

        self._allowed_id_set = set(allowed_tag_ids) if allowed_tag_ids else None

        half = tag_size_m / 2.0
        # Tag-local corners matching pupil-apriltags detection order:
        #   lower-left, lower-right, upper-right, upper-left
        # in tag frame (x right, y up, z out of tag).
        self._local_corners = np.array([
            [-half, -half, 0],  # lower-left  = corners[0]
            [ half, -half, 0],  # lower-right = corners[1]
            [ half,  half, 0],  # upper-right = corners[2]
            [-half,  half, 0],  # upper-left  = corners[3]
        ], dtype=np.float64)

    def draw(self, display_image: np.ndarray, camera_matrix: Optional[np.ndarray], key: int = -1) -> None:
        if camera_matrix is None:
            return
        curr_time = time.time()
        fps = 1.0 / (curr_time - self._prev_time) if (curr_time - self._prev_time) > 0 else 0.0
        self._prev_time = curr_time
        cv2.putText(display_image, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        gray = cv2.cvtColor(display_image, cv2.COLOR_RGB2GRAY)
        detections = self.detector.detect(gray)

        if not detections:
            return

        # Accumulate multi-marker PnP correspondences in world frame.
        all_obj_pts = []   # 3D corners in world frame
        all_img_pts = []   # corresponding 2D image corners
        valid_tags = []    # (tag_id, corners_2d, parent_frame, marker_t, marker_q)

        for det in detections:
            tag_id = int(det.tag_id)
            if self._allowed_id_set is not None and tag_id not in self._allowed_id_set:
                continue
            static_entry = self.ros.get_static_entry(tag_id)
            if static_entry is None:
                continue
            parent_frame, marker_t, marker_q = static_entry
            # Transform tag-local corners into world frame.
            world_corners = marker_t + Rotation.from_quat(marker_q).apply(self._local_corners)
            all_obj_pts.append(world_corners)
            all_img_pts.append(det.corners)  # already (4, 2) float64
            valid_tags.append((tag_id, det.corners, parent_frame, marker_t, marker_q))

        if not valid_tags:
            return

        obj_pts = np.vstack(all_obj_pts).astype(np.float32)  # (N*4, 3)
        img_pts = np.vstack(all_img_pts).astype(np.float32)  # (N*4, 2)

        # Joint PnP: solve for world-to-camera transform over all visible tags.
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

        # Draw each detected tag using its pose derived from the joint solution.
        for tag_id, corners_2d, _, marker_t_m, marker_q_m in valid_tags:
            R_mw = Rotation.from_quat(marker_q_m).as_matrix()
            # tag-in-camera: R_tc = R_wc @ R_mw,  t_tc = R_wc @ marker_t_m + tvec
            R_tc = R_wc @ R_mw
            t_tc = R_wc @ marker_t_m.reshape(3, 1) + tvec.reshape(3, 1)
            rvec_t, _ = cv2.Rodrigues(R_tc)
            self._draw_tag(display_image, tag_id, corners_2d, rvec_t, t_tc, camera_matrix)

        parent_frame = valid_tags[0][2]
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
        slerp = Slerp([0.0, 1.0], rotations)
        new_q = slerp(alpha).as_quat()
        self._ema_state[parent_frame] = (new_t, new_q)
        return new_t, new_q

    def _draw_tag(
        self,
        rgb_image: np.ndarray,
        tag_id: int,
        corners_2d: np.ndarray,
        rvec: np.ndarray,
        tvec: np.ndarray,
        camera_matrix: np.ndarray,
    ) -> None:
        pts = corners_2d.reshape(-1, 2).astype(np.int32)
        cv2.polylines(rgb_image, [pts], True, (0, 200, 255), 2)
        text_pos = (int(pts[3][0]), int(pts[3][1]) - 6)  # above upper-left corner
        cv2.putText(
            rgb_image, f"id:{tag_id}", text_pos,
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1, cv2.LINE_AA,
        )
        cv2.drawFrameAxes(rgb_image, camera_matrix, self.dist_coeffs, rvec, tvec, 0.03)


def run_rgb_apriltag_localization(
    device_ip: Optional[str] = None,
    tag_size_m: float = 0.13,
    tag_family: str = "tag36h11",
    update_iptables_rules: bool = False,
    allowed_tag_ids: Optional[list[int]] = None,
    use_ema: bool = True,
    ema_alpha: float = 0.2,
) -> None:
    static_tf_config_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "config", "apriltag_tf.json")
    )

    ros = RosPosePublisher(
        topic="/aria/cam_pose",
        marker_frame_prefix="apriltag_",
        camera_frame="aria_camera_rgb",
        static_tf_config_path=static_tf_config_path,
        camera_frame_correction_q_xyzw=None,
    )
    ros.setup()
    ros.publish_static_marker_tf()

    overlay = AprilTagOverlay(
        tag_size_m=tag_size_m,
        tag_family=tag_family,
        allowed_tag_ids=allowed_tag_ids,
        use_ema=use_ema,
        ema_alpha=ema_alpha,
        ros=ros,
    )

    stream = AriaRgbStream(
        device_ip=device_ip,
        update_iptables_rules=update_iptables_rules,
        window_name="Aria RGB AprilTag",
    )
    stream.add_overlay(overlay)
    try:
        stream.run()
    finally:
        ros.shutdown()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-ip", help="IP address to connect to the device")
    parser.add_argument("--tag-size-m", type=float, default=0.13)
    parser.add_argument("--tag-family", type=str, default="tag36h11",
                        help="AprilTag family: tag36h11, tag25h9, tag16h5, tagStandard41h12, ...")
    parser.add_argument("--update_iptables", default=True, action="store_true")
    parser.add_argument("--tag-ids", type=int, nargs="+", default=None)
    parser.add_argument("--ema-alpha", type=float, default=0.2)
    parser.add_argument("--disable-ema", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_rgb_apriltag_localization(
        device_ip=args.device_ip,
        tag_size_m=args.tag_size_m,
        tag_family=args.tag_family,
        update_iptables_rules=args.update_iptables,
        allowed_tag_ids=args.tag_ids,
        use_ema=not args.disable_ema,
        ema_alpha=args.ema_alpha,
    )


if __name__ == "__main__":
    main()
