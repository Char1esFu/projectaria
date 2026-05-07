#!/usr/bin/env python3
"""
SegService – ROS2 service node.  Call /seg/infer (std_srvs/Trigger) to run
YOLO + SAM3 on the latest RGB frame and publish a masked point cloud on
/seg/point_cloud.  Set the target_label parameter before calling.

Requirements:
  ultralytics >= 8.3.237   (pip install -U ultralytics)
  sam3.pt checkpoint downloaded from https://huggingface.co/facebook/sam3
    and placed at yolo_model/sam3.pt

Service:
  /seg/infer  (std_srvs/srv/Trigger)  – start inference; returns immediately

Subscriptions:
  /zedr/zed_node/rgb/image_rect_color      (sensor_msgs/Image)
  /zedr/zed_node/depth/depth_registered    (sensor_msgs/Image)
  /zedr/zed_node/rgb/camera_info           (sensor_msgs/CameraInfo)

Publications:
  /seg/point_cloud  (sensor_msgs/PointCloud2)

CLI arguments:
  --visualize  show SAM3 mask overlay

ROS parameters:
  target_label – object label to segment   (set before calling /seg/infer)
  yolo_model   – path to YOLO .pt          (default: yolo_model/mixed.pt)
  yolo_conf    – YOLO confidence threshold (default: 0.25)
  depth_scale  – raw-depth-to-metres divisor (default: 1000.0 for 16UC1/mm)

Usage:
  ros2 param set /seg_service target_label banana
  ros2 service call /seg/infer std_srvs/srv/Trigger '{}'
"""

import argparse
import queue
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from std_srvs.srv import Trigger

YOLO_MODEL_DEFAULT = str(Path(__file__).parent.parent / "yolo_model" / "mixed.pt")
SAM3_MODEL_DEFAULT = str(Path(__file__).parent.parent / "yolo_model" / "sam3.pt")
_VIZ_WIN = "SAM3 – mask preview  (close or press ESC to publish)"


def _build_pointcloud2(header, points_xyz: np.ndarray,
                       rgb_packed: np.ndarray | None = None) -> PointCloud2:
    """Build PointCloud2.  rgb_packed: uint32 array (R<<16|G<<8|B), same length as points_xyz."""
    msg = PointCloud2()
    msg.header = header
    msg.height = 1
    msg.width = len(points_xyz)
    msg.is_dense = False
    msg.is_bigendian = False

    if rgb_packed is not None:
        # XYZRGB: rgb stored as float32 bit-cast from uint32 (standard ROS convention)
        rgb_f = rgb_packed.astype(np.uint32).view(np.float32)
        data = np.column_stack([points_xyz.astype(np.float32), rgb_f])
        msg.fields = [
            PointField(name="x",   offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name="y",   offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name="z",   offset=8,  datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        msg.point_step = 16
    else:
        data = points_xyz.astype(np.float32)
        msg.fields = [
            PointField(name="x", offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8,  datatype=PointField.FLOAT32, count=1),
        ]
        msg.point_step = 12

    msg.row_step = msg.point_step * len(points_xyz)
    msg.data = np.ascontiguousarray(data).tobytes()
    return msg


def _make_overlay(bgr: np.ndarray, mask: np.ndarray,
                  label: str, best_box) -> np.ndarray:
    """Render mask + bbox + label onto bgr and return the overlay image."""
    overlay = bgr.copy()

    green = np.zeros_like(bgr)
    green[mask] = (0, 220, 80)
    overlay = cv2.addWeighted(overlay, 0.55, green, 0.45, 0)

    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(overlay, contours, -1, (0, 255, 100), 2)

    if best_box is not None:
        x1, y1, x2, y2 = (int(v) for v in best_box)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 200, 0), 2)

    cv2.putText(overlay, label, (12, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 100), 2, cv2.LINE_AA)
    return overlay


class SegServiceNode(Node):
    def __init__(self, visualize: bool) -> None:
        super().__init__("seg_service")

        self._visualize = visualize

        self.declare_parameter("target_label", "")
        self.declare_parameter("yolo_model",  YOLO_MODEL_DEFAULT)
        self.declare_parameter("yolo_conf",   0.25)
        self.declare_parameter("depth_scale", 1000.0)

        yolo_path    = self.get_parameter("yolo_model").get_parameter_value().string_value
        self._conf   = self.get_parameter("yolo_conf").get_parameter_value().double_value
        self._dscale = self.get_parameter("depth_scale").get_parameter_value().double_value

        import torch
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        self.get_logger().info(f"Loading YOLO: {yolo_path}")
        from ultralytics import YOLO
        self._yolo = YOLO(yolo_path)

        self.get_logger().info(f"Loading SAM3: {SAM3_MODEL_DEFAULT}")
        from ultralytics import SAM
        self._sam3 = SAM(SAM3_MODEL_DEFAULT)

        self.get_logger().info("Warming up models (first CUDA init)...")
        _dummy = np.zeros((644, 644, 3), dtype=np.uint8)
        self._yolo(_dummy, conf=self._conf, verbose=False, device=self._device)
        self._sam3(_dummy, bboxes=[[0, 0, 100, 100]], verbose=False)
        self.get_logger().info("Models ready.")

        self._bridge    = CvBridge()
        self._rgb_msg:   Image      | None = None
        self._depth_msg: Image      | None = None
        self._cam_info:  CameraInfo | None = None
        self._lock = threading.Lock()
        self._busy = False
        self._viz_queue: queue.Queue = queue.Queue()

        self.create_service(Trigger, "/seg/infer", self._handle_infer)
        self.create_subscription(Image,      "/camera/camera/color/image_raw",
                                 self._on_rgb, 10)
        self.create_subscription(Image,      "/camera/camera/aligned_depth_to_color/image_raw",
                                 self._on_depth, 10)
        self.create_subscription(CameraInfo, "/camera/camera/color/camera_info",
                                 self._on_cam_info, 10)

        self._pc_pub = self.create_publisher(PointCloud2, "/seg/point_cloud", 10)

        mode = "YOLO+SAM3-box" + (" +visualize" if self._visualize else "")
        self.get_logger().info(f"SegService ready [{mode}] – call /seg/infer to trigger")

    # ------------------------------------------------------------------ #
    #  Image / info callbacks                                              #
    # ------------------------------------------------------------------ #

    def _on_rgb(self, msg: Image) -> None:
        with self._lock:
            self._rgb_msg = msg

    def _on_depth(self, msg: Image) -> None:
        with self._lock:
            self._depth_msg = msg

    def _on_cam_info(self, msg: CameraInfo) -> None:
        with self._lock:
            self._cam_info = msg

    # ------------------------------------------------------------------ #
    #  Service handler                                                     #
    # ------------------------------------------------------------------ #

    def _handle_infer(self, request: Trigger.Request,
                      response: Trigger.Response) -> Trigger.Response:
        t0 = time.monotonic()
        label = self.get_parameter("target_label").get_parameter_value().string_value.strip()
        if not label:
            response.success = False
            response.message = "target_label parameter is empty – set it first"
            return response

        if self._busy:
            response.success = False
            response.message = "Previous inference still running"
            return response

        with self._lock:
            rgb_msg   = self._rgb_msg
            depth_msg = self._depth_msg
            cam_info  = self._cam_info

        if rgb_msg is None:
            response.success = False
            response.message = "RGB image not yet received"
            return response
        if depth_msg is None:
            response.success = False
            response.message = "Depth image not yet received"
            return response
        if cam_info is None:
            response.success = False
            response.message = "CameraInfo not yet received"
            return response

        self._busy = True
        threading.Thread(
            target=self._infer,
            args=(label, rgb_msg, depth_msg, cam_info),
            daemon=True,
        ).start()

        response.success = True
        response.message = f"Inference started for '{label}'"
        self.get_logger().info(f"_handle_infer returned in {(time.monotonic()-t0)*1000:.1f} ms")
        return response

    # ------------------------------------------------------------------ #
    #  Inference (background thread)                                       #
    # ------------------------------------------------------------------ #

    def _infer(
        self,
        label:     str,
        rgb_msg:   Image,
        depth_msg: Image,
        cam_info:  CameraInfo,
    ) -> None:
        released = False
        try:
            bgr   = self._bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
            depth = self._bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
            best_box = None  # kept for visualisation

            # ---- segmentation -----------------------------------------
            results = self._yolo(bgr, conf=self._conf, verbose=False, device=self._device)
            best_conf = 0.0
            label_lc  = label.lower()
            for r in results:
                if r.boxes is None:
                    continue
                for i, cls_id in enumerate(r.boxes.cls.int().tolist()):
                    cls_name = self._yolo.names[cls_id].lower()
                    if label_lc in cls_name or cls_name in label_lc:
                        c = float(r.boxes.conf[i])
                        if c > best_conf:
                            best_conf = c
                            best_box  = r.boxes.xyxy[i].cpu().tolist()

            if best_box is None:
                self.get_logger().warn(
                    f"YOLO found nothing for '{label}' "
                    f"(known: {list(self._yolo.names.values())})"
                )
                return
            self.get_logger().info(
                f"YOLO: '{label}'  conf={best_conf:.2f}  "
                f"box=[{', '.join(f'{v:.0f}' for v in best_box)}]"
            )

            results = self._sam3(bgr, bboxes=[best_box])
            if not results or results[0].masks is None:
                self.get_logger().warn("SAM3 returned no mask for the bbox.")
                return
            masks_data = results[0].masks.data.cpu().numpy()  # (N, H, W)
            # pick highest-score mask when multiple are returned
            scores = results[0].boxes.conf if results[0].boxes is not None else None
            best_idx = int(scores.cpu().argmax()) if scores is not None else 0
            mask = masks_data[best_idx].astype(bool)

            # ---- depth back-projection ---------------------------------
            K  = np.array(cam_info.k).reshape(3, 3)
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]

            if depth_msg.encoding == "32FC1":
                depth_f = depth.astype(np.float32)
            else:                               # 16UC1 – RealSense default (mm)
                depth_f = depth.astype(np.float32) / self._dscale

            ys, xs = np.where(mask)
            d_vals = depth_f[ys, xs]
            valid  = (d_vals > 0.05) & np.isfinite(d_vals)
            xs, ys, d_vals = xs[valid], ys[valid], d_vals[valid]

            if len(xs) == 0:
                self.get_logger().warn("Mask obtained but all depth values are invalid.")
                return

            pts = np.column_stack([
                (xs - cx) * d_vals / fx,
                (ys - cy) * d_vals / fy,
                d_vals,
            ]).astype(np.float32)

            # Extract RGB colors from the original image (OpenCV is BGR)
            bgr_vals = bgr[ys, xs]                           # (N, 3)  B G R
            rgb_packed = (
                bgr_vals[:, 2].astype(np.uint32) << 16 |     # R
                bgr_vals[:, 1].astype(np.uint32) << 8  |     # G
                bgr_vals[:, 0].astype(np.uint32)              # B
            )

            pc_msg = _build_pointcloud2(depth_msg.header, pts, rgb_packed)
            self._pc_pub.publish(pc_msg)
            self.get_logger().info(
                f"Published {len(pts)} points on /seg/point_cloud  (label='{label}')"
            )

            # release before viz so new triggers are accepted while window is open
            self._busy = False
            released = True

            if self._visualize:
                self._viz_queue.put((bgr, mask, label, best_box))

        except Exception as exc:
            self.get_logger().error(f"Inference error: {exc}")
        finally:
            if not released:
                self._busy = False


# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description="SegService: YOLO+SAM3 → PointCloud2")
    parser.add_argument("--visualize", action="store_true",
                        help="show SAM3 mask overlay; publish after window is closed")
    args, _ = parser.parse_known_args()   # leave --ros-args untouched

    rclpy.init()
    node = SegServiceNode(visualize=args.visualize)
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        if args.visualize:
            # OpenCV GUI must run on the main thread; spin ROS2 in background.
            spin_thread = threading.Thread(target=executor.spin, daemon=True)
            spin_thread.start()
            win_open = False
            while rclpy.ok() and spin_thread.is_alive():
                # Drain queue – show only the latest frame if multiple arrived
                frame = None
                while True:
                    try:
                        frame = node._viz_queue.get_nowait()
                    except queue.Empty:
                        break

                if frame is not None:
                    bgr, mask, label, best_box = frame
                    overlay = _make_overlay(bgr, mask, label, best_box)
                    if not win_open:
                        cv2.namedWindow(_VIZ_WIN, cv2.WINDOW_NORMAL)
                        win_open = True
                    cv2.imshow(_VIZ_WIN, overlay)

                key = cv2.waitKey(30)
                if key == 27:                                        # ESC
                    cv2.destroyWindow(_VIZ_WIN)
                    win_open = False
                elif win_open:
                    if cv2.getWindowProperty(_VIZ_WIN, cv2.WND_PROP_VISIBLE) < 1:
                        win_open = False                             # window X'd out
        else:
            executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
