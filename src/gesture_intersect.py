"""
Subscribe to /hand_markers (published by gesture_wilor.py), extract wrist (joint 0)
and index fingertip (joint 8) for each detected hand, transform them into base_link,
visualize the pointing ray and its intersection with the z=0 (XY) plane.
"""

import argparse
import numpy as np

import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA, Header
import tf2_ros


# Joint indices (matching gesture_wilor.py)
WRIST_IDX = 0
INDEX_TIP_IDX = 8

# Colours for left / right hand rays
HAND_COLORS = {
    "L": ColorRGBA(r=0.0, g=0.8, b=1.0, a=0.9),   # cyan-ish
    "R": ColorRGBA(r=1.0, g=0.4, b=0.0, a=0.9),    # orange
}
INTERSECTION_COLOR = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)  # red sphere


class GestureIntersectNode(Node):
    def __init__(self) -> None:
        super().__init__("gesture_intersect")

        # TF2 listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Subscribe to hand markers published by gesture_wilor.py
        self.create_subscription(MarkerArray, "/hand_markers", self._hand_markers_cb, 10)

        # Publisher for pointing-ray + intersection visualisation
        self.vis_pub = self.create_publisher(MarkerArray, "/hand_pointing", 10)

        self.get_logger().info(
            "gesture_intersect running – subscribing to /hand_markers, "
            "publishing rays on /hand_pointing (frame: base_link)"
        )

    # ------------------------------------------------------------------
    # Callback
    # ------------------------------------------------------------------
    def _hand_markers_cb(self, msg: MarkerArray) -> None:
        """Process incoming hand markers."""
        # Collect SPHERE_LIST markers (one per hand) – they contain the 21 joints.
        sphere_markers: list[Marker] = []
        text_markers: list[Marker] = []
        for m in msg.markers:
            if m.action == Marker.DELETEALL:
                continue
            if m.type == Marker.SPHERE_LIST:
                sphere_markers.append(m)
            elif m.type == Marker.TEXT_VIEW_FACING:
                text_markers.append(m)

        if not sphere_markers:
            # No hands – publish DELETEALL to clear previous visualisation
            self._publish_clear()
            return

        # Build a hand_idx -> label map from text markers
        label_map: dict[int, str] = {}
        for tm in text_markers:
            # ns = "hand_{idx}", text = "R" or "L"
            label_map[tm.id // 10] = tm.text

        out = MarkerArray()
        # Clear previous markers
        clear = Marker()
        clear.action = Marker.DELETEALL
        out.markers.append(clear)

        marker_id = 0
        for sm in sphere_markers:
            if len(sm.points) < INDEX_TIP_IDX + 1:
                continue  # incomplete joint set

            hand_idx = sm.id // 10
            label = label_map.get(hand_idx, "R")
            color = HAND_COLORS.get(label, HAND_COLORS["R"])

            wrist_cam = sm.points[WRIST_IDX]
            tip_cam = sm.points[INDEX_TIP_IDX]

            # Transform both points from aria_camera_rgb -> base_link
            source_frame = sm.header.frame_id or "aria_camera_rgb"
            stamp = sm.header.stamp
            wrist_bl = self._transform_point(wrist_cam, source_frame, "base_link")
            tip_bl = self._transform_point(tip_cam, source_frame, "base_link")
            if wrist_bl is None or tip_bl is None:
                continue

            # --- Ray line (wrist -> index tip, extended) ---
            direction = np.array([
                tip_bl.x - wrist_bl.x,
                tip_bl.y - wrist_bl.y,
                tip_bl.z - wrist_bl.z,
            ])
            norm = np.linalg.norm(direction)
            if norm < 1e-6:
                continue

            # Extend the ray for visual clarity (2 m beyond tip)
            ray_end = np.array([tip_bl.x, tip_bl.y, tip_bl.z]) + direction / norm * 2.0

            line = Marker()
            line.header = Header(stamp=stamp, frame_id="base_link")
            line.ns = f"ray_{label}_{hand_idx}"
            line.id = marker_id; marker_id += 1
            line.type = Marker.LINE_STRIP
            line.action = Marker.ADD
            line.scale.x = 0.005
            line.pose.orientation.w = 1.0
            line.color = color
            line.points.append(wrist_bl)
            line.points.append(tip_bl)
            line.points.append(Point(x=float(ray_end[0]), y=float(ray_end[1]), z=float(ray_end[2])))
            out.markers.append(line)

            # --- XY-plane intersection (z = 0) ---
            if abs(direction[2]) > 1e-6:
                t_intersect = -wrist_bl.z / direction[2]
                if t_intersect > 0:  # intersection is in the forward direction
                    ix = wrist_bl.x + direction[0] * t_intersect
                    iy = wrist_bl.y + direction[1] * t_intersect

                    sphere = Marker()
                    sphere.header = Header(stamp=stamp, frame_id="base_link")
                    sphere.ns = f"intersect_{label}_{hand_idx}"
                    sphere.id = marker_id; marker_id += 1
                    sphere.type = Marker.SPHERE
                    sphere.action = Marker.ADD
                    sphere.pose.position = Point(x=float(ix), y=float(iy), z=0.0)
                    sphere.pose.orientation.w = 1.0
                    sphere.scale.x = 0.03
                    sphere.scale.y = 0.03
                    sphere.scale.z = 0.03
                    sphere.color = INTERSECTION_COLOR
                    out.markers.append(sphere)

                    # Dashed line from tip to intersection point for clarity
                    dash = Marker()
                    dash.header = Header(stamp=stamp, frame_id="base_link")
                    dash.ns = f"dash_{label}_{hand_idx}"
                    dash.id = marker_id; marker_id += 1
                    dash.type = Marker.LINE_STRIP
                    dash.action = Marker.ADD
                    dash.scale.x = 0.003
                    dash.pose.orientation.w = 1.0
                    dash.color = ColorRGBA(r=color.r, g=color.g, b=color.b, a=0.4)
                    dash.points.append(tip_bl)
                    dash.points.append(Point(x=float(ix), y=float(iy), z=0.0))
                    out.markers.append(dash)

                    # Label showing coordinates
                    text = Marker()
                    text.header = Header(stamp=stamp, frame_id="base_link")
                    text.ns = f"label_{label}_{hand_idx}"
                    text.id = marker_id; marker_id += 1
                    text.type = Marker.TEXT_VIEW_FACING
                    text.action = Marker.ADD
                    text.pose.position = Point(x=float(ix), y=float(iy), z=0.03)
                    text.pose.orientation.w = 1.0
                    text.scale.z = 0.02
                    text.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
                    text.text = f"{label}({ix:.2f},{iy:.2f})"
                    out.markers.append(text)

        self.vis_pub.publish(out)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _transform_point(self, pt: Point, source: str, target: str) -> Point | None:
        """Transform a Point from source frame to target frame via TF2 (latest available TF)."""
        try:
            tf = self.tf_buffer.lookup_transform(target, source, rclpy.time.Time())
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as exc:
            self.get_logger().warn(f"TF {source}->{target} unavailable: {exc}", throttle_duration_sec=2.0)
            return None
        # Apply transform manually: R * p + t
        q = tf.transform.rotation
        t = tf.transform.translation
        from scipy.spatial.transform import Rotation
        rot = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        p = np.array([pt.x, pt.y, pt.z])
        p_out = rot @ p + np.array([t.x, t.y, t.z])
        return Point(x=float(p_out[0]), y=float(p_out[1]), z=float(p_out[2]))

    def _publish_clear(self) -> None:
        ma = MarkerArray()
        m = Marker()
        m.action = Marker.DELETEALL
        ma.markers.append(m)
        self.vis_pub.publish(ma)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualise hand pointing ray & XY-plane intersection")
    parser.parse_args()

    rclpy.init()
    node = GestureIntersectNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
