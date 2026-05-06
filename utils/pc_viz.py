#!/usr/bin/env python3
"""
pc_viz.py – Subscribes to /seg/point_cloud and visualizes it with Open3D.

Each new message replaces the previous result; the window stays open.

Usage:
  python3 src/pc_viz.py
"""

import queue
import threading

import numpy as np
import open3d as o3d
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2


def _parse_xyzrgb(msg: PointCloud2) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Extract XYZ and (optional) RGB from a PointCloud2 message.
    Returns (xyz: float32 (N,3), colors: float32 (N,3) or None)
    """
    field_names = {f.name for f in msg.fields}
    n = msg.width * msg.height
    has_rgb = "rgb" in field_names

    if has_rgb:
        # point_step=16: x(0) y(4) z(8) rgb(12)
        raw = np.frombuffer(bytes(msg.data), dtype=np.float32).reshape(n, -1)
        xyz = raw[:, :3]
        rgb_packed = raw[:, 3].view(np.uint32)
        r = ((rgb_packed >> 16) & 0xFF).astype(np.float32) / 255.0
        g = ((rgb_packed >>  8) & 0xFF).astype(np.float32) / 255.0
        b = ( rgb_packed        & 0xFF).astype(np.float32) / 255.0
        colors = np.column_stack([r, g, b])
    else:
        raw = np.frombuffer(bytes(msg.data), dtype=np.float32).reshape(n, -1)
        xyz = raw[:, :3]
        colors = None

    valid = np.isfinite(xyz).all(axis=1)
    return xyz[valid], (colors[valid] if colors is not None else None)


class PCVizNode(Node):
    def __init__(self, viz_queue: "queue.Queue") -> None:
        super().__init__("pc_viz")
        self._viz_queue = viz_queue
        self.create_subscription(PointCloud2, "/seg/point_cloud", self._on_pc, 10)
        self.get_logger().info("PCVizNode ready – listening on /seg/point_cloud ...")

    def _on_pc(self, msg: PointCloud2) -> None:
        xyz, colors = _parse_xyzrgb(msg)
        self.get_logger().info(
            f"Received point cloud: {len(xyz)} points  "
            f"({'with RGB' if colors is not None else 'no color'})"
        )
        # Drop any pending frames; keep only the latest
        while not self._viz_queue.empty():
            try:
                self._viz_queue.get_nowait()
            except queue.Empty:
                break
        self._viz_queue.put((xyz, colors))


def _run_visualizer(viz_queue: "queue.Queue") -> None:
    """Main thread: Open3D render loop. Open3D GUI must run on the main thread."""
    vis = o3d.visualization.Visualizer()
    vis.create_window("Point Cloud – /seg/point_cloud", width=1280, height=720)

    opt = vis.get_render_option()
    opt.background_color = np.array([0.1, 0.1, 0.1])
    opt.point_size = 2.0

    pcd = o3d.geometry.PointCloud()
    geometry_added = False

    while True:
        # Drain the queue, keep only the latest frame
        frame = None
        while True:
            try:
                frame = viz_queue.get_nowait()
            except queue.Empty:
                break

        if frame is not None:
            xyz, colors = frame
            if len(xyz) > 0:
                pcd.points = o3d.utility.Vector3dVector(xyz)
                if colors is not None:
                    pcd.colors = o3d.utility.Vector3dVector(colors)
                else:
                    # Fallback: depth-based gradient (near=green, far=blue)
                    z = xyz[:, 2]
                    t = (z - z.min()) / (z.max() - z.min() + 1e-6)
                    fb = np.zeros((len(xyz), 3), dtype=np.float32)
                    fb[:, 1] = 1.0 - t
                    fb[:, 2] = t
                    pcd.colors = o3d.utility.Vector3dVector(fb)

                if not geometry_added:
                    vis.add_geometry(pcd)
                    geometry_added = True
                else:
                    vis.update_geometry(pcd)
                vis.reset_view_point(True)

        # poll_events returns False when the window is closed
        if not vis.poll_events():
            break
        vis.update_renderer()

    vis.destroy_window()


def main() -> None:
    rclpy.init()
    viz_queue: queue.Queue = queue.Queue()
    node = PCVizNode(viz_queue)

    try:
        # Open3D GUI must run on the main thread; spin ROS2 in a background thread
        spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
        spin_thread.start()
        _run_visualizer(viz_queue)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
