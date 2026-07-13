#!/usr/bin/env python3
"""
bag_record.py – START via UDP (audio_record.py sends it on [B] press),
               STOP when /transcription message is received.

Also subscribes to /seg/point_cloud: each received cloud is saved as
  recordings/<participant>/<NN>/<target_label>.ply
using the seg_service target_label parameter as filename.

Bags are saved to:
  recordings/<participant>/<NN>/bag/
NN continues from the last existing session on restart with the same ID.

Usage:
  python3 src/bag_record.py --participant AB12
"""

import argparse
import signal
import subprocess
import threading
import open3d as o3d
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Empty, String

TOPICS_TO_RECORD = [
    "/yolo_result_image",
    "/finger/markers",
    "/hand/landmark_markers",
    "/yolo/bbox_centers",
    "/finger/mcp_to_bbox_lines",
    "/transcription",
    "/gaze_label", 
]

_proc: subprocess.Popen | None = None
_lock = threading.Lock()
_participant: str = ""
_current_subfolder: Path | None = None   # updated each time a new recording starts



def start_recording() -> None:
    global _proc, _current_subfolder
    with _lock:
        if _proc is not None:
            return
        base = Path("recordings") / _participant
        base.mkdir(parents=True, exist_ok=True)
        existing = sorted([int(p.name) for p in base.iterdir() if p.is_dir() and p.name.isdigit()])
        if existing:
            idx = existing[-1]  # use folder created by audio_record
            subfolder = base / f"{idx:02d}"
        else:
            idx = 1             # fallback: audio_record not used, create new
            subfolder = base / f"{idx:02d}"
            subfolder.mkdir()
        bag_path = str(subfolder / "bag")
        _proc = subprocess.Popen(
            ["ros2", "bag", "record", "--storage", "mcap", "-o", bag_path] + TOPICS_TO_RECORD
        )
        _current_subfolder = subfolder
        print(f"Recording... → {bag_path}/")


def stop_recording() -> None:
    global _proc
    with _lock:
        if _proc is None:
            return
        _proc.send_signal(signal.SIGINT)
        _proc.wait()
        _proc = None
        print("Stopped.")


def _get_target_label() -> str:
    try:
        r = subprocess.run(
            ["ros2", "param", "get", "/seg_service", "target_label"],
            capture_output=True, text=True, timeout=2.0,
        )
        line = r.stdout.strip()
        if "value is:" in line:
            return line.split("value is:")[-1].strip()
    except Exception:
        pass
    return "unknown"


def _pc2_to_o3d(msg: PointCloud2):
    n = msg.width * msg.height
    has_rgb = any(f.name == "rgb" for f in msg.fields)
    raw = np.frombuffer(bytes(msg.data), dtype=np.float32).reshape(n, -1)
    xyz = raw[:, :3]
    valid = np.isfinite(xyz).all(axis=1)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz[valid])

    if has_rgb:
        rgb_packed = raw[valid, 3].view(np.uint32)
        r = ((rgb_packed >> 16) & 0xFF).astype(np.float32) / 255.0
        g = ((rgb_packed >>  8) & 0xFF).astype(np.float32) / 255.0
        b = ( rgb_packed        & 0xFF).astype(np.float32) / 255.0
        pcd.colors = o3d.utility.Vector3dVector(np.column_stack([r, g, b]))

    return pcd


class RecorderNode(Node):
    def __init__(self) -> None:
        super().__init__("bag_recorder")
        self.create_subscription(Empty, "/recording/start", lambda _: start_recording(), 10)
        self.create_subscription(String, "/transcription", self._on_transcription, 10)
        self.create_subscription(PointCloud2, "/seg/point_cloud", self._on_point_cloud, 10)
        self.get_logger().info("Subscribed to /recording/start, /transcription and /seg/point_cloud")

    def _on_transcription(self, msg: String) -> None:
        self.get_logger().info(f"Transcription: {msg.data!r}")
        stop_recording()

    def _on_point_cloud(self, msg: PointCloud2) -> None:
        save_dir = _current_subfolder
        if save_dir is None:
            self.get_logger().warn("No session active, skipping point cloud save")
            return
        label = _get_target_label()
        path = save_dir / f"{label}.ply"
        try:
            pcd = _pc2_to_o3d(msg)
            o3d.io.write_point_cloud(str(path), pcd)
            self.get_logger().info(f"Saved {len(pcd.points)} pts → {path}")
        except Exception as exc:
            self.get_logger().error(f"Failed to save point cloud: {exc}")


def main() -> None:
    global _participant

    parser = argparse.ArgumentParser()
    parser.add_argument("--participant", required=True, help="Participant ID, e.g. AB12")
    args, _ = parser.parse_known_args()
    _participant = args.participant

    rclpy.init()
    node = RecorderNode()

    print(f"Participant: {_participant}  |  saving to recordings/{_participant}/NN/bag/")
    print("Ready. Hold [B] in audio_record to start. Ctrl+C to quit.")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        stop_recording()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
