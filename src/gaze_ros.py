import threading
from typing import Callable


class GazeRos:
    """ROS2 subscriptions and /gaze_label publisher wrapper."""

    def __init__(
        self,
        on_gaze: Callable[[float, float], None],
        on_manip_stamp: Callable[[int], None],
        on_recording_start: Callable[[], None],
        on_label_start: Callable[[], None],
        on_recording_stop: Callable[[], None],
    ) -> None:
        try:
            import rclpy
            from geometry_msgs.msg import Vector3
            from sensor_msgs.msg import CameraInfo
            from std_msgs.msg import Empty, String

            if not rclpy.ok():
                rclpy.init(args=None)
            self._rclpy = rclpy
            self._node = rclpy.create_node("gaze_rgb_visualizer")
            self._String = String

            self._node.create_subscription(
                Vector3, "/aria/gaze_euler",
                lambda msg: on_gaze(float(msg.x), float(msg.y)), 10,
            )
            self._node.create_subscription(
                CameraInfo, "/zedr/zed_node/rgb/camera_info",
                lambda msg: on_manip_stamp(
                    msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec
                ),
                10,
            )
            self._node.create_subscription(
                Empty, "/recording/start", lambda _: on_recording_start(), 10
            )
            self._node.create_subscription(
                Empty, "/gaze_label_recording_start", lambda _: on_label_start(), 10
            )
            self._node.create_subscription(
                Empty, "/key/b/release", lambda _: on_recording_stop(), 10
            )
            self._gaze_label_pub = self._node.create_publisher(
                String, "/gaze_label", 10
            )

            from rclpy.executors import SingleThreadedExecutor

            self._executor = SingleThreadedExecutor()
            self._executor.add_node(self._node)
            self._thread = threading.Thread(target=self._executor.spin, daemon=True)
            self._thread.start()
            print("ROS2 publishers started: /gaze_label")
        except Exception as exc:
            raise RuntimeError(f"ROS2 subscriber unavailable: {exc}") from exc

    def publish_gaze_label(self, data: str) -> None:
        msg = self._String()
        msg.data = data
        self._gaze_label_pub.publish(msg)

    def shutdown(self) -> None:
        try:
            self._executor.shutdown()
        except Exception:
            pass
        try:
            self._node.destroy_node()
            self._rclpy.shutdown()
        except Exception:
            pass
