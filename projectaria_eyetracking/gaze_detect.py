import argparse
import sys
import time

import aria.sdk as aria
import cv2
import numpy as np
import torch

from utils.common import quit_keypress, update_iptables
from projectaria_eyetracking.inference.infer import EyeGazeInference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_checkpoint_path",
        type=str,
        default=(
            "projectaria_eyetracking/inference/model/"
            "pretrained_weights/social_eyes_uncertainty_v1/weights.pth"
        ),
        help="Location of the model weights",
    )
    parser.add_argument(
        "--model_config_path",
        type=str,
        default=(
            "projectaria_eyetracking/inference/model/"
            "pretrained_weights/social_eyes_uncertainty_v1/config.yaml"
        ),
        help="Location of the model config",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run inference on (cpu or cuda:0)",
    )
    parser.add_argument(
        "--update_iptables",
        default=False,
        action="store_true",
        help="Update iptables to enable receiving the data stream, only for Linux.",
    )
    parser.add_argument(
        "--pitch-bias",
        type=float,
        default=0.1,
        help="Additive bias applied to predicted pitch (radians).",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    if args.update_iptables and sys.platform.startswith("linux"):
        update_iptables()

    aria.set_log_level(aria.Level.Info)

    try:
        import rclpy
        from geometry_msgs.msg import Vector3

        rclpy.init(args=None)
        ros_node = rclpy.create_node("eye_gaze_publisher")
        gaze_topic = "/aria/gaze_euler"
        gaze_publisher = ros_node.create_publisher(Vector3, gaze_topic, 10)
        print(f"ROS2 publishing enabled: {gaze_topic}")
    except Exception as exc:
        raise RuntimeError(f"ROS2 publisher unavailable: {exc}") from exc

    inference_model = EyeGazeInference(
        args.model_checkpoint_path, args.model_config_path, args.device
    )

    streaming_client = aria.StreamingClient()

    config = streaming_client.subscription_config
    config.subscriber_data_type = (
        aria.StreamingDataType.EyeTrack
    )
    config.message_queue_size[aria.StreamingDataType.EyeTrack] = 1

    options = aria.StreamingSecurityOptions()
    options.use_ephemeral_certs = True
    config.security_options = options
    streaming_client.subscription_config = config

    class StreamingClientObserver:
        def __init__(self):
            self.images = {}

        def on_image_received(self, image: np.ndarray, record) -> None:
            self.images[record.camera_id] = image

    observer = StreamingClientObserver()
    streaming_client.set_streaming_client_observer(observer)

    print("Start listening to image data")
    streaming_client.subscribe()

    eyetrack_window = "Aria EyeTrack"

    cv2.namedWindow(eyetrack_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(eyetrack_window, 640, 240)
    cv2.setWindowProperty(eyetrack_window, cv2.WND_PROP_TOPMOST, 1)
    cv2.moveWindow(eyetrack_window, 50, 800)

    try:
        while not quit_keypress():
            if aria.CameraId.EyeTrack in observer.images:
                eyetrack_image = observer.images[aria.CameraId.EyeTrack]
                del observer.images[aria.CameraId.EyeTrack]

                cv2.imshow(eyetrack_window, eyetrack_image)

                # The model expects a single grayscale image containing [left | right] eyes.
                eye_tensor = torch.from_numpy(eyetrack_image)

                preds, lower, upper = inference_model.predict(eye_tensor)
                yaw = -preds[0][0].item() # look left negative, right positive, to match convention
                pitch = preds[0][1].item() + args.pitch_bias

                print(
                    f"pitch={pitch:.4f}, yaw={yaw:.4f}"
                )

                gaze_msg = Vector3()
                gaze_msg.x = pitch
                gaze_msg.y = yaw
                gaze_msg.z = 0.0
                gaze_publisher.publish(gaze_msg)

            time.sleep(0.001)
    finally:
        print("Stop listening to image data")
        streaming_client.unsubscribe()
        try:
            ros_node.destroy_node()
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
