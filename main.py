"""
Combined Aria RGB visualizer — runs any combination of overlays in a single window.

Usage examples:
  # ArUco only
  python3 main.py --aruco --marker-ids 0 1 2

  # Gaze only
  python3 main.py --gaze

  # Both in one window
  python3 main.py --aruco --gaze --marker-ids 0 1 2
"""

import argparse
import os

from utils.aria_rgb_stream import AriaRgbStream


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aria RGB combined overlay visualizer.")
    parser.add_argument("--device-ip", help="IP address of the Aria device")
    parser.add_argument("--update_iptables", default=False, action="store_true")

    parser.add_argument("--aruco", action="store_true", help="Enable ArUco localization overlay")
    parser.add_argument("--marker-length-m", type=float, default=0.2)
    parser.add_argument("--dictionary", type=str, default="DICT_4X4_50")
    parser.add_argument("--marker-ids", type=int, nargs="+", default=None)
    parser.add_argument("--ema-alpha", type=float, default=0.2)
    parser.add_argument("--disable-ema", action="store_true")

    parser.add_argument("--gaze", action="store_true", help="Enable gaze visualization overlay")

    parser.add_argument("--hands", action="store_true", help="Enable hand gesture overlay")
    parser.add_argument("--max-hands", type=int, default=2)
    parser.add_argument("--detection-confidence", type=float, default=0.5)
    parser.add_argument("--tracking-confidence", type=float, default=0.5)

    parser.add_argument("--yolo", action="store_true", help="Enable YOLO detection overlay")
    parser.add_argument("--yolo-model", type=str, default=None, help="Path to YOLO model (default: yolo_model/best.pt)")
    parser.add_argument("--yolo-conf", type=float, default=0.25)
    parser.add_argument("--yolo-device", type=str, default=None, help="Inference device: cuda / cpu (default: auto)")
    parser.add_argument("--yolo-infer-size", type=int, default=640)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.aruco and not args.gaze and not args.hands and not args.yolo:
        print("No overlays selected. Use --aruco, --gaze, --hands, and/or --yolo.")
        return

    overlays = []

    if args.aruco:
        from src.aruco_localization import ArucoOverlay, RosPosePublisher

        static_tf_config_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "config", "aruco_tf.json"
        )
        ros = RosPosePublisher(
            topic="/aria/cam_pose",
            marker_frame_prefix="aruco_marker_",
            camera_frame="aria_camera_rgb",
            static_tf_config_path=static_tf_config_path,
            camera_frame_correction_q_xyzw=None,
        )
        ros.setup()
        ros.publish_static_marker_tf()
        overlays.append(("aruco", ArucoOverlay(
            marker_length_m=args.marker_length_m,
            dictionary_name=args.dictionary,
            allowed_marker_ids=args.marker_ids,
            use_ema=not args.disable_ema,
            ema_alpha=args.ema_alpha,
            ros=ros,
        ), ros))

    if args.gaze:
        from src.gaze_rgb_visualizer import GazeOverlay
        overlays.append(("gaze", GazeOverlay(), None))

    if args.hands:
        from src.hand_gesture import HandGestureOverlay
        overlays.append(("hands", HandGestureOverlay(
            max_hands=args.max_hands,
            detection_confidence=args.detection_confidence,
            tracking_confidence=args.tracking_confidence,
        ), None))

    if args.yolo:
        from pathlib import Path
        from src.yolo_rgb_detector import YoloOverlay
        model_path = Path(args.yolo_model) if args.yolo_model else Path(__file__).parent / "yolo_model" / "best.pt"
        overlays.append(("yolo", YoloOverlay(
            model_path=model_path,
            conf_threshold=args.yolo_conf,
            device=args.yolo_device,
            infer_size=args.yolo_infer_size,
        ), None))

    feature_names = "+".join(name for name, _, _ in overlays)
    stream = AriaRgbStream(
        device_ip=args.device_ip,
        update_iptables_rules=args.update_iptables,
        window_name=f"Aria RGB [{feature_names}]",
    )
    for _, overlay, _ in overlays:
        stream.add_overlay(overlay)

    try:
        stream.run()
    finally:
        for _, overlay, extra in overlays:
            if hasattr(overlay, "shutdown"):
                overlay.shutdown()
            if hasattr(overlay, "close"):
                overlay.close()
            if extra is not None and hasattr(extra, "shutdown"):
                extra.shutdown()


if __name__ == "__main__":
    main()
