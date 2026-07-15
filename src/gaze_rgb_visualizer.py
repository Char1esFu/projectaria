import argparse
from pathlib import Path
from typing import Optional

from utils.aria_rgb_stream import AriaRgbStream
from src.gaze_overlay import GazeOverlay
from src.gaze_rgb_config import (
    DEFAULT_GAZE_PEAK_RADIUS,
    DEFAULT_GAZE_PEAK_WINDOW,
    MODEL_PATH,
)


def run_gaze_rgb_visualizer(
    # device_ip: Optional[str] = None,
    update_iptables_rules: bool = False,
    homography_path: Optional[Path] = None,
    enable_yolo: bool = False,
    draw_gaze: bool = False,
    enable_capture: bool = False,
    model_path: Optional[Path] = None,
    conf_threshold: float = 0.25,
    device: Optional[str] = None,
    filter_labels: Optional[list[str]] = None,
    capture_interval: float = 0.0,
    dist_threshold: float = 1080.0,
    std_dist: float = 200.0,
    s_min: float = 0.3,
    participant: str = "",
    gaze_peak_window: int = DEFAULT_GAZE_PEAK_WINDOW,
    gaze_peak_radius: float = DEFAULT_GAZE_PEAK_RADIUS,
    rgb_buffer_delay_frames: int = 0,
    rgb_timestamp_source: str = "zedr",
) -> None:
    overlay = GazeOverlay(
        homography_path=homography_path,
        enable_yolo=enable_yolo,
        draw_gaze=draw_gaze,
        enable_capture=enable_capture,
        model_path=model_path,
        conf_threshold=conf_threshold,
        device=device,
        filter_labels=filter_labels,
        capture_interval=capture_interval,
        dist_threshold=dist_threshold,
        std_dist=std_dist,
        s_min=s_min,
        participant=participant,
        gaze_peak_window=gaze_peak_window,
        gaze_peak_radius=gaze_peak_radius,
        rgb_timestamp_source=rgb_timestamp_source,
    )
    stream = AriaRgbStream(
        # device_ip=device_ip,
        update_iptables_rules=update_iptables_rules,
        window_name="Aria RGB Gaze",
        rgb_buffer_delay_frames=rgb_buffer_delay_frames,
    )
    stream.add_overlay(overlay)
    stream.set_frame_callback(overlay.record_frame)
    try:
        stream.run()
    finally:
        overlay.shutdown()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize gaze direction on Aria RGB stream with optional YOLO detection."
    )
    # parser.add_argument("--device-ip",
    #     default="192.168.8.117",
    #     help="IP address of the Aria device"
    # )
    parser.add_argument("--update_iptables",
        default=False,
        action="store_true",
        help="Update iptables for DDS UDP stream (Linux only).",
    )
    parser.add_argument("--homography", 
        type=Path, 
        default="test_homography/homography.txt",
        help="Path to a homography matrix file (e.g. test_homography/homography.txt). "
             "If provided, the RGB image is warped before gaze overlay.",
    )
    parser.add_argument("--yolo",
        default=False,
        action="store_true",
        help="Enable YOLO detection. If not set, only gaze overlay is shown.",
    )
    parser.add_argument("--draw-gaze",
        default=False,
        action="store_true",
        help="Draw gaze marker (red circle and red dot) on RGB view.",
    )
    parser.add_argument("--capture",
        default=False,
        action="store_true",
        help="Enable runtime capture toggle. Press S to start/stop continuous saving to saved_images/.",
    )
    parser.add_argument("--model", 
        type=str, 
        default=str(MODEL_PATH), 
        help="YOLO model path"
    )
    parser.add_argument("--yolo-conf", 
        type=float, 
        default=0.8, 
        help="YOLO confidence threshold")
    parser.add_argument("--device", 
        type=str, 
        default=None, 
        help="Inference device: cuda / cpu / mps (default: auto)")
    parser.add_argument("--filter-label", 
        type=str, 
        nargs="+", 
        default=["beer bottle", "mayonnaise bottle", "oil bottle", "water bottle"],
        help="Labels to exclude from YOLO results (e.g. --filter-label 'beer bottle' 'water bottle').",
    )
    parser.add_argument("--capture-interval", 
        type=float, 
        default=1.0,
        help="Minimum time interval (seconds) between saved frames. 0 = save every frame.",
    )
    parser.add_argument("--dist-threshold", 
        type=float, 
        default=1080.0,
        help="Max distance (pixels) from bbox center to gaze point. Score = 0 when d >= dist_threshold. Default: 200.",
    )
    parser.add_argument("--std-dist", 
        type=float, 
        default=200.0, 
        dest="std_dist",
        help="Gaussian std (pixels) for score falloff within dist_threshold. Default: 80.",
    )
    parser.add_argument("--s-min", 
        type=float, 
        default=0.0, 
        dest="s_min",
        help="Minimum score threshold for publishing /gaze_label entries (default: 0.2).",
    )
    parser.add_argument("--participant", 
        type=str, 
        default="",
        help="Participant ID (e.g. AB12). Required for video recording to recordings/<participant>/NN/.",
    )
    parser.add_argument("--gaze-peak-window",
        type=int, 
        default=DEFAULT_GAZE_PEAK_WINDOW,
        help="Temporal window for stitched gaze peak detection. Fixed default: 3.",
    )
    parser.add_argument("--gaze-peak-radius",
        type=float, 
        default=DEFAULT_GAZE_PEAK_RADIUS,
        help="Pixel radius for filtering start/end fixation regions. Default: 15.",
    )
    parser.add_argument(
        "--rgb-buffer-delay-frames",
        type=int,
        default=0,
        help=(
            "Display/process the RGB frame this many received RGB frames behind "
            "the newest frame (0 displays immediately)."
        ),
    )
    parser.add_argument(
        "--rgb-timestamp-source",
        choices=("zedr", "hardware"),
        default="zedr",
        help=(
            "Timestamp recorded RGB frames with the latest ZED right camera_info "
            "stamp or with each Aria RGB frame's hardware capture timestamp."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_gaze_rgb_visualizer(
        # device_ip=args.device_ip,
        update_iptables_rules=args.update_iptables,
        homography_path=args.homography,
        enable_yolo=args.yolo,
        draw_gaze=args.draw_gaze,
        enable_capture=args.capture,
        model_path=Path(args.model) if args.yolo else None,
        conf_threshold=args.yolo_conf,
        device=args.device,
        filter_labels=args.filter_label,
        capture_interval=args.capture_interval,
        dist_threshold=args.dist_threshold,
        std_dist=args.std_dist,
        s_min=args.s_min,
        participant=args.participant,
        gaze_peak_window=args.gaze_peak_window,
        gaze_peak_radius=args.gaze_peak_radius,
        rgb_buffer_delay_frames=args.rgb_buffer_delay_frames,
        rgb_timestamp_source=args.rgb_timestamp_source,
    )


if __name__ == "__main__":
    main()
