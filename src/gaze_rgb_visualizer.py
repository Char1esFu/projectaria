import argparse
from pathlib import Path
from typing import Optional

from utils.aria_rgb_stream import AriaRgbStream
from src.gaze_overlay import GazeOverlay
from src.gaze_rgb_config import (
    DEFAULT_GAZE_BOUNDARY_RADIUS,
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
    boundary_radius: float = DEFAULT_GAZE_BOUNDARY_RADIUS,
    rgb_buffer_delay_frames: int = 0,
    rgb_timestamp_source: str = "zedr",
    gaze_var_window: int = 3,
    gaze_var_threshold: Optional[float] = None,
    gaze_var_top: Optional[int] = 1,
    gaze_var_force_endpoint_points: int = 1,
    hide_excluded: bool = False,
    detection_infill: bool = True,
    infill_min_observations: int = 2,
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
        boundary_radius=boundary_radius,
        gaze_var_window=gaze_var_window,
        gaze_var_threshold=gaze_var_threshold,
        gaze_var_top=gaze_var_top,
        gaze_var_force_endpoint_points=gaze_var_force_endpoint_points,
        hide_excluded=hide_excluded,
        rgb_timestamp_source=rgb_timestamp_source,
        detection_infill=detection_infill,
        infill_min_observations=infill_min_observations,
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
        default=True,
        action="store_true",
        help="Enable YOLO detection. If not set, only gaze overlay is shown.",
    )
    parser.add_argument("--draw-gaze",
        default=True,
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
    parser.add_argument(
        "--boundary-radius",
        type=float,
        default=DEFAULT_GAZE_BOUNDARY_RADIUS,
        help=("Pixel radius for filtering start/end fixation regions before "
              f"variance selection (default: {DEFAULT_GAZE_BOUNDARY_RADIUS})."),
    )
    parser.add_argument(
        "--rgb-buffer-delay-frames",
        type=int,
        default=5,
        help=(
            "Display/process the RGB frame this many received RGB frames behind "
            "the newest frame (0 displays immediately)."
        ),
    )
    parser.add_argument(
        "--rgb-timestamp-source",
        choices=("zedr", "hardware"),
        default="hardware",
        help=(
            "Timestamp recorded RGB frames with the latest ZED right camera_info "
            "stamp or with each Aria RGB frame's hardware capture timestamp."
        ),
    )
    parser.add_argument(
        "--gaze-var-window",
        type=int,
        default=3,
        help=(
            "Sliding window length in frames for label-score variance "
            "(default: 3)."
        ),
    )
    parser.add_argument(
        "--gaze-var-threshold",
        type=float,
        default=1.5e-3,
        help=(
            "Keep every window with label-score variance "
            "below this value (default: 1.5e-3)."
        ),
    )
    parser.add_argument(
        "--gaze-var-force-endpoint-points",
        type=int,
        default=1,
        help=(
            "When an endpoint has no fixation cluster, "
            "force-exclude this many outermost gaze points at that end "
            "(default: 1)."
        ),
    )
    parser.add_argument(
        "--gaze-var-top",
        type=int,
        default=1,
        help=(
            "If no interior window passes the threshold, "
            "keep the N lowest-variance interior windows "
            "(default: 1; <=0 keeps all). All passing windows are always kept."
        ),
    )
    parser.add_argument(
        "--hide-excluded",
        action="store_true",
        help=(
            "Hide endpoint-excluded windows from "
            "gaze_score_stability_variance.png instead of drawing grey crosses. "
            "Only the plot changes; selection and /gaze_label do not."
        ),
    )
    parser.add_argument(
        "--detection-infill",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "After recording stops, reproject a label seen in nearby frames into "
            "the frames where YOLO flickered it out, using the stitching poses, "
            "and score the filled points like real detections (default: on; "
            "--no-detection-infill keeps the raw log)."
        ),
    )
    parser.add_argument(
        "--infill-min-observations",
        type=int,
        default=2,
        help=(
            "in-fill only: a label must be detected at least this many times in "
            "the recording before it is filled into other frames, so a single "
            "spurious detection is not spread around (default: 2)."
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
        boundary_radius=args.boundary_radius,
        rgb_buffer_delay_frames=args.rgb_buffer_delay_frames,
        rgb_timestamp_source=args.rgb_timestamp_source,
        gaze_var_window=args.gaze_var_window,
        gaze_var_threshold=args.gaze_var_threshold,
        gaze_var_top=args.gaze_var_top if args.gaze_var_top > 0 else None,
        gaze_var_force_endpoint_points=args.gaze_var_force_endpoint_points,
        hide_excluded=args.hide_excluded,
        detection_infill=args.detection_infill,
        infill_min_observations=args.infill_min_observations,
    )


if __name__ == "__main__":
    main()
