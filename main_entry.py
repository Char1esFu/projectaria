"""
Combined entry point: one AriaStream serves both the gaze RGB overlay and
the audio record/transcribe handler.

Usage:
    python main_entry.py --device-ip 192.168.x.x --participant AB12 \
        [--yolo] [--draw-gaze] [--capture] [--homography path] ...

For RGB-only or audio-only standalone runs, use src/gaze_rgb_visualizer.py
or src/audio_record.py directly.
"""

import argparse
from pathlib import Path

import aria.sdk as aria

from src.audio_record import AudioHandler, ARIA_NUM_CHANNELS
from src.gaze_rgb_visualizer import GazeOverlay, MODEL_PATH
from utils.aria_rgb_stream import AriaStream


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combined Aria entry: gaze RGB overlay + audio transcription on a shared StreamingClient.",
    )

    # Shared
    parser.add_argument("--device-ip", default="192.168.8.117",
                        help="IP address of the Aria device")
    parser.add_argument("--update_iptables", default=False, action="store_true",
                        help="Update iptables for DDS UDP stream (Linux only).")
    parser.add_argument("--participant", type=str, default="",
                        help="Participant ID (e.g. AB12). Used by both gaze recording and audio session folders.")

    # Gaze / RGB
    parser.add_argument("--homography", type=Path, default="test_homography/homography.txt",
                        help="Path to homography matrix file. If provided, RGB is warped before overlay.")
    parser.add_argument("--yolo", default=False, action="store_true",
                        help="Enable YOLO detection.")
    parser.add_argument("--draw-gaze", default=False, action="store_true",
                        help="Draw gaze marker on RGB view.")
    parser.add_argument("--capture", default=False, action="store_true",
                        help="Enable runtime capture toggle (press S in RGB window).")
    parser.add_argument("--model", type=str, default=str(MODEL_PATH), help="YOLO model path")
    parser.add_argument("--yolo-conf", type=float, default=0.8, help="YOLO confidence threshold")
    parser.add_argument("--device", type=str, default=None,
                        help="YOLO inference device: cuda / cpu / mps (default: auto)")
    parser.add_argument("--filter-label", type=str, nargs="+",
                        default=["beer bottle", "mayonnaise bottle", "oil bottle", "water bottle"],
                        help="Labels to exclude from YOLO results.")
    parser.add_argument("--capture-interval", type=float, default=1.0,
                        help="Minimum interval (s) between saved frames.")
    parser.add_argument("--dist-threshold", type=float, default=1080.0,
                        help="Max distance (px) for gaze-to-bbox score.")
    parser.add_argument("--std-dist", type=float, default=200.0, dest="std_dist",
                        help="Gaussian std (px) for score falloff.")
    parser.add_argument("--s-min", type=float, default=0.0, dest="s_min",
                        help="Minimum score threshold for /gaze_label entries.")
    parser.add_argument("--label-hold", type=float, default=2.0, dest="label_hold",
                        help="Seconds to hold the last non-empty gaze label.")

    # Audio
    parser.add_argument("--channel", type=int, nargs="+", default=None,
                        choices=range(ARIA_NUM_CHANNELS), metavar="{0..6}",
                        help=f"Microphone channel(s) to mix (0-{ARIA_NUM_CHANNELS - 1}); default: all 7")
    parser.add_argument("--gain", type=float, default=2.0,
                        help="Extra gain multiplier after normalization.")
    parser.add_argument("--language", default="en",
                        help="Whisper transcription language.")
    parser.add_argument("--lowcut", type=float, default=100.0,
                        help="Bandpass lower cutoff (Hz).")
    parser.add_argument("--highcut", type=float, default=4000.0,
                        help="Bandpass upper cutoff (Hz).")
    parser.add_argument("--filter-order", type=int, default=6,
                        help="Butterworth filter order.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    overlay = GazeOverlay(
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
        label_hold_duration=args.label_hold,
    )
    audio = AudioHandler(
        channels=args.channel,
        gain=args.gain,
        language=args.language,
        participant=args.participant,
        lowcut=args.lowcut,
        highcut=args.highcut,
        filter_order=args.filter_order,
    )

    stream = AriaStream(
        device_ip=args.device_ip,
        update_iptables_rules=args.update_iptables,
        window_name="Aria RGB Gaze",
        data_types=aria.StreamingDataType.Rgb | aria.StreamingDataType.Audio,
    )
    stream.add_overlay(overlay)
    stream.set_frame_callback(overlay.record_frame)
    stream.add_audio_handler(audio)

    try:
        stream.run()
    finally:
        overlay.shutdown()
        audio.shutdown()


if __name__ == "__main__":
    main()
