import argparse
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO

from utils.aria_rgb_stream import AriaRgbStream, RgbOverlay

MODEL_PATH = Path(__file__).parent.parent / "yolo_model" / "best.pt"


class YoloOverlay(RgbOverlay):
    """Runs YOLO inference and draws detections on the pre-rotation RGB image."""

    def __init__(
        self,
        model_path: Path,
        conf_threshold: float = 0.25,
        device: Optional[str] = None,
        infer_size: int = 640,
    ) -> None:
        import torch
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading YOLO model from {model_path} on device={device}")
        self.model = YOLO(str(model_path))
        self.conf_threshold = conf_threshold
        self.infer_size = infer_size
        self.device = device

    def draw(self, display_image: np.ndarray, _camera_matrix: Optional[np.ndarray]) -> None:
        results = self.model(
            display_image,
            conf=self.conf_threshold,
            imgsz=self.infer_size,
            device=self.device,
            verbose=False,
        )
        np.copyto(display_image, results[0].plot())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO detection on Aria RGB stream.")
    parser.add_argument("--device-ip", help="IP address of the Aria device")
    parser.add_argument("--update_iptables", default=False, action="store_true")
    parser.add_argument("--model", type=str, default=str(MODEL_PATH))
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", type=str, default=None, help="Inference device: cuda / cpu / mps (default: auto)")
    parser.add_argument("--infer-size", type=int, default=640, help="YOLO input size (pixels)")
    parser.add_argument("--undistort-width", type=int, default=1408)
    parser.add_argument("--undistort-height", type=int, default=1408)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overlay = YoloOverlay(
        model_path=Path(args.model),
        conf_threshold=args.conf,
        device=args.device,
        infer_size=args.infer_size,
    )
    stream = AriaRgbStream(
        device_ip=args.device_ip,
        update_iptables_rules=args.update_iptables,
        undistort_width=args.undistort_width,
        undistort_height=args.undistort_height,
        window_name="Aria RGB YOLO",
    )
    stream.add_overlay(overlay)
    stream.run()


if __name__ == "__main__":
    main()
