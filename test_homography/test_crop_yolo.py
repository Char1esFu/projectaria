"""Read an image, center-crop it to a square, run YOLO inference, and save the annotated result."""

import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

MODEL_PATH = Path(__file__).parent.parent / "yolo_model" / "best.pt"


def center_crop(img: np.ndarray, target_size: int) -> np.ndarray:
    """Crop equally from all edges, keeping the center fixed, to produce a target_size x target_size image."""
    h, w = img.shape[:2]
    if target_size > h or target_size > w:
        raise ValueError(f"Target size {target_size} exceeds image dimensions {w}x{h}")
    y_start = (h - target_size) // 2
    x_start = (w - target_size) // 2
    cropped = img[y_start : y_start + target_size, x_start : x_start + target_size]
    return cropped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Center-crop an image and run YOLO inference.")
    parser.add_argument("--input", type=str, default="test_homography/images/frame1-1.png", help="Input image path")
    parser.add_argument("--output", type=str, default="test_homography/images/frame1-1_yolo.png", help="Output image path")
    parser.add_argument("--size", type=int, default=720, help="Output resolution (square, e.g. 720 -> 720x720)")
    parser.add_argument("--model", type=str, default=str(MODEL_PATH), help="YOLO model path")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--device", type=str, default=None, help="Inference device: cuda / cpu / mps (default: auto)")
    parser.add_argument("--infer-size", type=int, default=640, help="YOLO input size (pixels)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load image
    img = cv2.imread(args.input)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {args.input}")
    print(f"Original image size: {img.shape[1]}x{img.shape[0]}")

    # Center crop and resize
    cropped = center_crop(img, args.size)
    print(f"Cropped image size: {cropped.shape[1]}x{cropped.shape[0]}")
    cropped = cv2.resize(cropped, (1920, 1920), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite("cropped.png", cropped)
    
    # 降噪：双边滤波，保留边缘的同时去噪
    cropped = cv2.bilateralFilter(cropped, d=5, sigmaColor=15, sigmaSpace=15)
    
    # 锐化：unsharp mask
    blurred = cv2.GaussianBlur(cropped, (0, 0), sigmaX=5)
    cropped = cv2.addWeighted(cropped, 5, blurred, -4, 0)

    cv2.imwrite("resized.png", cropped)
    print(f"Saved cropped image to resized.png")
    # YOLO inference
    import torch
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading YOLO model from {args.model} on device={device}")
    model = YOLO(args.model)

    results = model(
        cropped,
        conf=args.conf,
        # imgsz=args.infer_size,
        device=device,
        verbose=False,
    )

    # Draw detections
    annotated = results[0].plot()

    # Save
    cv2.imwrite(args.output, annotated)
    print(f"Saved annotated image to {args.output}")


if __name__ == "__main__":
    main()
