import math
from typing import Optional

import cv2
import numpy as np

from src.gaze_rgb_config import RESIZE_SIZE


def compute_score(d: float, dist_threshold: float, std_dist: float) -> float:
    """Gaussian score based on distance from bbox center."""
    if d >= dist_threshold:
        return 0.0
    return math.exp(-(d ** 2) / (2 * std_dist ** 2))


def enhance_crop(crop_img: np.ndarray) -> np.ndarray:
    resized_crop = cv2.resize(
        crop_img, (RESIZE_SIZE, RESIZE_SIZE), interpolation=cv2.INTER_LINEAR
    )
    resized_crop = cv2.bilateralFilter(
        resized_crop, d=5, sigmaColor=15, sigmaSpace=15
    )
    blurred = cv2.GaussianBlur(resized_crop, (0, 0), sigmaX=5)
    return cv2.addWeighted(resized_crop, 5, blurred, -4, 0)


def draw_circles(
    img: np.ndarray, result, crop_transform: Optional[tuple] = None
) -> np.ndarray:
    """Draw a circle per detection on img."""
    out = img.copy()
    if result.boxes is None or len(result.boxes) == 0:
        return out
    names = result.names
    for xyxy, cid, conf in zip(
        result.boxes.xyxy.tolist(),
        result.boxes.cls.int().tolist(),
        result.boxes.conf.tolist(),
    ):
        x1, y1, x2, y2 = xyxy
        if crop_transform is not None:
            xs, ys, sc = crop_transform
            cx = int(xs + (x1 + x2) / 2 / sc)
            cy = int(ys + (y1 + y2) / 2 / sc)
            radius = int(math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) / 2 / sc)
        else:
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            radius = int(math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) / 2)
        label = names.get(cid, str(cid))
        cv2.circle(out, (cx, cy), radius, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(
            out,
            label,
            (cx - radius, max(cy - radius - 6, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    return out


def filter_results(result, filter_labels: set[str]) -> None:
    """Filter labels and keep only the highest-confidence detection per class."""
    if result.boxes is None or len(result.boxes) == 0:
        return
    names = result.names
    cls_ids = result.boxes.cls.int().tolist()
    confs = result.boxes.conf.tolist()

    if filter_labels:
        keep = [
            i
            for i, cid in enumerate(cls_ids)
            if names.get(cid, "").lower() not in filter_labels
        ]
        result.boxes = result.boxes[keep]
        if len(result.boxes) == 0:
            return
        cls_ids = result.boxes.cls.int().tolist()
        confs = result.boxes.conf.tolist()

    best: dict[int, tuple[int, float]] = {}
    for i, (cid, conf) in enumerate(zip(cls_ids, confs)):
        if cid not in best or conf > best[cid][1]:
            best[cid] = (i, conf)
    keep = [idx for idx, _ in best.values()]
    result.boxes = result.boxes[keep]


def summarize_detections(
    result,
    gaze_in_crop: tuple[float, float],
    x_start: int,
    y_start: int,
    scale: float,
    dist_threshold: float,
    std_dist: float,
) -> list[tuple[str, float, float, tuple[int, int]]]:
    det_summary = []
    if result.boxes is None:
        return det_summary
    names_map = result.names
    for xyxy, cid, conf in zip(
        result.boxes.xyxy.tolist(),
        result.boxes.cls.int().tolist(),
        result.boxes.conf.tolist(),
    ):
        x1, y1, x2, y2 = xyxy
        ocx = (x1 + x2) / 2.0
        ocy = (y1 + y2) / 2.0
        d = math.sqrt((gaze_in_crop[0] - ocx) ** 2 + (gaze_in_crop[1] - ocy) ** 2)
        score = compute_score(d, dist_threshold, std_dist)
        center_px = (
            int(round(x_start + ocx / scale)),
            int(round(y_start + ocy / scale)),
        )
        det_summary.append((names_map.get(cid, str(cid)), conf, score, center_px))
    return det_summary
