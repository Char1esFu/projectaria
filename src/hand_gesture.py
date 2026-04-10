"""
Hand gesture detection on Aria RGB stream using MediaPipe.
Estimates each hand's 6-DOF pose (position + orientation) relative to the camera
by combining hand_world_landmarks (3D metric) with hand_landmarks (2D image) via solvePnP.

Usage:
    python3 -m src.hand_gesture --device-ip 192.168.8.117
"""

import argparse
import sys
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

from utils.aria_rgb_stream import AriaRgbStream
from utils.common import update_iptables


class HandGestureOverlay:
    """MediaPipe hand detection drawn on the pre-rotation RGB image."""

    def __init__(
        self,
        max_hands: int = 2,
        detection_confidence: float = 0.5,
        tracking_confidence: float = 0.5,
    ) -> None:
        self.dist_coeffs = np.zeros((5, 1), dtype=np.float64)

        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )

    def draw(self, display_image: np.ndarray, camera_matrix: Optional[np.ndarray]) -> None:
        results = self.hands.process(display_image)
        if not results.multi_hand_landmarks:
            return

        h, w = display_image.shape[:2]

        for hand_landmarks, hand_world_landmarks, handedness in zip(
            results.multi_hand_landmarks,
            results.multi_hand_world_landmarks,
            results.multi_handedness,
        ):
            self.mp_drawing.draw_landmarks(
                display_image,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style(),
            )

            pts_2d = np.array(
                [[lm.x * w, lm.y * h] for lm in hand_landmarks.landmark],
                dtype=np.float64,
            )
            pts_3d = np.array(
                [[lm.x, lm.y, lm.z] for lm in hand_world_landmarks.landmark],
                dtype=np.float64,
            )
            success, rvec, tvec = cv2.solvePnP(
                pts_3d, pts_2d, camera_matrix, self.dist_coeffs,
                flags=cv2.SOLVEPNP_SQPNP,
            )
            if not success:
                continue

            cv2.drawFrameAxes(display_image, camera_matrix, self.dist_coeffs, rvec, tvec, 0.05)

            label = handedness.classification[0].label
            dist_m = float(np.linalg.norm(tvec))
            wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
            px, py = int(wrist.x * w), int(wrist.y * h)
            cv2.putText(
                display_image,
                f"{label}  {dist_m:.2f} m",
                (px, py - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )

    def close(self) -> None:
        self.hands.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MediaPipe hand gesture on Aria RGB stream")
    parser.add_argument("--device-ip", help="IP address of the Aria device")
    parser.add_argument("--update_iptables", default=True, action="store_true")
    parser.add_argument("--max-hands", type=int, default=2)
    parser.add_argument("--detection-confidence", type=float, default=0.5)
    parser.add_argument("--tracking-confidence", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.update_iptables and sys.platform.startswith("linux"):
        update_iptables()

    overlay = HandGestureOverlay(
        max_hands=args.max_hands,
        detection_confidence=args.detection_confidence,
        tracking_confidence=args.tracking_confidence,
    )
    stream = AriaRgbStream(
        device_ip=args.device_ip,
        update_iptables_rules=False,
        window_name="Aria RGB Hand Gesture",
    )
    stream.add_overlay(overlay)
    try:
        stream.run()
    finally:
        overlay.close()


if __name__ == "__main__":
    main()
