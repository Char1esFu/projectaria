"""
Save RGB frames from the Aria stream as static images.

Press 'S' to save a frame. Frames are saved in pairs:
  frame1-1.png, frame1-2.png, frame2-1.png, frame2-2.png, ...

Existing complete pairs are skipped automatically, so new frames fill
gaps first, then continue with the next available number.

All images are saved to test_homography/images/.
Press 'Q' or ESC to quit.
"""

import argparse
import re
from pathlib import Path

import cv2
import numpy as np

from utils.aria_rgb_stream import AriaRgbStream

SAVE_DIR = Path("test_homography/images")


def find_missing_pair_numbers() -> list[int]:
    """Find pair numbers that don't have both -1 and -2 files."""
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(r"^frame(\d+)-[12]\.png$")
    existing: dict[int, set[int]] = {}
    max_n = 0
    for f in SAVE_DIR.iterdir():
        m = pattern.match(f.name)
        if m:
            n = int(m.group(1))
            existing.setdefault(n, set())
            existing[n].add(int(f.stem.split("-")[1]))
            max_n = max(max_n, n)

    # Collect incomplete pairs (missing -1 or -2 or both)
    incomplete = []
    for n in range(1, max_n + 1):
        if n not in existing or len(existing[n]) < 2:
            incomplete.append(n)
    return incomplete, max_n


class FrameSaverOverlay:
    """Overlay that saves frames on 'S' key press."""

    def __init__(self) -> None:
        incomplete, max_n = find_missing_pair_numbers()
        self._incomplete = incomplete
        self._next_new = max_n + 1
        self._count = 0
        self._current_pair = None

    def _get_next_pair(self) -> int:
        if self._incomplete:
            return self._incomplete.pop(0)
        n = self._next_new
        self._next_new += 1
        return n

    def draw(self, display_image: np.ndarray, camera_matrix, key: int) -> None:
        if key not in (ord("s"), ord("S")):
            return
        self._count += 1
        sub = 1 if self._count % 2 == 1 else 2

        if sub == 1:
            self._current_pair = self._get_next_pair()

        SAVE_DIR.mkdir(parents=True, exist_ok=True)
        filename = SAVE_DIR / f"frame{self._current_pair}-{sub}.png"
        cv2.imwrite(str(filename), display_image)
        print(f"[{self._count}] Saved → {filename}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device-ip", help="IP address of the Aria device")
    parser.add_argument(
        "--update_iptables", action="store_true", default=False,
        help="Update iptables for DDS UDP stream (Linux only).",
    )
    args = parser.parse_args()

    stream = AriaRgbStream(
        device_ip=args.device_ip,
        update_iptables_rules=args.update_iptables,
        window_name="Save RGB Frames",
    )
    stream.add_overlay(FrameSaverOverlay())
    stream.run()


if __name__ == "__main__":
    main()
