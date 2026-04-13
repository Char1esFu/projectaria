import argparse
import re
from pathlib import Path

import cv2
import numpy as np

IMAGES_DIR = Path("test_homography/images")
RESULTS_DIR = Path("test_homography/results")


def find_all_pairs() -> list[tuple[int, Path, Path]]:
    pattern = re.compile(r"^frame(\d+)-1\.png$")
    pairs = []
    for f in sorted(IMAGES_DIR.iterdir()):
        m = pattern.match(f.name)
        if m:
            n = int(m.group(1))
            f2 = IMAGES_DIR / f"frame{n}-2.png"
            if f2.exists():
                pairs.append((n, f, f2))
    return pairs


def verify_pair(n: int, f1: Path, f2: Path, H: np.ndarray) -> None:
    frame0 = cv2.imread(str(f1))
    frame1 = cv2.imread(str(f2))
    h, w = frame0.shape[:2]
    frame1 = cv2.resize(frame1, (w, h))

    warped = cv2.warpPerspective(frame0, H, (w, h))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(RESULTS_DIR / f"warped_frame{n}.png"), warped)

    row1 = np.hstack([frame0, frame1, warped])
    cv2.imshow(f"pair {n}: 1st | 2nd | warped", row1)

    blend_f0_w = cv2.addWeighted(frame0, 0.5, warped, 0.5, 0)
    blend_f1_w = cv2.addWeighted(frame1, 0.5, warped, 0.5, 0)
    row2 = np.hstack([blend_f0_w, blend_f1_w])
    cv2.imshow(f"pair {n}: 1st+warped | 2nd+warped", row2)

    while True:
        k = cv2.waitKey(0) & 0xFF
        if k == ord(" "):
            break
    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--homography", type=Path, default=Path("test_homography/homography.txt"))
    args = parser.parse_args()

    H = np.loadtxt(args.homography)

    pairs = find_all_pairs()
    if not pairs:
        print(f"No frameN-1/frameN-2 pairs found in {IMAGES_DIR}")
        return

    for n, f1, f2 in pairs:
        print(f"Verifying pair {n}: {f1.name} ↔ {f2.name}")
        verify_pair(n, f1, f2, H)

    print(f"\n{len(pairs)} pairs verified.")


if __name__ == "__main__":
    main()
