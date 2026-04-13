"""
Compute a homography between image pairs via ORB feature matching.

All pairs share the same physical camera displacement, so --pair all pools
feature matches from every pair to compute a single, more robust homography.

Usage:
  --pair 1          → uses test_homography/images/frame1-1.png & frame1-2.png
  --pair all        → pools matches from ALL pairs → one homography
  --1st-frame/--2nd-frame  → explicit paths (override --pair)
"""

import argparse
import re
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

IMAGES_DIR = Path("test_homography/images")
RESULTS_DIR = Path("test_homography/results")


def _make_valid_mask(img: np.ndarray) -> np.ndarray:
    """Create a mask that excludes black border from undistorted fisheye images."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    # Pixels above threshold are considered valid (non-black)
    mask = (gray > 10).astype(np.uint8) * 255
    # Erode to push the boundary inward, avoiding edge artifacts
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.erode(mask, kernel)
    return mask


def extract_matches(img0: np.ndarray, img1: np.ndarray):
    """Return matched point arrays and draw info for visualization."""
    mask0 = _make_valid_mask(img0)
    mask1 = _make_valid_mask(img1)

    orb = cv2.ORB_create(nfeatures=4000)
    kp0, des0 = orb.detectAndCompute(img0, mask0)
    kp1, des1 = orb.detectAndCompute(img1, mask1)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    all_matches = sorted(matcher.match(des0, des1), key=lambda m: m.distance)

    # Filter out matches with large vertical displacement (pure horizontal shift expected)
    max_dy = 20  # pixels
    matches = [m for m in all_matches
               if abs(kp0[m.queryIdx].pt[1] - kp1[m.trainIdx].pt[1]) <= max_dy]

    pts0 = np.float32([kp0[m.queryIdx].pt for m in matches])
    pts1 = np.float32([kp1[m.trainIdx].pt for m in matches])
    print(f"  filtered: {len(all_matches)} → {len(matches)} (max_dy={max_dy}px)")
    return pts0, pts1, kp0, kp1, matches


def save_match_vis(img0: np.ndarray, img1: np.ndarray,
                   kp0, kp1, matches, output_path: Path) -> None:
    """Draw and save feature match visualization."""
    vis = cv2.drawMatches(
        img0, kp0, img1, kp1, matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), vis)
    print(f"Saved match visualisation → {output_path}")


def compute_homography(all_pts0: np.ndarray, all_pts1: np.ndarray, output: Path) -> Optional[np.ndarray]:
    """Compute homography from pooled point correspondences."""
    if len(all_pts0) < 4:
        print(f"Too few matches ({len(all_pts0)}) to compute homography.")
        return None

    H, mask = cv2.findHomography(all_pts0, all_pts1, cv2.RANSAC,
                                  ransacReprojThreshold=1.0, maxIters=10000, confidence=0.9999)

    if H is None:
        print("findHomography failed.")
        return None

    inliers = int(mask.sum())
    print(f"Homography computed: {inliers}/{len(all_pts0)} inliers")
    print(H)

    out_dir = output.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(output, H)
    print(f"Saved homography → {output}")
    return H


def find_all_pairs() -> list[tuple[Path, Path]]:
    """Discover all frameN-1 / frameN-2 pairs in IMAGES_DIR."""
    pattern = re.compile(r"^frame(\d+)-1\.png$")
    pairs = []
    for f in sorted(IMAGES_DIR.iterdir()):
        m = pattern.match(f.name)
        if m:
            n = m.group(1)
            f2 = IMAGES_DIR / f"frame{n}-2.png"
            if f2.exists():
                pairs.append((f, f2))
    return pairs


def load_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Cannot read {path}")
    return img


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pair", default=None,
                        help="Pair number (e.g. 1, 2) or 'all' to pool all pairs into one homography.")
    parser.add_argument("--1st-frame", type=Path, default=None,
                        dest="first_frame", help="Explicit path to the first image (overrides --pair).")
    parser.add_argument("--2nd-frame", type=Path, default=None,
                        dest="second_frame", help="Explicit path to the second image (overrides --pair).")
    parser.add_argument("--output", type=Path, default=Path("test_homography/homography.txt"),
                        help="Output path for the homography matrix.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Determine which image pairs to use
    if args.first_frame and args.second_frame:
        pairs = [(args.first_frame, args.second_frame)]
    elif args.pair and args.pair.lower() == "all":
        pairs = find_all_pairs()
        if not pairs:
            print(f"No frameN-1/frameN-2 pairs found in {IMAGES_DIR}")
            return
    else:
        n = args.pair or "1"
        pairs = [(IMAGES_DIR / f"frame{n}-1.png", IMAGES_DIR / f"frame{n}-2.png")]

    # Pool matches from all pairs and save per-pair visualizations
    all_pts0, all_pts1 = [], []
    for f1, f2 in pairs:
        img0 = load_image(f1)
        img1 = load_image(f2)
        pts0, pts1, kp0, kp1, matches = extract_matches(img0, img1)
        print(f"{f1.name} ↔ {f2.name}: {len(pts0)} matches")
        all_pts0.append(pts0)
        all_pts1.append(pts1)

        # Save match visualization for this pair
        pair_name = f1.stem.rsplit("-", 1)[0]  # e.g. "frame1"
        save_match_vis(img0, img1, kp0, kp1, matches,
                       RESULTS_DIR / f"matches_{pair_name}.png")

    all_pts0 = np.vstack(all_pts0)
    all_pts1 = np.vstack(all_pts1)
    print(f"\nTotal pooled matches: {len(all_pts0)}")

    compute_homography(all_pts0, all_pts1, args.output)


if __name__ == "__main__":
    main()
