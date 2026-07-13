"""
Create a short clip from ±N frames around a specified timestamp in a trial's frames directory.
"""

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np


RECORDINGS_ROOT = Path(__file__).parent.parent / "recordings"
DEFAULT_WINDOW = 10
DEFAULT_FPS_FALLBACK = 10.0

COLOR_CENTER = (0, 0, 255)    # red   (BGR)
COLOR_WINDOW = (0, 165, 255)  # orange (BGR)


def find_frames(frames_dir: Path) -> list[Path]:
    return sorted(frames_dir.glob("*.png"), key=lambda p: p.stem)


def compute_durations(stems: list[str]) -> list[float]:
    """Return per-frame duration (seconds). Uses timestamp diff if stems are nanoseconds."""
    try:
        values = [int(s) for s in stems]
    except ValueError:
        return [1.0 / DEFAULT_FPS_FALLBACK] * len(stems)

    if len(values) < 2:
        return [1.0 / DEFAULT_FPS_FALLBACK] * len(values)

    # Heuristic: nanosecond timestamps are > 1e15
    if values[-1] > 1e15:
        diffs = [(values[i + 1] - values[i]) / 1e9 for i in range(len(values) - 1)]
        avg = sum(diffs) / len(diffs)
        return diffs + [avg]
    else:
        return [1.0 / DEFAULT_FPS_FALLBACK] * len(values)


def annotate_frame(img: np.ndarray, stem: str, color: tuple) -> np.ndarray:
    """Draw the timestamp stem in the top-right corner."""
    out = img.copy()
    _, w = out.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.45, w / 2500)
    thickness = max(1, int(scale * 2))
    (tw, th), _ = cv2.getTextSize(stem, font, scale, thickness)
    x = w - tw - 8
    y = th + 8
    # Dark shadow for legibility on any background
    cv2.putText(out, stem, (x + 1, y + 1), font, scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(out, stem, (x, y), font, scale, color, thickness, cv2.LINE_AA)
    return out


def build_clip(
    subset: list[Path],
    center_stem: str,
    durations: list[float],
    out_path: Path,
) -> None:
    """Annotate frames, write ffconcat, and encode the clip with ffmpeg."""
    tmp_dir = Path(tempfile.mkdtemp(prefix="clip_frames_"))
    try:
        concat_path = tmp_dir / "ffconcat.txt"
        lines = ["ffconcat version 1.0"]

        for frame, dur in zip(subset, durations):
            img = cv2.imread(str(frame))
            if img is None:
                raise RuntimeError(f"Could not read {frame}")
            color = COLOR_CENTER if frame.stem == center_stem else COLOR_WINDOW
            annotated = annotate_frame(img, frame.stem, color)
            dst = tmp_dir / frame.name
            cv2.imwrite(str(dst), annotated)
            lines.append(f"file '{dst}'")
            lines.append(f"duration {dur:.9f}")

        concat_path.write_text("\n".join(lines) + "\n")

        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(concat_path),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(out_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"Clip saved → {out_path}")
        else:
            print(f"ffmpeg failed:\n{result.stderr[-800:]}")
            raise SystemExit(1)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode a short clip around a specific frame.")
    parser.add_argument("--participant", required=True, help="Participant ID, e.g. AB12")
    parser.add_argument("--trial", required=True, help="Trial number, e.g. 04")
    parser.add_argument(
        "--timestamp", required=True,
        help="Filename (with or without .png) of the center frame, e.g. 1779296840081608199",
    )
    parser.add_argument(
        "--window", type=int, default=DEFAULT_WINDOW,
        help=f"Number of frames before and after the center (default: {DEFAULT_WINDOW})",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output video path. Defaults to <trial_dir>/clip_<timestamp>.mp4",
    )
    args = parser.parse_args()

    trial_dir = RECORDINGS_ROOT / args.participant / args.trial
    frames_dir = trial_dir / "frames"
    if not frames_dir.is_dir():
        print(f"Frames directory not found: {frames_dir}")
        raise SystemExit(1)

    all_frames = find_frames(frames_dir)
    if not all_frames:
        print(f"No PNG frames found in {frames_dir}")
        raise SystemExit(1)

    target_stem = Path(args.timestamp).stem  # strips .png if provided
    stems = [f.stem for f in all_frames]

    if target_stem not in stems:
        print(f"Timestamp '{target_stem}' not found in {frames_dir}")
        print(f"Available range: {stems[0]} – {stems[-1]}")
        raise SystemExit(1)

    center_idx = stems.index(target_stem)
    start_idx = max(0, center_idx - args.window)
    end_idx = min(len(all_frames) - 1, center_idx + args.window)
    subset = all_frames[start_idx : end_idx + 1]

    all_durations = compute_durations(stems)
    subset_durations = all_durations[start_idx : end_idx + 1]

    out_path = args.output or trial_dir / f"clip_{target_stem}.mp4"

    print(
        f"Encoding {len(subset)} frames "
        f"(indices {start_idx}–{end_idx}, center={center_idx}) → {out_path}"
    )
    build_clip(subset, target_stem, subset_durations, out_path)


if __name__ == "__main__":
    main()
