"""Encode a folder of timestamp-named frames into an mp4.

Recorded frames are named ``<stamp_ns>.png`` (and ``<stamp_ns>_filled.png``
once detection in-fill has redrawn one), so playback timing comes from the file
names rather than a fixed frame rate: each frame is held for the gap to the next
stamp via an ffconcat demuxer list.

Shared by the live recorder and the offline in-fill tool, which is why it lives
outside both.
"""

import subprocess
from pathlib import Path
from typing import Optional


def frame_entries(frames_dir: Path) -> list[tuple[int, str]]:
    """``(stamp_ns, file name)`` for every ``<stamp_ns>[_suffix].png``, in time
    order. Unparseable names are ignored; one file per stamp wins."""
    by_stamp: dict[int, str] = {}
    for path in sorted(frames_dir.glob("*.png")):
        digits = path.stem.split("_", 1)[0]
        if not digits.isdigit():
            continue
        by_stamp[int(digits)] = path.name
    return sorted(by_stamp.items())


def encode_frame_video(
    session_dir: Path,
    frames_dir: Path,
    out_name: str,
    concat_name: Optional[str] = None,
) -> Optional[Path]:
    """Encode ``frames_dir``'s frames into ``session_dir/out_name``.

    The ffconcat list references the frames relative to ``session_dir``, so the
    frame folder must live inside it. Returns the video path, or ``None`` when
    there is nothing to encode or ffmpeg fails.
    """
    entries = frame_entries(frames_dir)
    if not entries:
        print(f"Video encoding skipped: no timestamped frames in {frames_dir}.")
        return None

    prefix = frames_dir.name
    lines = ["ffconcat version 1.0"]
    for i, (stamp_ns, file_name) in enumerate(entries):
        if i + 1 < len(entries):
            duration = (entries[i + 1][0] - stamp_ns) / 1e9
        elif len(entries) >= 2:
            duration = (entries[-1][0] - entries[-2][0]) / 1e9
        else:
            duration = 0.1
        lines.append(f"file '{prefix}/{file_name}'")
        lines.append(f"duration {duration:.9f}")

    concat_path = session_dir / (concat_name or f"ffconcat_{Path(out_name).stem}.txt")
    concat_path.write_text("\n".join(lines) + "\n")

    out_path = session_dir / out_name
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_path),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out_path),
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"ffmpeg encoding failed:\n{result.stderr[-800:]}")
        return None
    print(f"Video encoded ({len(entries)} frames) -> {out_path}")
    return out_path
