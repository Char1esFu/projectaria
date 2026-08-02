"""Re-run the variance-to-gaze-label pipeline for a saved recording.

``gaze_labels.json`` supplies immutable logged detections plus the existing
detection in-fill.  This tool never writes that file or regenerates in-fill. It
starts again at variance selection, replaces every variance visualization/export,
and stores the recomputed normalized result in ``published.json``.
"""

import argparse
import contextlib
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.detection_infill import merge_infilled  # noqa: E402
from src.gaze_recording import (  # noqa: E402
    compute_average_entries,
    normalize_entries,
    save_published_result,
)
from src.gaze_rgb_config import (  # noqa: E402
    DEFAULT_GAZE_BOUNDARY_RADIUS,
)
from src.gaze_score_stability import (  # noqa: E402
    export_selected_frames,
    generate_stability_plots,
    select_stable_windows,
)


def resolve_recording(path: Path) -> tuple[Path, Path]:
    """Return ``(recording_dir, gaze_labels.json)`` for either input form."""
    path = path.expanduser()
    labels_path = path / "gaze_labels.json" if path.is_dir() else path
    return labels_path.parent, labels_path


def compute_result(
    analysis_log: list[dict], selected_stamps: list[int]
) -> tuple[list[dict], bool]:
    """Apply the live selected-frame average and empty-selection fallback."""
    selected_set = set(selected_stamps) if selected_stamps else None
    averaged = compute_average_entries(analysis_log, selected_set)
    used_selected = bool(selected_stamps and averaged)
    if selected_stamps and not averaged:
        averaged = compute_average_entries(analysis_log, None)
    return normalize_entries(averaged), used_selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Re-run variance selection and the normalized gaze-label result; "
            "overwrite derived files while preserving detection in-fill."
        )
    )
    parser.add_argument(
        "input", type=Path,
        help="gaze_labels.json path, or the recording directory containing it",
    )
    parser.add_argument("-w", "--window", type=int, default=3)
    parser.add_argument("-t", "--threshold", type=float, default=1.5e-3)
    parser.add_argument(
        "-n", "--top", type=int, default=1,
        help=("if nothing passes threshold, keep N lowest-variance interior "
              "windows; <=0 keeps all (passing windows are always all kept)"),
    )
    parser.add_argument(
        "--boundary-radius", type=float, default=DEFAULT_GAZE_BOUNDARY_RADIUS,
    )
    parser.add_argument("--force-endpoint-points", type=int, default=1)
    parser.add_argument(
        "--hide-excluded", action="store_true",
        help=("hide endpoint-excluded windows from the variance timeline; "
              "does not change selection or the normalized result"),
    )
    parser.add_argument(
        "--pretty", action="store_true", help="pretty-print the final JSON result",
    )
    parser.add_argument(
        "--details", action="store_true",
        help="print selection and averaging diagnostics to stderr",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.window < 2:
        raise SystemExit("window must be >= 2")

    recording_dir, labels_path = resolve_recording(args.input)
    track_path = recording_dir / "stitched_gaze_track.json"
    try:
        data = json.loads(labels_path.read_text())
        frames = data.get("frames", [])
        if not isinstance(frames, list):
            raise ValueError("'frames' must be a JSON array")
        # Existing in-fill is input data here.  It is merged for analysis but
        # neither recalculated nor changed in gaze_labels.json.
        analysis_log = merge_infilled(frames, data.get("infilled_frames"))
        # Mirror the online guard: without a stitched track there is no
        # variance selection (and no endpoint filtering), so averaging falls
        # back to the whole recording.
        selected_windows = []
        if track_path.exists():
            selected_windows, _ = select_stable_windows(
                analysis_log,
                track_path,
                window=args.window,
                threshold=args.threshold,
                top=args.top if args.top > 0 else None,
                boundary_radius=args.boundary_radius,
                force_endpoint_points=args.force_endpoint_points,
            )
        selected_stamps = sorted({
            window["center_stamp_ns"] for window in selected_windows
        })
        result, used_selected = compute_result(analysis_log, selected_stamps)
    except (OSError, json.JSONDecodeError, TypeError, ValueError, KeyError) as exc:
        raise SystemExit(f"Cannot compute gaze label from {labels_path}: {exc}") from exc

    infilled_count = sum(
        len(frame.get("detected", []))
        for frame in data.get("infilled_frames", [])
    )
    source_note = (
        f"logged + in-filled ({infilled_count} merged detection(s))"
        if infilled_count else "logged only"
    )

    # Replace every output derived from variance selection.
    # Keep stdout machine-readable: plot/export progress goes to stderr and the
    # final line on stdout is only the normalized JSON topic value.
    with contextlib.redirect_stdout(sys.stderr):
        generate_stability_plots(
            recording_dir,
            analysis_log,
            selected_windows,
            window=args.window,
            threshold=args.threshold,
            boundary_radius=args.boundary_radius,
            force_endpoint_points=args.force_endpoint_points,
            hide_excluded=args.hide_excluded,
            source_note=source_note,
        )
        export_selected_frames(
            recording_dir,
            recording_dir / "frames",
            analysis_log,
            selected_stamps,
        )

    # gaze_labels.json is immutable analysis input. Only the derived topic
    # result is replaced, in its own file.
    published_path = save_published_result(recording_dir, result)

    averaging_frames = [
        frame for frame in analysis_log
        if frame.get("stamp_ns") is not None
        and (not used_selected or frame["stamp_ns"] in set(selected_stamps))
    ]
    details = {
        "selected_stamps_ns": selected_stamps,
        "averaging_source": (
            "VARIANCE_SELECTED_FRAMES" if used_selected
            else "WHOLE_RECORDING_FALLBACK"
        ),
        "averaging_frame_count": len(averaging_frames),
        "infilled_detection_count": infilled_count,
        "immutable_gaze_labels": str(labels_path),
        "overwritten_published": str(published_path),
        "selection_file": str(recording_dir / "variance_selected" / "scores.json"),
    }
    if args.details:
        print(json.dumps(details, ensure_ascii=False, indent=2), file=sys.stderr)
    print(json.dumps(result, ensure_ascii=False, indent=2 if args.pretty else None))


if __name__ == "__main__":
    main()
