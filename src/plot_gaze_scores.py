#!/usr/bin/env python3
"""Plot per-label detection score over time from gaze_labels.json.

Usage:
    python src/plot_gaze_scores.py test02/25   # save one figure into that run folder
    python src/plot_gaze_scores.py test02       # save a figure into every sub-run folder

X axis is time (seconds, relative to the first frame), Y axis is the detection
score. Each label is one line; the line breaks wherever the label was not
detected in a frame so no interpolation crosses the gaps. The figure is written
as gaze_scores.png inside the run's own folder (never displayed).
"""

import argparse
import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")  # headless: no display, just save files
import matplotlib.pyplot as plt

OUT_NAME = "gaze_scores.png"

RECORDINGS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "recordings"
)


def resolve_path(arg):
    """Return an absolute path for the argument, trying it as-is then under recordings/."""
    for cand in (arg, os.path.join(RECORDINGS_DIR, arg)):
        if os.path.exists(cand):
            return os.path.abspath(cand)
    sys.exit(f"Path not found: {arg} (also tried under {RECORDINGS_DIR})")


def find_runs(path):
    """Return a list of (name, gaze_labels.json path) for the given path.

    If path itself holds a gaze_labels.json it is a single run; otherwise every
    immediate subdirectory that holds one is treated as a run.
    """
    direct = os.path.join(path, "gaze_labels.json")
    if os.path.isfile(direct):
        return [(os.path.basename(path), direct)]

    runs = []
    for name in sorted(os.listdir(path)):
        sub = os.path.join(path, name, "gaze_labels.json")
        if os.path.isfile(sub):
            runs.append((name, sub))
    return runs


def load_series(labels_json):
    """Return (series, t0) where series maps label -> list of (t_sec, score).

    Missing detections are recorded as None so the line breaks across gaps.
    """
    with open(labels_json) as f:
        data = json.load(f)

    frames = data.get("frames", [])
    stamps = [f["stamp_ns"] for f in frames if "stamp_ns" in f]
    if not stamps:
        return {}, None
    t0 = min(stamps)

    # Collect every label that ever appears so all lines share a time base.
    all_labels = set()
    for fr in frames:
        for det in fr.get("detected", []):
            all_labels.add(det["label"])

    series = {label: [] for label in sorted(all_labels)}
    for fr in frames:
        t = (fr["stamp_ns"] - t0) / 1e9
        found = {det["label"]: det["score"] for det in fr.get("detected", [])}
        for label in series:
            series[label].append((t, found.get(label)))  # None -> break
    return series, t0


def save_run(name, labels_json):
    """Render one run's score-over-time figure and save it beside its json."""
    out_path = os.path.join(os.path.dirname(labels_json), OUT_NAME)
    series, _ = load_series(labels_json)
    if not series:
        print(f"skip {name}: no detections")
        return

    fig, ax = plt.subplots(figsize=(11, 6))
    for label, pts in series.items():
        xs = [t for t, _ in pts]
        # None scores become NaN so matplotlib breaks the line across gaps.
        ys = [s if s is not None else math.nan for _, s in pts]
        ax.plot(xs, ys, marker=".", markersize=4, linewidth=1.2, label=label)

    ax.set_title(f"Gaze detection scores — {name}")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("score")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize="small", loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path",
                        help="recording dir (e.g. test02) or a single run (e.g. test02/25)")
    args = parser.parse_args()

    path = resolve_path(args.path)
    runs = find_runs(path)
    if not runs:
        sys.exit(f"No gaze_labels.json found in {path}")

    for name, labels_json in runs:
        save_run(name, labels_json)


if __name__ == "__main__":
    main()
