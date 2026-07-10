"""Cluster stitched-canvas gaze points with a temporal window and pixel radius.

Input is the ``stitched_gaze_track.json`` written by ``src.frame_stitcher``.
For each gaze point, the analyzer looks at nearby samples in time:

  window=7 -> previous 3 points + next 3 points

It counts how many of those neighboring points fall within ``radius`` pixels
of the current point. Points with enough local support are treated as dense,
and consecutive dense points form clusters. A cluster's center is the sample
closest to the temporal middle of that consecutive dense run, not the first
sample and not the geometric centroid. If multiple clusters are found, only
the cluster whose center is closest to the middle of the whole track is kept.

Usage:
  python -m src.gaze_track_cluster recordings/test01/20/stitched_gaze_track.json \
      --window 7 --radius 30
"""

import argparse
import json
from pathlib import Path

import numpy as np


CLUSTER_COLORS = [
    (0, 255, 0),
    (255, 200, 0),
    (0, 200, 255),
    (255, 0, 255),
    (0, 128, 255),
    (255, 255, 0),
    (128, 255, 128),
    (255, 128, 128),
]


def _validate_params(window: int, radius: float) -> None:
    if window < 3:
        raise ValueError("window must be at least 3.")
    if window % 2 == 0:
        raise ValueError("window must be odd, e.g. 7 means 3 before + 3 after.")
    if radius <= 0:
        raise ValueError("radius must be positive.")


def count_neighbors_in_radius(points_xy: np.ndarray, window: int, radius: float) -> np.ndarray:
    """Count local temporal neighbors within radius for each point.

    Self is excluded from the count.
    """
    half = window // 2
    counts = np.zeros(len(points_xy), dtype=np.int64)
    for i, point in enumerate(points_xy):
        start = max(0, i - half)
        stop = min(len(points_xy), i + half + 1)
        distances = np.linalg.norm(points_xy[start:stop] - point, axis=1)
        counts[i] = int(np.count_nonzero(distances <= radius)) - 1
    return counts


def cluster_by_window_radius(
    points_xy: np.ndarray,
    window: int,
    radius: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return ``(cluster_ids, neighbor_counts, min_neighbors)``.

    ``cluster_ids`` is -1 for non-cluster transition samples. The only user
    supplied clustering parameters are ``window`` and ``radius``; the density
    threshold is derived as a majority of the available window neighborhood.
    For window=7 this requires at least 3 of the 6 neighboring samples.
    """
    _validate_params(window, radius)
    min_neighbors = window // 2
    counts = count_neighbors_in_radius(points_xy, window, radius)
    dense = counts >= min_neighbors

    cluster_ids = np.full(len(points_xy), -1, dtype=np.int64)
    cluster_id = -1
    was_dense = False
    for i, is_dense in enumerate(dense):
        if is_dense and not was_dense:
            cluster_id += 1
        if is_dense:
            cluster_ids[i] = cluster_id
        was_dense = bool(is_dense)

    return cluster_ids, counts, min_neighbors


def summarize_clusters(records: list[dict], points_xy: np.ndarray, cluster_ids: np.ndarray) -> list[dict]:
    summaries = []
    max_cluster_id = int(cluster_ids.max()) if len(cluster_ids) else -1
    for cluster_id in range(max_cluster_id + 1):
        mask = cluster_ids == cluster_id
        sample_indices = np.flatnonzero(mask)
        cluster_points = points_xy[mask]
        cluster_records = [record for record, keep in zip(records, mask) if keep]
        center_pos = (len(cluster_records) - 1) // 2
        center_index = int(sample_indices[center_pos])
        center_record = cluster_records[center_pos]
        center_xy = cluster_points[center_pos]
        distances = np.linalg.norm(cluster_points - center_xy, axis=1)
        start = cluster_records[0]
        end = cluster_records[-1]
        start_t = start.get("t_sec")
        end_t = end.get("t_sec")
        summaries.append(
            {
                "id": cluster_id,
                "n_points": int(mask.sum()),
                "stamp_ns_start": start.get("stamp_ns"),
                "stamp_ns_end": end.get("stamp_ns"),
                "t_start_sec": start_t,
                "t_end_sec": end_t,
                "duration_sec": (
                    round(float(end_t) - float(start_t), 6)
                    if start_t is not None and end_t is not None
                    else None
                ),
                "center_sample_index": center_index,
                "center_stamp_ns": center_record.get("stamp_ns"),
                "center_t_sec": center_record.get("t_sec"),
                "center_canvas_xy": [float(center_xy[0]), float(center_xy[1])],
                "mean_radius_px": round(float(distances.mean()), 2),
                "max_radius_px": round(float(distances.max()), 2),
            }
        )
    return summaries


def keep_track_middle_cluster(
    cluster_ids: np.ndarray,
    clusters: list[dict],
    n_points: int,
) -> tuple[np.ndarray, list[dict], int | None]:
    """Keep only the cluster centered closest to the whole track's time middle."""
    if not clusters:
        return cluster_ids, clusters, None

    track_middle_index = (n_points - 1) / 2.0
    keep = min(
        clusters,
        key=lambda cluster: abs(cluster["center_sample_index"] - track_middle_index),
    )
    filtered_ids = np.full(len(cluster_ids), -1, dtype=np.int64)
    filtered_ids[cluster_ids == keep["id"]] = 0
    kept_cluster = {**keep, "id": 0, "original_id": keep["id"]}
    return filtered_ids, [kept_cluster], int(keep["id"])


def save_visualization(
    points_xy: np.ndarray,
    cluster_ids: np.ndarray,
    clusters: list[dict],
    track_path: Path,
    output_path: Path,
) -> None:
    import cv2

    background_path = track_path.parent / "stitched_trajectory.png"
    canvas = cv2.imread(str(background_path)) if background_path.exists() else None
    if canvas is None:
        width = int(np.ceil(points_xy[:, 0].max())) + 20
        height = int(np.ceil(points_xy[:, 1].max())) + 20
        canvas = np.zeros((height, width, 3), dtype=np.uint8)

    pts = np.rint(points_xy).astype(np.int32)
    if len(pts) >= 2:
        cv2.polylines(canvas, [pts], False, (150, 150, 150), 1, cv2.LINE_AA)

    for pt, cluster_id in zip(pts, cluster_ids):
        color = (160, 160, 160) if cluster_id < 0 else CLUSTER_COLORS[cluster_id % len(CLUSTER_COLORS)]
        cv2.circle(canvas, tuple(pt), 3, color, -1, cv2.LINE_AA)

    for cluster in clusters:
        color = CLUSTER_COLORS[cluster["id"] % len(CLUSTER_COLORS)]
        cx, cy = (int(round(v)) for v in cluster["center_canvas_xy"])
        radius = max(8, int(round(cluster["max_radius_px"])))
        cv2.circle(canvas, (cx, cy), radius, color, 2, cv2.LINE_AA)
        cv2.putText(
            canvas,
            str(cluster["id"]),
            (cx + radius + 4, cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(output_path), canvas)


def analyze_track(track_path: Path, window: int, radius: float, save_png: bool = True) -> Path:
    data = json.loads(track_path.read_text())
    records = data.get("points", [])
    if not records:
        raise ValueError(f"No points found in {track_path}.")

    points_xy = np.asarray([record["canvas_xy"] for record in records], dtype=np.float64)
    cluster_ids, neighbor_counts, min_neighbors = cluster_by_window_radius(
        points_xy, window, radius
    )
    clusters = summarize_clusters(records, points_xy, cluster_ids)
    original_cluster_count = len(clusters)
    cluster_ids, clusters, kept_original_cluster_id = keep_track_middle_cluster(
        cluster_ids, clusters, len(records)
    )

    output_path = track_path.with_name(f"{track_path.stem}_window_cluster.json")
    output = {
        "source": str(track_path),
        "coordinate_frame": data.get("coordinate_frame"),
        "params": {
            "window": window,
            "radius_px": radius,
            "min_neighbors": min_neighbors,
        },
        "count": len(records),
        "original_cluster_count": original_cluster_count,
        "cluster_count": len(clusters),
        "kept_original_cluster_id": kept_original_cluster_id,
        "clusters": clusters,
        "points": [
            {
                **record,
                "neighbors_in_radius": int(count),
                "cluster_id": int(cluster_id),
            }
            for record, count, cluster_id in zip(records, neighbor_counts, cluster_ids)
        ],
    }
    output_path.write_text(json.dumps(output, indent=2))

    if save_png:
        png_path = output_path.with_suffix(".png")
        save_visualization(points_xy, cluster_ids, clusters, track_path, png_path)

    dense_count = int(np.count_nonzero(cluster_ids >= 0))
    print(
        f"{len(records)} points, window={window}, radius={radius}px, "
        f"min_neighbors={min_neighbors} -> {len(clusters)} clusters, "
        f"{dense_count} clustered points"
    )
    print(f"Saved cluster JSON: {output_path}")
    if save_png:
        print(f"Saved cluster PNG: {output_path.with_suffix('.png')}")

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cluster stitched_gaze_track.json with temporal window + pixel radius."
    )
    parser.add_argument("track", type=Path, help="Path to stitched_gaze_track.json")
    parser.add_argument(
        "--window",
        type=int,
        required=True,
        help="Odd window length. Example: 7 means previous 3 + next 3 points.",
    )
    parser.add_argument(
        "--radius",
        type=float,
        required=True,
        help="Pixel radius used when counting neighbors inside the temporal window.",
    )
    parser.add_argument(
        "--no-png",
        action="store_true",
        help="Only write JSON; skip the visualization PNG.",
    )
    args = parser.parse_args()
    analyze_track(args.track, args.window, args.radius, save_png=not args.no_png)


if __name__ == "__main__":
    main()
