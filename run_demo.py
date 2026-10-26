"""Chạy ví dụ Incremental DBSCAN 2D có thể lặp lại với seed cố định."""

from __future__ import annotations

import argparse
from collections import Counter

import numpy as np

from src.clustering.incremental_dbscan import IncrementalDBSCAN


EPS = 1.1
MIN_PTS = 4
DEFAULT_SEED = 42


def make_demo_data(seed: int = DEFAULT_SEED) -> np.ndarray:
    """Sinh hai cụm 2D dày đặc để điểm cầu nối có thể gộp chúng."""
    rng = np.random.default_rng(seed)
    left_cluster = rng.normal(loc=(-1.0, 0.0), scale=0.05, size=(6, 2))
    right_cluster = rng.normal(loc=(1.0, 0.0), scale=0.05, size=(6, 2))
    return np.vstack((left_cluster, right_cluster))


def print_summary(title: str, model: IncrementalDBSCAN) -> None:
    """In số cụm và loại điểm sau mỗi thao tác."""
    labels = model.get_labels()
    n_clusters = len({label for label in labels if label != -1})
    point_types = Counter(point.point_type.value for point in model.points.values())
    print(
        f"{title}: n={len(labels)}, clusters={n_clusters}, "
        f"noise={labels.count(-1)}, core={point_types['CORE']}, "
        f"border={point_types['BORDER']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Demo DBSCAN 2D: insert một điểm cầu nối rồi delete điểm đó."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"seed sinh dữ liệu; mặc định {DEFAULT_SEED}",
    )
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("--seed phải là số nguyên không âm")

    model = IncrementalDBSCAN(eps=EPS, min_pts=MIN_PTS)
    model.fit(make_demo_data(args.seed))
    print(f"Incremental DBSCAN 2D | seed={args.seed} | eps={EPS} | MinPts={MIN_PTS}")
    print_summary("Ban đầu", model)

    bridge_id = model.insert((0.0, 0.0))
    print_summary(f"Sau insert điểm cầu nối (id={bridge_id})", model)

    model.delete(bridge_id)
    print_summary("Sau delete điểm cầu nối", model)


if __name__ == "__main__":
    main()
