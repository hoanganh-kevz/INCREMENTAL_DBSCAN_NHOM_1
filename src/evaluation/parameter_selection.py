"""Small DBSCAN parameter grid and k-distance profile for task V05."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from baseline.sklearn_dbscan import fit_full_dbscan
from sklearn.metrics import adjusted_rand_score

from src.clustering.dbscan import DBSCAN
from src.evaluation.evaluation_library import silhouette_score as library_silhouette_score
from src.evaluation.metrics import silhouette_score as custom_silhouette_score
from src.visualization import k_distances, knee_value, plot_k_distance

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "data" / "processed" / "moons_n500_scaled.csv"
DEFAULT_EPS_VALUES = (0.14, 0.16, 0.18, 0.20, 0.22)
DEFAULT_MIN_PTS_VALUES = (4, 6, 8)
K_DISTANCE_MIN_PTS = 4
SELECTED_EPS = 0.18
SELECTED_MIN_PTS = 4


def _silhouette_or_none(
    metric, data: np.ndarray, labels: Iterable[int]
) -> float | None:
    labels = list(labels)
    if len(set(labels) - {-1}) < 2:
        return None
    try:
        return metric(data, labels)
    except ValueError:
        return None


def evaluate_parameter_grid(
    data: Iterable[Iterable[float]],
    eps_values: Iterable[float],
    min_pts_values: Iterable[int],
    *,
    truth_labels: Iterable[int] | None = None,
    selected: tuple[float, int] | None = None,
) -> pd.DataFrame:
    """Fit custom and sklearn DBSCAN for each identical (eps, MinPts) pair."""
    coords = np.asarray(data, dtype=float)
    if coords.ndim != 2 or not len(coords) or not coords.shape[1]:
        raise ValueError("data cần là ma trận hữu hạn có ít nhất một điểm và một chiều")
    if not np.isfinite(coords).all():
        raise ValueError("data chỉ được chứa giá trị hữu hạn")
    truth = None if truth_labels is None else np.asarray(list(truth_labels))
    if truth is not None and len(truth) != len(coords):
        raise ValueError("truth_labels phải có cùng số phần tử với data")

    rows = []
    for eps in eps_values:
        for min_pts in min_pts_values:
            custom_model = DBSCAN(float(eps), int(min_pts)).fit(coords)
            custom_labels = np.asarray(custom_model.get_labels(), dtype=int)
            sklearn_labels, _ = fit_full_dbscan(coords, float(eps), int(min_pts))
            custom_noise = custom_labels == -1
            sklearn_noise = sklearn_labels == -1
            row = {
                "eps": float(eps),
                "min_pts": int(min_pts),
                "selected": selected == (float(eps), int(min_pts)) if selected else False,
                "n_clusters_custom": len(set(custom_labels.tolist()) - {-1}),
                "n_noise_custom": int(custom_noise.sum()),
                "noise_fraction_custom": float(custom_noise.mean()),
                "silhouette_custom": _silhouette_or_none(
                    custom_silhouette_score, coords, custom_labels
                ),
                "n_clusters_sklearn": len(set(sklearn_labels.tolist()) - {-1}),
                "n_noise_sklearn": int(sklearn_noise.sum()),
                "silhouette_sklearn": _silhouette_or_none(
                    library_silhouette_score, coords, sklearn_labels
                ),
                "ari_custom_vs_sklearn": float(adjusted_rand_score(custom_labels, sklearn_labels)),
                "noise_equal": bool(np.array_equal(custom_noise, sklearn_noise)),
            }
            if truth is not None:
                row["ari_custom_vs_truth"] = float(adjusted_rand_score(truth, custom_labels))
                row["ari_sklearn_vs_truth"] = float(adjusted_rand_score(truth, sklearn_labels))
            rows.append(row)

    return pd.DataFrame(rows)


def k_distance_profile(
    data: Iterable[Iterable[float]], min_pts: int = K_DISTANCE_MIN_PTS
) -> tuple[pd.DataFrame, float]:
    """Return descending k-distances and the knee value for the requested MinPts."""
    coords = np.asarray(data, dtype=float)
    distances = np.sort(k_distances(coords, min_pts=min_pts))[::-1]
    knee_index, knee_eps = knee_value(distances)
    profile = pd.DataFrame(
        {
            "rank_descending": np.arange(1, len(distances) + 1),
            "k_distance": distances,
        }
    )
    return profile, knee_eps


def run_parameter_selection(data_path: Path = DEFAULT_DATA) -> tuple[pd.DataFrame, float]:
    """Run the documented moon-data grid and save its table and k-distance plot."""
    frame = pd.read_csv(data_path, encoding="utf-8")
    feature_columns = [column for column in frame.columns if column.endswith("_z")]
    if len(feature_columns) != 2 or "label" not in frame.columns:
        raise ValueError(f"Expected two *_z features and a label column in {data_path}")
    coords = frame[feature_columns].to_numpy(dtype=float)
    truth = frame["label"].to_numpy(dtype=int)
    profile, knee_eps = k_distance_profile(coords, K_DISTANCE_MIN_PTS)
    results = evaluate_parameter_grid(
        coords,
        DEFAULT_EPS_VALUES,
        DEFAULT_MIN_PTS_VALUES,
        truth_labels=truth,
        selected=(SELECTED_EPS, SELECTED_MIN_PTS),
    )

    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(
        results_dir / "parameter_selection.csv",
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )
    profile.to_csv(
        results_dir / "k_distance_moons.csv",
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )
    figure = plot_k_distance(
        coords,
        min_pts=K_DISTANCE_MIN_PTS,
        eps=SELECTED_EPS,
        title="Moons 2D: k-distance va Eps chon (MinPts = 4)",
        unit="don vi sau z-score",
    )
    figure.savefig(results_dir / "k_distance_moons.png", dpi=180, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(figure)
    return results, knee_eps


def main() -> int:
    results, knee_eps = run_parameter_selection()
    selected = results.loc[results["selected"]].iloc[0]
    print(
        "Wrote results/parameter_selection.csv, results/k_distance_moons.csv, "
        "and results/k_distance_moons.png"
    )
    print(
        f"k-distance knee (MinPts={K_DISTANCE_MIN_PTS}): Eps≈{knee_eps:.4f}; "
        f"selected Eps={selected['eps']:.2f}, MinPts={int(selected['min_pts'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
