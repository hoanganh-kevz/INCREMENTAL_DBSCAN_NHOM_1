"""Reference metrics from scikit-learn, isolated from the custom algorithm."""

from __future__ import annotations

from typing import Any, Iterable

from .metrics import _prepare_inputs


def silhouette_samples(
    data: Iterable[Iterable[float]],
    labels: Iterable[Any],
    *,
    noise_label: Any = -1,
) -> list[float | None]:
    """Return scikit-learn silhouette values aligned with the input points.

    Noise points are omitted from the score and represented by ``None`` so the
    output follows the same policy as :func:`metrics.silhouette_samples`.
    """
    from sklearn.metrics import silhouette_samples as sklearn_silhouette_samples

    rows, label_list = _prepare_inputs(data, labels)
    included = [index for index, label in enumerate(label_list) if label != noise_label]
    scores = sklearn_silhouette_samples(
        [rows[index] for index in included],
        [label_list[index] for index in included],
        metric="euclidean",
    )
    result: list[float | None] = [None] * len(rows)
    for index, score in zip(included, scores):
        result[index] = float(score)
    return result


def silhouette_score(
    data: Iterable[Iterable[float]],
    labels: Iterable[Any],
    *,
    noise_label: Any = -1,
) -> float:
    """Return scikit-learn's mean silhouette after excluding noise points."""
    from sklearn.metrics import silhouette_score as sklearn_silhouette_score

    rows, label_list = _prepare_inputs(data, labels)
    included = [index for index, label in enumerate(label_list) if label != noise_label]
    return float(
        sklearn_silhouette_score(
            [rows[index] for index in included],
            [label_list[index] for index in included],
            metric="euclidean",
        )
    )


def evaluate_library_metrics(
    data: Iterable[Iterable[float]],
    predicted_labels: Iterable[Any],
    *,
    true_labels: Iterable[Any] | None = None,
    noise_label: Any = -1,
) -> dict[str, float | int]:
    """Compute library quality metrics and optional external-label scores.

    Silhouette excludes noise. ARI and AMI include every observation, with
    ``noise_label`` treated as an ordinary predicted label. This keeps noise
    detection errors visible when comparing predictions with ground truth.
    """
    from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

    rows, predicted = _prepare_inputs(data, predicted_labels)
    included = [index for index, label in enumerate(predicted) if label != noise_label]
    result: dict[str, float | int] = {
        "silhouette_score": silhouette_score(rows, predicted, noise_label=noise_label),
        "n_clusters": len({predicted[index] for index in included}),
        "n_noise": len(predicted) - len(included),
    }

    if true_labels is not None:
        truth = list(true_labels)
        if len(truth) != len(predicted):
            raise ValueError("true_labels và predicted_labels phải có cùng số phần tử")
        result["adjusted_rand_index"] = float(adjusted_rand_score(truth, predicted))
        result["adjusted_mutual_information"] = float(
            adjusted_mutual_info_score(truth, predicted)
        )
    return result
