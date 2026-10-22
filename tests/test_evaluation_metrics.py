import math

import pytest
from sklearn.metrics import silhouette_score as sklearn_silhouette_score

from src.evaluation.evaluation_library import evaluate_library_metrics
from src.evaluation.evaluation_library import silhouette_samples as library_silhouette_samples
from src.evaluation.metrics import silhouette_samples, silhouette_score


DATA = [
    (0.0, 0.0),
    (0.0, 0.2),
    (0.2, 0.0),
    (5.0, 5.0),
    (5.0, 5.2),
    (5.2, 5.0),
    (12.0, 12.0),
]
TRUE_LABELS = [10, 10, 10, 20, 20, 20, -1]
PREDICTED_LABELS = [0, 0, -1, 1, 1, 1, -1]


def test_custom_silhouette_matches_sklearn_on_non_noise_points():
    custom_samples = silhouette_samples(DATA, PREDICTED_LABELS)
    library_samples = library_silhouette_samples(DATA, PREDICTED_LABELS)
    included = [index for index, label in enumerate(PREDICTED_LABELS) if label != -1]

    assert all(custom_samples[index] == pytest.approx(library_samples[index]) for index in included)
    assert custom_samples[2] is None
    assert custom_samples[6] is None
    assert silhouette_score(DATA, PREDICTED_LABELS) == pytest.approx(
        sklearn_silhouette_score(
            [DATA[index] for index in included],
            [PREDICTED_LABELS[index] for index in included],
            metric="euclidean",
        )
    )


def test_library_metrics_include_noise_for_external_scores():
    results = evaluate_library_metrics(
        DATA,
        PREDICTED_LABELS,
        true_labels=TRUE_LABELS,
    )

    assert results["n_clusters"] == 2
    assert results["n_noise"] == 2
    assert 0.0 <= results["adjusted_rand_index"] <= 1.0
    assert math.isfinite(results["adjusted_mutual_information"])


def test_singleton_cluster_has_zero_silhouette():
    values = silhouette_samples([(0.0,), (1.0,), (3.0,)], [0, 0, 1])

    assert values[2] == 0.0


@pytest.mark.parametrize(
    ("data", "labels"),
    [
        ([], []),
        ([(0.0,), (1.0,)], [0]),
        ([(0.0,), (1.0,)], [0, -1]),
        ([(0.0,), (1.0, 2.0)], [0, 1]),
    ],
)
def test_invalid_silhouette_inputs_raise_value_error(data, labels):
    with pytest.raises(ValueError):
        silhouette_score(data, labels)
