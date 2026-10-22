import pytest

from src.core.distance import DistanceMetric


@pytest.mark.parametrize(
    ("first", "second", "expected"),
    [
        ([0.0, 0.0], [3.0, 4.0], 5.0),
        ([1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], 8.0),
    ],
    ids=["2d", "4d"],
)
def test_euclidean_distance_in_two_and_n_dimensions(first, second, expected):
    assert DistanceMetric.euclidean(first, second) == pytest.approx(expected)


def test_euclidean_distance_rejects_mismatched_dimensions():
    with pytest.raises(ValueError, match="cùng số chiều"):
        DistanceMetric.euclidean([0.0, 1.0], [0.0, 1.0, 2.0])
