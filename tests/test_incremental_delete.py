import numpy as np
import pytest

from src.clustering.dbscan import DBSCAN
from src.clustering.incremental_dbscan import IncrementalDBSCAN
from src.core.data_point import PointType


def _core_groups(model):
    groups = {}
    for point_id, point in model.points.items():
        if point.point_type == PointType.CORE:
            groups.setdefault(point.cluster_id, set()).add(point_id)
    return {frozenset(ids) for ids in groups.values()}


def _assert_matches_refit(model):
    ordered_ids = list(model.points)
    data = np.asarray([model.points[point_id].coords for point_id in ordered_ids])
    reference = DBSCAN(model.eps, model.min_pts).fit(data)
    expected_groups = {
        frozenset(ordered_ids[index]
                  for index, expected in reference.points.items()
                  if expected.point_type == PointType.CORE and expected.cluster_id == cluster_id)
        for cluster_id in {
            expected.cluster_id
            for expected in reference.points.values()
            if expected.point_type == PointType.CORE
        }
    }
    assert _core_groups(model) == expected_groups

    for index, point_id in enumerate(ordered_ids):
        point = model.points[point_id]
        expected = reference.points[index]
        assert np.array_equal(point.coords, expected.coords)
        expected_neighbor_ids = {ordered_ids[neighbor_id] for neighbor_id in expected.neighbor_ids}
        assert point.neighbor_ids == expected_neighbor_ids
        assert point.point_type == expected.point_type
        if point.point_type == PointType.NOISE:
            assert point.cluster_id == -1
        elif point.point_type == PointType.BORDER:
            adjacent_core_labels = {
                model.points[neighbor_id].cluster_id
                for neighbor_id in point.neighbor_ids
                if model.points[neighbor_id].point_type == PointType.CORE
            }
            assert point.cluster_id in adjacent_core_labels


def test_delete_noise_preserves_cluster_and_neighbor_cache():
    data = np.asarray([[0, 0], [0.4, 0], [0, 0.4], [0.4, 0.4], [5, 5]], dtype=float)
    model = IncrementalDBSCAN(eps=0.7, min_pts=4).fit(data)

    model.delete_point(4)

    _assert_matches_refit(model)
    assert 4 not in model.points


def test_delete_reduction_without_split():
    data = np.asarray([[0.0], [0.9], [1.8], [2.7]])
    model = IncrementalDBSCAN(eps=1.0, min_pts=3).fit(data)
    assert len(_core_groups(model)) == 1

    model.delete(0)

    _assert_matches_refit(model)
    assert len(_core_groups(model)) == 1


def test_delete_last_core_turns_remaining_points_into_noise():
    data = np.asarray([[0, 0], [0.2, 0], [0, 0.2]], dtype=float)
    model = IncrementalDBSCAN(eps=0.5, min_pts=3).fit(data)
    assert len(_core_groups(model)) == 1

    model.delete_point(2)

    _assert_matches_refit(model)
    assert all(point.point_type == PointType.NOISE for point in model.points.values())


def test_delete_core_splits_cluster_and_relabels_borders():
    data = np.asarray([[0.9 * i] for i in range(8)])
    model = IncrementalDBSCAN(eps=1.0, min_pts=3).fit(data)
    assert len(_core_groups(model)) == 1

    model.delete_point(3)

    _assert_matches_refit(model)
    assert len(_core_groups(model)) == 2


def test_delete_missing_id_raises_value_error():
    model = IncrementalDBSCAN(eps=1.0, min_pts=2).fit(np.asarray([[0.0], [0.5]]))

    with pytest.raises(ValueError, match="Không tìm thấy"):
        model.delete_point(10)


def test_insert_point_alias_remains_available_after_delete():
    data = np.asarray([[0.0], [0.5], [10.0]])
    model = IncrementalDBSCAN(eps=0.75, min_pts=2).fit(data)
    model.delete_point(2)

    new_id = model.insert_point(np.asarray([0.25]))

    assert new_id == 3
    _assert_matches_refit(model)
