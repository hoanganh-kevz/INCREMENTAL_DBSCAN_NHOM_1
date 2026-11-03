"""Focused tests for incremental DBSCAN deletions."""

import numpy as np
import pytest

from src.clustering.dbscan import DBSCAN
from src.clustering.incremental_dbscan import IncrementalDBSCAN
from src.core.data_point import PointType


def _core_groups(model: DBSCAN) -> set[frozenset[int]]:
    groups: dict[int, set[int]] = {}
    for point_id, point in model.points.items():
        if point.point_type == PointType.CORE:
            assert point.cluster_id is not None and point.cluster_id >= 0
            groups.setdefault(point.cluster_id, set()).add(point_id)
    return {frozenset(group) for group in groups.values()}


def _assert_matches_refit(model: IncrementalDBSCAN) -> None:
    """Compare cached state with a fresh run of the custom DBSCAN."""
    point_ids = list(model.points)
    data = np.asarray([model.points[point_id].coords for point_id in point_ids])
    reference = DBSCAN(eps=model.eps, min_pts=model.min_pts).fit(data)

    expected_core_groups = {
        frozenset(
            point_ids[index]
            for index, expected in reference.points.items()
            if expected.point_type == PointType.CORE
            and expected.cluster_id == cluster_id
        )
        for cluster_id in {
            point.cluster_id
            for point in reference.points.values()
            if point.point_type == PointType.CORE
        }
    }
    assert _core_groups(model) == expected_core_groups

    for index, point_id in enumerate(point_ids):
        actual = model.points[point_id]
        expected = reference.points[index]
        assert np.array_equal(actual.coords, expected.coords)
        assert actual.point_type == expected.point_type
        assert actual.neighbor_count == expected.neighbor_count
        assert actual.neighbor_ids == {
            point_ids[neighbor_index] for neighbor_index in expected.neighbor_ids
        }
        if actual.point_type == PointType.NOISE:
            assert actual.cluster_id == -1
        elif actual.point_type == PointType.BORDER:
            adjacent_core_clusters = {
                model.points[neighbor_id].cluster_id
                for neighbor_id in actual.neighbor_ids
                if model.points[neighbor_id].point_type == PointType.CORE
            }
            assert actual.cluster_id in adjacent_core_clusters


def test_delete_noise_preserves_clusters_and_neighbor_cache():
    data = np.asarray([[0, 0], [0.4, 0], [0, 0.4], [0.4, 0.4], [5, 5]], dtype=float)
    model = IncrementalDBSCAN(eps=0.7, min_pts=4).fit(data)
    core_groups_before = _core_groups(model)

    model.delete_point(4)

    assert 4 not in model.points
    assert _core_groups(model) == core_groups_before
    _assert_matches_refit(model)


def test_delete_border_preserves_cluster_structure():
    data = np.asarray(
        [[0.0, 0.0], [0.4, 0.0], [0.0, 0.4], [0.4, 0.4], [0.9, 0.4]],
        dtype=float,
    )
    model = IncrementalDBSCAN(eps=0.7, min_pts=4).fit(data)
    border_id = 4
    assert model.points[border_id].point_type == PointType.BORDER
    core_groups_before = _core_groups(model)

    model.delete_point(border_id)

    assert border_id not in model.points
    assert _core_groups(model) == core_groups_before
    _assert_matches_refit(model)


def test_delete_reduction_keeps_one_cluster():
    data = np.asarray([[0.0], [0.9], [1.8], [2.7]])
    model = IncrementalDBSCAN(eps=1.0, min_pts=3).fit(data)
    assert len(_core_groups(model)) == 1

    model.delete_point(0)

    assert len(_core_groups(model)) == 1
    _assert_matches_refit(model)


def test_delete_last_core_turns_remaining_points_into_noise():
    data = np.asarray([[0, 0], [0.2, 0], [0, 0.2]], dtype=float)
    model = IncrementalDBSCAN(eps=0.5, min_pts=3).fit(data)
    assert len(_core_groups(model)) == 1

    model.delete_point(2)

    assert all(point.point_type == PointType.NOISE for point in model.points.values())
    _assert_matches_refit(model)


def test_delete_core_splits_one_cluster_in_two():
    data = np.asarray([[0.9 * index] for index in range(8)])
    model = IncrementalDBSCAN(eps=1.0, min_pts=3).fit(data)
    assert len(_core_groups(model)) == 1

    model.delete_point(3)

    assert 3 not in model.points
    assert len(_core_groups(model)) == 2
    _assert_matches_refit(model)


def test_delete_core_splits_one_cluster_in_three():
    names = [
        "R1", "R2", "R3", "R4", "L1", "L2", "L3", "L4",
        "T1", "T2", "T3", "T4", "p",
    ]
    data = np.asarray(
        [
            [0.95, -0.1], [0.95, 0.1], [1.15, -0.1], [1.15, 0.1],
            [-0.95, -0.1], [-0.95, 0.1], [-1.15, -0.1], [-1.15, 0.1],
            [-0.1, 0.95], [0.1, 0.95], [-0.1, 1.15], [0.1, 1.15], [0, 0],
        ],
        dtype=float,
    )
    model = IncrementalDBSCAN(eps=1.0, min_pts=4).fit(data)
    assert len(_core_groups(model)) == 1
    deleted_id = names.index("p")
    old_neighbors = model.points[deleted_id].neighbor_ids.copy()

    model.delete_point(deleted_id)

    id_to_name = dict(enumerate(names))
    update_seeds = {
        id_to_name[point_id]
        for point_id in old_neighbors - {deleted_id}
        if point_id in model.points and model.points[point_id].point_type == PointType.CORE
    }
    assert update_seeds == {"L1", "L2", "R1", "R2", "T1", "T2"}
    assert len(_core_groups(model)) == 3
    assert deleted_id not in model.points
    _assert_matches_refit(model)


def test_delete_bridge_core_keeps_cluster_when_long_alternative_path_exists():
    data = np.asarray(
        [
            [0, 0], [0.5, 0], [0, 0.5], [2, 0], [2.5, 0], [2, 0.5], [1.25, 0],
            [0, -0.75], [0, -1.5], [0, -2.25], [0.75, -2.25], [1.5, -2.25],
            [2, -1.5], [2, -0.75],
        ],
        dtype=float,
    )
    names = ["L1", "L2", "L3", "R1", "R2", "R3", "p", "d1", "d2", "d3", "d4", "d5", "d6", "d7"]
    model = IncrementalDBSCAN(eps=1.0, min_pts=3).fit(data)
    assert len(_core_groups(model)) == 1
    deleted_id = names.index("p")
    old_neighbors = model.points[deleted_id].neighbor_ids.copy()

    model.delete_point(deleted_id)

    id_to_name = dict(enumerate(names))
    update_seeds = {
        id_to_name[point_id]
        for point_id in old_neighbors - {deleted_id}
        if point_id in model.points and model.points[point_id].point_type == PointType.CORE
    }
    assert update_seeds == {"L2", "R1", "R3"}
    assert len(_core_groups(model)) == 1
    _assert_matches_refit(model)


def test_delete_unknown_id_raises_value_error():
    model = IncrementalDBSCAN(eps=1.0, min_pts=2).fit(np.asarray([[0.0], [0.5]]))

    with pytest.raises(ValueError, match="Không tìm thấy"):
        model.delete_point(10)
