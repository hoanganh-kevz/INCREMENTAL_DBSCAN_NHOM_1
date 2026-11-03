"""Focused tests for incremental DBSCAN insertions."""

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


@pytest.mark.parametrize(
    "data, inserted, core_ids, noise_ids, new_core_ids, seeds, new_type",
    [
        (
            [[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1]],
            [5, 5],
            {1, 4},
            {6},
            set(),
            set(),
            PointType.NOISE,
        ),
        (
            [[0, 0], [1, 0], [2, 0]],
            [1, 0.5],
            {1},
            set(),
            {1},
            {1},
            PointType.BORDER,
        ),
        (
            [[0, 0], [1, 0], [0, 1], [1, 1], [2, 0], [2, 1]],
            [2.2, 0.5],
            {1, 3, 4, 5},
            set(),
            {4, 5},
            {1, 3, 4, 5},
            PointType.BORDER,
        ),
        (
            [
                [-1, 0], [0, 0], [1, 0], [-1, 1], [0, 1], [1, 1],
                [2.8, 0], [3.8, 0], [4.8, 0], [2.8, 1], [3.8, 1], [4.8, 1],
            ],
            [1.9, 0.5],
            {1, 2, 4, 5, 6, 7, 9, 10, 12},
            set(),
            {2, 5, 6, 9, 12},
            {1, 2, 4, 5, 6, 7, 9, 10, 12},
            PointType.CORE,
        ),
    ],
    ids=["noise", "creation", "absorption", "merge"],
)
def test_insert_cases(data, inserted, core_ids, noise_ids, new_core_ids, seeds, new_type):
    model = IncrementalDBSCAN(eps=1.1, min_pts=4).fit(np.asarray(data, dtype=float))

    new_id = model.insert(np.asarray(inserted, dtype=float))

    assert new_id == len(data)
    assert model.points[new_id].point_type == new_type
    assert {
        point_id
        for point_id, point in model.points.items()
        if point.point_type == PointType.CORE
    } == core_ids
    assert {
        point_id
        for point_id, point in model.points.items()
        if point.point_type == PointType.NOISE
    } == noise_ids
    assert model._find_new_core_points(new_id) == new_core_ids
    assert model._get_update_seeds(new_core_ids) == seeds
    _assert_matches_refit(model)


@pytest.mark.parametrize(
    "coords, expected_type",
    [([1.5, 0], PointType.BORDER), ([1.5625, 0], PointType.NOISE)],
    ids=["exactly-eps", "just-outside-eps"],
)
def test_insert_respects_eps_boundary(coords, expected_type):
    data = np.asarray([[0, 0], [0.5, 0], [0, 0.5], [0.5, 0.5]], dtype=float)
    model = IncrementalDBSCAN(eps=1.0, min_pts=4).fit(data)

    new_id = model.insert(np.asarray(coords, dtype=float))

    assert model.points[new_id].point_type == expected_type
    if expected_type == PointType.BORDER:
        assert 1 in model.points[new_id].neighbor_ids
        assert new_id in model.points[1].neighbor_ids
    else:
        assert model.points[new_id].neighbor_ids == {new_id}
    _assert_matches_refit(model)


def test_insert_border_does_not_merge_two_clusters():
    data = np.asarray(
        [
            [0, 0], [0.5, 0], [0, 0.5], [0.5, 0.5],
            [2.4, 0.5], [2.9, 0.5], [2.4, 1], [2.9, 1],
        ],
        dtype=float,
    )
    model = IncrementalDBSCAN(eps=1.0, min_pts=4).fit(data)
    groups_before = _core_groups(model)
    assert len(groups_before) == 2

    new_id = model.insert(np.asarray([1.45, 0.45]))

    assert model.points[new_id].point_type == PointType.BORDER
    assert _core_groups(model) == groups_before
    _assert_matches_refit(model)


def test_insert_promotes_border_and_noise_neighbors_to_core():
    data = np.asarray(
        [
            [0, 0], [0.5, 0], [0, 0.5], [0.5, 0.5],
            [1.25, 0], [1.25, 1.4], [1.6, 1.3], [1.9, 1.0],
        ],
        dtype=float,
    )
    model = IncrementalDBSCAN(eps=1.0, min_pts=4).fit(data)
    assert model.points[4].point_type == PointType.BORDER
    assert all(model.points[i].point_type == PointType.NOISE for i in (5, 6, 7))

    new_id = model.insert(np.asarray([1.25, 0.5]))

    expected_core_ids = {4, 5, 6, 7, new_id}
    actual_core_ids = {
        point_id
        for point_id, point in model.points.items()
        if point.point_type == PointType.CORE
    }
    assert expected_core_ids <= actual_core_ids
    assert len(_core_groups(model)) == 1
    _assert_matches_refit(model)


def test_insert_bridge_merges_three_clusters():
    data = np.asarray(
        [
            [0.95, -0.1], [0.95, 0.1], [1.15, -0.1], [1.15, 0.1],
            [-0.95, -0.1], [-0.95, 0.1], [-1.15, -0.1], [-1.15, 0.1],
            [-0.1, 0.95], [0.1, 0.95], [-0.1, 1.15], [0.1, 1.15],
        ],
        dtype=float,
    )
    model = IncrementalDBSCAN(eps=1.0, min_pts=4).fit(data)
    assert len(_core_groups(model)) == 3

    new_id = model.insert(np.asarray([0, 0], dtype=float))

    assert model.points[new_id].point_type == PointType.CORE
    assert len(_core_groups(model)) == 1
    _assert_matches_refit(model)


def test_insert_updates_repeatedly_and_matches_refit():
    initial = np.asarray(
        [
            [-1, 0], [0, 0], [1, 0], [-1, 1], [0, 1], [1, 1],
            [2.8, 0], [3.8, 0], [4.8, 0], [2.8, 1], [3.8, 1], [4.8, 1],
        ],
        dtype=float,
    )
    model = IncrementalDBSCAN(eps=1.1, min_pts=4).fit(initial)
    updates = [
        ([10, 10], PointType.NOISE, 2, 1),
        ([1.9, 0.5], PointType.CORE, 1, 1),
        ([10, 10.5], PointType.NOISE, 1, 2),
        ([10, 11], PointType.NOISE, 1, 3),
        ([10.5, 10.5], PointType.CORE, 2, 0),
    ]

    for coords, expected_type, expected_clusters, expected_noise in updates:
        new_id = model.insert(np.asarray(coords, dtype=float))

        assert new_id == max(model.points)
        assert model.points[new_id].point_type == expected_type
        assert len(_core_groups(model)) == expected_clusters
        assert sum(p.point_type == PointType.NOISE for p in model.points.values()) == expected_noise
        _assert_matches_refit(model)


@pytest.mark.parametrize("dimension", [2, 4], ids=["2d", "4d"])
def test_insert_works_in_multiple_dimensions(dimension):
    rng = np.random.default_rng(20261019 + dimension)
    data = rng.normal(size=(20, dimension))
    model = IncrementalDBSCAN(eps=1.5, min_pts=4).fit(data)

    for coords in rng.normal(size=(5, dimension)):
        new_id = model.insert(coords)
        assert len(model.points[new_id].coords) == dimension
        _assert_matches_refit(model)
