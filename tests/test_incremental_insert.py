"""Test Incremental DBSCAN"""
import numpy as np
import pytest
from src.clustering.incremental_dbscan import IncrementalDBSCAN
from src.clustering.dbscan import DBSCAN
from src.core.data_point import PointType

def _check_neighborhood_cache(model: IncrementalDBSCAN) -> None:
    """Kiểm tra tính chính xác của tập láng giền sau phép chèn"""

    for point_id, point in model.points.items():
        expected = {other_id 
                    for other_id, other in model.points.items()
        if model.metric.euclidean(point.coords, other.coords) <= model.eps}
        assert point.neighbor_ids == expected, (
            f"Tập láng giềng {point_id} không khớp")
        assert point.neighbor_count == len(expected)

@pytest.mark.parametrize(
    "name, data, new_coords, expected_core, expected_noise, "
    "expected_new_core, expected_seeds, expected_type",
    [
        ("Noise",
            [[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1]],
            [5, 5], {1, 4}, {6}, set(), set(), PointType.NOISE,),
        ("Creation",
            [[0, 0], [1, 0], [2, 0]],
            [1, 0.5], {1}, set(), {1}, {1}, PointType.BORDER,),
        ("Absorption",
            [[0, 0], [1, 0], [0, 1], [1, 1], [2, 0], [2, 1]],
            [2.2, 0.5], {1, 3, 4, 5}, set(), {4, 5}, {1, 3, 4, 5}, PointType.BORDER,),
        ("Merge",[
                [-1, 0], [0, 0], [1, 0],
                [-1, 1], [0, 1], [1, 1],
                [2.8, 0], [3.8, 0], [4.8, 0],
                [2.8, 1], [3.8, 1], [4.8, 1]],
            [1.9, 0.5],{1, 2, 4, 5, 6, 7, 9, 10, 12},
            set(),{2, 5, 6, 9, 12},{1, 2, 4, 5, 6, 7, 9, 10, 12},
            PointType.CORE,),],
    ids = ["noise", "creation", "absorption", "merge"],)

def test_insert(
    name, data, new_coords, expected_core, expected_noise, expected_new_core, expected_seeds, expected_type):

    """Kiểm tra core, noise, UpdSeed và nhãn sau INSERT"""

    model = IncrementalDBSCAN(eps = 1.1, min_pts = 4)
    model.fit(np.asarray(data, dtype = float))
    #Thực hiện INSERT
    new_id = model.insert(np.asarray(new_coords, dtype = float))
    #Kiểm tra id và trạng thái của điểm mới
    assert new_id == len(data), f"{name}: ID mới không đúng"
    assert len(model.points) == len(data) + 1
    assert model.points[new_id].point_type == expected_type
    #xác định các core và noise sau cập nhật
    actual_core = {point_id
                   for point_id, point in model.points.items()
                   if point.point_type == PointType.CORE}
    actual_noise = {point_id
                    for point_id, point in model.points.items()
                    if point.point_type == PointType.NOISE}
    assert actual_core == expected_core
    assert actual_noise == expected_noise
    #cả 4 trường hợp đều có 1 cụm sau insert
    cluster_ids = {point.cluster_id
                   for point in model.points.values()
                   if point.cluster_id is not None and point.cluster_id >= 0}
    assert len(cluster_ids) == 1
    #Các điễm ko phải noise thuộc cụm
    assert all(point.cluster_id in cluster_ids
               for point_id, point in model.points.items()
               if point_id not in expected_noise)
    #Kiểm tra tập core mới và UpdSeedIns
    new_core_ids = model._find_new_core_points(new_id)
    assert new_core_ids == expected_new_core
    assert model._get_update_seeds(new_core_ids) == expected_seeds
    #Kiểm tra cache sau cập nhật
    _check_neighborhood_cache(model)

def test_insert_border_no_merge():
    """Border nằm gần 2 cụm nhưng ko gây hợp nhất cụm"""

    data = np.asarray([
        [0, 0], [0.5, 0], [0, 0.5], [0.5, 0.5],
        [2.4, 0.5], [2.9, 0.5], [2.4, 1], [2.9, 1]], 
        dtype=float)
    model = IncrementalDBSCAN(eps=1.0, min_pts = 4)
    model.fit(data)
    new_id = model.insert(np.asarray([1.45, 0.45]))
    #Điểm mới là border
    assert model.points[new_id].point_type == PointType.BORDER
    #2 cụm ban đầu vẫn riêng biệt
    left_labels = {model.points[i].cluster_id for i in range(4)}
    right_labels = {model.points[i].cluster_id for i in range(4, 8)}
    assert len(left_labels) == len(right_labels) == 1
    assert left_labels != right_labels
    #border được gán vào 1 trong 2 cụm
    assert model.points[new_id].cluster_id in left_labels | right_labels
    #ko có core mới và uupdseedins rỗng
    new_core_ids = model._find_new_core_points(new_id)
    assert model._get_update_seeds(new_core_ids) == set()

    _check_neighborhood_cache(model)

def _get_core_groups(model: DBSCAN) -> set[frozenset[int]]:
    """Lấy các nhóm core theo thành viên, không phụ thuộc vào nhãn cụm"""

    groups: dict[int, set[int]] = {}
    for point_id, point in model.points.items():
        if point.point_type == PointType.CORE:
            assert point.cluster_id is not None and point.cluster_id >=0
            groups.setdefault(point.cluster_id, set()).add(point_id)        

    return {frozenset(ids) for ids in groups.values()}

def _check_against_refit(model: IncrementalDBSCAN, all_data: np.ndarray) -> None:
    """So sánh trạng thái sau INSERT với DBSCAN chạy lại toàn bộ"""

    baseline = DBSCAN(eps=model.eps, min_pts = model.min_pts)
    baseline.fit(np.asarray(all_data, dtype = float))
    assert model.points.keys() == baseline.points.keys()
    for point_id, point in model.points.items():
        ref = baseline.points[point_id]
        #kiểm tra tọa độ và tập lân cận
        assert np.array_equal(point.coords, ref.coords)
        assert point.neighbor_ids == ref.neighbor_ids
        assert point.neighbor_count == ref.neighbor_count
        #Kiểm tra tính chất core, border, noise
        assert point.point_type == ref.point_type
        if point.point_type == PointType.NOISE:
            assert point.cluster_id == -1
        elif point.point_type == PointType.BORDER:
            #Border phải thuộc cụm của ít nhất 1 core lân cận
            allowed_clusters = {
                model.points[neighbor_id].cluster_id
                for neighbor_id in point.neighbor_ids
                if model.points[neighbor_id].point_type == PointType.CORE}
            assert point.cluster_id in allowed_clusters
        else: 
            assert point.point_type == PointType.CORE
            assert point.cluster_id is not None and point.cluster_id >= 0
    #so sánh các nhóm core ko phụ thuộc ID cụm
    assert _get_core_groups(model) == _get_core_groups(baseline)

def test_insert_multiple_updates() -> None:
    """Kiểm tra liên tiếp noise, merge, creation trên cùng mô hình"""

    initial = np.asarray([
        [-1, 0], [0, 0], [1, 0],
        [-1, 1], [0, 1], [1, 1],
        [2.8, 0], [3.8, 0], [4.8, 0],
        [2.8, 1], [3.8, 1], [4.8, 1]], dtype=float)
    model = IncrementalDBSCAN(eps = 1.1, min_pts =4)
    model.fit(initial)
    all_data = initial.copy()
    cases = [
        ([10, 10], PointType.NOISE, 2, 1),
        ([1.9, 0.5], PointType.CORE, 1, 1),
        ([10, 10.5], PointType.NOISE, 1, 2),
        ([10, 11], PointType.NOISE, 1, 3),
        ([10.5, 10.5], PointType.CORE, 2, 0)]
    for coords, expected_type, expected_clusters, expected_noise in cases:
        #Chèn điểm mới 
        point_id = model.insert(np.asarray(coords, dtype = float))
        assert point_id == len(all_data)
        #So sánh với DBSCAN chạy lại toàn bộ
        all_data = np.vstack((all_data, coords))
        _check_against_refit(model, all_data)
        #Xác định số cụm và số noise
        cluster_ids = {
            point.cluster_id
            for point in model.points.values()
            if point.cluster_id is not None and point.cluster_id >=0}
        noise_count = sum(
            point.point_type == PointType.NOISE
            for point in model.points.values())
        assert model.points[point_id].point_type == expected_type
        assert len(cluster_ids) == expected_clusters
        assert noise_count == expected_noise

@pytest.mark.parametrize("dimension", [2,4], ids=["2d", "4d"])

def test_insert_refit_nd(dimension: int) -> None:
    """So sánh INSERT với DBSCAN chạy lại trong ko gian n chiều"""

    rng = np.random.default_rng(20261019 + dimension)
    data = rng.normal(size=(30, dimension))
    model = IncrementalDBSCAN(eps = 1.5, min_pts = 4)
    model.fit(data)
    #Kiểm tra kết quả phân cụm ban đầu 
    _check_against_refit(model, data)
    #Kiểm tra 20 lần INSERT liên tiếp
    for coords in rng.normal(size=(20, dimension)):
        point_id = model.insert(coords)
        assert point_id == len(data)
        data = np.vstack((data, coords))
        _check_against_refit(model, data)

def test_insert_at_eps() -> None:
    """Điểm cách core đúng bằng eps vẫn thuộc lân cận eps"""

    data = np.asarray([
        [0, 0], [0.5, 0],
        [0, 0.5], [0.5, 0.5]], dtype=float)
    model = IncrementalDBSCAN(eps = 1.0, min_pts = 4)
    model.fit(data)
    new_id = model.insert(np.asarray([1.5, 0], dtype = float))
    #Điểm mới nằm đúng Eps của một Core nên là Border
    assert model.points[new_id].point_type == PointType.BORDER
    assert 1 in model.points[new_id].neighbor_ids
    assert new_id in model.points[1].neighbor_ids
    _check_against_refit(model, np.vstack((data, [1.5, 0])))

def test_insert_duplicate_coords() -> None: 
    """Hai điểm trùng tọa độ phải có 2 id khác nhau"""

    data = np.asarray([
        [0, 0], [0.5, 0],
        [0, 0.5]], dtype=float)

    model = IncrementalDBSCAN(eps=1.0, min_pts = 4)
    model.fit(data)
    new_id = model.insert(np.asarray([0,0], dtype = float))
    #Hai điểm trùng tọa độ nhưng được quản lý bằng ID riêng
    assert new_id != 0
    assert 0 in model.points[new_id].neighbor_ids
    assert new_id in model.points[0].neighbor_ids
    #Điểm mới thòa mãn điều kiện core
    assert model.points[new_id].point_type == PointType.CORE

    _check_against_refit(model, np.vstack((data, [0,0])))

def test_insert_merge_three_clusters() -> None:
    """Một phép chèn có thể hợp nhất 3 cụm cũ"""

    data = np.asarray([
        [0.95, -0.1], [0.95, 0.1],
        [1.15, -0.1], [1.15, 0.1],
        [-0.95, -0.1], [-0.95, 0.1],
        [-1.15, -0.1], [-1.15, 0.1],
        [-0.1, 0.95], [0.1, 0.95],
        [-0.1, 1.15], [0.1, 1.15]], dtype=float)

    model = IncrementalDBSCAN(eps = 1.0, min_pts = 4)
    model.fit(data)
    #trước INSERT có ba cụm riêng biệt
    assert len(_get_core_groups(model)) ==3
    #Chèn điểm mới kết nối 3 cụm
    new_id = model.insert(np.asarray([0,0], dtype = float))
    assert model.points[new_id].point_type == PointType.CORE
    #Sau insert, ba cụm thành một
    assert len(_get_core_groups(model)) == 1
    assert all(point.cluster_id is not None and point.cluster_id >=0
               for point in model.points.values())

    _check_against_refit(model, np.vstack((data, [0,0])))


def test_insert_two_creations_shared_border() -> None:
    """Hai Core mới không kết nối không được hợp nhất qua một Border chung."""

    data = np.asarray([
        [-1.4], [-1.3], [-0.9],
        [0.9], [1.3], [1.4]], dtype=float)
    model = IncrementalDBSCAN(eps=1.0, min_pts=4)
    model.fit(data)
    # Ban đầu chưa có cụm
    assert len(_get_core_groups(model)) == 0
    # Chèn điểm tạo ra hai Core mới nhưng không kết nối 
    new_id = model.insert(np.asarray([0.0], dtype=float))
    assert model.points[new_id].point_type == PointType.BORDER
    assert len(_get_core_groups(model)) == 2
    _check_against_refit(model, np.vstack((data, [0.0])))


def test_insert_just_outside_eps_stays_noise() -> None:
    """Điểm ngoài Eps một khoảng nhỏ không được nhận vào cụm qua điểm biên."""

    data = np.asarray([[0, 0], [0.5, 0], [0, 0.5], [0.5, 0.5]], dtype=float)
    model = IncrementalDBSCAN(eps=1.0, min_pts=4).fit(data)

    new_id = model.insert(np.asarray([1.5625, 0.0]))

    assert model.points[new_id].point_type == PointType.NOISE
    assert model.points[new_id].cluster_id == -1
    _check_against_refit(model, np.vstack((data, [1.5625, 0.0])))


def test_insert_at_eps_promotes_existing_noise_to_core() -> None:
    """Khoảng cách đúng Eps được tính vào lân cận và có thể tạo Core mới."""

    data = np.asarray([[0, 0], [0.5, 0], [0, 0.5]], dtype=float)
    model = IncrementalDBSCAN(eps=1.0, min_pts=4).fit(data)
    assert _get_core_groups(model) == set()

    new_id = model.insert(np.asarray([-1.0, 0.0]))

    assert model.points[0].point_type == PointType.CORE
    assert model.points[new_id].point_type == PointType.BORDER
    assert _get_core_groups(model) == {frozenset({0})}
    _check_against_refit(model, np.vstack((data, [-1.0, 0.0])))


def test_insert_promotes_border_and_noise_neighbors_to_core() -> None:
    """Một lần chèn có thể làm nhiều điểm kề cùng đạt ngưỡng MinPts."""

    data = np.asarray([
        [0, 0], [0.5, 0], [0, 0.5], [0.5, 0.5],
        [1.25, 0], [1.25, 1.4], [1.6, 1.3], [1.9, 1.0],
    ], dtype=float)
    model = IncrementalDBSCAN(eps=1.0, min_pts=4).fit(data)
    assert model.points[4].point_type == PointType.BORDER
    assert all(model.points[i].point_type == PointType.NOISE for i in (5, 6, 7))

    new_id = model.insert(np.asarray([1.25, 0.5]))

    expected_new_cores = {4, 5, 6, 7, new_id}
    actual_cores = {
        point_id for point_id, point in model.points.items()
        if point.point_type == PointType.CORE
    }
    assert expected_new_cores <= actual_cores
    assert all(model.points[i].cluster_id == model.points[0].cluster_id
               for i in expected_new_cores)
    _check_against_refit(model, np.vstack((data, [1.25, 0.5])))


def test_insert_merge_absorbs_one_noise_and_leaves_another() -> None:
    """Merge cụm đồng thời chỉ hấp thụ noise nằm trong Eps của Core mới."""

    data = np.asarray([
        [0, 0], [0.5, 0], [0, 0.5], [0.5, 0.5],
        [2, 0], [2.5, 0], [2, 0.5], [2.5, 0.5],
        [1.25, 1.2], [1.25, -0.9],
    ], dtype=float)
    model = IncrementalDBSCAN(eps=1.0, min_pts=4).fit(data)
    left_cluster = model.points[0].cluster_id
    right_cluster = model.points[4].cluster_id
    assert left_cluster != right_cluster
    assert model.points[8].point_type == model.points[9].point_type == PointType.NOISE

    new_id = model.insert(np.asarray([1.25, 0.25]))

    assert model.points[new_id].point_type == PointType.CORE
    assert model.points[0].cluster_id == model.points[4].cluster_id
    assert model.points[8].point_type == PointType.BORDER
    assert model.points[9].point_type == PointType.NOISE
    _check_against_refit(model, np.vstack((data, [1.25, 0.25])))


def test_insert_creates_cluster_exactly_at_min_pts() -> None:
    """Cụm chỉ xuất hiện sau lần chèn làm vùng lân cận đạt đúng MinPts."""

    data = np.asarray([[0, 0], [5, 5]], dtype=float)
    model = IncrementalDBSCAN(eps=1.0, min_pts=4).fit(data)
    updates = [[0.5, 0], [0, 0.5], [0.5, 0.5]]

    for index, coords in enumerate(updates):
        new_id = model.insert(np.asarray(coords, dtype=float))
        data = np.vstack((data, coords))
        if index < 2:
            assert model.points[new_id].point_type == PointType.NOISE
            assert _get_core_groups(model) == set()
        else:
            assert model.points[new_id].point_type == PointType.CORE
            assert len(_get_core_groups(model)) == 1
            assert all(model.points[i].point_type == PointType.CORE for i in (0, 2, 3, 4))
        assert model.points[1].point_type == PointType.NOISE
        _check_against_refit(model, data)


def test_insert_near_border_but_outside_every_core_is_noise() -> None:
    """Điểm biên không truyền mật độ sang điểm mới nằm ngoài mọi Core."""

    data = np.asarray([
        [0, 0], [0.5, 0], [0, 0.5], [0.5, 0.5], [1.4, 0],
    ], dtype=float)
    model = IncrementalDBSCAN(eps=1.0, min_pts=4).fit(data)
    assert model.points[4].point_type == PointType.BORDER

    new_id = model.insert(np.asarray([2.2, 0.0]))

    assert model.points[new_id].point_type == PointType.NOISE
    assert len(_get_core_groups(model)) == 1
    _check_against_refit(model, np.vstack((data, [2.2, 0.0])))


def test_insert_merges_clusters_in_four_dimensions() -> None:
    """Tính khoảng cách dùng đủ bốn tọa độ khi một điểm nối hai cụm."""

    data = np.asarray([
        [0, 0, 0, 0], [0.3, 0, 0, 0], [0, 0.3, 0, 0],
        [0, 0, 0.3, 0], [0, 0, 0, 0.3],
        [1.7, 0, 0, 0], [1.4, 0, 0, 0], [1.7, 0.3, 0, 0],
        [1.7, 0, 0.3, 0], [1.7, 0, 0, 0.3],
        [0.5, 0.5, 3, 3], [0.55, 0.5, 3, 3], [0.5, 0.55, 3, 3],
    ], dtype=float)
    model = IncrementalDBSCAN(eps=1.0, min_pts=4).fit(data)
    assert len(_get_core_groups(model)) == 2

    new_id = model.insert(np.asarray([0.85, 0, 0, 0]))

    assert model.points[new_id].point_type == PointType.CORE
    assert len(_get_core_groups(model)) == 1
    assert all(model.points[i].point_type == PointType.NOISE for i in (10, 11, 12))
    _check_against_refit(model, np.vstack((data, [0.85, 0, 0, 0])))


def test_insert_min_pts_two_connects_core_chain() -> None:
    """Với MinPts=2, các điểm kề tạo thành chuỗi lõi không có điểm biên."""

    data = np.asarray([
        [0, 0], [0.9, 0], [1.8, 0], [3.6, 0], [4.5, 0], [5.4, 0], [0, 3],
    ], dtype=float)
    model = IncrementalDBSCAN(eps=1.0, min_pts=2).fit(data)
    assert len(_get_core_groups(model)) == 2
    assert model.points[6].point_type == PointType.NOISE

    for coords, expected_cluster_count in (([2.7, 0], 1), ([6.3, 0], 1)):
        new_id = model.insert(np.asarray(coords, dtype=float))
        data = np.vstack((data, coords))
        assert model.points[new_id].point_type == PointType.CORE
        assert len(_get_core_groups(model)) == expected_cluster_count
        assert all(point.point_type != PointType.BORDER for point in model.points.values())
        assert model.points[6].point_type == PointType.NOISE
        _check_against_refit(model, data)

