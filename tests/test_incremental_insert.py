"""Test Incremental DBSCAN"""
import numpy as np
import pytest
from src.clustering.incremental_dbscan import IncrementalDBSCAN
from src.clustering.dbscan import DBSCAN
from src.core.data_point import PointType

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

def test_insert_multiple_updates() -> None:
    """Kiểm tra liên tiếp noise, merge, creation trên cùng mô hình"""

    initial = np.asarray([
        [-1, 0], [0, 0], [1, 0],
        [-1, 1], [0, 1], [1, 1],
        [2.8, 0], [3.8, 0], [4.8, 0],
        [2.8, 1], [3.8, 1], [4.8, 1]], dtype=float)
    model = IncrementalDBSCAN(eps = 1.1, min_pts =4)
    model.fit(initial)
    cases = [
        ([10, 10], PointType.NOISE, 2, 1),
        ([1.9, 0.5], PointType.CORE, 1, 1),
        ([10, 10.5], PointType.NOISE, 1, 2),
        ([10, 11], PointType.NOISE, 1, 3),
        ([10.5, 10.5], PointType.CORE, 2, 0)]
    for coords, expected_type, expected_clusters, expected_noise in cases:
        #Chèn điểm mới
        expected_id = len(model.points)
        point_id = model.insert(np.asarray(coords, dtype=float))
        assert point_id == expected_id
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
