"""C12 - Kiểm thử các trường hợp biên của Incremental DBSCAN"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from baseline.sklearn_dbscan import compare_partitions
from src.clustering.dbscan import DBSCAN
from src.clustering.incremental_dbscan import IncrementalDBSCAN
from src.core.data_point import PointType

DATA_DIR = Path(__file__).resolve().parents[1] / "data/raw/toy_edge"

@pytest.mark.parametrize("name, expected_type, expected_neighbors",
                         [("e01_insert_at_eps_inclusive", PointType.BORDER, {1,4}),
                          ("e02_insert_just_outside_eps", PointType.NOISE, {4})])

def test_eps_boundary(name, expected_type, expected_neighbors):
    """Kiểm tra biên Eps, UpdSeedIns và kết quả full re-fit"""

    folder = DATA_DIR / name
    data = pd.read_csv(folder / "points.csv", encoding="utf-8")
    update = pd.read_csv(folder / "update.csv", encoding="utf-8").iloc[0]
    expected = json.loads((folder / "expected.json").read_text(encoding="utf-8"))
    coords = data[["x", "y"]].to_numpy(dtype=float)
    new_coords = update[["x", "y"]].to_numpy(dtype=float)
    #Thực hiện insert và dbscan full re-fit
    model = IncrementalDBSCAN(eps = expected["eps"], min_pts = expected["minpts"]).fit(coords)
    new_id = model.insert(new_coords)
    names = data["point_id"].tolist() + [update["point_id"]]
    #Kiểm tra trạng thái sau insert
    assert new_id == len(coords)
    assert model.points[new_id].point_type == expected_type
    assert model.points[new_id].neighbor_ids == expected_neighbors
    assert model._get_update_seeds(model._find_new_core_points(new_id)) == set()
    assert {names[i] for i, p in model.points.items() if p.point_type == PointType.CORE} == set(expected["core_after"])
    #Đối chiếu với expected output và full re-fit
    assert compare_partitions(dict(zip(names, model.get_labels())), expected["labels_after"])["partition_equal"]
    _check_refit(model)


def _check_refit(model):
    """Đối chiếu với DBSCAN full re-fit."""
    
    ids = list(model.points)
    coords = np.asarray([model.points[i].coords for i in ids])
    refit = DBSCAN(eps=model.eps, min_pts=model.min_pts).fit(coords)
    assert compare_partitions(model.get_labels(), refit.get_labels())["partition_equal"]
    for j, i in enumerate(ids):
        assert model.points[i].point_type == refit.points[j].point_type
        assert model.points[i].neighbor_ids == {ids[k] for k in refit.points[j].neighbor_ids}
        assert model.points[i].neighbor_count == refit.points[j].neighbor_count

def test_border_core_roundtrip_r02():
    """INSERT: E, F chuyển Border -> Core. DELETE: Core -> Border"""
    
    coords = np.asarray([[0, 0], [1, 0], [0, 1], [1, 1], [2, 0], [2, 1]], dtype=float)
    model = IncrementalDBSCAN(eps=1.1, min_pts=4).fit(coords)
    assert all(model.points[i].point_type == PointType.BORDER and model.points[i].neighbor_count == 3 for i in (4, 5))

    #INSERT: E, F chuyển thành Core
    new_id = model.insert(np.asarray([2.2, 0.5]))
    assert all(model.points[i].point_type == PointType.CORE and model.points[i].neighbor_count == 4 for i in (4, 5))
    assert model._find_new_core_points(new_id) == {4, 5}
    assert model._get_update_seeds({4, 5}) == {1, 3, 4, 5}
    assert model.points[new_id].point_type == PointType.BORDER
    _check_refit(model)
    #DELETE: xác định LostCore và UpdSeedDel
    old_core = {i for i, p in model.points.items() if p.point_type == PointType.CORE}
    old_neighbors = {i: p.neighbor_ids.copy() for i, p in model.points.items()}
    model.delete_point(new_id)
    lost_core = old_core - {i for i, p in model.points.items() if p.point_type == PointType.CORE}
    updseed_del = {j for i in lost_core for j in old_neighbors[i] if j in model.points and model.points[j].point_type == PointType.CORE}
    assert lost_core == {4, 5} and updseed_del == {1, 3}
    assert new_id not in model.points
    assert all(model.points[i].point_type == PointType.BORDER and model.points[i].neighbor_count == 3 for i in (4, 5))
    _check_refit(model)

def test_border_core_cascade_e05():
    """Kiểm tra Border/Noise -> Core và khôi phục trạng thái sau DELETE"""
    
    folder = DATA_DIR / "e05_insert_border_to_core_cascade"
    data = pd.read_csv(folder / "points.csv")
    update = pd.read_csv(folder / "update.csv").iloc[0]
    expected = json.loads((folder / "expected.json").read_text(encoding="utf-8"))
    coords = data[["x", "y"]].to_numpy(dtype=float)
    names = data["point_id"].tolist()
    model = IncrementalDBSCAN(eps=expected["eps"], min_pts=expected["minpts"]).fit(coords)
    assert {names[i] for i, p in model.points.items() if p.point_type == PointType.CORE} == set(expected["core_before"])
    #INSERT: xác định Core mới, UpdSeedIns và kết quả gom cụm
    new_id = model.insert(update[["x", "y"]].to_numpy(dtype=float))
    names_after = names + [update["point_id"]]
    new_core = model._find_new_core_points(new_id)
    assert {names_after[i] for i in new_core} == {"b1", "n1", "n2", "n3", "p"}
    assert {names_after[i] for i in model._get_update_seeds(new_core)} == set(expected["updseed"])
    assert {names_after[i] for i, p in model.points.items() if p.point_type == PointType.CORE} == set(expected["core_after"])
    assert compare_partitions(dict(zip(names_after, model.get_labels())), expected["labels_after"])["partition_equal"]
    _check_refit(model)
    #DELETE: kiểm tra kết quả trở về trạng thái ban đầu
    model.delete_point(new_id)
    assert new_id not in model.points
    assert {names[i] for i, p in model.points.items() if p.point_type == PointType.CORE} == set(expected["core_before"])
    assert compare_partitions(dict(zip(names, model.get_labels())), expected["labels_before"])["partition_equal"]
    _check_refit(model)


def test_insert_bridge_three_clusters():
    """Kiểm tra INSERT bridge hợp nhất 3 cụm và UpdSeedIns"""
    
    folder = DATA_DIR / "e06_insert_bridges_three_clusters"
    data = pd.read_csv(folder / "points.csv")
    update = pd.read_csv(folder / "update.csv").iloc[0]
    expected = json.loads((folder / "expected.json").read_text(encoding="utf-8"))
    coords = data[["x", "y"]].to_numpy(dtype=float)
    names = data["point_id"].tolist()
    model = IncrementalDBSCAN(eps=expected["eps"], min_pts=expected["minpts"]).fit(coords)
    #Kiểm tra 3 cụm trước INSERT
    assert compare_partitions(dict(zip(names, model.get_labels())), expected["labels_before"])["partition_equal"]
    assert len({p.cluster_id for p in model.points.values() if p.cluster_id >= 0}) == 3
    #INSERT điểm p và xác định NewCore, UpdSeedIns
    new_id = model.insert(update[["x", "y"]].to_numpy(dtype=float))
    names_after = names + [update["point_id"]]
    new_core = model._find_new_core_points(new_id)
    assert new_id == len(coords)
    assert model.points[new_id].point_type == PointType.CORE
    assert model.points[new_id].neighbor_count == 7
    assert new_core == {new_id}
    assert {names_after[i] for i in model._get_update_seeds(new_core)} == set(expected["updseed"])
    #Kiểm tra kết quả Merge và đối chiếu full re-fit
    assert {names_after[i] for i, p in model.points.items() if p.point_type == PointType.CORE} == set(expected["core_after"])
    assert compare_partitions(dict(zip(names_after, model.get_labels())), expected["labels_after"])["partition_equal"]
    assert len({p.cluster_id for p in model.points.values() if p.cluster_id >= 0}) == 1
    _check_refit(model)


def test_delete_critical_core_r03():
    """Potential Split: xóa p làm một cụm tách thành hai, I thành Noise"""
    
    names = list("ABCDEFGHI") + ["p"]
    coords = np.asarray([[1, 0], [.2, 0], [.6, .6], [.6, -.6], [3, 0], [3.8, 0], [3.4, .6], [3.4, -.6], [2, 1], [2, 0]], dtype=float)
    model = IncrementalDBSCAN(eps=1.1, min_pts=4).fit(coords)
    assert len(set(model.get_labels()) - {-1}) == 1
    assert model.points[9].point_type == PointType.CORE
    old_neighbors = model.points[9].neighbor_ids.copy()
    #DELETE: kiểm tra UpdSeedDel và kết quả Split
    model.delete_point(9)
    updseed = {names[i] for i in old_neighbors - {9} if model.points[i].point_type == PointType.CORE}
    expected_labels = {**{n: 0 for n in "ABCD"}, **{n: 1 for n in "EFGH"}, "I": -1}
    assert updseed == {"A", "E"}
    assert 9 not in model.points and model.points[8].point_type == PointType.NOISE
    assert model.points[0].neighbor_count == model.points[4].neighbor_count == 4
    assert compare_partitions(dict(zip(names[:-1], model.get_labels())), expected_labels)["partition_equal"]
    assert len(set(model.get_labels()) - {-1}) == 2
    _check_refit(model)

def test_delete_critical_core_e13():
    """Potential Split: xóa Core nối ba cụm và đối chiếu expected.json"""
    
    folder = DATA_DIR / "e13_delete_splits_into_three"
    data = pd.read_csv(folder / "points.csv", encoding="utf-8")
    update = pd.read_csv(folder / "update.csv", encoding="utf-8").iloc[0]
    expected = json.loads((folder / "expected.json").read_text(encoding="utf-8"))
    names = data["point_id"].tolist()
    model = IncrementalDBSCAN(eps=expected["eps"], min_pts=expected["minpts"]).fit(data[["x", "y"]].to_numpy(dtype=float))
    assert compare_partitions(dict(zip(names, model.get_labels())), expected["labels_before"])["partition_equal"]
    delete_id = names.index(update["point_id"])
    assert model.points[delete_id].point_type == PointType.CORE
    old_neighbors = model.points[delete_id].neighbor_ids.copy()
    #DELETE: kiểm tra UpdSeedDel, Core, kết quả Split
    model.delete_point(delete_id)
    updseed = {names[i] for i in old_neighbors - {delete_id} if model.points[i].point_type == PointType.CORE}
    assert updseed == set(expected["updseed"])
    assert delete_id not in model.points
    assert {names[i] for i, p in model.points.items() if p.point_type == PointType.CORE} == set(expected["core_after"])
    assert compare_partitions({names[i]: p.cluster_id for i, p in model.points.items()}, expected["labels_after"])["partition_equal"]
    assert len(set(model.get_labels()) - {-1}) == 3
    _check_refit(model)


def test_delete_nonsplit_long_detour():
    """Potential Split: xóa Core nhưng vẫn giữ một cụm nhờ kết nối thay thế"""
    
    folder = DATA_DIR / "e14_delete_nonsplit_long_detour"
    data = pd.read_csv(folder / "points.csv", encoding="utf-8")
    update = pd.read_csv(folder / "update.csv", encoding="utf-8").iloc[0]
    expected = json.loads((folder / "expected.json").read_text(encoding="utf-8"))
    names = data["point_id"].tolist()
    model = IncrementalDBSCAN(eps=expected["eps"], min_pts=expected["minpts"]).fit(data[["x", "y"]].to_numpy(dtype=float))
    #Kiểm tra một cụm trước DELETE
    assert compare_partitions(dict(zip(names, model.get_labels())), expected["labels_before"])["partition_equal"]
    delete_id = names.index(update["point_id"])
    old_neighbors = model.points[delete_id].neighbor_ids.copy()
    assert model.points[delete_id].point_type == PointType.CORE
    #DELETE: xác định UpdSeedDel và kiểm tra Non-split
    model.delete_point(delete_id)
    updseed = {names[i] for i in old_neighbors - {delete_id} if model.points[i].point_type == PointType.CORE}
    assert updseed == set(expected["updseed"])
    assert delete_id not in model.points
    assert {names[i] for i, p in model.points.items() if p.point_type == PointType.CORE} == set(expected["core_after"])
    assert compare_partitions({names[i]: p.cluster_id for i, p in model.points.items()}, expected["labels_after"])["partition_equal"]
    assert len({p.cluster_id for p in model.points.values() if p.cluster_id >= 0}) == 1
    _check_refit(model)



