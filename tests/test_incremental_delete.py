import json
from pathlib import Path
from pyexpat import model
from turtle import left, right

import numpy as np
import pandas as pd

from baseline.sklearn_dbscan import compare_partitions
from src.clustering.incremental_dbscan import IncrementalDBSCAN
from src.core.data_point import PointType

DATA_DIR = Path(__file__).resolve().parent / "fixtures/toy_edge"


def _core_groups(model):
    groups = {}
    for point_id, point in model.points.items():
        if point.point_type == PointType.CORE:
            groups.setdefault(point.cluster_id, set()).add(point_id)
    return {frozenset(ids) for ids in groups.values()}

def test_delete_reduction_without_split():
    data = np.asarray([[0.0], [0.9], [1.8], [2.7]])
    model = IncrementalDBSCAN(eps=1.0, min_pts=3).fit(data)
    assert len(_core_groups(model)) == 1

    model.delete(0)
    assert 0 not in model.points
    assert len(_core_groups(model)) == 1
    assert all(p.cluster_id != -1 for p in model.points.values())

def test_delete_removal_no_core_remaining():
    data = np.asarray([[0, 0], [0.2, 0], [0, 0.2]], dtype=float)
    model = IncrementalDBSCAN(eps=0.5, min_pts=3).fit(data)
    assert len(_core_groups(model)) == 1

    model.delete_point(2)
    assert 2 not in model.points
    assert all(
    point.point_type == PointType.NOISE
    and point.cluster_id == -1
    for point in model.points.values()
)

def test_delete_core_splits_cluster_and_relabels_borders():
    data = np.asarray([[0.9 * i] for i in range(8)])
    model = IncrementalDBSCAN(eps=1.0, min_pts=3).fit(data)
    assert len(_core_groups(model)) == 1
    model.delete_point(3)
    assert 3 not in model.points
    assert len(_core_groups(model)) == 2
    left = {model.points[i].cluster_id for i in (0, 1, 2)}
    right = {model.points[i].cluster_id for i in (4, 5, 6, 7)}
    assert len(left) == len(right) == 1
    assert left != right
    #Các điểm ở hai đầu cụm trở thành border
    assert all(model.points[i].point_type == PointType.BORDER
    for i in (0, 2, 4, 7)
)

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

