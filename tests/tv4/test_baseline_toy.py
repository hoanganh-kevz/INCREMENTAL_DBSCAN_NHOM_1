"""RefitDBSCAN trên 8 kịch bản toy và 19 kịch bản toy_edge: áp dụng update.csv theo thứ tự,
so phân hoạch với expected.json (bỏ các điểm biên mơ hồ), so thêm tập lõi."""
import json

import numpy as np
import pandas as pd
import pytest

from baseline.sklearn_dbscan import PointType, RefitDBSCAN, compare_partitions
from .tv4_paths import RAW

TOY = sorted(p.name for p in (RAW / "toy").iterdir() if p.is_dir())
EDGE = sorted(p.name for p in (RAW / "toy_edge").iterdir() if p.is_dir())


def _state(engine, id_to_name):
    snap = engine.snapshot()
    names = [id_to_name[int(i)] for i in snap["ids"]]
    labels = dict(zip(names, snap["labels"].tolist()))
    cores = {n for n, t in zip(names, snap["point_types"]) if t == PointType.CORE}
    return labels, cores


def _check(engine, id_to_name, exp_labels, exp_cores, ignore, where):
    got, cores = _state(engine, id_to_name)
    cmp = compare_partitions(got, exp_labels, ignore_ids=ignore)
    assert cmp["partition_equal"], f"{where}: {cmp}"
    # lõi: so trên mọi điểm (điểm biên mơ hồ không bao giờ là lõi)
    assert cores == set(exp_cores), f"{where}: lõi {sorted(cores ^ set(exp_cores))}"


def _run(kind, name):
    base = RAW / kind / name
    exp = json.loads((base / "expected.json").read_text(encoding="utf-8"))
    pts = pd.read_csv(base / "points.csv", dtype={"point_id": str})
    upd = pd.read_csv(base / "update.csv", dtype={"point_id": str})
    cols = [c for c in pts.columns if c != "point_id"]
    eps, minpts = exp["eps"], exp["minpts"]
    eng = RefitDBSCAN(eps, minpts).fit(pts[cols].to_numpy(dtype=float))
    id_to_name = dict(enumerate(pts["point_id"]))
    name_to_id = {v: k for k, v in id_to_name.items()}
    global_amb = set(exp.get("ambiguous_to_ignore", []))
    _check(eng, id_to_name, exp["labels_before"], exp["core_before"], global_amb, f"{name} trước update")
    steps = exp.get("steps")
    for k, row in enumerate(upd.itertuples(index=False)):
        if row.op == "insert":
            new_id = eng.insert([getattr(row, c) for c in cols])
            id_to_name[new_id] = row.point_id
            name_to_id[row.point_id] = new_id
        else:
            eng.delete(name_to_id.pop(row.point_id))
        if steps:   # toy_edge: kiểm sau từng update
            s = steps[k]
            assert s["op"] == row.op and s["point_id"] == row.point_id
            _check(eng, id_to_name, s["labels_after"], s["core_after"],
                   set(s.get("ambiguous_border_ids", [])) | global_amb, f"{name} step {s['step']}")
    _check(eng, id_to_name, exp["labels_after"], exp["core_after"], global_amb, f"{name} sau update")


def test_counts():
    assert len(TOY) == 8 and len(EDGE) == 19


@pytest.mark.parametrize("name", TOY)
def test_toy(name):
    _run("toy", name)


@pytest.mark.parametrize("name", EDGE)
def test_toy_edge(name):
    _run("toy_edge", name)


# ---------------------------------------------------------------- RefitDBSCAN bề mặt công khai
def test_refit_surface_and_aliases():
    eng = RefitDBSCAN(1.5, 4).fit_initial(np.array([[0, 0], [1, 0], [0, 1], [1, 1], [9, 9.0]]))
    assert eng.get_labels() == [0, 0, 0, 0, -1]
    pid = eng.insert_point([0.5, 0.5])
    assert pid == 5 and eng.n_points == 6 and eng.last_update_seconds > 0
    eng.delete_point(4)
    snap = eng.snapshot()
    assert set(snap) == {"ids", "coords", "labels", "point_types"}
    assert list(snap["ids"]) == [0, 1, 2, 3, 5]
    assert all(t == PointType.CORE for t in snap["point_types"])
    with pytest.raises(KeyError):
        eng.delete(4)                       # id đã xóa không còn
    assert eng.insert([20, 20]) == 6        # id không dùng lại


def test_refit_empty_after_deleting_everything():
    eng = RefitDBSCAN(1.0, 2).fit(np.array([[0.0, 0.0], [0.5, 0.0]]))
    eng.delete(0)
    eng.delete(1)
    assert eng.n_points == 0 and eng.get_labels() == []
    assert eng.insert([1.0, 1.0]) == 2 and eng.get_labels() == [-1]


def test_compare_partitions():
    a = [0, 0, 1, 1, -1, 2]
    b = [5, 5, 3, 3, -1, 0]                      # cùng cách chia, id cụm khác
    r = compare_partitions(a, b)
    assert r["ari"] == 1.0 and r["noise_equal"] and r["partition_equal"]
    c = [0, 0, 1, 1, 1, 2]                       # điểm nhiễu thành cụm
    r = compare_partitions(a, c)
    assert r["ari"] < 1.0 and not r["noise_equal"] and r["noise_only_a"] == [4]
    # điểm 5 khác nhãn nhưng nằm trong ignore_ids thì bị bỏ
    d = [0, 0, 1, 1, -1, 0]
    assert not compare_partitions(a, d)["partition_equal"]
    r = compare_partitions(a, d, ignore_ids=[5])
    assert r["partition_equal"] and r["n_ignored"] == 1 and r["n_compared"] == 5
    # dict nhãn và so tập lõi
    ta = [3, 3, 2, 3, 1, 3]
    tb = [3, 3, 3, 3, 1, 3]
    assert compare_partitions({"u": 0, "v": 1}, {"u": 7, "v": 9})["partition_equal"]
    assert compare_partitions(a, b, types_a=ta, types_b=tb)["core_equal"] is False
    assert compare_partitions(a, b, types_a=ta, types_b=ta)["core_equal"] is True
    assert compare_partitions(a, b)["core_equal"] is None
    with pytest.raises(ValueError):
        compare_partitions([0, 1], [0])
