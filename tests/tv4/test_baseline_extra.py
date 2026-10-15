"""Baseline: adapter DataPoint, ánh xạ id stream-engine, đo thời gian lặp, và các điểm yếu mà review phát hiện."""
import shutil
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import baseline.sklearn_dbscan as bl
from baseline.sklearn_dbscan import IdMap, PointType, RefitDBSCAN, as_coords, as_matrix, compare_partitions, run_stream
from .tv4_paths import RAW

ATM = RAW / "bank_geo" / "atm_stream"
CARD = RAW / "bank_card" / "card_stream"


# ---------------------------------------------------------------- partition_equal và compare_partitions
def test_partition_equal_requires_same_noise_set():
    """ARI = 1 nhưng điểm nhiễu của bên này là một cụm của bên kia: KHÔNG được coi là cùng phân hoạch."""
    a = [0, 0, 1, 1, -1, -1]
    b = [0, 0, 1, 1, 2, 2]
    r = compare_partitions(a, b)
    assert r["ari"] == 1.0                                    # nhiễu tính như một lớp nên ARI không thấy khác biệt
    assert not r["noise_equal"] and r["noise_only_a"] == [4, 5] and r["noise_only_b"] == []
    assert not r["partition_equal"]


# ---------------------------------------------------------------- adapter
def test_adapter_accepts_coordinates_or_datapoint_like():
    dp = SimpleNamespace(id=77, coords=np.array([0.5, 0.5]))
    assert as_coords(dp).tolist() == [0.5, 0.5] and as_coords([1, 2]).tolist() == [1.0, 2.0]
    assert as_matrix([SimpleNamespace(coords=[0, 0]), SimpleNamespace(coords=[1, 1])]).shape == (2, 2)
    base = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [9, 9.0]])
    a = RefitDBSCAN(1.5, 4).fit(base)
    b = RefitDBSCAN(1.5, 4).fit([SimpleNamespace(coords=r) for r in base])
    assert a.get_labels() == b.get_labels()
    pa, pb = a.insert([0.5, 0.5]), b.insert(dp)
    assert pa == pb == 5                                      # engine cấp id, bỏ qua dp.id = 77
    assert a.get_labels() == b.get_labels() and a.snapshot()["coords"].tobytes() == b.snapshot()["coords"].tobytes()
    with pytest.raises(ValueError):
        b.insert(SimpleNamespace(coords=[1, 2, 3]))           # sai số chiều


def test_idmap_is_explicit_two_way_and_rejects_double_binding():
    m = IdMap()
    m.bind(500, 0)
    m.bind(501, 1)
    assert m.engine_id(500) == 0 and m.stream_id(1) == 501 and len(m) == 2
    with pytest.raises(KeyError):
        m.bind(500, 9)                                        # id stream đã gắn
    with pytest.raises(KeyError):
        m.bind(600, 1)                                        # id engine đã gắn
    assert m.release(500) == 0 and len(m) == 1
    with pytest.raises(KeyError):
        m.engine_id(500)
    with pytest.raises(KeyError):
        m.stream_id(0)


# ---------------------------------------------------------------- id stream khác id engine
def _offset_stream(src, dst, offset):
    dst.mkdir(parents=True)
    d0 = pd.read_csv(src / "d0.csv", dtype={"district_id": str})
    upd = pd.read_csv(src / "updates.csv", dtype={"ambiguous_border_ids": str, "district_id": str})
    d0["point_id"] += offset
    upd["point_id"] += offset
    d0.to_csv(dst / "d0.csv", index=False)
    upd.to_csv(dst / "updates.csv", index=False)
    shutil.copy(src / "params.json", dst / "params.json")
    return dst


class ShiftedIdEngine:
    """Engine bọc RefitDBSCAN nhưng cấp id khác hẳn (cộng 100000), để bắt lỗi trộn id stream với id engine."""
    SHIFT = 100000

    def __init__(self, eps, min_pts):
        self.inner = RefitDBSCAN(eps, min_pts)

    def fit(self, data):
        self.inner.fit(data)
        return self

    def insert(self, coords):
        return self.inner.insert(coords) + self.SHIFT

    def delete(self, point_id):
        self.inner.delete(point_id - self.SHIFT)

    def snapshot(self):
        s = self.inner.snapshot()
        s["ids"] = s["ids"] + self.SHIFT
        return s


@pytest.mark.parametrize("stream", [ATM, CARD], ids=["atm", "card"])
def test_labels_use_stream_ids_when_engine_ids_differ(stream, tmp_path):
    ref_t, ref_l = run_stream(RefitDBSCAN, stream, tmp_path / "ref", max_steps=40, repeats=1)
    # (1) id stream bị dời 7000 (không còn trùng id engine 0..n-1)
    moved = _offset_stream(stream, tmp_path / "moved" / stream.name, 7000)
    t1, l1 = run_stream(RefitDBSCAN, moved, tmp_path / "out1", max_steps=40, repeats=1)
    assert (l1["point_id"] - 7000).tolist() == ref_l["point_id"].tolist() and l1["label"].tolist() == ref_l["label"].tolist()
    assert l1["point_id"].min() >= 7000
    # (2) engine cấp id khác: nhãn xuất ra vẫn theo id stream
    t2, l2 = run_stream(ShiftedIdEngine, stream, tmp_path / "out2", max_steps=40, repeats=1)
    pd.testing.assert_frame_equal(l2, ref_l)
    assert l2["point_id"].max() < 100000
    keep = [c for c in bl.TIMING_COLUMNS if c != "update_seconds"]
    pd.testing.assert_frame_equal(t1[keep], ref_t[keep])
    pd.testing.assert_frame_equal(t2[keep], ref_t[keep])


# ---------------------------------------------------------------- đo thời gian
def test_timing_is_recorded_positive_and_median_of_repeats(tmp_path):
    timing, _ = run_stream(RefitDBSCAN, ATM, tmp_path, max_steps=8, repeats=5)
    assert (timing["update_seconds"] > 0).all()               # không ghi 0 (và không âm)
    assert (timing["n_repeats"] == 5).all() and (timing["n"] == timing["n_points"]).all()
    saved = pd.read_csv(tmp_path / "atm_stream" / "timing.csv")
    assert list(saved.columns) == bl.TIMING_COLUMNS and (saved["update_seconds"] > 0).all()
    t1, l1 = run_stream(RefitDBSCAN, ATM, tmp_path / "r1", max_steps=8, repeats=1)
    _, l5 = run_stream(RefitDBSCAN, ATM, tmp_path / "r5", max_steps=8, repeats=5)
    pd.testing.assert_frame_equal(l1, l5)                      # phần xác định giống hệt, bất kể số lần đo
    keep = ["step", "op", "n_points", "n_clusters", "n_noise"]
    pd.testing.assert_frame_equal(t1[keep], timing[keep])


class _Logger:
    def __init__(self):
        self.log = []


def test_timed_applies_for_real_once_and_reports_median_of_clone_runs(monkeypatch):
    ticks = iter([0.0, 3.0, 10.0, 11.0, 20.0, 22.0])           # ba lần đo: 3 s, 1 s, 2 s -> trung vị 2 s
    monkeypatch.setattr(bl.time, "perf_counter", lambda: next(ticks))
    eng = _Logger()
    out, secs, reps = bl._timed(eng, lambda e: e.log.append("x") or "kq", 3)
    assert (out, secs, reps) == ("kq", 2.0, 3)
    assert eng.log == ["x"]                                    # engine thật chỉ chịu một lần áp dụng; 3 lần đo nằm trên bản sao
    monkeypatch.undo()
    eng2 = _Logger()
    out, secs, reps = bl._timed(eng2, lambda e: e.log.append("y") or "kq2", 1)
    assert reps == 1 and eng2.log == ["y"] and secs >= 0


def test_benchmark_entry_point_small(tmp_path, monkeypatch):
    """run_benchmark với n = 500 (tự sinh data/raw/bench/ bằng bộ sinh dữ liệu nếu chưa có), ghi bench_timing.csv và README.txt."""
    res = bl.run_benchmark([500], out_dir=tmp_path, repeats=2)
    assert set(res["op"]) == {"fit", "insert", "delete"} and set(res["family"]) == {"atm", "card"}
    assert (res["median_ms"] > 0).all() and (tmp_path / "bench_timing.csv").exists() and (tmp_path / "README.txt").exists()
    assert "Phụ thuộc máy" in (tmp_path / "README.txt").read_text(encoding="utf-8")
