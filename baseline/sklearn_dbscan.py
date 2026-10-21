"""Baseline DBSCAN bằng scikit-learn (D04). sklearn CHỈ được dùng ở thư mục baseline/, không bao giờ trong src/.

Gồm năm phần:
  (a) fit_full_dbscan: một lần DBSCAN đầy đủ, trả nhãn và loại điểm (CORE, BORDER, NOISE).
  (b) RefitDBSCAN: lớp THAY THẾ TẠM cho IncrementalDBSCAN của nhóm; mỗi insert/delete fit lại toàn bộ.
  (c) compare_partitions: so hai phân hoạch (ARI, tập nhiễu, tập lõi), bỏ điểm biên mơ hồ.
  (d) replay_stream / run_stream: chạy lại một stream (d0.csv, updates.csv, params.json), ghi thời gian và nhãn.
  (e) benchmark: python -m baseline.sklearn_dbscan --bench, thời gian re-fit theo n = 500, 1000, 2000, 4000.

Quy ước nhóm (đầy đủ ở docs_extra/CONVENTIONS.md): lân cận Eps tính cả chính điểm đó, min_pts của nhóm = min_samples của sklearn,
khoảng cách Euclid, biên Eps bao hàm (d <= eps). Tọa độ stream lấy từ src/preprocessing/stream_scaling.py (dùng chung với phần còn lại).
"""
import copy
import json
import time
from enum import IntEnum
from pathlib import Path
from collections.abc import Mapping

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score

from src.preprocessing.stream_scaling import feature_columns, stream_coordinates

REPEATS = 5     # số lần đo mỗi update (trên bản sao sâu của engine), lấy trung vị


class PointType(IntEnum):
    """Bản sao PointType của src/core/data_point.py để baseline không phụ thuộc src/core (so với enum của nhóm theo tên)."""
    UNCLASSIFIED = 0
    NOISE = 1
    BORDER = 2
    CORE = 3


# ------------------------------------------------------------------ (a) DBSCAN đầy đủ
def fit_full_dbscan(coords, eps, min_pts):
    """DBSCAN của sklearn (min_samples = min_pts, lân cận gồm cả chính điểm).

    Trả (labels, point_types): labels là mảng int, -1 là nhiễu, cụm đánh số từ 0;
    point_types là mảng int theo PointType (CORE từ core_sample_indices_, NOISE khi nhãn -1, còn lại BORDER).
    Mảng rỗng trả hai mảng rỗng.
    """
    coords = np.asarray(coords, dtype=float)
    if len(coords) == 0:
        return np.empty(0, dtype=int), np.empty(0, dtype=int)
    model = DBSCAN(eps=eps, min_samples=min_pts).fit(coords)
    labels = model.labels_.astype(int)
    types = np.where(labels == -1, int(PointType.NOISE), int(PointType.BORDER))
    types[model.core_sample_indices_] = int(PointType.CORE)
    return labels, types


# ------------------------------------------------------------------ adapter đầu vào
def as_coords(point):
    """Vector tọa độ (float, 1-D) từ MỘT điểm, dù đó là dãy số/ndarray hay một đối tượng giống DataPoint (có thuộc tính .coords,
    như docs/oop_design.md: DataPoint.coords). Lớp IncrementalDBSCAN thật sẽ nhận DataPoint, còn code đo đạc của D05 đến D10 có thể
    đưa tọa độ trần; RefitDBSCAN chấp nhận cả hai để đổi engine không phải đổi nơi gọi."""
    raw = point.coords if hasattr(point, "coords") else point
    return np.asarray(raw, dtype=float).ravel()


def as_matrix(data):
    """Ma trận (n, d) từ ndarray/list 2-D, hoặc từ một dãy đối tượng giống DataPoint (mỗi phần tử có .coords)."""
    if isinstance(data, (list, tuple)) and len(data) and hasattr(data[0], "coords"):
        return np.vstack([as_coords(p) for p in data])
    return np.asarray(data, dtype=float)


class IdMap:
    """Ánh xạ RÕ RÀNG giữa id trong stream (cột point_id của d0.csv và updates.csv) và id do engine cấp.

    Hai id khác nhau về nguyên tắc: engine cấp id riêng (RefitDBSCAN: 0..n-1 cho fit, rồi tăng dần cho insert; engine thật của nhóm có
    thể cấp khác), còn nhãn xuất ra (labels_long.csv) luôn dùng id của STREAM để các thuật toán so được với nhau.
    """

    def __init__(self):
        self._to_engine = {}
        self._to_stream = {}

    def bind(self, stream_id, engine_id):
        stream_id, engine_id = int(stream_id), int(engine_id)
        if stream_id in self._to_engine or engine_id in self._to_stream:
            raise KeyError(f"id đã gắn: stream {stream_id} hoặc engine {engine_id}")
        self._to_engine[stream_id], self._to_stream[engine_id] = engine_id, stream_id

    def engine_id(self, stream_id):
        return self._to_engine[int(stream_id)]

    def stream_id(self, engine_id):
        return self._to_stream[int(engine_id)]

    def release(self, stream_id):
        """Gỡ cặp id khi điểm bị xóa; trả id engine để gọi engine.delete."""
        engine_id = self._to_engine.pop(int(stream_id))
        del self._to_stream[engine_id]
        return engine_id

    def __len__(self):
        return len(self._to_engine)


# ------------------------------------------------------------------ (b) lớp thay thế tạm
class RefitDBSCAN:
    """THAY THẾ TẠM cho IncrementalDBSCAN: mỗi insert hoặc delete fit lại toàn bộ bằng sklearn.

    Mục đích: để D05 đến D10 (thí nghiệm, trực quan, benchmark, demo) chạy được trước khi TV2 (insert)
    và TV3 (delete) viết xong thuật toán thật. Bề mặt công khai là thứ lớp thật sẽ có; chỉ cần đổi lớp.

    Tên phương thức CHƯA CHỐT trong nhóm (bảng task: fit_initial/insert/delete; tài liệu R04:
    fit/insert_point/delete_point; docs/oop_design.md: fit/insert/delete). Vì vậy `fit_initial`,
    `insert_point` và `delete_point` chỉ là bí danh mỏng, TẠM THỜI; xóa khi nhóm chốt tên.

    Bề mặt: fit(data) -> self; insert(coords hoặc DataPoint) -> id mới; delete(point_id); get_labels(); snapshot();
    last_update_seconds (thời gian của lần fit/insert/delete gần nhất); n_points, n_clusters, n_noise.
    id điểm: fit gán 0..n-1 theo thứ tự dòng, insert gán id tăng dần, id đã xóa không dùng lại.
    """

    def __init__(self, eps, min_pts):
        self.eps = float(eps)
        self.min_pts = int(min_pts)
        self._ids = []
        self._coords = np.empty((0, 0))
        self._labels = np.empty(0, dtype=int)
        self._types = np.empty(0, dtype=int)
        self._next_id = 0
        self.last_update_seconds = 0.0

    def _refit(self):
        t0 = time.perf_counter()
        self._labels, self._types = fit_full_dbscan(self._coords, self.eps, self.min_pts)
        self.last_update_seconds = time.perf_counter() - t0

    def fit(self, data):
        data = as_matrix(data)
        if data.ndim != 2:
            raise ValueError("fit cần ma trận 2-D (n_điểm, n_chiều)")
        self._coords = data.copy()
        self._ids = list(range(len(data)))
        self._next_id = len(data)
        self._refit()
        return self

    def insert(self, coords):
        """Thêm một điểm, fit lại toàn bộ, trả id của điểm mới. `coords` là tọa độ trần hoặc đối tượng có .coords (DataPoint)."""
        p = as_coords(coords)
        if self._coords.size and p.shape[0] != self._coords.shape[1]:
            raise ValueError(f"điểm mới cần {self._coords.shape[1]} chiều, nhận được {p.shape[0]}")
        self._coords = p[None, :] if self._coords.size == 0 else np.vstack([self._coords, p])
        pid = self._next_id
        self._next_id += 1
        self._ids.append(pid)
        self._refit()
        return pid

    def delete(self, point_id):
        """Xóa điểm theo id, fit lại toàn bộ. KeyError nếu id không tồn tại."""
        try:
            pos = self._ids.index(point_id)
        except ValueError:
            raise KeyError(f"không có điểm id={point_id}") from None
        self._ids.pop(pos)
        self._coords = np.delete(self._coords, pos, axis=0)
        self._refit()

    # bí danh tạm thời, xem docstring lớp
    fit_initial = fit
    insert_point = insert
    delete_point = delete

    def get_labels(self):
        """Nhãn theo thứ tự id hiện tại (list int; -1 là nhiễu)."""
        return self._labels.tolist()

    def snapshot(self):
        """Ảnh chụp trạng thái: dict gồm ids, coords, labels, point_types (mảng numpy, bản sao)."""
        return {"ids": np.array(self._ids, dtype=int), "coords": self._coords.copy(),
                "labels": self._labels.copy(), "point_types": self._types.copy()}

    @property
    def n_points(self):
        return len(self._ids)

    @property
    def n_clusters(self):
        return len(set(self._labels.tolist()) - {-1})

    @property
    def n_noise(self):
        return int((self._labels == -1).sum())


# ------------------------------------------------------------------ (c) so sánh phân hoạch
def _align(labels_a, labels_b, ids):
    if isinstance(labels_a, Mapping) or isinstance(labels_b, Mapping):
        if not (isinstance(labels_a, Mapping) and isinstance(labels_b, Mapping)):
            raise ValueError("hai nhãn phải cùng là dict hoặc cùng là dãy")
        if set(labels_a) != set(labels_b):
            raise ValueError(f"hai dict nhãn khác tập id: {sorted(set(labels_a) ^ set(labels_b), key=str)}")
        ids = list(labels_a) if ids is None else list(ids)
        return ids, np.array([labels_a[i] for i in ids]), np.array([labels_b[i] for i in ids])
    a, b = np.asarray(labels_a), np.asarray(labels_b)
    if a.shape != b.shape:
        raise ValueError(f"độ dài khác nhau: {a.shape} và {b.shape}")
    ids = list(range(len(a))) if ids is None else list(ids)
    if len(ids) != len(a):
        raise ValueError("ids phải cùng độ dài với nhãn")
    return ids, a, b


def compare_partitions(labels_a, labels_b, ids=None, ignore_ids=None, types_a=None, types_b=None):
    """So hai phân hoạch (id cụm không cần giống nhau, chỉ so cách chia).

    labels_a, labels_b: dãy cùng độ dài hoặc hai dict {id: nhãn}; -1 là nhiễu.
    ids: id của từng vị trí (mặc định 0..n-1); ignore_ids: id bị bỏ trước khi so (cột
    ambiguous_border_ids của updates.csv, điểm biên mà hai cài đặt đúng có thể gán khác cụm).
    types_a, types_b (tùy chọn): mảng PointType, để so thêm tập lõi.

    Trả dict: n_compared, n_ignored, ari (nhiễu tính là một lớp), noise_equal, noise_only_a,
    noise_only_b, core_equal (None nếu thiếu types), partition_equal (ari = 1 và tập nhiễu giống nhau).
    """
    ids, a, b = _align(labels_a, labels_b, ids)
    ignore = set(ignore_ids or [])
    keep = np.array([i not in ignore for i in ids], dtype=bool)
    ids_k = [i for i, k in zip(ids, keep) if k]
    a, b = a[keep], b[keep]
    na, nb = {i for i, v in zip(ids_k, a) if v == -1}, {i for i, v in zip(ids_k, b) if v == -1}
    ari = 1.0 if len(a) == 0 else float(adjusted_rand_score(a, b))
    core_equal = None
    if types_a is not None and types_b is not None:
        ta, tb = np.asarray(types_a)[keep], np.asarray(types_b)[keep]
        core_equal = {i for i, v in zip(ids_k, ta) if v == PointType.CORE} == \
                     {i for i, v in zip(ids_k, tb) if v == PointType.CORE}
    return {"n_compared": int(len(a)), "n_ignored": int(len(ids) - len(a)), "ari": ari,
            "noise_equal": na == nb, "noise_only_a": sorted(na - nb, key=str),
            "noise_only_b": sorted(nb - na, key=str), "core_equal": core_equal,
            "partition_equal": bool(np.isclose(ari, 1.0) and na == nb)}


# ------------------------------------------------------------------ (d) chạy lại stream
def read_stream(stream_dir):
    """Đọc d0.csv, updates.csv, params.json; trả (d0, updates, params, feature_cols)."""
    stream_dir = Path(stream_dir)
    d0 = pd.read_csv(stream_dir / "d0.csv", encoding="utf-8")
    upd = pd.read_csv(stream_dir / "updates.csv", dtype={"ambiguous_border_ids": str}, encoding="utf-8")
    params = json.loads((stream_dir / "params.json").read_text(encoding="utf-8"))
    return d0, upd, params, feature_columns(d0)


def ambiguous_by_step(params, upd):
    """Điểm biên mơ hồ theo từng step, dạng {step: [point_id]}. Step 0 lấy từ params.json, các step sau từ cột
    ambiguous_border_ids của updates.csv."""
    amb = {0: [int(v) for v in params.get("d0_ambiguous_border_ids", [])]}
    for step, ids in zip(upd["step"], upd["ambiguous_border_ids"]):
        amb[int(step)] = [int(v) for v in str(ids).split(";") if v and v != "nan"]
    return amb


def _timed(engine, apply_fn, repeats):
    """Áp dụng apply_fn(engine) THẬT một lần và trả (kết quả, giây, số lần đo).

    repeats <= 1: giây = thời gian của chính lần áp dụng thật. repeats > 1: giây = TRUNG VỊ của `repeats` lần chạy trên
    bản sao sâu (deepcopy) của engine trước update, rồi mới áp dụng thật (không tính giờ), nên một lần chạy nguội hoặc
    một lần bị hệ điều hành chen ngang không làm lệch con số."""
    if repeats <= 1:
        t0 = time.perf_counter()
        out = apply_fn(engine)
        return out, time.perf_counter() - t0, 1
    secs = []
    for _ in range(repeats):
        clone = copy.deepcopy(engine)
        t0 = time.perf_counter()
        apply_fn(clone)
        secs.append(time.perf_counter() - t0)
    return apply_fn(engine), float(np.median(secs)), repeats


TIMING_COLUMNS = ["step", "op", "n_points", "n_clusters", "n_noise", "update_seconds", "n_repeats", "n"]


def replay_stream(engine, d0, upd, params, feats, repeats=REPEATS, max_steps=None):
    """Chạy d0 rồi từng dòng updates (theo step) trên `engine`; trả (timing DataFrame, labels DataFrame).

    engine: lớp hoặc hàm tạo nhận (eps=, min_pts=) như RefitDBSCAN, hoặc một đối tượng đã tạo sẵn (khi đó
    eps/minpts của params không được đặt lại). Cần fit(data), insert(coords) -> id, delete(id), snapshot().
    Tọa độ lấy từ stream_coordinates (stream thẻ: frozen_scaling = cửa sổ tham chiếu cố định, xem stream_scaling.py).
    timing: step 0 là lần fit đầu trên D0; update_seconds là trung vị của n_repeats lần đo; n = số điểm engine xử lý (= n_points).
    labels: (step, point_id, label) với point_id là id của STREAM (IdMap đổi từ id engine).
    """
    is_factory = isinstance(engine, type) or not hasattr(engine, "fit")
    eng = engine(eps=params.get("eps", params.get("eps_km")), min_pts=params["minpts"]) if is_factory else engine
    ids = IdMap()
    timing, labels = [], []

    def record(step, op, secs, reps):
        snap = eng.snapshot()
        lab = np.asarray(snap["labels"])
        timing.append((step, op, len(lab), len(set(lab.tolist()) - {-1}), int((lab == -1).sum()), secs, reps, len(lab)))
        labels.extend((step, ids.stream_id(e), int(l)) for e, l in zip(snap["ids"], lab))

    coords0 = stream_coordinates(params, d0, feats)
    _, secs, reps = _timed(eng, lambda e: e.fit(coords0), repeats)
    for sid, eid in zip(d0["point_id"].tolist(), eng.snapshot()["ids"].tolist()):
        ids.bind(sid, eid)
    record(0, "fit", secs, reps)
    rows = upd.sort_values("step")
    if max_steps is not None:
        rows = rows[rows["step"] <= max_steps]
    coords = stream_coordinates(params, rows, feats)
    for k, (_, r) in enumerate(rows.iterrows()):
        if r["op"] == "insert":
            eid, secs, reps = _timed(eng, lambda e, x=coords[k]: e.insert(x), repeats)
            ids.bind(r["point_id"], eid)
        elif r["op"] == "delete":
            eid = ids.release(r["point_id"])
            _, secs, reps = _timed(eng, lambda e, i=eid: e.delete(i), repeats)
        else:
            raise ValueError(f"op lạ ở step {r['step']}: {r['op']!r}")
        record(int(r["step"]), r["op"], secs, reps)
    return (pd.DataFrame(timing, columns=TIMING_COLUMNS), pd.DataFrame(labels, columns=["step", "point_id", "label"]))


def run_stream(engine, stream_dir, out_dir, max_steps=None, repeats=REPEATS):
    """Đọc một thư mục stream (d0.csv, updates.csv, params.json), chạy replay_stream và ghi
    out_dir/<tên thư mục stream>/timing.csv (cột TIMING_COLUMNS) và labels_long.csv (step, point_id, label).
    Trả (timing, labels). Phần xác định (mọi cột trừ update_seconds) không phụ thuộc repeats."""
    stream_dir = Path(stream_dir)
    d0, upd, params, feats = read_stream(stream_dir)
    timing, labels = replay_stream(engine, d0, upd, params, feats, repeats=repeats, max_steps=max_steps)
    out = Path(out_dir) / stream_dir.name
    out.mkdir(parents=True, exist_ok=True)
    timing.to_csv(out / "timing.csv", index=False, encoding="utf-8", lineterminator="\n")
    labels.to_csv(out / "labels_long.csv", index=False, encoding="utf-8", lineterminator="\n")
    return timing, labels


# ------------------------------------------------------------------ (e) benchmark theo n
BENCH_NS = (500, 1000, 2000, 4000)
BENCH_UPDATES = 20      # 10 insert xen kẽ 10 delete điểm cũ nhất; D0 là phần còn lại của bảng


def _bench_frames(family, table, params_n, n_updates=BENCH_UPDATES):
    """Dựng (d0, updates, params, feats) trong bộ nhớ từ một bảng bench (data/raw/bench). Thời gian không phụ thuộc cách
    chuẩn hóa nên thẻ dùng cửa sổ tham chiếu cố định = chính bảng bench (log10 amount rồi z-score, như frozen_scaling của stream)."""
    if family == "card":
        feats = ["amount_vnd", "hour_of_day", "distance_from_home_km", "txns_last_7d", "merchant_risk_score"]
        table = table.drop_duplicates().dropna(subset=feats)       # bảng bench còn ô thiếu và dòng trùng như bảng thẻ chính
        x = table[feats].to_numpy(float).copy()
        x[:, 0] = np.log10(x[:, 0])
        cols = ["log10_amount_vnd"] + feats[1:]
        p = {"eps": params_n["card"]["recommended_eps"], "minpts": params_n["card"]["recommended_minpts"],
             "frozen_scaling": {"columns": cols, "mean": x.mean(axis=0).tolist(), "std": x.std(axis=0).tolist()}}
    else:
        feats = ["x_km", "y_km"]
        p = {"eps": params_n["atm"]["eps_km"], "minpts": params_n["atm"]["minpts"]}
    tab = table[feats].reset_index(drop=True)
    n_ins = n_updates // 2
    d0 = tab.iloc[: len(tab) - n_ins].copy()
    d0.insert(0, "point_id", range(len(d0)))
    d0["ref_label"] = 0
    rows, oldest, nxt = [], 0, len(d0)
    for step in range(1, n_updates + 1):
        if step % 2:
            rows.append({"step": step, "op": "insert", "point_id": nxt, **tab.iloc[nxt].to_dict()})
            nxt += 1
        else:
            rows.append({"step": step, "op": "delete", "point_id": oldest, **tab.iloc[oldest].to_dict()})
            oldest += 1
    return d0, pd.DataFrame(rows), p, feats


def _ensure_bench_tables(ns, bench_dir, root):
    """Tạo data/raw/bench/ bằng bộ sinh dữ liệu (generate_data.write_bench, giống `--n`) nếu thiếu file; không đụng file khác."""
    if all((bench_dir / f"{kind}_n{n}.csv").exists() for n in ns for kind in ("card_txn_5d", "atm_txn_2d")) \
            and all((bench_dir / f"params_n{n}.json").exists() for n in ns):
        return
    import importlib.util
    import sys
    sys.path.insert(0, str(root / "data"))      # generate_data import các gen_* cùng thư mục
    try:
        spec = importlib.util.spec_from_file_location("generate_data", root / "data" / "generate_data.py")
        gen = importlib.util.module_from_spec(spec)
        sys.modules["generate_data"] = gen
        spec.loader.exec_module(gen)
        gen.write_bench(list(ns))
    finally:
        sys.path.remove(str(root / "data"))


def run_benchmark(ns=BENCH_NS, out_dir=None, repeats=REPEATS, root=None):
    """Đo thời gian re-fit của RefitDBSCAN theo n (bảng bench 2D ATM và 5D thẻ) và ghi data/processed/baseline_bench/.

    Mỗi dòng bench_timing.csv: family, n, op (fit, insert, delete), eps, minpts, n_updates, repeats, median_ms, min_ms, max_ms.
    File thời gian phụ thuộc máy (CPU, hệ điều hành, phiên bản sklearn) nên không đưa lên git (xem data/.gitignore); README.txt trong thư mục ghi cấu hình máy.
    """
    import platform
    import sklearn
    root = Path(root) if root else Path(__file__).resolve().parents[1]
    out = Path(out_dir) if out_dir else root / "data" / "processed" / "baseline_bench"
    bench_dir = root / "data" / "raw" / "bench"
    _ensure_bench_tables(ns, bench_dir, root)
    rows = []
    for n in ns:
        params_n = json.loads((bench_dir / f"params_n{n}.json").read_text(encoding="utf-8"))
        for family, fname in (("atm", f"atm_txn_2d_n{n}.csv"), ("card", f"card_txn_5d_n{n}.csv")):
            table = pd.read_csv(bench_dir / fname, encoding="utf-8")
            d0, upd, p, feats = _bench_frames(family, table, params_n)
            timing, _ = replay_stream(RefitDBSCAN, d0, upd, p, feats, repeats=repeats)
            for op, g in timing.groupby("op", sort=False):
                ms = g["update_seconds"] * 1000
                rows.append((family, n, op, p["eps"], p["minpts"], len(g), repeats, round(ms.median(), 3),
                             round(ms.min(), 3), round(ms.max(), 3)))
            print(f"bench {family} n={n}: " + ", ".join(f"{r[2]} {r[7]} ms" for r in rows[-3:]), flush=True)
    res = pd.DataFrame(rows, columns=["family", "n", "op", "eps", "minpts", "n_updates", "repeats",
                                      "median_ms", "min_ms", "max_ms"])
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "bench_timing.csv", index=False, encoding="utf-8", lineterminator="\n")
    (out / "README.txt").write_text(
        "Thời gian re-fit của baseline (sklearn) theo n. Phụ thuộc máy, không đưa lên git (xem data/.gitignore).\n"
        f"Máy: {platform.platform()}, Python {platform.python_version()}, scikit-learn {sklearn.__version__}, numpy {np.__version__}.\n"
        f"Mỗi update đo {repeats} lần trên bản sao sâu, lấy trung vị; {BENCH_UPDATES} update mỗi (family, n); op fit là lần fit đầu trên D0.\n"
        "Chạy lại: python -m baseline.sklearn_dbscan --bench\n", encoding="utf-8", newline="\n")
    return res


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Baseline sklearn: chạy ba stream (mặc định) hoặc benchmark theo n.")
    ap.add_argument("--bench", action="store_true", help="đo thời gian theo n, ghi data/processed/baseline_bench/ (phụ thuộc máy)")
    ap.add_argument("--n", type=int, nargs="+", default=list(BENCH_NS), help="các n cho --bench (mặc định 500 1000 2000 4000)")
    ap.add_argument("--repeats", type=int, default=REPEATS, help="số lần đo mỗi update (trung vị)")
    a = ap.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    if a.bench:
        print(run_benchmark(a.n, repeats=a.repeats).to_string(index=False))
        return
    for stream in ("bank_geo/atm_stream", "bank_card/card_stream", "bank_card/card_stream_hard"):
        timing, _ = run_stream(RefitDBSCAN, root / "data/raw" / stream, root / "data/processed/baseline", repeats=a.repeats)
        print(f"{stream}: {len(timing) - 1} update, trung vị {timing['update_seconds'].iloc[1:].median() * 1000:.2f} ms/update "
              f"({a.repeats} lần đo mỗi update), số cụm {timing['n_clusters'].min()} đến {timing['n_clusters'].max()}")


if __name__ == "__main__":
    # python -m baseline.sklearn_dbscan  (chạy từ thư mục gốc repo): ba stream, ghi data/processed/baseline/
    main()
