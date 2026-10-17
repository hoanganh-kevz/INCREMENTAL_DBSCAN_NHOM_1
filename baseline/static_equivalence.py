"""D08: so DBSCAN.fit của nhóm với DBSCAN của sklearn trên dữ liệu của TV4.

Hai kết quả tương đương khi mọi điểm có cùng loại (lõi, biên, nhiễu) và các điểm lõi được chia cụm giống nhau. Điểm biên được phép
khác cụm nếu nó mơ hồ: trong Eps có lõi của từ hai cụm trở lên, và cụm được gán là một trong các cụm đó.

Chỉ dùng trong baseline/ và test, không import từ src/.
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from .sklearn_dbscan import PointType, fit_full_dbscan, read_stream
from src.preprocessing.stream_scaling import stream_coordinates
from src.visualization.scenarios import list_scenarios, load_scenario

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

# Eps, MinPts của từng bộ dữ liệu (xem data/README_data.md, mục 6, 7 và 9a)
ATM = (0.5, 4)
ATM_HARD = (0.4, 4)
CARD = (0.65, 10)
CARD_HARD = (0.30, 15)
ND_BLOBS = (0.8, 10)        # bộ 5 chiều của D01 sau chuẩn hóa: ARI 0.994 so với nhãn sinh, 4 cụm


def _code(point_type):
    """Mã loại điểm theo tên enum (CORE, BORDER, NOISE), dùng được cho cả enum của nhóm lẫn của baseline."""
    name = str(getattr(point_type, "name", point_type)).upper()
    return {"CORE": int(PointType.CORE), "BORDER": int(PointType.BORDER), "NOISE": int(PointType.NOISE)}.get(name, int(PointType.UNCLASSIFIED))


def labels_and_types(model):
    """Nhãn và loại điểm theo thứ tự id. Nhận model có snapshot() (RefitDBSCAN) hoặc có points và get_labels() (DBSCAN của nhóm,
    get_labels() trả nhãn theo thứ tự các điểm trong dict points)."""
    if hasattr(model, "snapshot"):
        snap = model.snapshot()
        return np.asarray(snap["labels"], dtype=int), np.asarray(snap["point_types"], dtype=int)
    points, labels = model.points, list(model.get_labels())
    if len(labels) != len(points):
        raise ValueError(f"get_labels() trả {len(labels)} nhãn cho {len(points)} điểm")
    if not isinstance(points, dict):
        return np.asarray(labels, dtype=int), np.array([_code(p.point_type) for p in points], dtype=int)
    by_id = dict(zip(points, labels))
    order = sorted(points)
    return np.array([by_id[k] for k in order], dtype=int), np.array([_code(points[k].point_type) for k in order], dtype=int)


def compare_static_fit(engine_factory, coords, eps, min_pts):
    """Fit engine_factory(eps, min_pts) rồi so với sklearn. Trả dict: n, dims, số cụm hai bên, types_equal, cores_equal,
    noise_equal, core_partition_ari, border_diff (số điểm biên khác cụm), border_diff_all_ambiguous, thời gian hai bên (đo một
    lần) và equivalent."""
    coords = np.asarray(coords, dtype=float)
    model = engine_factory(eps, min_pts)
    start = time.perf_counter()
    model.fit(coords)
    seconds = time.perf_counter() - start
    la, ta = labels_and_types(model)
    start = time.perf_counter()
    lb, tb = fit_full_dbscan(coords, eps, min_pts)
    sk_seconds = time.perf_counter() - start
    core_a, core_b = ta == int(PointType.CORE), tb == int(PointType.CORE)
    types_equal = bool((ta == tb).all())
    cores_equal = bool((core_a == core_b).all())
    noise_equal = bool(((ta == int(PointType.NOISE)) == (tb == int(PointType.NOISE))).all())
    ari = float(adjusted_rand_score(lb[core_b], la[core_b])) if core_b.sum() > 1 else 1.0
    border_diff, all_ambiguous = 0, True
    if cores_equal and core_b.any():
        mapping = {}
        for a, b in zip(la[core_b], lb[core_b]):
            mapping.setdefault(b, a)
        for i in np.where(tb == int(PointType.BORDER))[0]:
            if mapping.get(lb[i]) != la[i]:
                border_diff += 1
                near = np.linalg.norm(coords[core_b] - coords[i], axis=1) <= eps + 1e-12
                all_ambiguous &= len(set(lb[core_b][near])) >= 2 and la[i] in set(la[core_b][near])
    equivalent = types_equal and cores_equal and noise_equal and ari == 1.0 and all_ambiguous
    return dict(n=len(coords), dims=coords.shape[1], clusters_engine=len(set(la) - {-1}), clusters_sklearn=len(set(lb) - {-1}),
                types_equal=types_equal, cores_equal=cores_equal, noise_equal=noise_equal, core_partition_ari=round(ari, 6),
                border_diff=border_diff, border_diff_all_ambiguous=all_ambiguous, engine_seconds=round(seconds, 3),
                sklearn_seconds=round(sk_seconds, 4), equivalent=equivalent)


def _final_state(scenario):
    """Tọa độ trước và sau khi áp dụng mọi dòng update của một kịch bản (khóa là tên điểm trong points.csv)."""
    current = dict(zip(scenario["names"], map(np.asarray, scenario["coords"])))
    before = list(current.values())
    for u in scenario["updates"]:
        if u["op"] == "insert":
            current[u["point_id"]] = np.asarray(u["coords"])
        else:
            current.pop(u["point_id"], None)
    return before, list(current.values())


def iter_cases(raw_dir=RAW, processed_dir=PROCESSED):
    """Các trường hợp của D08, mỗi cái là (tên, tọa độ, eps, min_pts): đầu và cuối của các kịch bản toy, toy_edge, bốn bảng ATM và
    thẻ, bộ blobs 5 chiều và D0 của ba stream."""
    for kind in ("toy", "toy_edge"):
        for name in list_scenarios(kind, raw_dir):
            sc = load_scenario(kind, name, raw_dir)
            before, after = _final_state(sc)
            yield f"{kind}/{name} (đầu)", before, sc["eps"], sc["min_pts"]
            yield f"{kind}/{name} (cuối)", after, sc["eps"], sc["min_pts"]
    raw_dir, processed_dir = Path(raw_dir), Path(processed_dir)
    for fname, (eps, mp) in (("atm_txn_2d.csv", ATM), ("atm_txn_2d_hard.csv", ATM_HARD)):
        frame = pd.read_csv(raw_dir / "bank_geo" / fname, encoding="utf-8")
        yield fname.removesuffix(".csv"), frame[["x_km", "y_km"]].to_numpy(), eps, mp
    for fname, (eps, mp) in (("card_txn_5d_scaled.csv", CARD), ("card_txn_5d_hard_scaled.csv", CARD_HARD), ("blobs5d_n860_scaled.csv", ND_BLOBS)):
        frame = pd.read_csv(processed_dir / fname, encoding="utf-8")
        yield fname.removesuffix(".csv"), frame[[c for c in frame.columns if c.endswith("_z")]].to_numpy(), eps, mp
    for stream in ("bank_geo/atm_stream", "bank_card/card_stream", "bank_card/card_stream_hard"):
        d0, _, params, feats = read_stream(raw_dir / stream)
        yield f"{stream} D0", stream_coordinates(params, d0, feats), params.get("eps", params.get("eps_km")), params["minpts"]


def run_all(engine_factory):
    """Chạy hết các trường hợp, trả DataFrame mỗi dòng một trường hợp."""
    rows = []
    for name, coords, eps, min_pts in iter_cases():
        rows.append({"case": name, **compare_static_fit(engine_factory, coords, eps, min_pts)})
    return pd.DataFrame(rows)
