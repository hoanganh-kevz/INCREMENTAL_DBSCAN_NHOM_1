"""Đọc kịch bản toy, toy_edge và stream rồi chạy trên MỘT engine bất kỳ để lấy snapshot (không sklearn, không scipy).

Engine được truyền vào dưới dạng hàm tạo `engine_factory(eps, min_pts)` (RefitDBSCAN hôm nay, IncrementalDBSCAN của nhóm sau này), cần các
phương thức fit(data), insert(coords) -> id, delete(id) và một snapshot lấy được bằng `snapshot_from_model`. Nhờ vậy mọi hình trong
báo cáo đổi sang thuật toán thật chỉ bằng cách đổi đối số `engine_factory`.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.preprocessing.stream_scaling import feature_columns, stream_coordinates
from .plotter import align_labels, snapshot_from_model

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"


def list_scenarios(kind, raw_dir=RAW):
    """Tên các thư mục kịch bản (kind là 'toy' hoặc 'toy_edge'), theo thứ tự tên."""
    return sorted(p.name for p in (Path(raw_dir) / kind).iterdir() if p.is_dir())


def load_scenario(kind, name, raw_dir=RAW):
    """Đọc points.csv, update.csv, expected.json của một kịch bản. Trả dict: names, coords, updates (list dict op, point_id, coords),
    eps, min_pts, expected, seed_ids (UpdSeed theo expected.json; kịch bản nhiều bước lấy hợp các bước), coord_cols."""
    base = Path(raw_dir) / kind / name
    exp = json.loads((base / "expected.json").read_text(encoding="utf-8"))
    pts = pd.read_csv(base / "points.csv", dtype={"point_id": str}, encoding="utf-8")
    upd = pd.read_csv(base / "update.csv", dtype={"point_id": str}, encoding="utf-8")
    cols = [c for c in pts.columns if c != "point_id"]
    updates = []
    for _, r in upd.iterrows():
        xy = None if r["op"] == "delete" else r[cols].to_numpy(dtype=float)
        updates.append({"op": r["op"], "point_id": r["point_id"], "coords": xy})
    seeds = exp.get("updseed")
    if seeds is None:
        seeds = sorted({s for step in exp.get("updseed_by_step", []) for s in step})
    return {"kind": kind, "name": name, "names": pts["point_id"].tolist(), "coords": pts[cols].to_numpy(dtype=float), "updates": updates,
            "eps": float(exp["eps"]), "min_pts": int(exp["minpts"]), "expected": exp, "seed_ids": list(seeds), "coord_cols": cols}


def run_scenario(engine_factory, sc):
    """Chạy kịch bản trên engine: fit các điểm đầu, rồi từng dòng update theo thứ tự.

    Trả dict: before (snapshot sau fit), after (snapshot sau mọi update), steps (snapshot sau từng update), update (danh sách để truyền
    cho plot_before_after), seed_ids, eps. Id trong snapshot là TÊN điểm trong points.csv và update.csv (như A1, p, q), không phải id engine."""
    eng = engine_factory(sc["eps"], sc["min_pts"])
    eng.fit(sc["coords"])
    snap0 = snapshot_from_model(eng)
    to_name = dict(zip(snap0["ids"].tolist(), sc["names"]))
    to_engine = {n: e for e, n in to_name.items()}

    def named(snap):
        out = dict(snap)
        out["ids"] = np.array([to_name[i] for i in snap["ids"].tolist()])
        return out
    before, steps = named(snap0), []
    for u in sc["updates"]:
        if u["op"] == "insert":
            eid = eng.insert(u["coords"])
            to_name[eid], to_engine[u["point_id"]] = u["point_id"], eid
        else:
            eng.delete(to_engine.pop(u["point_id"]))
        steps.append(named(snapshot_from_model(eng)))
    return {"before": before, "after": steps[-1] if steps else before, "steps": steps, "update": sc["updates"],
            "seed_ids": sc["seed_ids"], "eps": sc["eps"]}


# ------------------------------------------------------------------ stream
def stream_snapshots(engine_factory, stream_dir, max_steps=None):
    """Chạy một stream (d0.csv, updates.csv, params.json) và trả danh sách khung, mỗi khung ứng với một step (step 0 là sau fit D0).

    Mỗi khung là dict: step, op ('fit', 'insert', 'delete'), point_id (id của stream), coords (tọa độ điểm cập nhật hoặc None),
    snapshot (ids là id của STREAM, coords theo stream_coordinates), case (verified_case hoặc None), eps, min_pts."""
    stream_dir = Path(stream_dir)
    d0 = pd.read_csv(stream_dir / "d0.csv", encoding="utf-8")
    upd = pd.read_csv(stream_dir / "updates.csv", encoding="utf-8").sort_values("step")
    params = json.loads((stream_dir / "params.json").read_text(encoding="utf-8"))
    feats = feature_columns(d0)
    eps, min_pts = params.get("eps", params.get("eps_km")), params["minpts"]
    eng = engine_factory(eps, min_pts)
    eng.fit(stream_coordinates(params, d0, feats))
    snap = snapshot_from_model(eng)
    to_stream = dict(zip(snap["ids"].tolist(), d0["point_id"].tolist()))
    to_engine = {s: e for e, s in to_stream.items()}

    def named(sn):
        out = dict(sn)
        out["ids"] = np.array([to_stream[i] for i in sn["ids"].tolist()])
        return out
    frames = [{"step": 0, "op": "fit", "point_id": None, "coords": None, "snapshot": named(snap), "case": None, "eps": eps, "min_pts": min_pts}]
    if max_steps is not None:
        upd = upd[upd["step"] <= max_steps]
    coords = stream_coordinates(params, upd, feats)
    for k, (_, r) in enumerate(upd.iterrows()):
        sid = int(r["point_id"])
        if r["op"] == "insert":
            eid = eng.insert(coords[k])
            to_stream[eid], to_engine[sid] = sid, eid
        else:
            eng.delete(to_engine.pop(sid))
        frames.append({"step": int(r["step"]), "op": r["op"], "point_id": sid, "coords": coords[k], "snapshot": named(snapshot_from_model(eng)),
                       "case": r.get("verified_case"), "eps": eps, "min_pts": min_pts})
    return frames


def equivalence_series(labels_a, labels_b, compare, ambiguous=None):
    """ARI từng step giữa hai bảng nhãn dạng dài (cột step, point_id, label), bỏ các điểm biên mơ hồ.

    compare: hàm so phân hoạch có chữ ký như baseline.sklearn_dbscan.compare_partitions(labels_a, labels_b, ids=None, ignore_ids=None)
    và trả dict có khóa ari (truyền vào để src/ không phụ thuộc sklearn). ambiguous: dict {step: [point_id]} (tùy chọn).
    Trả DataFrame cột step, ari, n_compared, n_ignored."""
    ambiguous = ambiguous or {}
    ga, gb = labels_a.groupby("step"), labels_b.groupby("step")
    rows = []
    for step in sorted(set(ga.groups) & set(gb.groups)):
        a = dict(zip(ga.get_group(step)["point_id"], ga.get_group(step)["label"]))
        b = dict(zip(gb.get_group(step)["point_id"], gb.get_group(step)["label"]))
        r = compare(a, b, ignore_ids=ambiguous.get(step, []))
        rows.append((int(step), r["ari"], r["n_compared"], r["n_ignored"]))
    return pd.DataFrame(rows, columns=["step", "ari", "n_compared", "n_ignored"])


def label_changes(labels_long, timing=None):
    """Số điểm ĐỔI cụm ở mỗi step so với step trước (cụm của step sau được khớp về step trước bằng align_labels, nên đánh số lại nhãn không bị tính).

    Không tính điểm vừa chèn hay vừa xóa. Điểm từ nhiễu sang cụm hoặc ngược lại tính là đổi. timing (tùy chọn, bảng baseline có cột step, op, n_points)
    thêm cột op và n_points. Trả DataFrame cột step, n_affected (+ op, n_points)."""
    groups = {s: g for s, g in labels_long.groupby("step")}
    steps = sorted(groups)
    rows = []
    for prev, cur in zip(steps[:-1], steps[1:]):
        gp, gc = groups[prev], groups[cur]
        ref = {"ids": gp["point_id"].to_numpy(), "labels": gp["label"].to_numpy()}
        snap = {"ids": gc["point_id"].to_numpy(), "labels": gc["label"].to_numpy()}
        new = align_labels(ref, snap)
        before = dict(zip(gp["point_id"].tolist(), gp["label"].tolist()))
        changed = sum(1 for i, l in zip(snap["ids"].tolist(), new.tolist()) if i in before and before[i] != l)
        rows.append((cur, changed))
    out = pd.DataFrame(rows, columns=["step", "n_affected"])
    if timing is not None:
        out = out.merge(timing[["step", "op", "n_points"]], on="step", how="left")
    return out
