"""D05 (thí nghiệm INSERT) và D06 (thí nghiệm DELETE): IncrementalDBSCAN của nhóm so với fit lại toàn bộ.

Ba engine chạy cùng dữ liệu, cùng Eps và MinPts:
  incremental    IncrementalEngine (src/experiments/engine.py), tức IncrementalDBSCAN của nhóm;
  own_refit      OwnRefitDBSCAN, DBSCAN của nhóm fit lại sau mỗi update (cùng ngôn ngữ, cùng hàm khoảng cách);
  sklearn_refit  RefitDBSCAN, DBSCAN của sklearn fit lại sau mỗi update.

Bốn phần:
  scenario_table   27 kịch bản toy và toy_edge, mỗi bước update một dòng (case, UpdSeed, số điểm đổi, thời gian ba engine, khớp đáp án);
  stream_table     ba stream (atm_stream, card_stream, card_stream_hard), mỗi step một dòng, kèm so tương đương với sklearn;
  batch_insert     chèn k điểm liên tiếp vào bản đồ ATM rồi so với một lần fit lại;
  case_summary     gộp stream_table theo (op, case).

Chạy từ gốc repo: python -m baseline.update_experiments [--repeats 5] [--no-own-refit-cards]
Ghi bảng vào data/processed/experiments/d05_d06/ và hình vào report/figures/. Cột thời gian phụ thuộc máy.
"""
import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.experiments.engine import IncrementalEngine
from src.experiments.update_cases import describe_update, update_seeds
from src.visualization.gallery import build_gallery_cases
from src.visualization.plotter import _HALO
from src.visualization.scenarios import list_scenarios, load_scenario, run_scenario, stream_snapshots
from .own_refit import OwnRefitDBSCAN
from .sklearn_dbscan import REPEATS, RefitDBSCAN, _timed, ambiguous_by_step, compare_partitions, read_stream, replay_stream

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed" / "experiments" / "d05_d06"
FIG = ROOT / "report" / "figures"

ENGINES = {"incremental": IncrementalEngine, "own_refit": OwnRefitDBSCAN, "sklearn_refit": RefitDBSCAN}
STREAMS = {"atm_stream": "bank_geo", "card_stream": "bank_card", "card_stream_hard": "bank_card"}

# case của 8 kịch bản toy (toy_edge ghi case trong expected.json)
TOY_CASE = {"r02_1_noise": "Noise", "r02_2_creation": "Creation", "r02_3_absorption": "Absorption", "r02_4_merge": "Merge",
            "r03_1_removal": "Removal", "r03_2_reduction": "Reduction", "r03_3a_split": "PotentialSplit_Split",
            "r03_3b_nonsplit": "PotentialSplit_NonSplit"}
INSERT_CASES = ["Noise", "Creation", "Absorption", "Merge"]
DELETE_CASES = ["DeleteNoise", "Removal", "Reduction", "PotentialSplit_NonSplit", "PotentialSplit_Split"]


# ------------------------------------------------------------------ so hai snapshot
def snapshots_equivalent(a, b, eps, ignore_ids=()):
    """Hai snapshot (cùng tập id) tương đương khi cùng loại điểm, cùng cách chia điểm lõi, và điểm biên khác cụm chỉ khi nó mơ hồ
    (trong Eps có lõi của từ hai cụm trở lên và cụm được gán là một trong số đó). ignore_ids: điểm biên mơ hồ đã biết, bỏ khỏi phép so nhãn."""
    ia, ib = np.asarray(a["ids"]).tolist(), np.asarray(b["ids"]).tolist()
    if sorted(ia) != sorted(ib):
        return False
    pa, pb = {i: k for k, i in enumerate(ia)}, {i: k for k, i in enumerate(ib)}
    ta = {i: int(a["point_types"][pa[i]]) for i in ia}
    tb = {i: int(b["point_types"][pb[i]]) for i in ia}
    if ta != tb:
        return False
    la = {i: int(a["labels"][pa[i]]) for i in ia}
    lb = {i: int(b["labels"][pb[i]]) for i in ia}
    cores = [i for i in ia if ta[i] == 3]
    fwd, bwd = {}, {}
    for i in cores:
        if fwd.setdefault(la[i], lb[i]) != lb[i] or bwd.setdefault(lb[i], la[i]) != la[i]:
            return False
    coords = np.asarray(a["coords"], dtype=float)
    core_xy = coords[[pa[i] for i in cores]] if cores else np.empty((0, coords.shape[1]))
    ignore = set(ignore_ids)
    for i in ia:
        if ta[i] != 2 or i in ignore or fwd.get(la[i]) == lb[i]:
            continue
        near = np.sqrt(((core_xy - coords[pa[i]]) ** 2).sum(axis=1)) <= eps
        near_labels = {lb[c] for c, k in zip(cores, near) if k}
        if len(near_labels) < 2 or la[i] not in {la[c] for c, k in zip(cores, near) if k}:
            return False
    return True


# ------------------------------------------------------------------ kịch bản toy và toy_edge
def _timed_steps(factory, sc, repeats):
    """Thời gian từng update (giây) của một engine trên một kịch bản; id engine được đổi từ tên điểm như run_scenario."""
    eng = factory(sc["eps"], sc["min_pts"])
    eng.fit(sc["coords"])
    to_engine = {n: i for i, n in zip(eng.snapshot()["ids"].tolist(), sc["names"])}
    secs = []
    for u in sc["updates"]:
        if u["op"] == "insert":
            eid, s, _ = _timed(eng, lambda e, x=u["coords"]: e.insert(x), repeats)
            to_engine[u["point_id"]] = eid
        else:
            _, s, _ = _timed(eng, lambda e, i=to_engine.pop(u["point_id"]): e.delete(i), repeats)
        secs.append(s)
    return secs


def _expected_after(sc, k):
    exp = sc["expected"]
    if exp.get("steps"):
        s = exp["steps"][k]
        return s["labels_after"], set(s["core_after"]), set(s.get("ambiguous_border_ids", [])) | set(exp.get("ambiguous_to_ignore", [])), s["case"]
    return exp["labels_after"], set(exp["core_after"]), set(exp.get("ambiguous_to_ignore", [])), TOY_CASE[sc["name"]]


def scenario_table(kinds=("toy", "toy_edge"), repeats=REPEATS):
    """Mỗi bước update của 27 kịch bản một dòng: kind, scenario, step, op, point_id, case, n_before, n_updseed, updseed_ok (khớp expected.json),
    n_core_changed, n_changed, n_reprocessed, matches_expected (nhãn và tập lõi), equivalent_sklearn, incremental_ms, own_refit_ms, sklearn_refit_ms."""
    rows = []
    for kind in kinds:
        for name in list_scenarios(kind):
            sc = load_scenario(kind, name)
            inc = run_scenario(IncrementalEngine, sc)
            ref = run_scenario(RefitDBSCAN, sc)
            times = {k: _timed_steps(f, sc, repeats) for k, f in ENGINES.items()}
            exp = sc["expected"]
            seeds_by_step = exp.get("updseed_by_step") or [exp.get("updseed")]
            snaps = [inc["before"]] + inc["steps"]
            for k, u in enumerate(sc["updates"]):
                before, after = snaps[k], snaps[k + 1]
                d = describe_update(before, after, u["op"], u["point_id"], sc["eps"])
                seeds, _ = update_seeds(before, after, u["op"], sc["eps"])
                labels_exp, cores_exp, amb, case = _expected_after(sc, k)
                got = dict(zip(after["ids"].tolist(), after["labels"].tolist()))
                cores = {i for i, t in zip(after["ids"].tolist(), after["point_types"].tolist()) if t == 3}
                ok = compare_partitions(got, labels_exp, ignore_ids=amb)["partition_equal"] and cores == cores_exp
                rows.append({"kind": kind, "scenario": name, "step": k + 1, "op": u["op"], "point_id": u["point_id"], "case": case, **d,
                             "updseed_ok": seeds == set(seeds_by_step[k] or []) if k < len(seeds_by_step) else None,
                             "matches_expected": ok,
                             "equivalent_sklearn": snapshots_equivalent(after, ref["steps"][k], sc["eps"], amb),
                             **{f"{e}_ms": round(times[e][k] * 1000, 4) for e in ENGINES}})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ stream
def stream_table(stream, repeats=REPEATS, own_refit=True, max_steps=None):
    """Mỗi step của một stream một dòng: stream, step, op, point_id, case (verified_case), n_before, n_updseed, n_core_changed, n_changed,
    n_reprocessed, equivalent_sklearn, incremental_ms, sklearn_refit_ms, own_refit_ms (NaN nếu own_refit=False).
    Thời gian lấy từ replay_stream (trung vị của `repeats` lần đo; own_refit đo một lần vì mỗi lần fit lại mất vài trăm ms)."""
    sdir = RAW / STREAMS[stream] / stream
    d0, upd, params, feats = read_stream(sdir)
    eps = params.get("eps", params.get("eps_km"))
    t_inc, _ = replay_stream(IncrementalEngine, d0, upd, params, feats, repeats=repeats, max_steps=max_steps)
    t_sk, _ = replay_stream(RefitDBSCAN, d0, upd, params, feats, repeats=repeats, max_steps=max_steps)
    t_own = replay_stream(OwnRefitDBSCAN, d0, upd, params, feats, repeats=1, max_steps=max_steps)[0] if own_refit else None
    f_inc = stream_snapshots(IncrementalEngine, sdir, max_steps=max_steps)
    f_sk = stream_snapshots(RefitDBSCAN, sdir, max_steps=max_steps)
    amb = ambiguous_by_step(params, upd)
    rows = []
    for k in range(1, len(f_inc)):
        fr = f_inc[k]
        d = describe_update(f_inc[k - 1]["snapshot"], fr["snapshot"], fr["op"], fr["point_id"], eps)
        rows.append({"stream": stream, "step": fr["step"], "op": fr["op"], "point_id": fr["point_id"], "case": fr["case"], **d,
                     "equivalent_sklearn": snapshots_equivalent(fr["snapshot"], f_sk[k]["snapshot"], eps, amb.get(fr["step"], [])),
                     "incremental_ms": t_inc["update_seconds"].iloc[k] * 1000, "sklearn_refit_ms": t_sk["update_seconds"].iloc[k] * 1000,
                     "own_refit_ms": t_own["update_seconds"].iloc[k] * 1000 if t_own is not None else np.nan})
    fit = {"stream": stream, "n_d0": len(d0), "dims": len(feats), "eps": eps, "minpts": params["minpts"],
           "incremental_fit_ms": t_inc["update_seconds"].iloc[0] * 1000, "sklearn_fit_ms": t_sk["update_seconds"].iloc[0] * 1000,
           "own_fit_ms": t_own["update_seconds"].iloc[0] * 1000 if t_own is not None else np.nan}
    return pd.DataFrame(rows), fit


def case_summary(steps):
    """Gộp bảng stream theo (stream, op, case): số lần, trung vị thời gian ba engine, tỉ lệ fit lại / gia tăng, trung vị n_updseed, n_changed,
    n_reprocessed và số điểm, cùng tỉ lệ step tương đương với sklearn."""
    g = steps.groupby(["stream", "op", "case"], sort=False)
    out = g.agg(count=("step", "size"), n=("n_before", "median"), incremental_ms=("incremental_ms", "median"),
                own_refit_ms=("own_refit_ms", "median"), sklearn_refit_ms=("sklearn_refit_ms", "median"),
                n_updseed=("n_updseed", "median"), n_changed=("n_changed", "median"), n_reprocessed=("n_reprocessed", "median"),
                equivalent=("equivalent_sklearn", "mean")).reset_index()
    out["own_refit_over_incremental"] = out["own_refit_ms"] / out["incremental_ms"]
    out["sklearn_over_incremental"] = out["sklearn_refit_ms"] / out["incremental_ms"]
    order = {c: i for i, c in enumerate(INSERT_CASES + DELETE_CASES)}
    key = {"case": lambda s: s.map(order), "op": lambda s: s.map({"insert": 0, "delete": 1})}
    return out.sort_values(["stream", "op", "case"], key=lambda s: key.get(s.name, lambda v: v)(s)).reset_index(drop=True)


# ------------------------------------------------------------------ batch insert
BATCH_KS = (1, 5, 10, 20, 50, 100, 200)


def batch_insert(ks=BATCH_KS, n0=1000, repeats=REPEATS):
    """Bản đồ ATM 2D (Eps 0,5 km, MinPts 4) xếp theo thời gian: D0 = n0 giao dịch đầu, rồi chèn k giao dịch kế tiếp.

    So ba cách có cùng kết quả cuối: gia tăng (k lần insert, đo tổng, trung vị của `repeats` lần trên bản sao), một lần fit lại sklearn trên
    D0 + k điểm, một lần fit lại bằng DBSCAN của nhóm (đo một lần). Trả DataFrame cột k, n_final, incremental_total_ms, incremental_per_point_ms,
    sklearn_refit_once_ms, own_refit_once_ms, sklearn_refit_each_ms (k lần fit lại, cộng thật), equivalent_sklearn."""
    atm = pd.read_csv(RAW / "bank_geo" / "atm_txn_2d.csv", encoding="utf-8").sort_values(["timestamp", "txn_id"]).reset_index(drop=True)
    xy = atm[["x_km", "y_km"]].to_numpy(float)
    eps, min_pts = 0.5, 4
    base = IncrementalEngine(eps, min_pts).fit(xy[:n0])
    rows = []
    for k in ks:
        batch = xy[n0:n0 + k]

        def apply(e, pts=batch):
            for p in pts:
                e.insert(p)
        eng = copy.deepcopy(base)
        _, inc_s, _ = _timed(eng, apply, repeats)
        full = xy[:n0 + k]
        _, sk_once, _ = _timed(RefitDBSCAN(eps, min_pts), lambda e: e.fit(full), repeats)
        t0 = time.perf_counter()
        OwnRefitDBSCAN(eps, min_pts).fit(full)
        own_once = time.perf_counter() - t0
        sk = RefitDBSCAN(eps, min_pts).fit(xy[:n0])
        t0 = time.perf_counter()
        for p in batch:
            sk.insert(p)
        sk_each = time.perf_counter() - t0
        rows.append({"k": k, "n_final": n0 + k, "incremental_total_ms": inc_s * 1000, "incremental_per_point_ms": inc_s * 1000 / k,
                     "sklearn_refit_once_ms": sk_once * 1000, "own_refit_once_ms": own_once * 1000, "sklearn_refit_each_ms": sk_each * 1000,
                     "equivalent_sklearn": snapshots_equivalent(eng.snapshot(), sk.snapshot(), eps)})
    return pd.DataFrame(rows), {"atm": atm, "xy": xy, "n0": n0, "eps": eps, "min_pts": min_pts}


# ------------------------------------------------------------------ hình
def crop_snapshot(snap, center, radius):
    """Chỉ giữ các điểm cách `center` không quá `radius` (để phóng to một vùng của bản đồ lớn)."""
    keep = np.sqrt(((np.asarray(snap["coords"]) - np.asarray(center)) ** 2).sum(axis=1)) <= radius
    return {k: np.asarray(v)[keep] for k, v in snap.items()}


def _titled(cases, titles):
    for c, (header, note) in zip(cases, titles):
        c["header"], c["note"] = header, note
    return cases


def insert_gallery(engine=IncrementalEngine):
    """6 ca chèn: 4 kịch bản toy (Noise, Creation, Absorption, Merge), e06 (một điểm nối 3 cụm), e07 (hợp nhất và hút nhiễu)."""
    cases = build_gallery_cases("toy", engine, names=[n for n in TOY_CASE if n.startswith("r02")]) + \
        build_gallery_cases("toy_edge", engine, names=["e06_insert_bridges_three_clusters", "e07_insert_merge_absorbs_noise"])
    return _titled(cases, [("1. Noise", "chèn p, p là nhiễu"), ("2. Creation", "chèn p, tạo cụm mới"),
                           ("3. Absorption", "chèn p, p vào cụm cũ"), ("4. Merge", "chèn p, hai cụm hợp nhất"),
                           ("5. Merge ba cụm", "một điểm chèn nối ba cụm"), ("6. Merge và hút nhiễu", "hai cụm hợp nhất, một nhiễu thành biên")])


def delete_gallery(engine=IncrementalEngine):
    """6 ca xóa: 4 kịch bản toy (Removal, Reduction, Split, Non-split), e13 (tách 3), e16 (xóa cầu, còn biên mơ hồ)."""
    cases = build_gallery_cases("toy", engine, names=[n for n in TOY_CASE if n.startswith("r03")]) + \
        build_gallery_cases("toy_edge", engine, names=["e13_delete_splits_into_three", "e16_delete_bridge_leaves_ambiguous_border"])
    return _titled(cases, [("1. Removal", "xóa p, cụm biến mất"), ("2. Reduction", "xóa p, cụm thu nhỏ"),
                           ("3a. Split", "xóa p, cụm tách đôi"), ("3b. Non-split", "xóa p, cụm vẫn liền nhờ đường vòng"),
                           ("4. Split thành ba", "xóa lõi trung tâm, cụm tách ba"), ("5. Split, còn biên mơ hồ", "xóa cầu nối, một điểm biên kề cả hai cụm")])


def stream_example(stream_dir, case, radius=1.6, engine=IncrementalEngine, which=0):
    """Khung trước và sau của lần thứ `which` mà stream có verified_case = case, cắt quanh điểm cập nhật.
    Trả dict before, after, update, seed_ids (UpdSeed tính từ hai snapshot đầy đủ), eps, step."""
    frames = stream_snapshots(engine, stream_dir)
    hits = [k for k, f in enumerate(frames) if f["case"] == case]
    k = hits[which]
    fr = frames[k]
    seeds, _ = update_seeds(frames[k - 1]["snapshot"], fr["snapshot"], fr["op"], fr["eps"])
    return {"before": crop_snapshot(frames[k - 1]["snapshot"], fr["coords"], radius),
            "after": crop_snapshot(fr["snapshot"], fr["coords"], radius),
            "update": {"op": fr["op"], "point_id": fr["point_id"], "coords": fr["coords"]}, "seed_ids": sorted(seeds),
            "eps": fr["eps"], "step": fr["step"]}


def plot_case_times(summary, stream="atm_stream"):
    """Cột trung vị thời gian mỗi update theo case (thang log): gia tăng, DBSCAN của nhóm fit lại, sklearn fit lại."""
    import matplotlib.pyplot as plt
    from src.visualization.plotter import report_style
    s = summary[summary["stream"] == stream].reset_index(drop=True)
    with report_style():
        fig, ax = plt.subplots(figsize=(7.4, 3.5))
        x = np.arange(len(s))
        w = 0.27
        for j, (col, lab, color) in enumerate((("incremental_ms", "IncrementalDBSCAN của nhóm", "#1f77b4"),
                                               ("own_refit_ms", "DBSCAN của nhóm, fit lại", "#ff7f0e"),
                                               ("sklearn_refit_ms", "sklearn, fit lại", "#7f7f7f"))):
            if s[col].notna().any():
                bars = ax.bar(x + (j - 1) * w, s[col], w, label=lab, color=color)
                if col == "incremental_ms":
                    for b, v in zip(bars, s[col]):
                        txt = "< 0.01" if v < 0.01 else (f"{v:.2f}" if v < 10 else f"{v:.0f}")
                        ax.annotate(txt, (b.get_x() + b.get_width() / 2, max(v, 0.012)), xytext=(0, 2), textcoords="offset points",
                                    ha="center", fontsize=5.6, color="#1f4e79")
        ax.set_yscale("log")
        ax.set_ylim(bottom=0.01)
        ax.set_xticks(x)
        short = {"DeleteNoise": "Xóa nhiễu", "PotentialSplit_NonSplit": "Non-split", "PotentialSplit_Split": "Split"}
        ax.set_xticklabels([f"{short.get(c, c)}\n{'chèn' if o == 'insert' else 'xóa'}, {n} lần" for c, o, n in zip(s["case"], s["op"], s["count"])],
                           fontsize=6.2)
        ax.set_ylabel("ms mỗi update (trung vị, log)")
        dims = 2 if stream.startswith("atm") else 5
        ax.set_title(f"{stream} ({int(s['n'].median())} điểm, {dims} chiều): thời gian mỗi update theo case, trung vị 5 lần đo; số trên cột xanh là ms",
                     fontsize=8, loc="left")
        ax.legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
        fig.tight_layout()
    return fig


def plot_batch(batch):
    """Tổng thời gian chèn k điểm: gia tăng so với một lần fit lại (sklearn, DBSCAN của nhóm) và k lần fit lại sklearn (log-log)."""
    import matplotlib.pyplot as plt
    from src.visualization.plotter import report_style
    with report_style():
        fig, ax = plt.subplots(figsize=(5.6, 3.6))
        for col, lab, style in (("incremental_total_ms", "Gia tăng: k lần insert", "o-"), ("own_refit_once_ms", "DBSCAN của nhóm: fit lại 1 lần", "s--"),
                                ("sklearn_refit_once_ms", "sklearn: fit lại 1 lần", "^--"), ("sklearn_refit_each_ms", "sklearn: fit lại sau mỗi điểm", "v:")):
            ax.plot(batch["k"], batch[col], style, label=lab, markersize=4, linewidth=1.2)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(list(batch["k"]))
        ax.set_xticklabels([str(v) for v in batch["k"]])
        ax.minorticks_off()
        ax.set_xlabel("số điểm chèn liên tiếp k (D0 = 1000 giao dịch ATM, 2 chiều)")
        ax.set_ylabel("tổng thời gian (ms, log)")
        ax.set_title("Chèn theo lô: tổng thời gian để có kết quả sau k điểm mới", fontsize=8.5, loc="left")
        ax.legend(fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.93))
        fig.tight_layout()
    return fig


def plot_update_pair(ex, title, panel_w=3.3, axis_labels=("x (km)", "y (km)")):
    """Trước và sau của một update: ex là dict before, after, update, seed_ids, eps (như stream_example hoặc run_scenario).
    Kích thước hình theo tỉ lệ khung dữ liệu nên không thừa lề."""
    import matplotlib.pyplot as plt
    from src.visualization import ClusterVisualizer
    from src.visualization.plotter import report_style
    xy = np.vstack([ex["before"]["coords"], ex["after"]["coords"]])
    dx, dy = np.ptp(xy[:, 0]) + 2 * ex["eps"], np.ptp(xy[:, 1]) + 2 * ex["eps"]
    h = float(np.clip(panel_w * max(dy / dx, 0.5), 1.5, 4.4))       # plot_before_after giữ khung cao ít nhất bằng nửa bề rộng
    with report_style():
        fig, axes = plt.subplots(1, 2, figsize=(2 * panel_w + 0.9, h + 1.2))
        fig.subplots_adjust(left=0.09, right=0.98, top=1 - 0.5 / (h + 1.2), wspace=0.28)
        ClusterVisualizer(point_size=34).plot_before_after(ex["before"], ex["after"], ex["update"], eps=ex["eps"], seed_ids=ex["seed_ids"],
                                                           axes=axes, suptitle=title, axis_labels=axis_labels, fill=None)
        fig.subplots_adjust(bottom=0.6 / (h + 1.2))
    return fig


def plot_batch_map(xy, n0=1000, k=20, eps=0.5, min_pts=4):
    """Một panel: bản đồ ATM sau khi chèn k giao dịch kế tiếp vào D0 = n0 giao dịch; sao đen là điểm mới, tiêu đề đếm số điểm mới vào cụm."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from src.visualization import ClusterVisualizer
    from src.visualization.plotter import NOISE_COLOR, report_style
    eng = IncrementalEngine(eps, min_pts).fit(xy[:n0])
    c0 = eng.n_clusters
    new_ids = [eng.insert(p) for p in xy[n0:n0 + k]]
    snap = eng.snapshot()
    lab = dict(zip(snap["ids"].tolist(), snap["labels"].tolist()))
    joined = sum(1 for i in new_ids if lab[i] >= 0)
    with report_style():
        fig, ax = plt.subplots(figsize=(6.4, 5.6))
        ClusterVisualizer(point_size=13).plot_2d(snap, ax=ax, fill=None, legend=False, axis_labels=("x (km)", "y (km)"),
                                                 title=f"Chèn {k} giao dịch vào D0 = {n0}: {joined} điểm vào cụm, {k - joined} điểm nhiễu; "
                                                       f"số cụm {c0} → {eng.n_clusters}")
        pts = xy[n0:n0 + k]
        ax.scatter(pts[:, 0], pts[:, 1], marker="*", s=95, c="#111111", edgecolors="white", linewidths=0.6, zorder=8)
        ink = {"markerfacecolor": "#BBBBBB", "markeredgecolor": "#222222", "markeredgewidth": 0.5, "linestyle": ""}
        handles = [Line2D([], [], marker="o", markersize=5, label="Điểm lõi", **ink), Line2D([], [], marker="^", markersize=5, label="Điểm biên", **ink),
                   Line2D([], [], marker="x", color=NOISE_COLOR, markersize=5, linestyle="", label="Nhiễu"),
                   Line2D([], [], marker="*", markersize=10, markerfacecolor="#111111", markeredgecolor="white", linestyle="", label=f"{k} điểm vừa chèn")]
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=4, frameon=False, fontsize=7.5)
        ax.title.set_fontsize(8.5)
        fig.tight_layout()
    return fig


def _step_panels(title_rows, draw_rows, legend_items, suptitle, height=6.6):
    """Khung 2 x 2 cho hình từng bước: mỗi panel một bước, chú giải chung ở dưới."""
    import matplotlib.pyplot as plt
    from src.visualization.plotter import report_style
    with report_style():
        fig, axes = plt.subplots(2, 2, figsize=(7.4, height))
        for ax, title, draw in zip(axes.ravel(), title_rows, draw_rows):
            draw(ax)
            ax.set_title(title, fontsize=7.4, loc="left")
        fig.legend(handles=legend_items, loc="lower center", ncol=4, frameon=False, fontsize=6.8, bbox_to_anchor=(0.5, 0.0))
        fig.suptitle(suptitle, fontsize=9.5, fontweight="bold")
        fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    return fig


def _step_legend(extra, noise=True):
    from matplotlib.lines import Line2D
    from src.visualization.plotter import INK, NOISE_COLOR
    ink = {"markerfacecolor": "#BBBBBB", "markeredgecolor": INK, "markeredgewidth": 0.5, "linestyle": ""}
    base = [Line2D([], [], marker="o", markersize=5, label="Điểm lõi", **ink), Line2D([], [], marker="^", markersize=5, label="Điểm biên", **ink),
            Line2D([], [], marker="x", color=NOISE_COLOR, markersize=5, linestyle="", label="Nhiễu"),
            Line2D([], [], color=INK, linestyle=(0, (3, 2)), linewidth=0.9, label="Vòng Eps quanh p")]
    return [h for h in base if noise or h.get_label() != "Nhiễu"] + extra


def _ring(ax, xy, color, size=150, lw=1.5, style="-"):
    ax.scatter(xy[:, 0], xy[:, 1], s=size, facecolors="none", edgecolors=color, linewidths=lw, linestyle=style, zorder=6)


def plot_insert_steps(name="r02_4_merge", engine=IncrementalEngine):
    """Ví dụ từng bước của phép chèn (mục 1.2 của yêu cầu): trước, lân cận Eps của p, lõi mới và UpdSeed, kết quả. Chạy trên engine thật."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle
    from src.visualization import ClusterVisualizer, make_snapshot
    from src.visualization.plotter import INK
    sc = load_scenario("toy", name)
    run = run_scenario(engine, sc)
    b, a, eps = run["before"], run["after"], sc["eps"]
    u = sc["updates"][0]
    pxy = np.asarray(u["coords"], float)
    seeds, new_core = update_seeds(b, a, "insert", eps)
    ids_a = a["ids"].tolist()
    nb = [i for i, c in zip(ids_a, a["coords"]) if np.linalg.norm(c - pxy) <= eps and i != u["point_id"]]
    xy_of = dict(zip(ids_a, a["coords"]))
    lim_pts = np.vstack([a["coords"], pxy - eps, pxy + eps])
    pad = 0.4
    limits = ((lim_pts[:, 0].min() - pad, lim_pts[:, 0].max() + pad), (lim_pts[:, 1].min() - pad, lim_pts[:, 1].max() + pad))
    vis = ClusterVisualizer(point_size=34)
    lab_b = dict(zip(b["ids"].tolist(), b["labels"].tolist()))
    lab_a = dict(zip(ids_a, a["labels"].tolist()))
    mixed = make_snapshot(ids_a, a["coords"], [-1 if lab_a[i] == -1 else lab_b.get(i, 0) for i in ids_a], a["point_types"])   # loại điểm sau, màu cụm trước
    old_names = [i for i in ids_a if i != u["point_id"]]

    def star(ax, filled=True):
        ax.scatter([pxy[0]], [pxy[1]], marker="*", s=230, facecolors=INK if filled else "none", edgecolors="white" if filled else INK,
                   linewidths=0.8 if filled else 1.1, zorder=8)
        ax.annotate("p", pxy, xytext=(5, 5), textcoords="offset points", fontsize=8, fontweight="bold", zorder=9, path_effects=_HALO)

    def eps_ring(ax):
        ax.add_patch(Circle(pxy, eps, fill=False, linestyle=(0, (3, 2)), edgecolor=INK, linewidth=0.9, zorder=5))

    def p0(ax):
        vis.plot_2d(b, ax=ax, legend=False, fill=None, annotate_ids=b["ids"].tolist(), limits=limits)
        star(ax, filled=False)

    def p1(ax):
        vis.plot_2d(b, ax=ax, legend=False, fill=None, limits=limits)
        star(ax)
        eps_ring(ax)
        _ring(ax, np.array([xy_of[i] for i in nb]), "#E69F00", lw=1.8)
        for i in nb:
            ax.annotate(i, xy_of[i], xytext=(4, -10), textcoords="offset points", fontsize=6.5, path_effects=_HALO)

    def p2(ax):
        vis.plot_2d(mixed, ax=ax, legend=False, fill=None, seed_ids=sorted(seeds), limits=limits)
        star(ax)
        eps_ring(ax)
        _ring(ax, np.array([xy_of[i] for i in sorted(new_core)]), "#D55E00", size=260, lw=1.6, style="--")

    def p3(ax):
        vis.plot_2d(a, ax=ax, legend=False, fill=None, limits=limits)
        star(ax)

    k_old = len(set(lab_b[i] for i in seeds if i in lab_b and i not in new_core))
    titles = [f"Bước 0. Trước khi chèn: {len(set(b['labels'].tolist()) - {-1})} cụm, Eps = {eps:g}, MinPts = {sc['min_pts']}",
              f"Bước 1. Chèn p, tìm N_Eps(p) = {{{', '.join(sorted(nb))}, p}}",
              f"Bước 2. Vừa thành lõi: {', '.join(sorted(new_core))}\nUpdSeed = lõi nằm trong Eps của lõi mới ({len(seeds)} điểm)",
              f"Bước 3. UpdSeed có lõi của {k_old} cụm cũ: Merge thành {len(set(a['labels'].tolist()) - {-1})} cụm"]
    legend = _step_legend([Line2D([], [], marker="*", markersize=10, markerfacecolor=INK, markeredgecolor="white", linestyle="", label="Điểm chèn p"),
                           Line2D([], [], marker="o", markersize=9, markerfacecolor="none", markeredgecolor="#E69F00", markeredgewidth=1.8,
                                  linestyle="", label="Láng giềng của p"),
                           Line2D([], [], marker="o", markersize=11, markerfacecolor="none", markeredgecolor="#D55E00", markeredgewidth=1.4,
                                  linestyle="", label="Vừa thành lõi (vòng nét đứt)"),
                           Line2D([], [], marker="o", markersize=8, markerfacecolor="none", markeredgecolor=INK, markeredgewidth=1.3,
                                  linestyle="", label="UpdSeed")], noise=bool((b["labels"] == -1).any() or (a["labels"] == -1).any()))
    return _step_panels(titles, [p0, p1, p2, p3], legend, f"Ví dụ từng bước: chèn một điểm làm hai cụm hợp nhất (Merge), kịch bản {name}",
                        height=5.3)


def plot_delete_steps(name="r03_3a_split", engine=IncrementalEngine):
    """Ví dụ từng bước của phép xóa: trước, điểm mất tính lõi và UpdSeed, kiểm tra liên thông theo chuỗi lõi, kết quả. Chạy trên engine thật."""
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle
    from src.visualization import ClusterVisualizer, make_snapshot
    from src.visualization.plotter import INK, PALETTE
    sc = load_scenario("toy", name)
    run = run_scenario(engine, sc)
    b, a, eps = run["before"], run["after"], sc["eps"]
    u = sc["updates"][0]
    ids_b = b["ids"].tolist()
    pxy = b["coords"][ids_b.index(u["point_id"])]
    seeds, lost = update_seeds(b, a, "delete", eps)
    ids_a = a["ids"].tolist()
    xy_of = dict(zip(ids_a, a["coords"]))
    lab_b = dict(zip(ids_b, b["labels"].tolist()))
    mixed = make_snapshot(ids_a, a["coords"], [-1 if l == -1 else lab_b[i] for i, l in zip(ids_a, a["labels"].tolist())], a["point_types"])
    lim_pts = np.vstack([b["coords"], pxy - eps, pxy + eps])
    pad = 0.35
    limits = ((lim_pts[:, 0].min() - pad, lim_pts[:, 0].max() + pad), (lim_pts[:, 1].min() - pad, lim_pts[:, 1].max() + pad))
    vis = ClusterVisualizer(point_size=34)
    cores = [i for i, t in zip(ids_a, a["point_types"].tolist()) if t == 3]
    lab_a = dict(zip(ids_a, a["labels"].tolist()))
    edges = [(i, j) for k, i in enumerate(cores) for j in cores[k + 1:] if np.linalg.norm(xy_of[i] - xy_of[j]) <= eps]

    def xmark(ax, filled=True):
        ax.scatter([pxy[0]], [pxy[1]], marker="X", s=140, facecolors=INK if filled else "none", edgecolors="white" if filled else INK,
                   linewidths=0.8 if filled else 1.1, zorder=8)
        ax.annotate("p", pxy, xytext=(5, 5), textcoords="offset points", fontsize=8, fontweight="bold", zorder=9, path_effects=_HALO)

    def eps_ring(ax):
        ax.add_patch(Circle(pxy, eps, fill=False, linestyle=(0, (3, 2)), edgecolor=INK, linewidth=0.9, zorder=5))

    def p0(ax):
        vis.plot_2d(b, ax=ax, legend=False, fill=None, annotate_ids=[i for i in ids_b if i != u["point_id"]], limits=limits)
        xmark(ax)
        eps_ring(ax)

    def p1(ax):
        vis.plot_2d(mixed, ax=ax, legend=False, fill=None, seed_ids=sorted(seeds), annotate_ids=sorted(seeds), limits=limits)
        xmark(ax, filled=False)
        eps_ring(ax)

    def p2(ax):
        vis.plot_2d(a, ax=ax, legend=False, fill=None, seed_ids=sorted(seeds), annotate_ids=sorted(seeds), limits=limits)
        segs = [[xy_of[i], xy_of[j]] for i, j in edges]
        cols = [PALETTE[lab_a[i] % len(PALETTE)] for i, _ in edges]
        ax.add_collection(LineCollection(segs, colors=cols, linewidths=1.6, zorder=2))
        xmark(ax, filled=False)

    def p3(ax):
        vis.plot_2d(a, ax=ax, legend=False, fill=None, annotate_ids=[i for i in ids_a if lab_a[i] == -1], limits=limits)
        xmark(ax, filled=False)

    noise_after = [i for i in ids_a if lab_a[i] == -1]
    titles = [f"Bước 0. Trước khi xóa: {len(set(b['labels'].tolist()) - {-1})} cụm, Eps = {eps:g}, MinPts = {sc['min_pts']}",
              f"Bước 1. Xóa p. Mất tính lõi: {', '.join(sorted(lost))}\nUpdSeed = lõi còn lại trong Eps của p: {{{', '.join(sorted(seeds))}}}",
              f"Bước 2. Đi theo cạnh lõi–lõi (khoảng cách ≤ Eps) từ {' và '.join(sorted(seeds))}:\nkhông đến được nhau, nên cụm bị tách",
              f"Bước 3. Split: còn {len(set(a['labels'].tolist()) - {-1})} cụm; " + (f"{', '.join(noise_after)} thành nhiễu" if noise_after else "không có nhiễu")]
    legend = _step_legend([Line2D([], [], marker="X", markersize=8, markerfacecolor=INK, markeredgecolor="white", linestyle="", label="Điểm bị xóa p"),
                           Line2D([], [], marker="X", markersize=8, markerfacecolor="none", markeredgecolor=INK, linestyle="", label="Vị trí p đã xóa"),
                           Line2D([], [], marker="o", markersize=8, markerfacecolor="none", markeredgecolor=INK, markeredgewidth=1.3,
                                  linestyle="", label="UpdSeed"),
                           Line2D([], [], color=PALETTE[0], linewidth=1.6, label="Cạnh lõi–lõi")])
    return _step_panels(titles, [p0, p1, p2, p3], legend, f"Ví dụ từng bước: xóa một điểm làm cụm tách đôi (Split), kịch bản {name}")


def save_figures(summary, batch, k=20):
    """Ghi các hình D05, D06 vào report/figures/ từ bảng case_summary và batch_insert (đọc lại CSV được, không cần chạy lại thí nghiệm)."""
    from src.visualization import ClusterVisualizer
    vis = ClusterVisualizer()
    vis.save(vis.plot_case_gallery(insert_gallery(), n_pair_cols=2, fill=None), FIG / "d05_insert_cases.png")
    vis.save(plot_insert_steps(), FIG / "d05_steps_insert_merge.png")
    vis.save(plot_delete_steps(), FIG / "d06_steps_delete_split.png")
    vis.save(vis.plot_case_gallery(delete_gallery(), n_pair_cols=2, fill=None), FIG / "d06_delete_cases.png")
    for stream in STREAMS:
        vis.save(plot_case_times(summary, stream), FIG / f"d05_d06_case_times_{stream}.png")
    vis.save(plot_batch(batch), FIG / "d05_batch_insert_time.png")
    atm = pd.read_csv(RAW / "bank_geo" / "atm_txn_2d.csv", encoding="utf-8").sort_values(["timestamp", "txn_id"])
    vis.save(plot_batch_map(atm[["x_km", "y_km"]].to_numpy(float), k=k), FIG / "d05_batch_insert_map.png")
    for case, fname in (("PotentialSplit_Split", "d06_stream_split_example.png"), ("Merge", "d05_stream_merge_example.png")):
        ex = stream_example(RAW / "bank_geo" / "atm_stream", case)
        label = "Split: xóa một điểm, cụm tách đôi" if case.endswith("Split") else "Merge: chèn một điểm, hai cụm hợp nhất"
        vis.save(plot_update_pair(ex, f"Luồng ATM, step {ex['step']}. {label}"), FIG / fname)


# ------------------------------------------------------------------ chạy tất cả
def main(argv=None):
    import argparse
    import platform
    import matplotlib
    matplotlib.use("Agg")
    ap = argparse.ArgumentParser(description="D05, D06: thí nghiệm chèn và xóa")
    ap.add_argument("--repeats", type=int, default=REPEATS)
    ap.add_argument("--no-own-refit-cards", action="store_true", help="bỏ OwnRefitDBSCAN ở hai stream thẻ (mỗi stream thêm vài phút)")
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)

    sc = scenario_table(repeats=a.repeats)
    sc.to_csv(OUT / "scenario_steps.csv", index=False, encoding="utf-8", lineterminator="\n")
    print(f"kịch bản: {len(sc)} bước, khớp đáp án {sc['matches_expected'].sum()}, tương đương sklearn {sc['equivalent_sklearn'].sum()}, "
          f"UpdSeed đúng {sc['updseed_ok'].sum()}", flush=True)

    steps, fits = [], []
    for name in STREAMS:
        st, fit = stream_table(name, repeats=a.repeats, own_refit=(name == "atm_stream" or not a.no_own_refit_cards))
        steps.append(st)
        fits.append(fit)
        print(f"{name}: {len(st)} step, tương đương sklearn {st['equivalent_sklearn'].sum()}/{len(st)}, "
              f"trung vị {st['incremental_ms'].median():.2f} ms (gia tăng) và {st['sklearn_refit_ms'].median():.2f} ms (sklearn)", flush=True)
    steps = pd.concat(steps, ignore_index=True)
    steps.to_csv(OUT / "stream_steps.csv", index=False, encoding="utf-8", lineterminator="\n")
    pd.DataFrame(fits).to_csv(OUT / "stream_fit.csv", index=False, encoding="utf-8", lineterminator="\n")
    summ = case_summary(steps)
    summ.to_csv(OUT / "case_summary.csv", index=False, encoding="utf-8", lineterminator="\n")

    batch, _ = batch_insert(repeats=a.repeats)
    batch.to_csv(OUT / "batch_insert.csv", index=False, encoding="utf-8", lineterminator="\n")
    print(batch.round(2).to_string(index=False), flush=True)

    save_figures(summ, batch)
    import sklearn
    (OUT / "README.txt").write_text(
        "Bảng của D05 và D06 (python -m baseline.update_experiments). Cột thời gian phụ thuộc máy, các cột còn lại tất định.\n"
        f"Máy: {platform.platform()}, Python {platform.python_version()}, scikit-learn {sklearn.__version__}, numpy {np.__version__}.\n"
        f"Mỗi update đo {a.repeats} lần trên bản sao sâu, lấy trung vị (own_refit trên stream đo một lần).\n", encoding="utf-8", newline="\n")
    print(json.dumps({"out": str(OUT), "figures": str(FIG)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
