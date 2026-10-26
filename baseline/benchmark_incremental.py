"""D09: benchmark thời gian và bộ nhớ của IncrementalDBSCAN so với fit lại toàn bộ.

Cùng máy, cùng bảng bench (data/raw/bench, ATM 2D và thẻ 5D), cùng Eps và MinPts cho mọi engine:
  incremental    IncrementalEngine (IncrementalDBSCAN của nhóm);
  own_refit      DBSCAN của nhóm fit lại sau mỗi update (đối chứng cùng ngôn ngữ: phép so "gia tăng có nhanh hơn không" dựa vào cặp này);
  sklearn_refit  DBSCAN của sklearn fit lại sau mỗi update (thư viện viết bằng C/Cython, có cây KD).

Hai thí nghiệm:
  theo n   n = 500, 1000, 2000, 4000; mỗi (family, n) chạy 20 update (10 insert xen kẽ 10 delete điểm cũ nhất, như baseline --bench);
           ghi trung vị và trung bình của từng loại op, kèm bộ nhớ giữ lại sau fit và đỉnh bộ nhớ của một insert, một delete (tracemalloc).
  theo m   n = 2000, m = 1 đến 500 update: phát lại toàn bộ chuỗi nhiều lần; báo cáo trung vị và trung bình tổng thời gian của m update
           bằng gia tăng, bằng fit lại sau mỗi update, và bằng một lần fit lại cuối.

Chạy từ gốc repo: python -m baseline.benchmark_incremental [--n 500 1000 2000 4000] [--repeats 5] [--m-n 2000]
Ghi data/processed/experiments/d09/ (thời gian phụ thuộc máy) và report/figures/d09_*.png.
"""
import copy
import json
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd

from src.experiments.engine import IncrementalEngine
from .own_refit import OwnRefitDBSCAN
from .sklearn_dbscan import BENCH_NS, REPEATS, IdMap, RefitDBSCAN, _bench_frames, _ensure_bench_tables, _timed
from src.preprocessing.stream_scaling import stream_coordinates

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "data" / "raw" / "bench"
OUT = ROOT / "data" / "processed" / "experiments" / "d09"
FIG = ROOT / "report" / "figures"

ENGINES = {"incremental": IncrementalEngine, "own_refit": OwnRefitDBSCAN, "sklearn_refit": RefitDBSCAN}
LABELS = {"incremental": "IncrementalDBSCAN của nhóm", "own_refit": "DBSCAN của nhóm, fit lại", "sklearn_refit": "sklearn, fit lại"}
COLORS = {"incremental": "#1f77b4", "own_refit": "#ff7f0e", "sklearn_refit": "#7f7f7f"}
M_VALUES = (1, 10, 50, 100, 200, 500)


def load_bench(family, n, n_updates=20):
    """(d0, updates, params, feats) của bảng bench như baseline.sklearn_dbscan._bench_frames (tạo bảng nếu thiếu)."""
    _ensure_bench_tables([n], BENCH, ROOT)
    params_n = json.loads((BENCH / f"params_n{n}.json").read_text(encoding="utf-8"))
    fname = f"atm_txn_2d_n{n}.csv" if family == "atm" else f"card_txn_5d_n{n}.csv"
    return _bench_frames(family, pd.read_csv(BENCH / fname, encoding="utf-8"), params_n, n_updates=n_updates)


def run_ops(factory, d0, upd, p, feats, repeats=REPEATS, fit_repeats=1, max_updates=None):
    """Fit D0 rồi áp dụng từng update; trả (engine sau cùng, DataFrame op, seconds, n). fit đo `fit_repeats` lần, mỗi update đo `repeats` lần trên
    bản sao sâu (trung vị). max_updates giới hạn số update được đo (dùng cho own_refit ở n lớn: mỗi update là một lần fit đầy đủ)."""
    eng = factory(p["eps"], p["minpts"])
    ids = IdMap()
    x0 = stream_coordinates(p, d0, feats)
    _, secs, _ = _timed(eng, lambda e: e.fit(x0), fit_repeats)
    for sid, eid in zip(d0["point_id"].tolist(), eng.snapshot()["ids"].tolist()):
        ids.bind(sid, eid)
    rows = [("fit", secs, len(x0))]
    xs = stream_coordinates(p, upd, feats)
    for k, (_, r) in enumerate(upd.iterrows()):
        if max_updates is not None and k >= max_updates:
            break
        if r["op"] == "insert":
            eid, secs, _ = _timed(eng, lambda e, x=xs[k]: e.insert(x), repeats)
            ids.bind(r["point_id"], eid)
        else:
            _, secs, _ = _timed(eng, lambda e, i=ids.release(r["point_id"]): e.delete(i), repeats)
        rows.append((r["op"], secs, eng.n_points))
    return eng, pd.DataFrame(rows, columns=["op", "seconds", "n"])


def memory_of(eng, d0_next, delete_id, peaks=True):
    """Bộ nhớ (KB, tracemalloc) của engine: retained = cấp phát khi sao sâu engine (xấp xỉ kích thước trạng thái giữ lại),
    insert_peak và delete_peak = đỉnh cấp phát của một insert, một delete trên bản sao. peaks=False bỏ hai đỉnh (NaN): với own_refit ở n lớn,
    mỗi update là một lần fit đầy đủ, chạy dưới tracemalloc mất vài phút."""
    tracemalloc.start()
    base = tracemalloc.get_traced_memory()[0]
    clone = copy.deepcopy(eng)
    retained = tracemalloc.get_traced_memory()[0] - base
    out = {"retained_kb": retained / 1024}
    for op, fn in (("insert", lambda e: e.insert(d0_next)), ("delete", lambda e: e.delete(delete_id))):
        if not peaks:
            out[f"{op}_peak_kb"] = np.nan
            continue
        c = copy.deepcopy(clone)
        tracemalloc.reset_peak()
        start = tracemalloc.get_traced_memory()[0]
        fn(c)
        out[f"{op}_peak_kb"] = (tracemalloc.get_traced_memory()[1] - start) / 1024
        del c
    tracemalloc.stop()
    return out


def bench_by_n(ns=BENCH_NS, repeats=REPEATS, own_max_updates=None, log=print):
    """Thí nghiệm theo n. Trả (timing, memory): timing một dòng mỗi (family, n, engine, op); memory một dòng mỗi (family, n, engine).
    own_max_updates: {n: số update đo cho own_refit}; mặc định 20 với n <= 1000, 4 với n = 2000, 2 với n lớn hơn."""
    own_max_updates = own_max_updates or {}
    timing, memory = [], []
    for n in ns:
        for family in ("atm", "card"):
            d0, upd, p, feats = load_bench(family, n)
            for name, factory in ENGINES.items():
                limit = own_max_updates.get(n, 20 if n <= 1000 else 4 if n <= 2000 else 2) if name == "own_refit" else None
                fit_rep = repeats if name == "sklearn_refit" else 1
                t0 = time.perf_counter()
                eng, ops = run_ops(factory, d0, upd, p, feats, repeats=repeats if name != "own_refit" else 1, fit_repeats=fit_rep,
                                   max_updates=limit)
                for op, g in ops.groupby("op", sort=False):
                    ms = g["seconds"] * 1000
                    timing.append({"family": family, "dims": len(feats), "n": n, "engine": name, "op": op, "count": len(g),
                                   "median_ms": ms.median(), "mean_ms": ms.mean(), "min_ms": ms.min(), "max_ms": ms.max(),
                                   "repeats": fit_rep if op == "fit" else (1 if name == "own_refit" else repeats),
                                   "eps": p["eps"], "minpts": p["minpts"]})
                nxt = stream_coordinates(p, upd.iloc[[0]], feats)[0]
                memory.append({"family": family, "dims": len(feats), "n": n, "engine": name,
                               **memory_of(eng, nxt, int(eng.snapshot()["ids"][0]), peaks=name != "own_refit" or n <= 1000)})
                log(f"{family} n={n} {name}: " + ", ".join(f"{r['op']} {r['median_ms']:.2f} ms" for r in timing[-ops['op'].nunique():])
                    + f" ({time.perf_counter() - t0:.0f} s)")
    return pd.DataFrame(timing), pd.DataFrame(memory)


def bench_by_m(n=2000, ms=M_VALUES, families=("atm", "card"), repeats=REPEATS, log=print):
    """Phát lại cùng stream độc lập repeats lần và tổng hợp thời gian theo số update m.

    Các cột thời gian update là trung vị; các cột *_mean_ms, *_std_ms, *_min_ms, *_max_ms
    mô tả độ phân tán. own_refit_each_ms_est là ước tính m lần fit đầy đủ từ thời gian fit D0.
    """
    if repeats < 1:
        raise ValueError("repeats phải lớn hơn hoặc bằng 1")
    if not ms or any(int(m) < 1 for m in ms):
        raise ValueError("ms phải chứa ít nhất một số nguyên dương")

    rows = []
    mmax = max(ms)
    for family in families:
        d0, upd, p, feats = load_bench(family, n, n_updates=mmax)
        x0 = stream_coordinates(p, d0, feats)
        xs = stream_coordinates(p, upd, feats)

        def replay_update_times(engine_template):
            """Phát lại stream từ cùng trạng thái D0; chỉ tính giờ insert/delete."""
            engine = copy.deepcopy(engine_template)
            ids = IdMap()
            for stream_id, engine_id in zip(d0["point_id"].tolist(), engine.snapshot()["ids"].tolist()):
                ids.bind(stream_id, engine_id)
            seconds = []
            for index, (_, row) in enumerate(upd.iterrows()):
                if row["op"] == "insert":
                    started = time.perf_counter()
                    engine_id = engine.insert(xs[index])
                    seconds.append(time.perf_counter() - started)
                    ids.bind(row["point_id"], engine_id)
                else:
                    engine_id = ids.release(row["point_id"])
                    started = time.perf_counter()
                    engine.delete(engine_id)
                    seconds.append(time.perf_counter() - started)
            return seconds

        incremental_base = IncrementalEngine(p["eps"], p["minpts"])
        fit_started = time.perf_counter()
        incremental_base.fit(x0)
        incremental_fit_ms = (time.perf_counter() - fit_started) * 1000
        sklearn_base = RefitDBSCAN(p["eps"], p["minpts"])
        sklearn_base.fit(x0)

        update_runs = {
            "incremental": np.asarray(
                [replay_update_times(incremental_base) for _ in range(repeats)], dtype=float
            ),
            "sklearn": np.asarray(
                [replay_update_times(sklearn_base) for _ in range(repeats)], dtype=float
            ),
        }
        # Fit của DBSCAN nhóm làm mốc ước tính cho một lần refit đầy đủ.
        own_fit_ms = incremental_fit_ms

        for m in ms:
            final = _final_coords(d0, upd.iloc[:m], p, feats)
            _, sk_once, _ = _timed(
                RefitDBSCAN(p["eps"], p["minpts"]), lambda engine: engine.fit(final), repeats
            )
            started = time.perf_counter()
            OwnRefitDBSCAN(p["eps"], p["minpts"]).fit(final)
            own_once = time.perf_counter() - started

            totals = {
                name: update_runs[name][:, :m].sum(axis=1) * 1000
                for name in ("incremental", "sklearn")
            }
            inc, sk = totals["incremental"], totals["sklearn"]
            rows.append({
                "family": family, "dims": len(feats), "n_d0": len(d0), "m": m,
                "incremental_ms": float(np.median(inc)),
                "incremental_mean_ms": float(np.mean(inc)),
                "incremental_std_ms": float(np.std(inc, ddof=1)) if repeats > 1 else 0.0,
                "incremental_min_ms": float(np.min(inc)), "incremental_max_ms": float(np.max(inc)),
                "incremental_fit_ms": incremental_fit_ms,
                "sklearn_refit_each_ms": float(np.median(sk)),
                "sklearn_refit_each_mean_ms": float(np.mean(sk)),
                "sklearn_refit_each_std_ms": float(np.std(sk, ddof=1)) if repeats > 1 else 0.0,
                "sklearn_refit_each_min_ms": float(np.min(sk)), "sklearn_refit_each_max_ms": float(np.max(sk)),
                "own_refit_each_ms_est": m * own_fit_ms,
                "sklearn_refit_once_ms": sk_once * 1000, "own_refit_once_ms": own_once * 1000,
                "repeats": repeats,
            })
            log(f"m-sweep {family} m={m}: gia tăng {rows[-1]['incremental_ms']:.1f} ms, "
                f"sklearn mỗi lần {rows[-1]['sklearn_refit_each_ms']:.1f} ms (trung vị của {repeats} lượt)")
    return pd.DataFrame(rows)


def _final_coords(d0, upd, p, feats):
    """Tọa độ của tập điểm sau khi áp dụng các dòng `upd` lên D0 (theo point_id)."""
    cur = dict(zip(d0["point_id"].tolist(), stream_coordinates(p, d0, feats)))
    xs = stream_coordinates(p, upd, feats)
    for k, (_, r) in enumerate(upd.iterrows()):
        if r["op"] == "insert":
            cur[int(r["point_id"])] = xs[k]
        else:
            cur.pop(int(r["point_id"]))
    return np.vstack(list(cur.values()))


# ------------------------------------------------------------------ hình
def plot_by_n(timing):
    """Lưới 2 x 3: hàng là ATM 2D và thẻ 5D, cột là fit, insert, delete; trục y log là trung vị ms, trục x là n."""
    import matplotlib.pyplot as plt
    from src.visualization.plotter import report_style
    with report_style():
        fig, axes = plt.subplots(2, 3, figsize=(7.4, 4.8), sharex=True)
        for r, family in enumerate(("atm", "card")):
            for c, op in enumerate(("fit", "insert", "delete")):
                ax = axes[r, c]
                for name in ENGINES:
                    g = timing[(timing["family"] == family) & (timing["op"] == op) & (timing["engine"] == name)].sort_values("n")
                    if op != "fit" or name != "own_refit":
                        ax.plot(g["n"], g["median_ms"], "o-", color=COLORS[name], label=LABELS[name], markersize=3.5, linewidth=1.2)
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_xticks(sorted(timing["n"].unique()))
                ax.set_xticklabels([str(v) for v in sorted(timing["n"].unique())], fontsize=6.5)
                ax.minorticks_off()
                if op != "fit":
                    sel = (timing["family"] == family) & (timing["op"] == op)
                    nmax = timing.loc[sel, "n"].max()
                    at = timing[sel & (timing["n"] == nmax)].set_index("engine")["median_ms"]
                    ax.text(0.03, 0.97, f"n = {nmax}: nhanh hơn fit lại\ncủa nhóm {at['own_refit'] / at['incremental']:.0f} lần;\n"
                                        f"sklearn / gia tăng = {at['sklearn_refit'] / at['incremental']:.1f}",
                            transform=ax.transAxes, va="top", fontsize=6.0, color="#1f4e79")
                name = {"fit": "fit ban đầu", "insert": "một insert", "delete": "một delete"}[op]
                ax.set_title(f"{'ATM 2D' if family == 'atm' else 'Thẻ 5D'}: {name}", fontsize=8)
                if c == 0:
                    ax.set_ylabel("ms (trung vị, log)")
                if r == 1:
                    ax.set_xlabel("n")
        h, l = axes[0, 1].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=7, bbox_to_anchor=(0.5, 0.025))
        fig.text(0.5, 0.0, "Cột fit ban đầu: bản gia tăng dùng chính DBSCAN.fit của nhóm, nên không vẽ riêng đường fit lại của nhóm.",
                 ha="center", va="bottom", fontsize=6.6, color="#444444")
        fig.tight_layout(rect=(0, 0.07, 1, 1))
    return fig


def plot_by_m(sweep):
    """Tổng thời gian xử lý m update theo m (log-log), hai panel ATM 2D và thẻ 5D."""
    import matplotlib.pyplot as plt
    from src.visualization.plotter import report_style
    lines = (("incremental_ms", "Gia tăng: m update", "o-", COLORS["incremental"]),
             ("own_refit_each_ms_est", "Nhóm: fit lại sau mỗi update (ước tính)", "s:", COLORS["own_refit"]),
             ("own_refit_once_ms", "Nhóm: fit lại 1 lần cuối", "s--", COLORS["own_refit"]),
             ("sklearn_refit_each_ms", "sklearn: fit lại sau mỗi update", "^:", COLORS["sklearn_refit"]),
             ("sklearn_refit_once_ms", "sklearn: fit lại 1 lần cuối", "^--", COLORS["sklearn_refit"]))
    with report_style():
        fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.4))
        for ax, family in zip(axes, ("atm", "card")):
            g = sweep[sweep["family"] == family]
            for col, lab, style, color in lines:
                ax.plot(g["m"], g[col], style, color=color, label=lab, markersize=3.5, linewidth=1.1)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xticks(list(g["m"]))
            ax.set_xticklabels([str(v) for v in g["m"]], fontsize=6.5)
            ax.minorticks_off()
            ax.set_xlabel("số update m")
            ax.set_title(f"{'ATM 2D' if family == 'atm' else 'Thẻ 5D'}, D0 = {int(g['n_d0'].iloc[0])} điểm", fontsize=8)
        axes[0].set_ylabel("tổng ms (log)")
        h, l = axes[0].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=6.6, bbox_to_anchor=(0.5, -0.02))
        fig.tight_layout(rect=(0, 0.12, 1, 1))
    return fig


def plot_memory(memory):
    """Bộ nhớ giữ lại sau fit (KB, log) theo n, hai panel ATM 2D và thẻ 5D."""
    import matplotlib.pyplot as plt
    from src.visualization.plotter import report_style
    with report_style():
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True)
        for ax, family in zip(axes, ("atm", "card")):
            for name, lab in (("incremental", LABELS["incremental"] + " (giữ cả tập láng giềng của mọi điểm)"),
                              ("sklearn_refit", "Fit lại, của nhóm hay sklearn (chỉ giữ tọa độ, hai đường trùng nhau)")):
                g = memory[(memory["family"] == family) & (memory["engine"] == name)].sort_values("n")
                ax.plot(g["n"], g["retained_kb"], "o-", color=COLORS[name], label=lab, markersize=3.5, linewidth=1.2)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xticks(sorted(memory["n"].unique()))
            ax.set_xticklabels([str(v) for v in sorted(memory["n"].unique())], fontsize=6.5)
            ax.minorticks_off()
            ax.set_xlabel("n")
            ax.set_title(f"{'ATM 2D' if family == 'atm' else 'Thẻ 5D'}: bộ nhớ giữ lại sau fit", fontsize=8)
        axes[0].set_ylabel("KB (log)")
        h, l = axes[0].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=1, frameon=False, fontsize=7, bbox_to_anchor=(0.5, 0.0))
        fig.tight_layout(rect=(0, 0.14, 1, 1))
    return fig


def main(argv=None):
    import argparse
    import platform
    import matplotlib
    matplotlib.use("Agg")
    import sklearn
    from src.visualization import ClusterVisualizer
    ap = argparse.ArgumentParser(description="D09: benchmark thời gian và bộ nhớ")
    ap.add_argument("--n", type=int, nargs="+", default=list(BENCH_NS))
    ap.add_argument("--repeats", type=int, default=REPEATS)
    ap.add_argument("--m-n", type=int, default=2000, help="n của thí nghiệm theo m")
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    timing, memory = bench_by_n(a.n, repeats=a.repeats, log=lambda s: print(s, flush=True))
    timing.to_csv(OUT / "bench_by_n.csv", index=False, encoding="utf-8", lineterminator="\n")
    memory.to_csv(OUT / "memory_by_n.csv", index=False, encoding="utf-8", lineterminator="\n")
    sweep = bench_by_m(a.m_n, repeats=a.repeats, log=lambda s: print(s, flush=True))
    sweep.to_csv(OUT / "bench_by_m.csv", index=False, encoding="utf-8", lineterminator="\n")
    vis = ClusterVisualizer()
    vis.save(plot_by_n(timing), FIG / "d09_runtime_by_n.png")
    vis.save(plot_by_m(sweep), FIG / "d09_cumulative_by_m.png")
    vis.save(plot_memory(memory), FIG / "d09_memory_by_n.png")
    (OUT / "README.txt").write_text(
        "Benchmark D09 (python -m baseline.benchmark_incremental). Cột thời gian phụ thuộc máy.\n"
        f"Máy: {platform.platform()}, {platform.processor() or platform.machine()}, Python {platform.python_version()}, "
        f"scikit-learn {sklearn.__version__}, numpy {np.__version__}.\n"
        f"Theo n: 20 update mỗi (family, n); incremental và sklearn_refit đo mỗi update {a.repeats} lần trên bản sao sâu (trung vị), "
        "own_refit đo một lần và chỉ vài update ở n lớn (mỗi update là một lần fit đầy đủ); fit incremental đo một lần, fit sklearn lặp lại.\n"
        "Bộ nhớ: tracemalloc; retained_kb là cấp phát khi sao sâu engine, insert_peak_kb và delete_peak_kb là đỉnh của một update.\n"
        f"Theo m: n = {a.m_n}, phát lại chuỗi {a.repeats} lần độc lập; báo cáo trung vị, trung bình, độ lệch chuẩn, nhỏ nhất và lớn nhất "
        "của tổng thời gian m update. own_refit_each_ms_est là ước tính m x thời gian một lần fit đầy đủ của nhóm.\n",
        encoding="utf-8", newline="\n")
    print(timing.round(3).to_string(index=False))
    print(memory.round(1).to_string(index=False))
    print(sweep.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
