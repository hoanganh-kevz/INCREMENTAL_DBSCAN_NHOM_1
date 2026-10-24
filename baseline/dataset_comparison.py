"""Mục 2.2 của yêu cầu đồ án: IncrementalDBSCAN của nhóm so với DBSCAN của sklearn trên tám bộ dữ liệu.

Mỗi bộ: xáo thứ tự (seed 0), fit 60% đầu, chèn từng điểm của 40% còn lại, xóa từng điểm của 20% chọn ngẫu nhiên, rồi chèn lại chính
các điểm đó (id mới), nên tập cuối là toàn bộ dữ liệu và ARI so được với một lần fit cả bộ. Sau MỖI update,
trạng thái của bản gia tăng được so với sklearn fit lại trên đúng tập điểm hiện có (snapshots_equivalent: cùng loại điểm, cùng cách chia
lõi, điểm biên chỉ được khác cụm khi mơ hồ). Cuối cùng đo ARI của cả hai so với nhãn sinh của bộ dữ liệu.

Eps và MinPts: các bộ ATM và thẻ dùng tham số của D08 (baseline/static_equivalence.py); moons và blobs_noise lấy Eps ở khuỷu k-distance với
MinPts 8; rings lấy Eps 0.30, giữa vùng ổn định 0.28 đến 0.40 (khuỷu k-distance cho 0.24, nằm sát mép vùng đó và tách vòng ngoài thành
nhiều mảnh).

Chạy từ gốc repo: python -m baseline.dataset_comparison. Ghi data/processed/experiments/d22/ và report/figures/d22_*.png.
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from src.experiments.engine import IncrementalEngine
from .sklearn_dbscan import RefitDBSCAN
from .static_equivalence import ATM, ATM_HARD, CARD, CARD_HARD, ND_BLOBS
from .update_experiments import snapshots_equivalent

ROOT = Path(__file__).resolve().parents[1]
RAW, PROCESSED = ROOT / "data" / "raw", ROOT / "data" / "processed"
OUT = ROOT / "data" / "processed" / "experiments" / "d22"
FIG = ROOT / "report" / "figures"

# tên hiển thị, file, cột nhãn, Eps, MinPts
DATASETS = [
    ("moons", PROCESSED / "moons_n500_scaled.csv", "label", 0.21, 8),
    ("rings", PROCESSED / "rings_n500_scaled.csv", "label", 0.30, 8),
    ("blobs có nhiễu", PROCESSED / "blobs_noise_n500_scaled.csv", "label", 0.24, 8),
    ("ATM 2D", RAW / "bank_geo" / "atm_txn_2d.csv", "hotspot_id", *ATM),
    ("ATM 2D khó", RAW / "bank_geo" / "atm_txn_2d_hard.csv", "hotspot_id", *ATM_HARD),
    ("blobs 5D", PROCESSED / "blobs5d_n860_scaled.csv", "label", *ND_BLOBS),
    ("thẻ 5D", PROCESSED / "card_txn_5d_scaled.csv", "segment_id", *CARD),
    ("thẻ 5D khó", PROCESSED / "card_txn_5d_hard_scaled.csv", "segment_id", *CARD_HARD),
]


def load(path, label_col):
    """(tọa độ, nhãn sinh): cột _z nếu có (bảng đã chuẩn hóa), không thì x_km, y_km (bản đồ ATM theo km)."""
    frame = pd.read_csv(path, encoding="utf-8")
    cols = [c for c in frame.columns if c.endswith("_z")] or ["x_km", "y_km"]
    return frame[cols].to_numpy(float), frame[label_col].to_numpy()


def plan_updates(n, seed=0, d0_share=0.6, delete_share=0.2):
    """Thứ tự xáo, số điểm của D0, và danh sách update (op, chỉ số dòng) theo đúng trình tự."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    n0 = int(round(d0_share * n))
    ups = [("insert", int(i)) for i in order[n0:]]
    gone = [int(i) for i in rng.choice(order, int(round(delete_share * n)), replace=False)]
    ups += [("delete", i) for i in gone] + [("insert", i) for i in gone]
    return order, n0, ups


def run_dataset(name, path, label_col, eps, min_pts, seed=0):
    """Chạy một bộ, trả dict kết quả và (tọa độ, nhãn gia tăng cuối, nhãn sklearn cuối, chỉ số dòng còn lại) để vẽ."""
    X, y = load(path, label_col)
    order, n0, ups = plan_updates(len(X), seed)
    inc, ref = IncrementalEngine(eps, min_pts), RefitDBSCAN(eps, min_pts)
    t0 = time.perf_counter()
    inc.fit(X[order[:n0]])
    fit_s = time.perf_counter() - t0
    ref.fit(X[order[:n0]])
    row_of = dict(zip(range(n0), order[:n0].tolist()))          # id engine -> chỉ số dòng (hai engine cấp id giống nhau)
    id_of = {r: i for i, r in row_of.items()}
    t_inc, t_ref, ok, ops = [], [], 0, []
    for op, r in ups:
        if op == "insert":
            t = time.perf_counter(); eid = inc.insert(X[r]); t_inc.append(time.perf_counter() - t)
            t = time.perf_counter(); ref.insert(X[r]); t_ref.append(time.perf_counter() - t)
            row_of[eid], id_of[r] = r, eid
        else:
            eid = id_of.pop(r)
            t = time.perf_counter(); inc.delete(eid); t_inc.append(time.perf_counter() - t)
            t = time.perf_counter(); ref.delete(eid); t_ref.append(time.perf_counter() - t)
            row_of.pop(eid)
        ops.append(op)
        ok += snapshots_equivalent(inc.snapshot(), ref.snapshot(), eps)
    si, sr = inc.snapshot(), ref.snapshot()
    rows = np.array([row_of[i] for i in si["ids"].tolist()])
    t_inc, t_ref, ops = np.array(t_inc), np.array(t_ref), np.array(ops)
    t = time.perf_counter()
    IncrementalEngine(eps, min_pts).fit(X[rows])
    own_fit_s = time.perf_counter() - t
    res = {"dataset": name, "dims": X.shape[1], "n": len(X), "eps": eps, "minpts": min_pts, "n_d0": n0,
           "n_insert": int((ops == "insert").sum()), "n_delete": int((ops == "delete").sum()), "equivalent_steps": ok, "n_updates": len(ups),
           "ari_incremental_truth": adjusted_rand_score(y[rows], si["labels"]), "ari_sklearn_truth": adjusted_rand_score(y[rows], sr["labels"]),
           "clusters": len(set(si["labels"].tolist()) - {-1}), "noise": int((si["labels"] == -1).sum()),
           "insert_ms_median": np.median(t_inc[ops == "insert"]) * 1000, "delete_ms_median": np.median(t_inc[ops == "delete"]) * 1000,
           "incremental_total_s": t_inc.sum(), "sklearn_refit_each_total_s": t_ref.sum(),
           "own_refit_each_total_s_est": own_fit_s * len(ups), "initial_fit_s": fit_s}
    return res, (X[rows], si["labels"], sr["labels"], y[rows])


def plot_2d_results(results, draws):
    """Kết quả cuối của bản gia tăng trên năm bộ 2 chiều, mỗi bộ một panel; tiêu đề ghi số step khớp sklearn và ARI so với nhãn sinh."""
    import matplotlib.pyplot as plt
    from src.visualization import ClusterVisualizer, make_snapshot
    from src.visualization.plotter import report_style
    two_d = [(r, d) for r, d in zip(results, draws) if r["dims"] == 2]
    with report_style():
        fig, axes = plt.subplots(2, 3, figsize=(7.6, 6.2))
        for ax, (r, (X, lab, _, _)) in zip(axes.ravel(), two_d):
            snap = make_snapshot(range(len(X)), X, lab, None)
            unit = "km" if r["dataset"].startswith("ATM") else "z"
            vis = ClusterVisualizer(point_size=8 if len(X) <= 1200 else 4)           # bộ 2000 điểm vẽ nhỏ hơn cho đỡ dày
            vis.plot_2d(snap, ax=ax, legend=False, fill=None, axis_labels=(f"x ({unit})", f"y ({unit})"))
            ax.set_title(f"{r['dataset']} (Eps {r['eps']:g}, MinPts {r['minpts']})\n"
                         f"khớp sklearn {r['equivalent_steps']}/{r['n_updates']} update; ARI {r['ari_incremental_truth']:.2f}", fontsize=7.2)
            ax.xaxis.label.set_size(7)
            ax.yaxis.label.set_size(7)
            ax.tick_params(labelsize=6)
        ax = axes.ravel()[-1]
        ax.axis("off")
        ax.text(0.0, 0.95, "Mỗi bộ: fit 60% điểm, chèn từng điểm\ncủa 40% còn lại, xóa 20% điểm chọn\nngẫu nhiên rồi chèn lại chúng.\n"
                           "Hình là trạng thái cuối của\nIncrementalDBSCAN (đủ mọi điểm).\n\nMàu là cụm, dấu x xám là nhiễu.\n"
                           "\"khớp sklearn\": số update mà kết quả\ngiống sklearn fit lại trên cùng tập điểm.\nARI so với nhãn sinh của bộ dữ liệu.",
                transform=ax.transAxes, va="top", fontsize=7, color="#333333", linespacing=1.4)
        fig.tight_layout()
    return fig


def plot_times(table):
    """Tổng thời gian xử lý toàn bộ update của mỗi bộ (thang log): gia tăng, sklearn fit lại sau mỗi update, DBSCAN của nhóm fit lại (ước tính)."""
    import matplotlib.pyplot as plt
    from src.visualization.plotter import report_style
    t = table.reset_index(drop=True)
    with report_style():
        fig, ax = plt.subplots(figsize=(7.4, 3.4))
        x = np.arange(len(t))
        w = 0.27
        for j, (col, lab, color) in enumerate((("incremental_total_s", "IncrementalDBSCAN của nhóm", "#1f77b4"),
                                               ("own_refit_each_total_s_est", "DBSCAN của nhóm fit lại sau mỗi update (ước tính)", "#ff7f0e"),
                                               ("sklearn_refit_each_total_s", "sklearn fit lại sau mỗi update", "#7f7f7f"))):
            bars = ax.bar(x + (j - 1) * w, t[col], w, color=color, label=lab)
            if col == "incremental_total_s":
                for b, v in zip(bars, t[col]):
                    ax.annotate(f"{v:.1f}", (b.get_x() + b.get_width() / 2, v), xytext=(0, 2), textcoords="offset points", ha="center",
                                fontsize=5.8, color="#1f4e79")
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{d}\n{n} update" for d, n in zip(t["dataset"], t["n_updates"])], fontsize=6.3)
        ax.set_ylabel("tổng thời gian (giây, log)")
        ax.set_title("Thời gian xử lý toàn bộ chuỗi update trên từng bộ dữ liệu (số trên cột xanh là giây)", fontsize=8.5, loc="left")
        ax.legend(fontsize=6.8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
        fig.tight_layout()
    return fig


def main():
    import platform
    import matplotlib
    matplotlib.use("Agg")
    import sklearn
    from src.visualization import ClusterVisualizer
    OUT.mkdir(parents=True, exist_ok=True)
    results, draws = [], []
    for spec in DATASETS:
        t = time.perf_counter()
        r, d = run_dataset(*spec)
        results.append(r)
        draws.append(d)
        print(f"{r['dataset']}: khớp {r['equivalent_steps']}/{r['n_updates']}, ARI {r['ari_incremental_truth']:.3f} "
              f"(sklearn {r['ari_sklearn_truth']:.3f}), {time.perf_counter() - t:.0f} s", flush=True)
    table = pd.DataFrame(results)
    table.to_csv(OUT / "dataset_comparison.csv", index=False, encoding="utf-8", lineterminator="\n")
    ClusterVisualizer.save(plot_2d_results(results, draws), FIG / "d22_datasets_2d.png")
    ClusterVisualizer.save(plot_times(table), FIG / "d22_datasets_time.png")
    (OUT / "README.txt").write_text(
        "Mục 2.2: IncrementalDBSCAN so với sklearn trên tám bộ dữ liệu (python -m baseline.dataset_comparison). Cột thời gian phụ thuộc máy.\n"
        f"Máy: {platform.platform()}, Python {platform.python_version()}, scikit-learn {sklearn.__version__}, numpy {np.__version__}.\n"
        "Mỗi update đo một lần. own_refit_each_total_s_est = số update x một lần fit của DBSCAN của nhóm trên tập cuối (ước tính).\n",
        encoding="utf-8", newline="\n")
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
