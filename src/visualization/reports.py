"""Hình cho báo cáo nhận DataFrame hoặc mảng (D07, dùng lại cho D09 và D10). Chỉ numpy, pandas, matplotlib. Mọi hàm trả Figure.

plot_equivalence_timeline   ARI từng step so với re-fit toàn bộ (kỳ vọng phẳng ở 1,0), đánh dấu step bị lệch
plot_runtime_comparison     thời gian mỗi update: incremental so với re-fit
plot_affected_points        số điểm bị ảnh hưởng mỗi update
plot_algorithm_contrast     KMeans, DBSCAN, HAC (hoặc thuật toán khác) cạnh nhãn thật
plot_k_distance             đồ thị k-distance chọn Eps, khoảng cách tính bằng numpy
plot_scaling_effect         trước và sau chuẩn hóa
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter

from .plotter import ClusterVisualizer, INK, NOISE_COLOR, PALETTE, align_labels, make_snapshot, report_style

BLUE, ORANGE, RED = PALETTE[0], PALETTE[1], PALETTE[3]
_PLAIN = FuncFormatter(lambda v, _: f"{v:g}")


def _plain_log_ticks(axis):
    """Nhãn trục log dạng số thường (0,9; 1; 10) thay vì 10^k. Khi trục trải dưới 30 lần, tick phụ có nhãn ở các mốc 2 và 5
    (dưới 10 lần thì thêm 3), để trục luôn có ít nhất hai số mà không chồng chữ. Gọi sau khi đã vẽ dữ liệu."""
    axis.set_major_formatter(_PLAIN)
    axis.axes.autoscale_view()                  # khoảng trục chỉ được cập nhật khi vẽ, nên co giãn trước khi đọc
    lo, hi = sorted(abs(v) for v in axis.get_view_interval())
    if lo <= 0 or hi / lo >= 30:
        axis.set_minor_formatter(NullFormatter())
        return
    keep = {2, 3, 5} if hi / lo < 10 else {2, 5}
    axis.set_minor_formatter(FuncFormatter(lambda v, _: f"{v:g}" if round(v / 10 ** np.floor(np.log10(v))) in keep else ""))


def find_dips(results, tol=1e-9):
    """Các dòng của `results` có ari < 1 - tol (step bị lệch so với re-fit). Cột series (nếu có) được giữ."""
    return results[results["ari"] < 1.0 - tol].copy()


def plot_equivalence_timeline(results, tol=1e-9, title=None, ax=None):
    """ARI từng step giữa thuật toán đang kiểm tra và re-fit toàn bộ (đã bỏ điểm biên mơ hồ). Kỳ vọng: đường phẳng ở 1,0.

    results: DataFrame cột step, ari; thêm cột series nếu muốn vẽ nhiều đường (mỗi giá trị một đường). Step có ari < 1 - tol được
    khoanh tròn đỏ và ghi số step (tối đa 12 nhãn). Trả Figure."""
    df = results.copy()
    if "series" not in df.columns:
        df["series"] = "ARI so với fit lại"
    with report_style():
        if ax is None:
            fig, ax = plt.subplots(figsize=(7.0, 3.2))
        else:
            fig = ax.figure
        ax.axhline(1.0, color=INK, linestyle=(0, (4, 3)), linewidth=0.9, label="Kỳ vọng: ARI = 1")
        for k, (name, g) in enumerate(df.groupby("series", sort=False)):
            g = g.sort_values("step")
            ax.plot(g["step"], g["ari"], color=PALETTE[k % len(PALETTE)], linewidth=1.4, marker="o", markersize=2.6, label=str(name), zorder=3)
        dips = find_dips(df, tol)
        if len(dips):
            ax.scatter(dips["step"], dips["ari"], s=60, facecolors="none", edgecolors=RED, linewidths=1.4, zorder=5,
                       label=f"Step bị lệch ({len(dips)})")
            for k, (_, r) in enumerate(dips.sort_values("step").head(12).iterrows()):
                ax.annotate(f"{int(r['step'])}", (r["step"], r["ari"]), xytext=(-12 if k % 2 == 0 else 5, -11), textcoords="offset points",
                            fontsize=6.5, color=RED)
        lo = min(df["ari"].min(), 1.0)
        ax.set_ylim((0.95 if lo >= 0.95 else lo - 0.04), 1.015 if lo >= 0.95 else 1.03)
        ax.set_xlabel("Step cập nhật (chèn hoặc xóa một điểm)")
        ax.set_ylabel("ARI so với fit lại toàn bộ")
        ax.set_title(title or "Tính tương đương với fit lại theo từng step (đã bỏ điểm biên mơ hồ)", loc="left")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), frameon=False, ncol=2)
        ax.margins(x=0.01)
        fig.tight_layout()
    return fig


def plot_runtime_comparison(timing_incremental, timing_refit, labels=("Incremental DBSCAN", "Re-fit toàn bộ (sklearn)"), logy=True,
                            ratio_label="re-fit / incremental"):
    """Hai bảng thời gian dạng baseline (cột step, op, update_seconds, tùy chọn n_points). Bỏ step 0 (lần fit đầu).

    Trái: thời gian mỗi update theo step (mili giây). Phải: trung vị theo loại update (insert, delete) và tỉ số trung vị bảng thứ
    hai / bảng thứ nhất, đặt tên bằng ratio_label. Trả Figure."""
    series = [(labels[0], timing_incremental, BLUE), (labels[1], timing_refit, ORANGE)]
    clean = [(n, t[t["step"] >= 1].copy(), c) for n, t, c in series]
    with report_style():
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.1), gridspec_kw={"width_ratios": [2.1, 1]})
        (n0, t0, c0), (n1, t1, c1) = clean
        a1.plot(t1["step"], t1["update_seconds"] * 1000, color=c1, linewidth=1.0, label=f"{n1} (trung vị {t1['update_seconds'].median() * 1000:.2f} ms)")
        for op, mk in (("insert", "o"), ("delete", "x")):
            g = t0[t0["op"] == op]
            if len(g):
                a1.scatter(g["step"], g["update_seconds"] * 1000, marker=mk, s=12, color=c0, linewidths=1.0, zorder=3,
                           label=f"{n0}: {op} (trung vị {g['update_seconds'].median() * 1000:.2f} ms)")
        if logy:
            a1.set_yscale("log")
            _plain_log_ticks(a1.yaxis)
        a1.set_xlabel("Step cập nhật")
        a1.set_ylabel("Thời gian mỗi update (ms)")
        a1.set_title("Thời gian từng update", loc="left")
        a1.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=1, fontsize=6.8)
        ops = sorted(set(clean[0][1]["op"]) | set(clean[1][1]["op"]))
        w = 0.36
        for k, (n, t, c) in enumerate(clean):
            med = [t[t["op"] == o]["update_seconds"].median() * 1000 for o in ops]
            bars = a2.bar(np.arange(len(ops)) + (k - 0.5) * w, med, w, color=c, label=n)
            for b, m in zip(bars, med):
                a2.annotate(f"{m:.2f}", (b.get_x() + b.get_width() / 2, m), xytext=(0, 2), textcoords="offset points", ha="center", fontsize=6.5)
        a2.set_xticks(np.arange(len(ops)))
        a2.set_xticklabels(ops)
        a2.set_ylabel("Trung vị (ms)")
        ratios = []
        for o in ops:
            r = clean[1][1][clean[1][1]["op"] == o]["update_seconds"].median() / max(clean[0][1][clean[0][1]["op"] == o]["update_seconds"].median(), 1e-12)
            ratios.append(f"{o} {r:.2f}" if r < 10 else f"{o} {r:.0f}")
        a2.set_title(f"Trung vị theo loại\n{ratio_label}: " + ", ".join(ratios) + " lần", loc="left", fontsize=7.5)
        fig.tight_layout()
    return fig


def plot_affected_points(counts, title=None):
    """Số điểm bị ảnh hưởng mỗi update. counts: DataFrame cột step, n_affected; tùy chọn op (tô màu theo insert/delete) và
    n_points (đường nét đứt: re-fit luôn chạm toàn bộ n điểm). Trục tung symlog để thấy cả 0 lẫn vài trăm. Trả Figure."""
    with report_style():
        fig, ax = plt.subplots(figsize=(7.0, 3.2))
        colors = {"insert": BLUE, "delete": ORANGE}
        if "op" in counts.columns:
            for op, g in counts.groupby("op"):
                ax.bar(g["step"], g["n_affected"], color=colors.get(op, NOISE_COLOR), width=0.85, label=f"{op} ({len(g)} step, trung bình {g['n_affected'].mean():.1f})")
        else:
            ax.bar(counts["step"], counts["n_affected"], color=BLUE, width=0.85, label="số điểm bị ảnh hưởng")
        if "n_points" in counts.columns:
            ax.plot(counts["step"], counts["n_points"], color=INK, linestyle=(0, (4, 3)), linewidth=1.0, label="Fit lại phải xử lý cả n điểm")
        ax.set_yscale("symlog", linthresh=1)
        _plain_log_ticks(ax.yaxis)
        ax.set_xlabel("Step cập nhật")
        ax.set_ylabel("Số điểm đổi cụm (symlog)")
        ax.set_title(title or "Số điểm bị ảnh hưởng mỗi update", loc="left")
        ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
        ax.margins(x=0.01)
        fig.tight_layout()
    return fig


def plot_algorithm_contrast(coords, truth, labels_by_algorithm, scores=None, ncols=2, axis_labels=("x (km)", "y (km)"), point_size=7):
    """Nhãn thật và nhãn của từng thuật toán trên cùng một bản đồ 2-D, cùng bảng màu (nhãn được khớp về nhãn thật để màu có nghĩa).

    coords: (n, 2). truth: nhãn thật (-1 là nhiễu). labels_by_algorithm: dict tên -> nhãn (-1 là nhiễu). scores: dict tên -> số (ví dụ ARI)
    in vào tiêu đề (tùy chọn). Trả Figure."""
    coords = np.asarray(coords, float)
    names = ["Nhãn thật (hotspot)"] + list(labels_by_algorithm)
    all_labels = [np.asarray(truth)] + [np.asarray(l) for l in labels_by_algorithm.values()]
    n = len(names)
    rows = -(-n // ncols)
    ids = np.arange(len(coords))
    ref = {"ids": ids, "labels": all_labels[0]}
    vis = ClusterVisualizer(point_size=point_size)
    with report_style():
        fig, axes = plt.subplots(rows, ncols, figsize=(3.6 * ncols, 3.2 * rows), squeeze=False)
        for k, (name, lab) in enumerate(zip(names, all_labels)):
            ax = axes[k // ncols][k % ncols]
            lab2 = lab if k == 0 else align_labels(ref, {"ids": ids, "labels": lab})
            snap = make_snapshot(ids, coords, lab2, None)
            vis.plot_2d(snap, ax=ax, legend=False, fill=None, axis_labels=axis_labels)
            sub = f"\nARI = {scores[name]:.2f}" if scores and name in scores else ""
            c = len(set(np.asarray(lab).tolist()) - {-1})
            nz = int((np.asarray(lab) == -1).sum())
            ax.set_title(f"{name}{sub}\n{c} cụm, {nz} điểm nhiễu", fontsize=8)
        for k in range(n, rows * ncols):
            axes[k // ncols][k % ncols].axis("off")
        fig.tight_layout()
    return fig


def k_distances(coords, min_pts=4, block=512):
    """Khoảng cách từ mỗi điểm đến láng giềng thứ (min_pts - 1) KHÔNG tính chính nó (lân cận Eps của nhóm tính cả chính điểm,
    nên điểm lõi cần đủ min_pts điểm kể cả nó). Numpy thuần, chia khối để không dựng ma trận n x n. Trả mảng (n,) chưa sắp xếp."""
    x = np.asarray(coords, float)
    n = len(x)
    k = int(min_pts) - 1
    if k < 1:
        return np.zeros(n)
    if n <= k:
        raise ValueError(f"cần ít nhất {k + 1} điểm cho min_pts = {min_pts}")
    sq = (x ** 2).sum(axis=1)
    out = np.empty(n)
    for s in range(0, n, block):
        e = min(s + block, n)
        d2 = sq[s:e, None] + sq[None, :] - 2.0 * x[s:e] @ x.T
        np.maximum(d2, 0.0, out=d2)
        d2[np.arange(e - s), np.arange(s, e)] = np.inf          # bỏ chính nó
        out[s:e] = np.sqrt(np.partition(d2, k - 1, axis=1)[:, k - 1])
    return out


def knee_value(sorted_desc):
    """Điểm khuỷu của đường k-distance đã sắp giảm dần: điểm xa nhất so với dây cung nối hai đầu (sau khi chuẩn hóa về [0, 1]). Trả (chỉ số, giá trị)."""
    y = np.asarray(sorted_desc, float)
    if len(y) < 3 or y[0] == y[-1]:
        return 0, float(y[0]) if len(y) else float("nan")
    t = np.linspace(0, 1, len(y))
    yn = (y - y[-1]) / (y[0] - y[-1])
    i = int(np.argmax(np.abs(yn - (1 - t))))
    return i, float(y[i])


def plot_k_distance(coords, min_pts=4, eps=None, ax=None, title=None, unit="km"):
    """Đồ thị k-distance (sắp giảm dần) để chọn Eps; đánh dấu điểm khuỷu và (nếu có) Eps đang dùng. Trả Figure.

    Khoảng cách tính bằng numpy (k_distances), không dùng sklearn."""
    d = np.sort(k_distances(coords, min_pts))[::-1]
    i, v = knee_value(d)
    with report_style():
        if ax is None:
            fig, ax = plt.subplots(figsize=(6.2, 3.4))
        else:
            fig = ax.figure
        ax.plot(np.arange(1, len(d) + 1), d, color=BLUE, linewidth=1.5)
        ax.scatter([i + 1], [v], s=40, color=RED, zorder=4, label=f"Điểm khuỷu: {v:.2f} {unit}")
        if eps is not None:
            ax.axhline(eps, color=INK, linestyle=(0, (4, 3)), linewidth=0.9, label=f"Eps đang dùng: {eps:g} {unit}")
        ax.set_xlabel("Điểm, sắp theo khoảng cách giảm dần")
        ax.set_ylabel(f"Khoảng cách tới láng giềng thứ {min_pts - 1} ({unit})")
        ax.set_title(title or f"Đồ thị k-distance, MinPts = {min_pts}: điểm lõi cần {min_pts - 1} láng giềng cộng chính nó", loc="left")
        ax.legend(frameon=False)
        ax.margins(x=0.01)
        fig.tight_layout()
    return fig


def plot_scaling_effect(before, after, pair=(0, 1), names=None, names_after=None, units=None):
    """So dữ liệu trước và sau chuẩn hóa (cùng số dòng, cùng thứ tự cột). before, after: DataFrame hoặc mảng (n, d).

    Trái: độ lệch chuẩn từng cột (thang log): trước chuẩn hóa một cột có thể lấn át khoảng cách Euclid. Giữa và phải: tán xạ hai cột
    `pair` trước và sau (names_after: tên cột sau chuẩn hóa, mặc định như names). units: dict chỉ số cột -> (hệ số chia, tên đơn vị) cho
    panel trước chuẩn hóa, ví dụ {0: (1e6, "triệu VND")}, để trục không hiện dạng 1e8. Trả Figure."""
    names = names or (list(before.columns) if hasattr(before, "columns") else [f"x{j + 1}" for j in range(np.asarray(before).shape[1])])
    b, a = np.asarray(before, float), np.asarray(after, float)
    if b.shape != a.shape:
        raise ValueError(f"before {b.shape} và after {a.shape} phải cùng hình dạng")
    names_after = names_after or names
    i, j = pair
    with report_style():
        fig, axes = plt.subplots(1, 3, figsize=(7.6, 3.0), gridspec_kw={"width_ratios": [1.15, 1, 1]})
        w = 0.38
        x = np.arange(b.shape[1])
        axes[0].bar(x - w / 2, b.std(axis=0), w, color=ORANGE, label="Trước")
        axes[0].bar(x + w / 2, a.std(axis=0), w, color=BLUE, label="Sau")
        axes[0].set_yscale("log")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(names, rotation=35, ha="right", fontsize=6.5)
        axes[0].set_ylabel("Độ lệch chuẩn (thang log)")
        axes[0].set_title("Thang đo từng cột", loc="left")
        axes[0].legend(frameon=False)
        units = units or {}
        for ax, m, t, c, nm, u in ((axes[1], b, "Trước chuẩn hóa", ORANGE, names, units), (axes[2], a, "Sau chuẩn hóa", BLUE, names_after, {})):
            fi, li = u.get(i, (1.0, None))
            fj, lj = u.get(j, (1.0, None))
            ax.scatter(m[:, i] / fi, m[:, j] / fj, s=3, color=c, alpha=0.5, linewidths=0)
            ax.set_xlabel(f"{nm[i]} ({li})" if li else nm[i], fontsize=7)
            ax.set_ylabel(f"{nm[j]} ({lj})" if lj else nm[j], fontsize=7)
            ax.set_title(t, loc="left")
        fig.tight_layout()
    return fig
