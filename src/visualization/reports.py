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
