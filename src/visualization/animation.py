"""Hoạt ảnh một stream chèn và xóa (D07, nice-to-have): bản đồ toàn cảnh, ô phóng to quanh điểm vừa cập nhật, bộ đếm cụm và nhiễu.

Nhận danh sách khung từ `scenarios.stream_snapshots` (engine nào cũng được). Ghi GIF bằng matplotlib.animation.PillowWriter, nên cần Pillow.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from .plotter import INK, ClusterVisualizer, align_labels, report_style, type_counts

OP_VI = {"fit": "Khởi tạo D0 (fit lần đầu)", "insert": "Chèn điểm", "delete": "Xóa điểm"}


def persistent_labels(frames):
    """Nhãn của từng khung được khớp về khung trước để cụm giữ nguyên màu qua các step; cụm mới nhận nhãn chưa từng dùng. Trả list mảng nhãn."""
    out, prev, top = [], None, -1
    for fr in frames:
        snap = fr["snapshot"]
        lab = snap["labels"].copy() if prev is None else align_labels(prev, snap, min_new_label=top + 1)
        top = max(top, int(lab.max()) if len(lab) else -1)
        out.append(lab)
        prev = {"ids": snap["ids"], "labels": lab}
    return out


def animate_stream(frames, path, every=1, fps=5, zoom_km=2.5, figsize=(6.0, 6.3), dpi=100, axis_labels=("x (km)", "y (km)")):
    """Ghi GIF của stream. every: lấy một khung mỗi `every` step (khung cuối luôn được giữ). Trả đường dẫn.

    Khung trên: bản đồ toàn thành phố, vùng tô = hợp hình tròn Eps quanh điểm lõi; ngôi sao = điểm vừa chèn, X = điểm vừa xóa (có vòng ngoài).
    Khung dưới trái: phóng to ±zoom_km quanh điểm đó, có vòng Eps và UpdSeed không vẽ (stream không kèm UpdSeed). Khung dưới phải: bộ đếm."""
    from pathlib import Path
    labels = persistent_labels(frames)
    keep = sorted(set(range(0, len(frames), every)) | {len(frames) - 1})
    allxy = np.vstack([fr["snapshot"]["coords"] for fr in frames] + [fr["coords"][None, :] for fr in frames if fr["coords"] is not None])
    lim = ((allxy[:, 0].min() - 0.5, allxy[:, 0].max() + 0.5), (allxy[:, 1].min() - 0.5, allxy[:, 1].max() + 0.5))
    big, small = ClusterVisualizer(point_size=11), ClusterVisualizer(point_size=75)
    last_xy = [None]
    n_steps = frames[-1]["step"]
    with report_style():
        fig = plt.figure(figsize=figsize, dpi=dpi)
        gs = fig.add_gridspec(2, 2, height_ratios=[1.35, 1], hspace=0.22, wspace=0.12, left=0.1, right=0.97, top=0.95, bottom=0.04)
        ax_map, ax_zoom, ax_txt = fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])

        def draw(i):
            fr = frames[keep[i]]
            snap = dict(fr["snapshot"])
            snap["labels"] = labels[keep[i]]
            eps = fr["eps"]
            xy = fr["coords"]
            if xy is not None:
                last_xy[0] = xy
            mark = [{"kind": fr["op"], "xy": xy, "label": None}] if xy is not None else []
            for ax in (ax_map, ax_zoom, ax_txt):
                ax.clear()
            big._draw(ax_map, snap, eps, None, None, mark, "eps", None, False, axis_labels, False, lim, "Bản đồ giao dịch ATM mô phỏng", None)
            if xy is not None:
                ax_map.scatter([xy[0]], [xy[1]], s=330, facecolors="none", edgecolors=INK, linewidths=1.2, zorder=7)
            centre = last_xy[0] if last_xy[0] is not None else allxy.mean(axis=0)
            zl = ((centre[0] - zoom_km, centre[0] + zoom_km), (centre[1] - zoom_km, centre[1] + zoom_km))
            small._draw(ax_zoom, snap, eps, None, None, mark, "eps", None, False, None, True, zl, f"Phóng to ±{zoom_km:g} km", None)
            c = type_counts(snap)
            ax_txt.axis("off")
            lines = [(f"Step {fr['step']} / {n_steps}", 12, "bold"), (OP_VI[fr["op"]] + (f" #{fr['point_id']}" if fr["point_id"] is not None else ""), 9, "normal")]
            if fr["case"] and isinstance(fr["case"], str):
                lines.append((f"Trường hợp: {fr['case']}", 9, "normal"))
            lines += [("", 4, "normal"), (f"Số điểm: {c['n']}", 10, "normal"), (f"Số cụm: {c['clusters']}", 10, "bold"), (f"Điểm nhiễu: {c['noise']}", 10, "bold"),
                      ("", 4, "normal"), ("★ điểm vừa chèn", 8, "normal"), ("✖ điểm vừa xóa", 8, "normal")]
            y = 0.97
            for text, size, weight in lines:
                ax_txt.text(0.02, y, text, fontsize=size, fontweight=weight, va="top", transform=ax_txt.transAxes)
                y -= size * 1.5 / 72 / 2.1
            return []

        anim = FuncAnimation(fig, draw, frames=len(keep), blit=False)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        anim.save(path, writer=PillowWriter(fps=fps), dpi=dpi)
        plt.close(fig)
    return path
