"""Dựng lại các hình D07 từ dữ liệu thật (chạy từ thư mục gốc repo): python tools/make_d07_figures.py [--no-gif]

Ghi vào report/figures/: case_gallery_insert_delete.png (8 kịch bản toy), case_gallery_edge.png (18 kịch bản toy_edge 2 chiều),
atm_stream.gif (stream ATM). Engine là IncrementalDBSCAN của nhóm (qua IncrementalEngine).
Các hình khác (ARI theo step, thời gian, k-distance, so sánh thuật toán, chuẩn hóa) do notebooks/D07_truc_quan_hoa.ipynb tạo.
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.experiments.engine import IncrementalEngine as ENGINE      # IncrementalDBSCAN của nhóm, có snapshot()
from src.visualization import ClusterVisualizer, stream_snapshots
from src.visualization.animation import animate_stream
from src.visualization.gallery import build_gallery_cases

OUT = ROOT / "report" / "figures"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-gif", action="store_true", help="bỏ qua hoạt ảnh (cần Pillow, mất khoảng 15 giây)")
    a = ap.parse_args()
    vis = ClusterVisualizer()
    for kind, name, cols in (("toy", "case_gallery_insert_delete.png", 2), ("toy_edge", "case_gallery_edge.png", 3)):
        cases = [c for c in build_gallery_cases(kind, ENGINE) if c["name"] != "e18_insert_4d_merge"]      # e18 là 4 chiều, vẽ riêng ở D07 mục 4
        fig = vis.plot_case_gallery(cases, n_pair_cols=cols, fill=None)
        print(vis.save(fig, OUT / name))
    if not a.no_gif:
        print(animate_stream(stream_snapshots(ENGINE, ROOT / "data" / "raw" / "bank_geo" / "atm_stream"), OUT / "atm_stream.gif"))


if __name__ == "__main__":
    main()
