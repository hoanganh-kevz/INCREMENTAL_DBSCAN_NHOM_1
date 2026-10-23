"""IncrementalEngine: IncrementalDBSCAN của nhóm cộng thêm snapshot() và vài thuộc tính đếm.

Lớp không đổi thuật toán, chỉ đọc self.points. Nhờ có snapshot() mà replay_stream, run_stream (baseline) và bộ vẽ (src/visualization)
chạy với thuật toán thật giống hệt như đã chạy với RefitDBSCAN. Không dùng sklearn.
"""
import numpy as np

from src.clustering.incremental_dbscan import IncrementalDBSCAN
from src.core.data_point import PointType

# mã loại điểm giống baseline.sklearn_dbscan.PointType và src.visualization.plotter
TYPE_CODE = {PointType.UNCLASSIFIED: 0, PointType.NOISE: 1, PointType.BORDER: 2, PointType.CORE: 3}


class IncrementalEngine(IncrementalDBSCAN):
    """IncrementalDBSCAN với snapshot() theo thứ tự id tăng dần; cluster_id None hoặc -1 đều ghi là nhiễu (-1)."""

    def snapshot(self):
        ids = sorted(self.points)
        pts = [self.points[i] for i in ids]
        dim = pts[0].dimension if pts else 0
        coords = np.vstack([p.coords for p in pts]) if pts else np.empty((0, dim))
        labels = np.array([-1 if p.cluster_id is None else int(p.cluster_id) for p in pts], dtype=int)
        types = np.array([TYPE_CODE[p.point_type] for p in pts], dtype=int)
        return {"ids": np.array(ids, dtype=int), "coords": coords, "labels": labels, "point_types": types}

    @property
    def n_points(self):
        return len(self.points)

    @property
    def n_clusters(self):
        return len({p.cluster_id for p in self.points.values() if p.cluster_id is not None and p.cluster_id >= 0})

    @property
    def n_noise(self):
        return sum(1 for p in self.points.values() if p.cluster_id is None or p.cluster_id < 0)
