from collections import deque
from typing import Dict, List, Optional, Set

import numpy as np

from src.core.data_point import DataPoint, PointType
from src.core.distance import DistanceMetric


class DBSCAN:
    """
    Thuật toán phân cụm dựa trên mật độ DBSCAN (Density-Based Spatial Clustering).
    Lưu trữ trạng thái lân cận để hỗ trợ cập nhật động trong Incremental DBSCAN.
    """
    def __init__(self, eps: float, min_pts: int, metric: Optional[DistanceMetric] = None):
        self.eps: float = eps
        self.min_pts: int = min_pts
        self.metric: DistanceMetric = metric if metric is not None else DistanceMetric()
        self.points: Dict[int, DataPoint] = {}
        self._next_point_id: int = 0
        self._next_cluster_id: int = 0

    def _region_query(self, point_id: int) -> Set[int]:
        """Trả ID các điểm nằm trong epsilon-lân cận của điểm được xét."""
        try:
            target_point = self.points[point_id]
        except KeyError as error:
            raise ValueError(f"Không tìm thấy DataPoint có ID = {point_id}") from error

        return {
            candidate_id
            for candidate_id, candidate in self.points.items()
            if self.metric.euclidean(target_point.coords, candidate.coords) <= self.eps
        }

    def fit(self, data: np.ndarray) -> 'DBSCAN':
        """Thực hiện phân cụm dữ liệu ban đầu và khởi tạo trạng thái cho các điểm dữ liệu."""
        self.points = {i: DataPoint(i, row) for i, row in enumerate(data)}
        self._next_point_id = len(self.points)
        self._next_cluster_id = 0

        for point_id, point in self.points.items():
            point.set_neighborhood(self._region_query(point_id))

        for point_id, pt in self.points.items():
            if pt.cluster_id is not None:
                continue

            if pt.neighbor_count < self.min_pts:
                pt.cluster_id = -1
                pt.point_type = PointType.NOISE
            else:
                pt.point_type = PointType.CORE
                self._expand_cluster(point_id, pt.neighbor_ids, self._next_cluster_id)
                self._next_cluster_id += 1

        return self

    def _expand_cluster(self, seed_id: int, neighbor_ids: Set[int], cluster_id: int) -> None:
        """Loang cụm từ một điểm lõi (Core Point) sang các điểm lân cận."""
        seed = self.points[seed_id]
        seed.cluster_id = cluster_id
        seed.point_type = PointType.CORE
        queue = deque(sorted(neighbor_ids))

        while queue:
            current_id = queue.popleft()
            current_point = self.points[current_id]

            if current_point.cluster_id == -1:
                current_point.cluster_id = cluster_id
                current_point.point_type = PointType.BORDER

            if current_point.cluster_id is not None:
                continue

            current_point.cluster_id = cluster_id

            if current_point.neighbor_count >= self.min_pts:
                current_point.point_type = PointType.CORE
                queue.extend(sorted(current_point.neighbor_ids))
            else:
                current_point.point_type = PointType.BORDER

    def get_labels(self) -> List[int]:
        """Trả về danh sách nhãn cụm tương ứng với các điểm dữ liệu (-1 đại diện cho Noise)."""
        return [
            point.cluster_id if point.cluster_id is not None else -1
            for point in self.points.values()
        ]
