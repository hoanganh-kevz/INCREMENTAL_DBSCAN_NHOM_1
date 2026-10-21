import numpy as np
from collections import deque
from typing import List, Optional, Dict
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
        self.points: List[DataPoint] = []
        self._next_cluster_id: int = 0
        self.neighborhoods: Dict[int, List[int]] = {}

    def region_query(self, point_idx: int) -> List[int]:
        """Xác định danh sách chỉ số các điểm nằm trong bán kính epsilon (eps) của điểm xét."""
        neighbors = []
        target_point = self.points[point_idx]
        for idx, pt in enumerate(self.points):
            if self.metric.calculate(target_point, pt) <= self.eps:
                neighbors.append(idx)
        return neighbors

    def fit(self, data: np.ndarray) -> 'DBSCAN':
        """Thực hiện phân cụm dữ liệu ban đầu và khởi tạo trạng thái cho các điểm dữ liệu."""
        self.points = [DataPoint(i, row) for i, row in enumerate(data)]
        self._next_cluster_id = 0
        self.neighborhoods.clear()

        for idx in range(len(self.points)):
            neighbors = self.region_query(idx)
            self.neighborhoods[idx] = neighbors
            self.points[idx].neighborhood = neighbors
            self.points[idx].neighbor_count = len(neighbors)

        for idx, pt in enumerate(self.points):
            if pt.cluster_id is not None:
                continue

            neighbors = self.neighborhoods[idx]

            if pt.neighbor_count < self.min_pts:
                pt.cluster_id = -1
                pt.point_type = PointType.NOISE
            else:
                pt.point_type = PointType.CORE
                self._expand_cluster(idx, neighbors, self._next_cluster_id)
                self._next_cluster_id += 1

        return self

    def _expand_cluster(self, seed_idx: int, neighbors: List[int], cluster_id: int):
        """Loang cụm từ một điểm lõi (Core Point) sang các điểm lân cận."""
        self.points[seed_idx].cluster_id = cluster_id
        self.points[seed_idx].point_type = PointType.CORE
        queue = deque(neighbors)

        while queue:
            curr_idx = queue.popleft()
            curr_pt = self.points[curr_idx]

            if curr_pt.cluster_id == -1:
                curr_pt.cluster_id = cluster_id
                curr_pt.point_type = PointType.BORDER

            if curr_pt.cluster_id is not None:
                continue

            curr_pt.cluster_id = cluster_id
            curr_neighbors = self.neighborhoods[curr_idx]

            if curr_pt.neighbor_count >= self.min_pts:
                curr_pt.point_type = PointType.CORE
                queue.extend(curr_neighbors)
            else:
                curr_pt.point_type = PointType.BORDER

    def get_labels(self) -> List[int]:
        """Trả về danh sách nhãn cụm tương ứng với các điểm dữ liệu (-1 đại diện cho Noise)."""
        return [pt.cluster_id if pt.cluster_id is not None else -1 for pt in self.points]


    
