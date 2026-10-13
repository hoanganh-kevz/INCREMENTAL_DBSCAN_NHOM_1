from collections import deque
from typing import List

import numpy as np

from ..core import DataPoint, DistanceMetric, PointType


class DBSCAN:
    """Thuật toán DBSCAN tĩnh ban đầu."""

    def __init__(self, eps: float, min_pts: int):
        self.eps: float = eps
        self.min_pts: int = min_pts
        self.points: List[DataPoint] = []
        self._next_cluster_id: int = 0

    def _region_query(self, point_idx: int) -> List[int]:
        """Tìm chỉ số của các điểm trong bán kính eps."""
        neighbors = []
        target_point = self.points[point_idx]
        for idx, point in enumerate(self.points):
            if DistanceMetric.euclidean(target_point.coords, point.coords) <= self.eps:
                neighbors.append(idx)
        target_point.neighbor_count = len(neighbors)
        return neighbors

    def fit(self, data: np.ndarray) -> "DBSCAN":
        """Huấn luyện DBSCAN trên tập dữ liệu ban đầu."""
        self.points = [DataPoint(i, row) for i, row in enumerate(data)]
        self._next_cluster_id = 0

        for idx, point in enumerate(self.points):
            if point.cluster_id is not None:
                continue

            neighbors = self._region_query(idx)
            if len(neighbors) < self.min_pts:
                point.cluster_id = -1
                point.point_type = PointType.NOISE
            else:
                point.point_type = PointType.CORE
                self._expand_cluster(idx, neighbors, self._next_cluster_id)
                self._next_cluster_id += 1
        return self

    def _expand_cluster(self, seed_idx: int, neighbors: List[int], cluster_id: int) -> None:
        """Loang cụm từ điểm lõi ban đầu."""
        self.points[seed_idx].cluster_id = cluster_id
        queue = deque(neighbors)

        while queue:
            curr_idx = queue.popleft()
            curr_point = self.points[curr_idx]

            if curr_point.cluster_id == -1:
                curr_point.cluster_id = cluster_id
                curr_point.point_type = PointType.BORDER

            if curr_point.cluster_id is not None:
                continue

            curr_point.cluster_id = cluster_id
            curr_neighbors = self._region_query(curr_idx)

            if len(curr_neighbors) >= self.min_pts:
                curr_point.point_type = PointType.CORE
                queue.extend(curr_neighbors)
            else:
                curr_point.point_type = PointType.BORDER

    def get_labels(self) -> List[int]:
        """Trả về danh sách nhãn cụm hiện tại."""
        return [point.cluster_id if point.cluster_id is not None else -1 for point in self.points]
