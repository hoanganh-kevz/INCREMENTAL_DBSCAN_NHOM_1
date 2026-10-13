import numpy as np
from collections import deque
from enum import Enum
from math import sqrt
from typing import List, Optional


class PointType(Enum):
    """Trạng thái của một điểm trong quá trình phân cụm."""

    UNCLASSIFIED = "unclassified"
    NOISE = "noise"
    BORDER = "border"
    CORE = "core"


class DataPoint:
    """Đại diện cho một điểm dữ liệu trong không gian N chiều."""

    def __init__(self, point_id: int, coords: np.ndarray):
        self.id: int = point_id
        self.coords: np.ndarray = np.array(coords, dtype=float)
        self.cluster_id: Optional[int] = None  # None: Chưa duyệt, -1: Noise, >=0: Cluster ID
        self.point_type: PointType = PointType.UNCLASSIFIED
        self.neighbor_count: int = 0

    @property
    def is_core(self) -> bool:
        """Giữ thuộc tính cũ để các đoạn code đang đọc is_core không bị gãy."""
        return self.point_type is PointType.CORE

    @is_core.setter
    def is_core(self, value: bool) -> None:
        """Tương thích với code cũ; trạng thái chính được lưu trong point_type."""
        if value:
            self.point_type = PointType.CORE
        elif self.point_type is PointType.CORE:
            self.point_type = (
                PointType.BORDER
                if self.cluster_id is not None and self.cluster_id >= 0
                else PointType.UNCLASSIFIED
            )

    def __repr__(self):
        return (
            f"DataPoint(id={self.id}, coords={self.coords.tolist()}, "
            f"cluster={self.cluster_id}, type={self.point_type.value})"
        )


class DistanceMetric:
    """Tính khoảng cách độc lập với dữ liệu điểm và thuật toán gom cụm."""

    @staticmethod
    def euclidean(first: np.ndarray, second: np.ndarray) -> float:
        first_coords = np.asarray(first, dtype=float).reshape(-1)
        second_coords = np.asarray(second, dtype=float).reshape(-1)
        if first_coords.size != second_coords.size:
            raise ValueError("Hai điểm phải có cùng số chiều.")

        squared_distance = sum(
            (float(first_value) - float(second_value)) ** 2
            for first_value, second_value in zip(first_coords, second_coords)
        )
        return sqrt(squared_distance)


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
        for idx, pt in enumerate(self.points):
            if DistanceMetric.euclidean(target_point.coords, pt.coords) <= self.eps:
                neighbors.append(idx)
        target_point.neighbor_count = len(neighbors)
        return neighbors

    def fit(self, data: np.ndarray) -> 'DBSCAN':
        """Huấn luyện DBSCAN trên tập dữ liệu ban đầu."""
        self.points = [DataPoint(i, row) for i, row in enumerate(data)]
        self._next_cluster_id = 0

        for idx, pt in enumerate(self.points):
            if pt.cluster_id is not None:
                continue

            neighbors = self._region_query(idx)
            if len(neighbors) < self.min_pts:
                pt.cluster_id = -1
                pt.point_type = PointType.NOISE
            else:
                pt.point_type = PointType.CORE
                self._expand_cluster(idx, neighbors, self._next_cluster_id)
                self._next_cluster_id += 1
        return self

    def _expand_cluster(self, seed_idx: int, neighbors: List[int], cluster_id: int):
        """Loang cụm từ điểm lõi ban đầu."""
        self.points[seed_idx].cluster_id = cluster_id
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
            curr_neighbors = self._region_query(curr_idx)

            if len(curr_neighbors) >= self.min_pts:
                curr_pt.point_type = PointType.CORE
                queue.extend(curr_neighbors)
            else:
                curr_pt.point_type = PointType.BORDER

    def get_labels(self) -> List[int]:
        """Trả về danh sách nhãn cụm hiện tại."""
        return [pt.cluster_id if pt.cluster_id is not None else -1 for pt in self.points]


class IncrementalDBSCAN(DBSCAN):
    """Mở rộng DBSCAN hỗ trợ cập nhật động (Insert / Delete)."""

    def insert_point(self, coords: np.ndarray) -> int:
        """
        [Do TV2 / Nguyễn Hoàng Anh cài đặt]
        Thêm một điểm mới vào mô hình và cập nhật cụm.
        """
        raise NotImplementedError("Phương thức insert_point sẽ được tích hợp bởi TV2.")

    def delete_point(self, point_id: int):
        """
        [Do TV3 / Từ Thiên Gia Bảo cài đặt]
        Xóa điểm dữ liệu theo point_id và cập nhật tái phân cụm địa phương.
        """
        # 1. Định vị điểm cần xóa
        target_idx = None
        for idx, pt in enumerate(self.points):
            if pt.id == point_id:
                target_idx = idx
                break

        if target_idx is None:
            raise ValueError(f"Không tìm thấy DataPoint có ID = {point_id}")

        deleted_point = self.points[target_idx]
        old_cluster_id = deleted_point.cluster_id
        affected_points = [
            self.points[idx]
            for idx in self._region_query(target_idx)
            if idx != target_idx
        ]

        # 2. Xóa điểm khỏi tập dữ liệu
        self.points.pop(target_idx)
        for point in affected_points:
            point.neighbor_count = max(0, point.neighbor_count - 1)

        # Nếu điểm xóa là điểm nhiễu hoặc chưa phân cụm -> Không ảnh hưởng cấu trúc
        if old_cluster_id is None or old_cluster_id == -1:
            return

        # 3. Cập nhật trạng thái Core của các điểm còn lại thuộc cụm bị ảnh hưởng
        cluster_pts_indices = [i for i, pt in enumerate(self.points) if pt.cluster_id == old_cluster_id]
        
        for idx in cluster_pts_indices:
            curr_neighbors = self._region_query(idx)
            self.points[idx].point_type = (
                PointType.CORE if len(curr_neighbors) >= self.min_pts else PointType.BORDER
            )

        # 4. Tái cấu trúc lại cụm bị ảnh hưởng (Xử lý trường hợp Split Cụm)
        self._recluster_affected_subgraph(old_cluster_id, cluster_pts_indices)

    def _recluster_affected_subgraph(self, old_cluster_id: int, cluster_indices: List[int]):
        """Xử lý phân tách cụm sau khi xóa điểm lõi."""
        # Reset nhãn của các điểm thuộc cụm bị ảnh hưởng
        for idx in cluster_indices:
            self.points[idx].cluster_id = None
            self.points[idx].point_type = PointType.UNCLASSIFIED

        new_cluster_id = self._next_cluster_id

        for idx in cluster_indices:
            pt = self.points[idx]
            if pt.cluster_id is not None:
                continue

            all_neighbors = self._region_query(idx)
            if len(all_neighbors) >= self.min_pts:
                pt.point_type = PointType.CORE
                self._expand_local(idx, cluster_indices, new_cluster_id)
                new_cluster_id += 1
            else:
                pt.cluster_id = -1
                pt.point_type = PointType.NOISE

        self._next_cluster_id = new_cluster_id

    def _expand_local(self, seed_idx: int, allowed_indices: List[int], cluster_id: int):
        """Loang cụm giới hạn trong phạm vi các điểm thuộc cụm bị ảnh hưởng."""
        allowed_set = set(allowed_indices)
        self.points[seed_idx].cluster_id = cluster_id
        queue = deque([seed_idx])

        while queue:
            curr_idx = queue.popleft()
            all_neighbors = self._region_query(curr_idx)
            curr_neighbors = [n for n in all_neighbors if n in allowed_set]

            if len(all_neighbors) >= self.min_pts:
                self.points[curr_idx].point_type = PointType.CORE
                for n_idx in curr_neighbors:
                    n_pt = self.points[n_idx]
                    if n_pt.cluster_id is None or n_pt.cluster_id == -1:
                        n_pt.cluster_id = cluster_id
                        n_pt.point_type = PointType.BORDER
                        if n_idx not in queue:
                            queue.append(n_idx)
            else:
                self.points[curr_idx].point_type = PointType.BORDER
