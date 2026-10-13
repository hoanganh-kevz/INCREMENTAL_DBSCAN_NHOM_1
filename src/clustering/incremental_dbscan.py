from collections import deque
from typing import List

import numpy as np

from ..core import PointType
from .dbscan import DBSCAN


class IncrementalDBSCAN(DBSCAN):
    """Mở rộng DBSCAN hỗ trợ cập nhật động (Insert / Delete)."""

    def insert_point(self, coords: np.ndarray) -> int:
        """
        [Do TV2 / Nguyễn Hoàng Anh cài đặt]
        Thêm một điểm mới vào mô hình và cập nhật cụm.
        """
        raise NotImplementedError("Phương thức insert_point sẽ được tích hợp bởi TV2.")

    def delete_point(self, point_id: int) -> None:
        """
        [Do TV3 / Từ Thiên Gia Bảo cài đặt]
        Xóa điểm dữ liệu theo point_id và cập nhật tái phân cụm địa phương.
        """
        # 1. Định vị điểm cần xóa
        target_idx = None
        for idx, point in enumerate(self.points):
            if point.id == point_id:
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
        cluster_pts_indices = [
            idx for idx, point in enumerate(self.points) if point.cluster_id == old_cluster_id
        ]

        for idx in cluster_pts_indices:
            curr_neighbors = self._region_query(idx)
            self.points[idx].point_type = (
                PointType.CORE if len(curr_neighbors) >= self.min_pts else PointType.BORDER
            )

        # 4. Tái cấu trúc lại cụm bị ảnh hưởng (xử lý trường hợp tách cụm)
        self._recluster_affected_subgraph(old_cluster_id, cluster_pts_indices)

    def _recluster_affected_subgraph(
        self, old_cluster_id: int, cluster_indices: List[int]
    ) -> None:
        """Xử lý phân tách cụm sau khi xóa điểm lõi."""
        # Reset nhãn của các điểm thuộc cụm bị ảnh hưởng
        for idx in cluster_indices:
            self.points[idx].cluster_id = None
            self.points[idx].point_type = PointType.UNCLASSIFIED

        new_cluster_id = self._next_cluster_id

        for idx in cluster_indices:
            point = self.points[idx]
            if point.cluster_id is not None:
                continue

            all_neighbors = self._region_query(idx)
            if len(all_neighbors) >= self.min_pts:
                point.point_type = PointType.CORE
                self._expand_local(idx, cluster_indices, new_cluster_id)
                new_cluster_id += 1
            else:
                point.cluster_id = -1
                point.point_type = PointType.NOISE

        self._next_cluster_id = new_cluster_id

    def _expand_local(
        self, seed_idx: int, allowed_indices: List[int], cluster_id: int
    ) -> None:
        """Loang cụm giới hạn trong phạm vi các điểm thuộc cụm bị ảnh hưởng."""
        allowed_set = set(allowed_indices)
        self.points[seed_idx].cluster_id = cluster_id
        queue = deque([seed_idx])

        while queue:
            curr_idx = queue.popleft()
            all_neighbors = self._region_query(curr_idx)
            curr_neighbors = [idx for idx in all_neighbors if idx in allowed_set]

            if len(all_neighbors) >= self.min_pts:
                self.points[curr_idx].point_type = PointType.CORE
                for neighbor_idx in curr_neighbors:
                    neighbor = self.points[neighbor_idx]
                    if neighbor.cluster_id is None or neighbor.cluster_id == -1:
                        neighbor.cluster_id = cluster_id
                        neighbor.point_type = PointType.BORDER
                        if neighbor_idx not in queue:
                            queue.append(neighbor_idx)
            else:
                self.points[curr_idx].point_type = PointType.BORDER
