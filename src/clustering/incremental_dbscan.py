import numpy as np
from collections import deque
from typing import List, Optional
from src.core.data_point import DataPoint, PointType
from src.core.distance import DistanceMetric
from src.clustering.dbscan import DBSCAN
from src.clustering.utils import ClusterUtils


class IncrementalDBSCAN(DBSCAN):
    """
    Thuật toán Incremental DBSCAN hỗ trợ cập nhật động (Insert/Delete)
    dựa trên việc xử lý vùng ảnh hưởng cục bộ mà không refit toàn bộ dữ liệu.
    """

    def insert_point(self, coords: np.ndarray) -> int:
        """Thêm một điểm mới vào mô hình và cập nhật cụm cục bộ."""
        raise NotImplementedError("Phương thức insert_point sẽ được tích hợp bởi nhóm.")

    def delete_point(self, point_id: int):
        """
        Xóa điểm dữ liệu theo point_id và thực hiện cập nhật phân cụm cục bộ.
        Xử lý các trường hợp: Removal, Reduction và Potential Split.
        """
        target_idx = None
        for idx, pt in enumerate(self.points):
            if pt.id == point_id:
                target_idx = idx
                break

        if target_idx is None:
            raise ValueError(f"Không tìm thấy DataPoint có ID = {point_id}")

        deleted_point = self.points[target_idx]
        old_cluster_id = deleted_point.cluster_id

        self.points.pop(target_idx)
        if target_idx in self.neighborhoods:
            del self.neighborhoods[target_idx]

        self._update_neighborhood_cache(target_idx)

        if old_cluster_id is None or old_cluster_id == -1:
            return

        affected_indices = [
            idx for idx, pt in enumerate(self.points)
            if pt.cluster_id == old_cluster_id
        ]

        core_demoted = False
        for idx in affected_indices:
            neighbors = self.region_query(idx)
            self.neighborhoods[idx] = neighbors
            pt = self.points[idx]
            pt.neighborhood = neighbors
            pt.neighbor_count = len(neighbors)

            old_type = pt.point_type
            if pt.neighbor_count >= self.min_pts:
                pt.point_type = PointType.CORE
            else:
                pt.point_type = PointType.BORDER
                if old_type == PointType.CORE:
                    core_demoted = True

        if not core_demoted:
            ClusterUtils.relabel_unreachable_points(self.points, set(affected_indices), self.region_query)
            return

        self._handle_potential_split(old_cluster_id, affected_indices)

    def _update_neighborhood_cache(self, removed_idx: int):
        """Đồng bộ lại bộ nhớ cache lân cận sau khi một điểm bị xóa khỏi danh sách."""
        updated_cache = {}
        for idx, neighbors in self.neighborhoods.items():
            new_key = idx - 1 if idx > removed_idx else idx
            new_neighbors = [
                n_idx - 1 if n_idx > removed_idx else n_idx
                for n_idx in neighbors if n_idx != removed_idx
            ]
            updated_cache[new_key] = new_neighbors
        self.neighborhoods = updated_cache

    def _handle_potential_split(self, old_cluster_id: int, cluster_indices: List[int]):
        """Kiểm tra tính liên thông giữa các điểm CORE và phân tách cụm nếu bị đứt gãy."""
        for idx in cluster_indices:
            self.points[idx].cluster_id = None

        remaining_cores = [idx for idx in cluster_indices if self.points[idx].is_core]

        if not remaining_cores:
            for idx in cluster_indices:
                self.points[idx].cluster_id = -1
                self.points[idx].point_type = PointType.NOISE
            return

        visited_cores = set()
        first_subcluster = True

        for core_idx in remaining_cores:
            if core_idx in visited_cores:
                continue

            current_cluster_id = old_cluster_id if first_subcluster else self._next_cluster_id
            if not first_subcluster:
                self._next_cluster_id += 1
            first_subcluster = False

            queue = deque([core_idx])
            visited_cores.add(core_idx)

            while queue:
                curr_idx = queue.popleft()
                self.points[curr_idx].cluster_id = current_cluster_id

                for n_idx in self.neighborhoods[curr_idx]:
                    if self.points[n_idx].is_core and n_idx not in visited_cores:
                        visited_cores.add(n_idx)
                        queue.append(n_idx)

        ClusterUtils.relabel_unreachable_points(self.points, set(cluster_indices), self.region_query)