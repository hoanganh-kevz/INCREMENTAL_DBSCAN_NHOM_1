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
        """
Module cài đặt thuật toán Incremental DBSCAN hỗ trợ cập nhật động (Insert/Delete).
"""

from collections import deque
from typing import List, Set
import numpy as np

from src.clustering.dbscan import DBSCAN
from src.clustering.utils import ClusterUtils
from src.core.data_point import DataPoint, PointType


class IncrementalDBSCAN(DBSCAN):
    """Lớp mở rộng DBSCAN hỗ trợ cập nhật động cục bộ mà không refit lại toàn bộ tập dữ liệu."""

    def insert_point(self, coords: np.ndarray) -> int:
        """Thêm một điểm dữ liệu mới vào mô hình và cập nhật cụm liên quan.

        Args:
            coords (np.ndarray): Tọa độ không gian n-D của điểm mới.

        Returns:
            int: ID cụm được phân cho điểm mới.

        Raises:
            NotImplementedError: Phương thức này do TV2 phát triển.
        """
        raise NotImplementedError("Phương thức insert_point được tích hợp bởi thành viên TV2.")

    def delete_point(self, point_id: int) -> None:
        """Xóa một điểm dữ liệu theo ID và thực hiện cập nhật cục bộ.

        Xử lý 3 trường hợp theo lý thuyết Incremental DBSCAN:
        1. Removal: Điểm xóa là NOISE hoặc chưa phân cụm.
        2. Reduction: Điểm xóa thuộc cụm nhưng không làm mất trạng thái CORE của điểm nào.
        3. Potential Split: Điểm xóa làm mất trạng thái CORE, dẫn đến nguy cơ tách cụm.

        Args:
            point_id (int): Định danh ID của điểm dữ liệu cần xóa.

        Raises:
            ValueError: Nếu không tìm thấy điểm dữ liệu khớp với point_id.
        """
        target_idx = self._find_point_index_by_id(point_id)
        if target_idx is None:
            raise ValueError(f"Không tìm thấy DataPoint có ID = {point_id}")

        deleted_point = self.points[target_idx]
        old_cluster_id = deleted_point.cluster_id

        # 1. Loại bỏ điểm khỏi danh sách và đồng bộ cache lân cận
        self.points.pop(target_idx)
        self._update_neighborhood_cache(target_idx)

        # Trường hợp 1: Removal (không ảnh hưởng cấu trúc cụm)
        if old_cluster_id is None or old_cluster_id == -1:
            return

        # 2. Thu thập vùng ảnh hưởng
        affected_indices = [
            idx for idx, pt in enumerate(self.points)
            if pt.cluster_id == old_cluster_id
        ]

        # 3. Cập nhật số lân cận và kiểm tra đứt gãy điểm CORE
        core_demoted = self._reevaluate_affected_points(affected_indices)

        # Trường hợp 2: Reduction (không đứt gãy CORE)
        if not core_demoted:
            ClusterUtils.relabel_unreachable_points(
                self.points, set(affected_indices), self.region_query
            )
            return

        # Trường hợp 3: Potential Split (kiểm tra phân tách cụm)
        self._handle_potential_split(old_cluster_id, affected_indices)

    def _find_point_index_by_id(self, point_id: int) -> Optional[int]:
        """Tìm chỉ số index của điểm trong danh sách dựa trên ID."""
        for idx, pt in enumerate(self.points):
            if pt.id == point_id:
                return idx
        return None

    def _update_neighborhood_cache(self, removed_idx: int) -> None:
        """Cập nhật lại chỉ số index trong bộ nhớ cache lân cận sau khi xóa một điểm.

        Args:
            removed_idx (int): Chỉ số index của điểm vừa bị loại bỏ khỏi self.points.
        """
        updated_cache = {}
        for idx, neighbors in self.neighborhoods.items():
            if idx == removed_idx:
                continue

            new_key = idx - 1 if idx > removed_idx else idx
            new_neighbors = [
                n_idx - 1 if n_idx > removed_idx else n_idx
                for n_idx in neighbors if n_idx != removed_idx
            ]
            updated_cache[new_key] = new_neighbors
        self.neighborhoods = updated_cache

    def _reevaluate_affected_points(self, affected_indices: List[int]) -> bool:
        """Đánh giá lại số lượng lân cận và loại điểm (CORE/BORDER) vùng bị ảnh hưởng.

        Args:
            affected_indices (List[int]): Danh sách index các điểm cần đánh giá lại.

        Returns:
            bool: True nếu có ít nhất một điểm CORE bị chuyển thành BORDER, ngược lại False.
        """
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
        return core_demoted

    def _handle_potential_split(self, old_cluster_id: int, cluster_indices: List[int]) -> None:
        """Xử lý kiểm tra liên thông và phân tách cụm khi phát hiện điểm CORE bị đứt gãy.

        Args:
            old_cluster_id (int): ID cụm cũ trước khi thực hiện xóa.
            cluster_indices (List[int]): Danh sách index thuộc cụm bị ảnh hưởng.
        """
        for idx in cluster_indices:
            self.points[idx].cluster_id = None

        remaining_cores = [idx for idx in cluster_indices if self.points[idx].is_core]

        if not remaining_cores:
            for idx in cluster_indices:
                self.points[idx].cluster_id = -1
                self.points[idx].point_type = PointType.NOISE
            return

        visited_cores: Set[int] = set()
        first_subcluster = True
        active_clusters: Set[int] = set()

        for core_idx in remaining_cores:
            if core_idx in visited_cores:
                continue

            current_cluster_id = old_cluster_id if first_subcluster else self._next_cluster_id
            if not first_subcluster:
                self._next_cluster_id += 1
            first_subcluster = False
            active_clusters.add(current_cluster_id)

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
        ClusterUtils.clean_stale_cluster_ids(self.points, active_clusters)