from collections import deque
from typing import Dict, List, Set, Tuple, Union

import numpy as np

from src.clustering.dbscan import DBSCAN
from src.clustering.utils import ClusterUtils
from src.core.data_point import DataPoint, PointType

GraphNode = Tuple[str, int]

class IncrementalDBSCAN(DBSCAN):
    """Mở rộng DBSCAN để hỗ trợ cập nhật cụm khi thay đổi dữ liệu"""

    def insert(self, point: Union[DataPoint, np.ndarray]) -> int:
        """Chèn một điểm mới và cập nhật kết quả phân cụm."""

        # Lấy tọa độ của điểm được chèn.
        coords = point.coords if isinstance(point, DataPoint) else point
        coords = np.array(coords, dtype=float).reshape(-1)
        #Kiểm tra tọa độ
        if coords.size == 0 or not np.all(np.isfinite(coords)):
            raise ValueError("Tọa độ phải khác rỗng và chứa các giá trị hữu hạn.")
        #Kiểm tra số chiều
        if self.points:
            expected_dim = next(iter(self.points.values())).dimension
            if coords.size != expected_dim:
                raise ValueError(f"Điểm mới phải có {expected_dim} chiều, nhưng đang có {coords.size} chiều.")
        #Chèn điểm mới và cập nhật cache lân cận
        new_id = self._next_point_id
        self.points[new_id] = DataPoint(new_id, coords)
        self._update_neighborhood_after_insert(new_id)
        self._next_point_id += 1
        #Xác định core mới và UpdSeedIns
        new_core_ids = self._find_new_core_points(new_id)
        update_seeds_ids = self._get_update_seeds(new_core_ids)
        #Cập nhật kết quả gom cụm
        if not update_seeds_ids:
            self._assign_border_or_noise(new_id)
        else:
            self._expand_new_connections(new_core_ids, update_seeds_ids)

        return new_id

    def insert_point(self, point: Union[DataPoint, np.ndarray]) -> int:
        """Tên tương thích với các caller cũ; cùng xử lý như ``insert``."""
        return self.insert(point)

    def _update_neighborhood_after_insert(self, new_point_id: int) -> Set[int]:
        """Tính lân cận của điểm mới và cập nhật neighbor_ids 2 chiều"""

        #Tìm các điểm thuộc lân cận Eps của điểm mới
        neighbor_ids = self._region_query(new_point_id)
        #Lưu cache lân cận cho điểm mới
        self.points[new_point_id].set_neighborhood(neighbor_ids)
        #Thêm id điểm mới vào cache của lân cận
        for neighbor_id in sorted(neighbor_ids - {new_point_id}):
            self.points[neighbor_id].neighbor_ids.add(new_point_id)

        return neighbor_ids

    def _find_new_core_points(self, new_point_id: int) -> Set[int]:
        """Tìm các đối tượng mới trở thành lõi sau khi chèn"""

        new_core_ids: Set[int] = set()
        new_point = self.points[new_point_id]
        #Kiểm tra điểm mới
        if new_point.neighbor_count >= self.min_pts:
            new_core_ids.add(new_point_id)
        #Kiểm tra lân cận
        for neighbor_id in new_point.neighbor_ids:
            neighbor = self.points[neighbor_id]
            #Điểm cũ có lân cận tăng lên bằng MinPts
            if neighbor.neighbor_count == self.min_pts:
                new_core_ids.add(neighbor_id)

        return new_core_ids

    def _get_update_seeds(self, new_core_ids: Set[int]) -> Set[int]:
        """Xác định Upd SeedIns từ các đối tượng lõi mới"""

        update_seed_ids: Set[int] = set()
        #Xét các lõi mới sau phép chèn
        for core_id in sorted(new_core_ids):
            #tìm các lõi thuộc lân cận của lõi mới
            for neighbor_id in sorted(self.points[core_id].neighbor_ids):
                neighbor = self.points[neighbor_id]
                if neighbor.neighbor_count >= self.min_pts:
                    update_seed_ids.add(neighbor_id)

        return update_seed_ids

    def _assign_border_or_noise(self, new_point_id: int) -> None:
        """Gán điểm mới thành border hoặc noise nếu ko phải là core mới"""

        new_point = self.points[new_point_id]
        #Tìm các cụm chứa core trong lân cận của điểm mới
        core_cluster_ids = {
            self.points[neighbor_id].cluster_id
            for neighbor_id in new_point.neighbor_ids - {new_point_id}
            if self.points[neighbor_id].point_type == PointType.CORE}
        if core_cluster_ids:
            #Điểm mới thuộc lân cận của ít nhất 1 core, gán thành border
            new_point.cluster_id = min(core_cluster_ids)
            new_point.point_type = PointType.BORDER
        else:
            # Điểm mới không thuộc cụm nào.
            new_point.cluster_id = -1
            new_point.point_type = PointType.NOISE

    def _build_insert_graph(self, new_core_ids: Set[int], update_seed_ids: Set[int]) -> Dict[GraphNode, Set[GraphNode]]:
        """Biểu diễn Core mới và các cụm cũ liên quan trong đồ thị thu gọn."""

        graph: Dict[GraphNode, Set[GraphNode]] = {("new", core_id): set() for core_id in new_core_ids}
        for core_id in sorted(new_core_ids):
            core_node = ("new", core_id)
            for neighbor_id in sorted(self.points[core_id].neighbor_ids):
                if neighbor_id not in update_seed_ids:
                    continue
                if neighbor_id in new_core_ids:
                    neighbor_node = ("new", neighbor_id)
                else:
                    old_core = self.points[neighbor_id]
                    if old_core.point_type != PointType.CORE:
                        raise RuntimeError("UpdSeedIns chứa điểm cũ không phải Core.")
                    if old_core.cluster_id is None or old_core.cluster_id < 0:
                        raise RuntimeError("Core cũ không có nhãn cụm hợp lệ.")
                    neighbor_node = ("cluster", old_core.cluster_id)
                if core_node != neighbor_node:
                    graph[core_node].add(neighbor_node)
                    graph.setdefault(neighbor_node, set()).add(core_node)

        return graph
    def _get_connected_insert_groups(self, graph: Dict[GraphNode, Set[GraphNode]]) -> List[Set[GraphNode]]:
        """Tìm các nhóm đỉnh kết nối trong đồ thị thu gọn bằng BFS."""

        groups: List[Set[GraphNode]] = []
        visited: Set[GraphNode] = set()
        for start_node in sorted(graph):
            if start_node in visited:
                continue
            group: Set[GraphNode] = set()
            queue = deque([start_node])
            visited.add(start_node)
            while queue:
                node = queue.popleft()
                group.add(node)
                for neighbor_node in sorted(graph[node]):
                    if neighbor_node not in visited:
                        visited.add(neighbor_node)
                        queue.append(neighbor_node)
            groups.append(group)
        return groups

    def _expand_new_connections(self, new_core_ids: Set[int], update_seed_ids: Set[int]) -> None:
        """Xử lý Creation, Absorption và Merge bằng đồ thị thu gọn."""

        graph = self._build_insert_graph(new_core_ids, update_seed_ids)
        groups = self._get_connected_insert_groups(graph)
        relabel: Dict[int, int] = {}
        new_core_labels: Dict[int, int] = {}
        for group in groups:
            old_clusters = sorted(node_id for kind, node_id in group if kind == "cluster")
            # Chọn nhãn cụm cho từng nhóm kết nối
            if old_clusters:
                target_cluster = old_clusters[0]
                for cluster_id in old_clusters[1:]:
                    relabel[cluster_id] = target_cluster
            else:
                target_cluster = self._next_cluster_id
                self._next_cluster_id += 1
            # Lưu nhãn cần gán cho các Core mới
            for kind, node_id in group:
                if kind == "new":
                    new_core_labels[node_id] = target_cluster
        # Chỉ đổi nhãn của cụm cũ nếu thực sự xảy ra Merge
        if relabel:
            for source_id, target_id in relabel.items():
                ClusterUtils.merge_clusters(self.points, source_id, target_id)
        # Cập nhật nhãn và trạng thái của các Core mới
        for core_id, cluster_id in new_core_labels.items():
            core = self.points[core_id]
            core.cluster_id = cluster_id
            core.point_type = PointType.CORE
        # Xác định các Border trong lân cận của Core mới
        affected_border_ids: Set[int] = set()
        for core_id in new_core_ids:
            for neighbor_id in self.points[core_id].neighbor_ids:
                if self.points[neighbor_id].neighbor_count < self.min_pts:
                    affected_border_ids.add(neighbor_id)
        # Gán nhãn cho Border mới hoặc Noise chuyển thành Border
        for border_id in sorted(affected_border_ids):
            border = self.points[border_id]
            if border.cluster_id is None or border.cluster_id == -1:
                adjacent_clusters: Set[int] = set()
                for neighbor_id in border.neighbor_ids:
                    neighbor = self.points[neighbor_id]
                    if (neighbor.point_type == PointType.CORE
                        and neighbor.cluster_id is not None
                        and neighbor.cluster_id >= 0):
                        adjacent_clusters.add(neighbor.cluster_id)
                if not adjacent_clusters:
                    raise RuntimeError("Border không nằm trong lân cận của Core nào.")
                border.cluster_id = min(adjacent_clusters)
            border.point_type = PointType.BORDER

    def delete_point(self, point_id: int) -> None:
        """Xóa một điểm và phân cụm lại các cụm có thể bị ảnh hưởng.

        Bộ nhớ lân cận lưu ID ổn định, vì vậy việc xóa không làm thay đổi ID
        hay cần đánh lại chỉ số cho những điểm còn lại.
        """
        if point_id not in self.points:
            raise ValueError(f"Không tìm thấy DataPoint có ID = {point_id}")

        deleted_point = self.points.pop(point_id)
        affected_cluster_ids = {
            self.points[neighbor_id].cluster_id
            for neighbor_id in deleted_point.neighbor_ids
            if neighbor_id != point_id
            and neighbor_id in self.points
            and self.points[neighbor_id].cluster_id is not None
            and self.points[neighbor_id].cluster_id >= 0
        }
        if deleted_point.cluster_id is not None and deleted_point.cluster_id >= 0:
            affected_cluster_ids.add(deleted_point.cluster_id)

        for neighbor_id in deleted_point.neighbor_ids:
            neighbor = self.points.get(neighbor_id)
            if neighbor is not None:
                neighbor.neighbor_ids.discard(point_id)

        if not affected_cluster_ids:
            return

        affected_ids = {
            candidate_id
            for candidate_id, point in self.points.items()
            if point.cluster_id in affected_cluster_ids
        }
        old_cluster_by_id = {
            candidate_id: self.points[candidate_id].cluster_id
            for candidate_id in affected_ids
        }

        # Xóa nhãn cũ trong vùng ảnh hưởng và xác định lại Core sau khi
        # neighbor_ids đã được cập nhật.
        for candidate_id in affected_ids:
            point = self.points[candidate_id]
            point.cluster_id = None
            point.point_type = (
                PointType.CORE
                if point.neighbor_count >= self.min_pts
                else PointType.BORDER
            )

        core_ids = {
            candidate_id
            for candidate_id in affected_ids
            if self.points[candidate_id].is_core
        }
        components: List[Set[int]] = []
        visited: Set[int] = set()
        for start_id in sorted(core_ids):
            if start_id in visited:
                continue
            component: Set[int] = set()
            queue = deque([start_id])
            visited.add(start_id)
            while queue:
                current_id = queue.popleft()
                component.add(current_id)
                for neighbor_id in sorted(self.points[current_id].neighbor_ids & core_ids):
                    if neighbor_id not in visited:
                        visited.add(neighbor_id)
                        queue.append(neighbor_id)
            components.append(component)

        # Giữ lại ID cũ cho một thành phần của mỗi cụm. Những thành phần mới
        # hình thành do split nhận ID tăng dần, không tái sử dụng ID đã cấp.
        used_cluster_ids: Set[int] = set()
        for component in components:
            previous_ids = sorted({
                old_cluster_by_id[candidate_id]
                for candidate_id in component
                if old_cluster_by_id[candidate_id] is not None
                and old_cluster_by_id[candidate_id] >= 0
            })
            cluster_id = next(
                (old_id for old_id in previous_ids if old_id not in used_cluster_ids),
                None,
            )
            if cluster_id is None:
                cluster_id = self._next_cluster_id
                self._next_cluster_id += 1
            used_cluster_ids.add(cluster_id)
            for candidate_id in component:
                self.points[candidate_id].cluster_id = cluster_id

        ClusterUtils.relabel_unreachable_points(
            self.points, affected_ids, self._region_query
        )
        active_cluster_ids = {
            point.cluster_id
            for point in self.points.values()
            if point.is_core and point.cluster_id is not None and point.cluster_id >= 0
        }
        ClusterUtils.clean_stale_cluster_ids(self.points, active_cluster_ids)

    def delete(self, point_id: int) -> None:
        """Tên tương thích với giao diện tài liệu; cùng xử lý như ``delete_point``."""
        self.delete_point(point_id)
