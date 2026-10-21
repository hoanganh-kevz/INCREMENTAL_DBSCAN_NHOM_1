from collections import deque #cho BFS
from typing import Dict, List, Set, Tuple, Union
import numpy as np
from src.clustering.dbscan import DBSCAN
from src.core.data_point import DataPoint, PointType 

GraphNode = Tuple[str, int]

class IncrementalDBSCAN(DBSCAN):
    """Mở rộng DBSCAN để hỗ trợ cập nhật cụm khi thay đổi dữ liệu"""

    def insert(self, point: Union[DataPoint, np.ndarray]) -> int: #trả về ID của điểm thêm thành công
        """Chèn một điểm mới và cập nhật kết quả phân cụm."""
        
        #Lấy tọa độ của điểm được chèn 
        coords = point.coords if isinstance(point, DataPoint) else point
        coords = np.array(coords, dtype = float).reshape(-1)
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
            #Điểm mới ko ở trong cụm 
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
            for point in self.points.values():
                if point.cluster_id in relabel:
                    point.cluster_id = relabel[point.cluster_id]
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
