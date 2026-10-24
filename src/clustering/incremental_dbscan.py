"""Cập nhật cụm DBSCAN cục bộ sau mỗi lần chèn hoặc xóa điểm."""

from collections import deque
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union

import numpy as np

from src.clustering.dbscan import DBSCAN
from src.clustering.utils import ClusterUtils
from src.core.data_point import DataPoint, PointType

GraphNode = Tuple[str, int]


class IncrementalDBSCAN(DBSCAN):
    """DBSCAN hỗ trợ chèn và xóa điểm mà không chạy lại ``fit`` toàn bộ."""

    def insert(self, point: Union[DataPoint, Iterable[float]]) -> int:
        """Chèn một điểm và cập nhật các điểm nằm trong vùng ảnh hưởng.

        Args:
            point: ``DataPoint`` hoặc vector tọa độ của điểm mới.

        Returns:
            ID ổn định được cấp cho điểm mới.

        Raises:
            ValueError: Nếu tọa độ rỗng, không hữu hạn hoặc sai số chiều.
        """
        coords = self._coerce_coordinates(point)
        new_point_id = self._next_point_id
        self.points[new_point_id] = DataPoint(new_point_id, coords)
        self._update_neighborhood_after_insert(new_point_id)
        self._next_point_id += 1

        new_core_ids = self._find_new_core_points(new_point_id)
        update_seed_ids = self._get_update_seeds(new_core_ids)
        if update_seed_ids:
            self._expand_new_connections(new_core_ids, update_seed_ids)
        else:
            self._assign_border_or_noise(new_point_id)

        return new_point_id

    def _coerce_coordinates(self, point: Union[DataPoint, Iterable[float]]) -> np.ndarray:
        """Chuẩn hóa và kiểm tra tọa độ trước khi thêm điểm.

        Args:
            point: Điểm dữ liệu hoặc vector tọa độ.

        Returns:
            Vector tọa độ một chiều kiểu số thực.

        Raises:
            ValueError: Nếu vector rỗng, chứa giá trị không hữu hạn hoặc
                không cùng số chiều với dữ liệu hiện tại.
        """
        raw_coords = point.coords if isinstance(point, DataPoint) else point
        coords = np.asarray(raw_coords, dtype=float).reshape(-1)
        if coords.size == 0 or not np.all(np.isfinite(coords)):
            raise ValueError("Tọa độ phải khác rỗng và chứa các giá trị hữu hạn.")

        if self.points:
            expected_dimension = next(iter(self.points.values())).dimension
            if coords.size != expected_dimension:
                raise ValueError(
                    f"Điểm mới phải có {expected_dimension} chiều, "
                    f"nhưng đang có {coords.size} chiều."
                )
        return coords

    def insert_point(self, point: Union[DataPoint, Iterable[float]]) -> int:
        """Tên tương thích cũ; chuyển tiếp tới :meth:`insert`.

        Args:
            point: ``DataPoint`` hoặc vector tọa độ của điểm mới.

        Returns:
            ID ổn định được cấp cho điểm mới.
        """
        return self.insert(point)

    def _update_neighborhood_after_insert(self, new_point_id: int) -> Set[int]:
        """Cập nhật cache lân cận hai chiều sau khi thêm điểm.

        Args:
            new_point_id: ID của điểm vừa được thêm vào ``points``.

        Returns:
            Tập ID nằm trong epsilon-lân cận của điểm mới, bao gồm chính nó.
        """
        neighbor_ids = self._region_query(new_point_id)
        self.points[new_point_id].set_neighborhood(neighbor_ids)

        for neighbor_id in sorted(neighbor_ids - {new_point_id}):
            self.points[neighbor_id].neighbor_ids.add(new_point_id)
        return neighbor_ids

    def _find_new_core_points(self, new_point_id: int) -> Set[int]:
        """Tìm điểm mới và điểm cũ vừa đạt ngưỡng ``min_pts``.

        Args:
            new_point_id: ID của điểm vừa được chèn.

        Returns:
            ID các điểm mới trở thành core.
        """
        new_core_ids: Set[int] = set()
        new_point = self.points[new_point_id]
        if new_point.neighbor_count >= self.min_pts:
            new_core_ids.add(new_point_id)

        for neighbor_id in sorted(new_point.neighbor_ids):
            neighbor = self.points[neighbor_id]
            if neighbor.neighbor_count == self.min_pts:
                new_core_ids.add(neighbor_id)
        return new_core_ids

    def _get_update_seeds(self, new_core_ids: Set[int]) -> Set[int]:
        """Tạo tập ``UpdSeedIns`` từ các core vừa phát sinh.

        Args:
            new_core_ids: ID các điểm vừa trở thành core.

        Returns:
            ID các điểm core trong lân cận của core mới.
        """
        update_seed_ids: Set[int] = set()
        for core_id in sorted(new_core_ids):
            for neighbor_id in sorted(self.points[core_id].neighbor_ids):
                neighbor = self.points[neighbor_id]
                if neighbor.neighbor_count >= self.min_pts:
                    update_seed_ids.add(neighbor_id)
        return update_seed_ids

    def _assign_border_or_noise(self, new_point_id: int) -> None:
        """Gán điểm mới thành border nếu kề core, nếu không thì thành noise.

        Args:
            new_point_id: ID điểm mới không phát sinh cập nhật seed.
        """
        new_point = self.points[new_point_id]
        core_cluster_ids: Set[int] = set()
        for neighbor_id in new_point.neighbor_ids - {new_point_id}:
            neighbor = self.points[neighbor_id]
            if (
                neighbor.point_type == PointType.CORE
                and neighbor.cluster_id is not None
                and neighbor.cluster_id >= 0
            ):
                core_cluster_ids.add(neighbor.cluster_id)
        if core_cluster_ids:
            new_point.cluster_id = min(core_cluster_ids)
            new_point.point_type = PointType.BORDER
        else:
            new_point.cluster_id = -1
            new_point.point_type = PointType.NOISE

    def _build_insert_graph(
        self,
        new_core_ids: Set[int],
        update_seed_ids: Set[int],
    ) -> Dict[GraphNode, Set[GraphNode]]:
        """Lập đồ thị rút gọn giữa core mới và các cụm cũ liên quan.

        Args:
            new_core_ids: ID các điểm vừa trở thành core.
            update_seed_ids: ID các core thuộc ``UpdSeedIns``.

        Returns:
            Đồ thị vô hướng có đỉnh dạng ``("new", point_id)`` hoặc
            ``("cluster", cluster_id)``.

        Raises:
            RuntimeError: Nếu một update seed cũ không có trạng thái core hợp lệ.
        """
        graph: Dict[GraphNode, Set[GraphNode]] = {
            ("new", core_id): set() for core_id in new_core_ids
        }
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

    def _get_connected_insert_groups(
        self,
        graph: Dict[GraphNode, Set[GraphNode]],
    ) -> List[Set[GraphNode]]:
        """Tìm các thành phần liên thông trong đồ thị cập nhật insert.

        Args:
            graph: Đồ thị do :meth:`_build_insert_graph` tạo.

        Returns:
            Danh sách các tập đỉnh liên thông.
        """
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

    def _expand_new_connections(
        self,
        new_core_ids: Set[int],
        update_seed_ids: Set[int],
    ) -> None:
        """Cập nhật nhãn cụm, core và border sau khi phát sinh update seed.

        Args:
            new_core_ids: ID các điểm vừa trở thành core.
            update_seed_ids: ID các core trong ``UpdSeedIns``.
        """
        graph = self._build_insert_graph(new_core_ids, update_seed_ids)
        groups = self._get_connected_insert_groups(graph)
        cluster_relabels, new_core_labels = self._get_insert_cluster_labels(groups)

        for source_id, target_id in cluster_relabels.items():
            ClusterUtils.merge_clusters(self.points, source_id, target_id)

        for core_id, cluster_id in new_core_labels.items():
            core = self.points[core_id]
            core.cluster_id = cluster_id
            core.point_type = PointType.CORE

        self._relabel_borders_near_new_cores(new_core_ids)

    def _get_insert_cluster_labels(
        self,
        groups: List[Set[GraphNode]],
    ) -> Tuple[Dict[int, int], Dict[int, int]]:
        """Tính ánh xạ nhãn cụm cũ và nhãn cho core mới sau insert.

        Args:
            groups: Các thành phần liên thông trong đồ thị insert.

        Returns:
            Cặp ``(cluster_relabels, new_core_labels)``.
        """
        cluster_relabels: Dict[int, int] = {}
        new_core_labels: Dict[int, int] = {}
        for group in groups:
            old_cluster_ids = sorted(
                node_id for kind, node_id in group if kind == "cluster"
            )
            if old_cluster_ids:
                target_cluster_id = old_cluster_ids[0]
                for source_cluster_id in old_cluster_ids[1:]:
                    cluster_relabels[source_cluster_id] = target_cluster_id
            else:
                target_cluster_id = self._next_cluster_id
                self._next_cluster_id += 1

            for node_type, node_id in group:
                if node_type == "new":
                    new_core_labels[node_id] = target_cluster_id
        return cluster_relabels, new_core_labels

    def _relabel_borders_near_new_cores(self, new_core_ids: Set[int]) -> None:
        """Gán nhãn cho border mới và noise được core mới hấp thụ.

        Args:
            new_core_ids: ID các core vừa phát sinh.

        Raises:
            RuntimeError: Nếu một điểm được xác định là border nhưng không kề core.
        """
        affected_border_ids = {
            neighbor_id
            for core_id in new_core_ids
            for neighbor_id in self.points[core_id].neighbor_ids
            if self.points[neighbor_id].neighbor_count < self.min_pts
        }
        for border_id in sorted(affected_border_ids):
            border = self.points[border_id]
            if border.cluster_id is None or border.cluster_id == -1:
                adjacent_cluster_ids: Set[int] = set()
                for neighbor_id in border.neighbor_ids:
                    neighbor = self.points[neighbor_id]
                    if (
                        neighbor.point_type == PointType.CORE
                        and neighbor.cluster_id is not None
                        and neighbor.cluster_id >= 0
                    ):
                        adjacent_cluster_ids.add(neighbor.cluster_id)
                if not adjacent_cluster_ids:
                    raise RuntimeError("Border không nằm trong lân cận của Core nào.")
                border.cluster_id = min(adjacent_cluster_ids)
            border.point_type = PointType.BORDER

    def delete(self, point_id: int) -> None:
        """Xóa điểm theo ID và phân cụm lại các vùng có thể bị ảnh hưởng.

        Args:
            point_id: ID ổn định của điểm cần xóa.

        Raises:
            ValueError: Nếu ID không tồn tại trong mô hình.
        """
        if point_id not in self.points:
            raise ValueError(f"Không tìm thấy DataPoint có ID = {point_id}")

        deleted_point = self.points.pop(point_id)
        affected_cluster_ids = self._find_affected_cluster_ids(deleted_point)
        self._remove_from_neighbor_caches(point_id, deleted_point.neighbor_ids)
        if not affected_cluster_ids:
            return

        affected_point_ids = self._find_affected_point_ids(affected_cluster_ids)
        previous_cluster_by_point = {
            affected_id: self.points[affected_id].cluster_id
            for affected_id in affected_point_ids
        }
        self._reset_affected_points(affected_point_ids)

        components = self._find_core_components(affected_point_ids)
        self._assign_component_cluster_ids(components, previous_cluster_by_point)
        self._relabel_after_delete(affected_point_ids)

    def _find_affected_cluster_ids(self, deleted_point: DataPoint) -> Set[int]:
        """Lấy các cụm cũ có điểm nằm trong lân cận điểm bị xóa.

        Args:
            deleted_point: Điểm đã được gỡ khỏi mô hình nhưng còn cache lân cận.

        Returns:
            ID các cụm cần cập nhật sau khi xóa.
        """
        affected_cluster_ids: Set[int] = set()
        for neighbor_id in deleted_point.neighbor_ids:
            neighbor = self.points.get(neighbor_id)
            if (
                neighbor is not None
                and neighbor.cluster_id is not None
                and neighbor.cluster_id >= 0
            ):
                affected_cluster_ids.add(neighbor.cluster_id)

        if deleted_point.cluster_id is not None and deleted_point.cluster_id >= 0:
            affected_cluster_ids.add(deleted_point.cluster_id)
        return affected_cluster_ids

    def _remove_from_neighbor_caches(
        self,
        deleted_point_id: int,
        neighbor_ids: Set[int],
    ) -> None:
        """Loại bỏ ID đã xóa khỏi cache lân cận của các điểm còn lại.

        Args:
            deleted_point_id: ID điểm vừa bị xóa.
            neighbor_ids: Lân cận được lưu trước khi xóa.
        """
        for neighbor_id in neighbor_ids:
            neighbor = self.points.get(neighbor_id)
            if neighbor is not None:
                neighbor.neighbor_ids.discard(deleted_point_id)

    def _find_affected_point_ids(self, cluster_ids: Set[int]) -> Set[int]:
        """Lấy thành viên của những cụm cần tái cấu trúc.

        Args:
            cluster_ids: ID các cụm chịu ảnh hưởng.

        Returns:
            ID điểm hiện thuộc các cụm đã cho.
        """
        return {
            point_id
            for point_id, point in self.points.items()
            if point.cluster_id in cluster_ids
        }

    def _reset_affected_points(self, point_ids: Set[int]) -> None:
        """Xóa nhãn cũ và xác định lại core/border theo cache đã cập nhật.

        Args:
            point_ids: ID các điểm thuộc cụm đang tái cấu trúc.
        """
        for point_id in point_ids:
            point = self.points[point_id]
            point.cluster_id = None
            point.point_type = (
                PointType.CORE
                if point.neighbor_count >= self.min_pts
                else PointType.BORDER
            )

    def _find_core_components(self, point_ids: Set[int]) -> List[Set[int]]:
        """Tìm các thành phần liên thông của core trong vùng bị ảnh hưởng.

        Args:
            point_ids: ID điểm thuộc các cụm cần tái cấu trúc.

        Returns:
            Các nhóm core còn kết nối với nhau; mỗi nhóm tương ứng một cụm.
        """
        core_ids = {
            point_id for point_id in point_ids if self.points[point_id].is_core
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
                core_neighbors = self.points[current_id].neighbor_ids & core_ids
                for neighbor_id in sorted(core_neighbors):
                    if neighbor_id not in visited:
                        visited.add(neighbor_id)
                        queue.append(neighbor_id)
            components.append(component)
        return components

    def _assign_component_cluster_ids(
        self,
        components: List[Set[int]],
        previous_cluster_by_point: Dict[int, Optional[int]],
    ) -> None:
        """Giữ nhãn cũ khi có thể và cấp nhãn mới cho thành phần tách ra.

        Args:
            components: Các thành phần core còn lại sau xóa.
            previous_cluster_by_point: Nhãn trước khi tái cấu trúc theo ID điểm.
        """
        used_cluster_ids: Set[int] = set()
        for component in components:
            previous_cluster_ids: Set[int] = set()
            for point_id in component:
                previous_cluster_id = previous_cluster_by_point[point_id]
                if previous_cluster_id is not None and previous_cluster_id >= 0:
                    previous_cluster_ids.add(previous_cluster_id)
            cluster_id = next(
                (
                    old_cluster_id
                    for old_cluster_id in sorted(previous_cluster_ids)
                    if old_cluster_id not in used_cluster_ids
                ),
                None,
            )
            if cluster_id is None:
                cluster_id = self._next_cluster_id
                self._next_cluster_id += 1

            used_cluster_ids.add(cluster_id)
            for point_id in component:
                self.points[point_id].cluster_id = cluster_id

    def _relabel_after_delete(self, affected_point_ids: Set[int]) -> None:
        """Đồng bộ border/noise và loại bỏ nhãn cụm không còn core.

        Args:
            affected_point_ids: ID điểm cần gán lại trạng thái sau khi tái cấu trúc.
        """
        ClusterUtils.relabel_unreachable_points(
            self.points, affected_point_ids, self._region_query
        )
        active_cluster_ids = {
            point.cluster_id
            for point in self.points.values()
            if point.is_core and point.cluster_id is not None and point.cluster_id >= 0
        }
        ClusterUtils.clean_stale_cluster_ids(self.points, active_cluster_ids)

    def delete_point(self, point_id: int) -> None:
        """Tên tương thích cũ; chuyển tiếp tới :meth:`delete`.

        Args:
            point_id: ID ổn định của điểm cần xóa.

        Raises:
            ValueError: Nếu ID không tồn tại trong mô hình.
        """
        self.delete(point_id)
