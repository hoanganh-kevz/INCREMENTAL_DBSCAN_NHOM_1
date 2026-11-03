"""Cập nhật cụm DBSCAN cục bộ sau mỗi lần chèn hoặc xóa điểm."""

from collections import deque
from typing import Dict, Iterable, List, Optional, Set, Union

import numpy as np

from src.clustering.dbscan import DBSCAN
from src.clustering.utils import ClusterUtils
from src.core.data_point import DataPoint, PointType


class IncrementalDBSCAN(DBSCAN):
    """Cập nhật DBSCAN theo thuật toán IncrementalDBSCAN của Ester et al. (1998).

    Insert mở rộng từ UpdSeedIns bằng stack; delete xử lý Removal, Reduction
    và Potential Split bằng BFS bắt đầu từ UpdSeedDel.
    """

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

    def _update_neighborhood_after_insert(self, new_point_id: int) -> None:
        """Cập nhật cache lân cận hai chiều sau khi thêm điểm.

        Args:
            new_point_id: ID của điểm vừa được thêm vào ``points``.

        """
        neighbor_ids = self._region_query(new_point_id)
        self.points[new_point_id].set_neighborhood(neighbor_ids)

        for neighbor_id in sorted(neighbor_ids - {new_point_id}):
            self.points[neighbor_id].neighbor_ids.add(new_point_id)

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

    def _expand_new_connections(
        self,
        new_core_ids: Set[int],
        update_seed_ids: Set[int],
    ) -> None:
        """Mở rộng DBSCAN trong vùng bị ảnh hưởng, bắt đầu từ UpdSeedIns."""
        components = self._get_insert_core_components(new_core_ids, update_seed_ids)
        parents = list(range(len(components)))

        def find(component_id: int) -> int:
            while parents[component_id] != component_id:
                parents[component_id] = parents[parents[component_id]]
                component_id = parents[component_id]
            return component_id

        def union(first_id: int, second_id: int) -> None:
            first_root = find(first_id)
            second_root = find(second_id)
            if first_root != second_root:
                parents[max(first_root, second_root)] = min(first_root, second_root)

        old_clusters_by_component: Dict[int, Set[int]] = {}
        component_by_old_cluster: Dict[int, int] = {}
        for component_id, component in enumerate(components):
            old_cluster_ids: Set[int] = set()
            for core_id in component:
                previous_cluster_id = self.points[core_id].cluster_id
                if previous_cluster_id is not None and previous_cluster_id >= 0:
                    old_cluster_ids.add(previous_cluster_id)

                old_core_seed_ids = (
                    self.points[core_id].neighbor_ids & update_seed_ids
                ) - new_core_ids
                for seed_id in old_core_seed_ids:
                    old_seed = self.points[seed_id]
                    if (
                        old_seed.is_core
                        and old_seed.cluster_id is not None
                        and old_seed.cluster_id >= 0
                    ):
                        old_cluster_ids.add(old_seed.cluster_id)

            old_clusters_by_component[component_id] = old_cluster_ids
            for cluster_id in sorted(old_cluster_ids):
                previous_component = component_by_old_cluster.setdefault(
                    cluster_id, component_id
                )
                # Các nhóm lõi mới cùng nối vào một cụm cũ thuộc cùng một cụm sau cập nhật.
                union(component_id, previous_component)

        core_ids_by_group: Dict[int, Set[int]] = {}
        old_cluster_ids_by_group: Dict[int, Set[int]] = {}
        for component_id, component in enumerate(components):
            root_id = find(component_id)
            core_ids_by_group.setdefault(root_id, set()).update(component)
            old_cluster_ids_by_group.setdefault(root_id, set()).update(
                old_clusters_by_component[component_id]
            )

        for group_id in sorted(core_ids_by_group):
            old_cluster_ids = old_cluster_ids_by_group[group_id]
            if old_cluster_ids:
                target_cluster_id = min(old_cluster_ids)
                # Absorption giữ một cụm cũ; Merge hợp nhất các cụm cũ được nối.
                for source_cluster_id in sorted(old_cluster_ids - {target_cluster_id}):
                    ClusterUtils.merge_clusters(
                        self.points, source_cluster_id, target_cluster_id
                    )
            else:
                # Creation: nhóm lõi mới chưa nối với cụm nào trước đó.
                target_cluster_id = self._next_cluster_id
                self._next_cluster_id += 1

            for core_id in core_ids_by_group[group_id]:
                self.points[core_id].cluster_id = target_cluster_id
                self.points[core_id].point_type = PointType.CORE

        self._relabel_borders_near_new_cores(new_core_ids)

    def _get_insert_core_components(
        self,
        new_core_ids: Set[int],
        update_seed_ids: Set[int],
    ) -> List[Set[int]]:
        """Duyệt các điểm NewCore trong UpdSeedIns bằng stack LIFO."""
        remaining_ids = set(new_core_ids) & set(update_seed_ids)
        components: List[Set[int]] = []
        while remaining_ids:
            start_id = min(remaining_ids)
            remaining_ids.remove(start_id)
            stack = [start_id]
            component: Set[int] = set()
            while stack:
                current_id = stack.pop()
                component.add(current_id)
                for neighbor_id in sorted(
                    self.points[current_id].neighbor_ids & remaining_ids,
                    reverse=True,
                ):
                    remaining_ids.remove(neighbor_id)
                    stack.append(neighbor_id)
            components.append(component)
        return components

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
        """Xóa một điểm và cập nhật cụm theo tập ``UpdSeedDel``.

        Args:
            point_id: ID ổn định của điểm cần xóa.

        Raises:
            ValueError: Nếu ID không tồn tại trong mô hình.
        """
        if point_id not in self.points:
            raise ValueError(f"Không tìm thấy DataPoint có ID = {point_id}")

        deleted_point = self.points[point_id]
        deleted_neighbor_ids = deleted_point.neighbor_ids - {point_id}
        lost_core_ids = {point_id} if deleted_point.is_core else set()

        # Một Core kề p mất tính Core khi số láng giềng trước khi xóa đúng MinPts.
        for neighbor_id in deleted_neighbor_ids:
            neighbor = self.points[neighbor_id]
            if neighbor.is_core and neighbor.neighbor_count == self.min_pts:
                lost_core_ids.add(neighbor_id)

        affected_cluster_ids = {
            self.points[lost_id].cluster_id
            for lost_id in lost_core_ids - {point_id}
            if self.points[lost_id].cluster_id is not None
            and self.points[lost_id].cluster_id >= 0
        }
        if (
            deleted_point.is_core
            and deleted_point.cluster_id is not None
            and deleted_point.cluster_id >= 0
        ):
            affected_cluster_ids.add(deleted_point.cluster_id)

        self.points.pop(point_id)
        self._remove_from_neighbor_caches(point_id, deleted_point.neighbor_ids)
        if not lost_core_ids:
            return

        for lost_id in lost_core_ids - {point_id}:
            self.points[lost_id].point_type = PointType.BORDER

        update_seed_ids = self._get_delete_update_seeds(deleted_point, lost_core_ids)
        seeds_by_cluster: Dict[int, Set[int]] = {}
        for seed_id in update_seed_ids:
            cluster_id = self.points[seed_id].cluster_id
            if cluster_id is not None and cluster_id >= 0:
                seeds_by_cluster.setdefault(cluster_id, set()).add(seed_id)

        # N_Eps(p) và N_Eps(q') là các điểm có thể đổi Core/Border/Noise.
        affected_point_ids = set(deleted_neighbor_ids)
        for lost_id in lost_core_ids:
            if lost_id == point_id:
                affected_point_ids.update(deleted_neighbor_ids)
            else:
                affected_point_ids.add(lost_id)
                affected_point_ids.update(self.points[lost_id].neighbor_ids)

        for cluster_id in sorted(affected_cluster_ids):
            seed_ids = seeds_by_cluster.get(cluster_id, set())
            if not seed_ids:
                # Removal: không còn Core nào trong UpdSeedDel.
                continue

            if self._are_seeds_directly_connected(seed_ids):
                # Reduction: các seed trực tiếp nối nhau nên cụm không thể tách.
                continue

            components = self._find_delete_components(seed_ids, cluster_id)
            if components is None:
                # BFS tìm được đường nối qua các Core trung gian: Non-split.
                continue

            # Potential Split: giữ ID cũ cho thành phần đầu tiên, cấp ID mới
            # cho các thành phần còn lại.
            components.sort(key=min)
            for index, component in enumerate(components):
                new_cluster_id = cluster_id if index == 0 else self._next_cluster_id
                if index > 0:
                    self._next_cluster_id += 1
                for core_id in component:
                    self.points[core_id].cluster_id = new_cluster_id
                    affected_point_ids.update(self.points[core_id].neighbor_ids)

        self._relabel_after_delete(affected_point_ids)

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

    def _get_delete_update_seeds(
        self,
        deleted_point: DataPoint,
        lost_core_ids: Set[int],
    ) -> Set[int]:
        """Tính UpdSeedDel từ các Core vừa mất tính Core."""
        update_seed_ids: Set[int] = set()
        for lost_core_id in lost_core_ids:
            neighbor_ids = (
                deleted_point.neighbor_ids - {deleted_point.id}
                if lost_core_id == deleted_point.id
                else self.points[lost_core_id].neighbor_ids
            )
            update_seed_ids.update(
                neighbor_id
                for neighbor_id in neighbor_ids
                if self.points[neighbor_id].is_core
            )
        return update_seed_ids

    def _are_seeds_directly_connected(self, seed_ids: Set[int]) -> bool:
        """Kiểm tra trường hợp Reduction: các seed đôi một kề trực tiếp."""
        ordered_seeds = sorted(seed_ids)
        return all(
            second_id in self.points[first_id].neighbor_ids
            for index, first_id in enumerate(ordered_seeds)
            for second_id in ordered_seeds[index + 1 :]
        )

    def _find_delete_components(
        self,
        seed_ids: Set[int],
        cluster_id: int,
    ) -> Optional[List[Set[int]]]:
        """Chạy BFS song song từ UpdSeedDel để phân loại Split/Non-split.

        Trả None khi các seed đã nối được với nhau; nếu không, trả các thành
        phần lõi tìm được trong cụm cũ.
        """
        parents = {seed_id: seed_id for seed_id in seed_ids}
        owner = {seed_id: seed_id for seed_id in seed_ids}
        queue = deque(sorted(seed_ids))

        def find(seed_id: int) -> int:
            while parents[seed_id] != seed_id:
                parents[seed_id] = parents[parents[seed_id]]
                seed_id = parents[seed_id]
            return seed_id

        def union(first_seed: int, second_seed: int) -> None:
            first_root = find(first_seed)
            second_root = find(second_seed)
            if first_root != second_root:
                parents[max(first_root, second_root)] = min(first_root, second_root)

        while queue:
            current_id = queue.popleft()
            current_seed = find(owner[current_id])
            for neighbor_id in sorted(self.points[current_id].neighbor_ids):
                neighbor = self.points[neighbor_id]
                if not neighbor.is_core or neighbor.cluster_id != cluster_id:
                    continue

                if neighbor_id not in owner:
                    owner[neighbor_id] = current_seed
                    queue.append(neighbor_id)
                else:
                    union(current_seed, owner[neighbor_id])
                    if len({find(seed_id) for seed_id in seed_ids}) == 1:
                        return None

        components_by_seed: Dict[int, Set[int]] = {}
        for point_id, seed_id in owner.items():
            components_by_seed.setdefault(find(seed_id), set()).add(point_id)
        return list(components_by_seed.values())

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
