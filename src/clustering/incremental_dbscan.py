from collections import deque #cho BFS
from typing import Set, Union # Union ho phép khai báo tham số có thể nhận nhiều kiểu dữ liệu
import numpy as np
from src.clustering.dbscan import DBSCAN
from src.core.data_point import DataPoint, PointType #D, tọa độ, loại điểm, nhãn cụm và tập lân cận

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
            self._expand_new_connections(update_seeds_ids)

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

    def _expand_new_connections(self, update_seed_ids: Set[int]) -> None:
        """Cập nhật các cụm dựa trên quan hệ khả đạt theo mật độ sau phép chèn."""

        visited_core_ids: Set[int] = set()
        for seed_id in sorted(update_seed_ids):
            if seed_id in visited_core_ids:
                continue
            core_ids: Set[int] = set()
            border_ids: Set[int] = set()
            old_cluster_ids: Set[int] = set()
            queue = deque([seed_id])
            #duyệt qua các lõi liên kết với nhau trong lân cận Eps
            while queue:
                core_id = queue.popleft()
                if core_id in visited_core_ids:
                    continue
                visited_core_ids.add(core_id)
                core_ids.add(core_id)
                core = self.points[core_id]
                #Lưu cụm cũ từ các điểm lõi cũ
                if core.point_type == PointType.CORE and core.cluster_id is not None:
                    old_cluster_ids.add(core.cluster_id)
                for neighbor_id in sorted(core.neighbor_ids - {core_id}):
                    neighbor = self.points[neighbor_id]
                    if neighbor.neighbor_count >= self.min_pts:
                        if neighbor_id not in visited_core_ids:
                            queue.append(neighbor_id)
                    else:
                        border_ids.add(neighbor_id)
            #Chọn id cụm sau khi cập nhật
            if old_cluster_ids:
                target_cluster_id = min(old_cluster_ids)
            else:
                target_cluster_id = self._next_cluster_id
                self._next_cluster_id += 1
            # Cập nhật nhãn cụm và tính chất lõi của các đối tượng Core tìm được
            for core_id in core_ids:
                core = self.points[core_id]
                core.cluster_id = target_cluster_id
                core.point_type = PointType.CORE
            #Cập nhật các điểm border
            for border_id in border_ids:
                border = self.points[border_id]
                border.point_type = PointType.BORDER
                if border.cluster_id is None or border.cluster_id == -1:
                    border.cluster_id = target_cluster_id
                elif border.cluster_id in old_cluster_ids:
                    border.cluster_id = target_cluster_id
