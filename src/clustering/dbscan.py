from collections import deque
from typing import Dict, List, Optional, Set

import numpy as np

from src.core.data_point import DataPoint, PointType
from src.core.distance import DistanceMetric


class DBSCAN:
    """DBSCAN tự cài đặt theo giả mã DBSCAN 1996.

    Dùng hàng đợi FIFO cho danh sách seeds như giả mã nguồn gốc mô tả bằng
    ``first``/``append``; cache lân cận được giữ để lớp IncrementalDBSCAN dùng.

    Attributes:
        points: Các điểm, được tra cứu theo ID ổn định.
        eps: Bán kính lân cận.
        min_pts: Số điểm tối thiểu trong lân cận để một điểm là core.
    """

    def __init__(
        self,
        eps: float,
        min_pts: int,
        metric: Optional[DistanceMetric] = None,
    ) -> None:
        """Tạo mô hình DBSCAN.

        Args:
            eps: Bán kính dùng để xác định epsilon-lân cận.
            min_pts: Số điểm tối thiểu để xác định điểm core.
            metric: Bộ tính khoảng cách; mặc định dùng ``DistanceMetric``.
        """
        self.eps: float = eps
        self.min_pts: int = min_pts
        self.metric: DistanceMetric = metric if metric is not None else DistanceMetric()
        self.points: Dict[int, DataPoint] = {}
        self._next_point_id: int = 0
        self._next_cluster_id: int = 0

    def _region_query(self, point_id: int) -> Set[int]:
        """Tìm các ID trong epsilon-lân cận của điểm.

        Args:
            point_id: ID điểm trung tâm của vùng truy vấn.

        Returns:
            Tập ID của các điểm cách điểm trung tâm không quá ``eps``; bao gồm
            chính điểm trung tâm.

        Raises:
            ValueError: Nếu ID không tồn tại.
        """
        try:
            target_point = self.points[point_id]
        except KeyError as error:
            raise ValueError(f"Không tìm thấy DataPoint có ID = {point_id}") from error

        return {
            candidate_id
            for candidate_id, candidate in self.points.items()
            if self.metric.euclidean(target_point.coords, candidate.coords) <= self.eps
        }

    def fit(self, data: np.ndarray) -> "DBSCAN":
        """Phân cụm dữ liệu ban đầu và khởi tạo trạng thái lân cận.

        Args:
            data: Ma trận có dạng ``(số điểm, số chiều)``.

        Returns:
            Chính mô hình đã được huấn luyện.
        """
        self.points = {i: DataPoint(i, row) for i, row in enumerate(data)}
        self._next_point_id = len(self.points)
        self._next_cluster_id = 0

        for point_id, point in self.points.items():
            point.set_neighborhood(self._region_query(point_id))

        for point_id, point in self.points.items():
            if point.cluster_id is not None:
                continue

            if point.neighbor_count < self.min_pts:
                point.cluster_id = -1
                point.point_type = PointType.NOISE
            else:
                point.point_type = PointType.CORE
                self._expand_cluster(
                    point_id,
                    point.neighbor_ids,
                    self._next_cluster_id,
                )
                self._next_cluster_id += 1

        return self

    def _expand_cluster(self, seed_id: int, neighbor_ids: Set[int], cluster_id: int) -> None:
        """Mở rộng cụm bằng cách duyệt các core và border tiếp cận được.

        Args:
            seed_id: ID core khởi tạo cụm.
            neighbor_ids: Lân cận của seed.
            cluster_id: Nhãn cụm cần gán.
        """
        seed = self.points[seed_id]
        seed.cluster_id = cluster_id
        seed.point_type = PointType.CORE
        queue = deque(sorted(neighbor_ids))

        while queue:
            current_id = queue.popleft()
            current_point = self.points[current_id]

            if current_point.cluster_id == -1:
                current_point.cluster_id = cluster_id
                current_point.point_type = PointType.BORDER

            if current_point.cluster_id is not None:
                continue

            current_point.cluster_id = cluster_id

            if current_point.neighbor_count >= self.min_pts:
                current_point.point_type = PointType.CORE
                queue.extend(sorted(current_point.neighbor_ids))
            else:
                current_point.point_type = PointType.BORDER

    def get_labels(self) -> List[int]:
        """Trả nhãn cụm theo thứ tự điểm hiện lưu trong mô hình.

        Returns:
            Danh sách nhãn; ``-1`` biểu thị noise hoặc điểm chưa được phân cụm.
        """
        return [
            point.cluster_id if point.cluster_id is not None else -1
            for point in self.points.values()
        ]
