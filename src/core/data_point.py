from enum import Enum
from typing import Iterable, Optional, Set

import numpy as np


class PointType(Enum):
    """Trạng thái của một điểm theo DBSCAN."""

    UNCLASSIFIED = "UNCLASSIFIED"
    CORE = "CORE"
    BORDER = "BORDER"
    NOISE = "NOISE"


class DataPoint:
    """Biểu diễn một điểm dữ liệu trong không gian n chiều tổng quát."""

    def __init__(self, point_id: int, coords: Iterable[float]) -> None:
        """Khởi tạo điểm với ID ổn định và tọa độ n chiều.

        Args:
            point_id: ID duy nhất do mô hình quản lý.
            coords: Các tọa độ của điểm.

        Raises:
            ValueError: Nếu không có tọa độ nào.
        """
        self.id: int = point_id
        self.coords: np.ndarray = np.array(coords, dtype=float, copy=True).reshape(-1)
        if self.coords.size == 0:
            raise ValueError("Tọa độ của điểm phải có ít nhất một chiều.")

        self.cluster_id: Optional[int] = None  # None: Chưa duyệt, -1: Noise, >=0: Cluster ID
        self.point_type: PointType = PointType.UNCLASSIFIED
        self.neighbor_ids: Set[int] = set()

    @property
    def neighbor_count(self) -> int:
        """Số láng giềng, suy ra từ cache để tránh lệch bộ đếm."""
        return len(self.neighbor_ids)

    def set_neighborhood(self, neighbor_ids: Iterable[int]) -> None:
        """Thay cache lân cận bằng các ID ổn định của điểm láng giềng.

        Args:
            neighbor_ids: ID điểm nằm trong epsilon-lân cận.
        """
        self.neighbor_ids = set(neighbor_ids)

    @property
    def is_core(self) -> bool:
        """Kiểm tra điểm có phải là điểm lõi (CORE Point) hay không."""
        return self.point_type == PointType.CORE

    @property
    def dimension(self) -> int:
        """Trả về số chiều dữ liệu của điểm."""
        return self.coords.size

    def __repr__(self) -> str:
        """Trả về biểu diễn ngắn gọn để kiểm tra trạng thái điểm."""
        return (f"DataPoint(id={self.id}, coords={self.coords.tolist()}, "
                f"cluster={self.cluster_id}, type={self.point_type.value}, "
                f"neighbor_ids={sorted(self.neighbor_ids)})")
