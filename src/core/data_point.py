from enum import Enum
from typing import Iterable, Optional, Set

import numpy as np


class PointType(Enum):
    """Phân loại trạng thái của điểm dữ liệu trong thuật toán DBSCAN."""
    UNCLASSIFIED = "UNCLASSIFIED"
    CORE = "CORE"
    BORDER = "BORDER"
    NOISE = "NOISE"


class DataPoint:
    """Biểu diễn một điểm dữ liệu trong không gian n chiều tổng quát."""

    def __init__(self, point_id: int, coords: np.ndarray):
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
        """Thay cache lân cận bằng các ID ổn định của điểm láng giềng."""
        self.neighbor_ids = set(neighbor_ids)

    @property
    def is_core(self) -> bool:
        """Kiểm tra điểm có phải là điểm lõi (CORE Point) hay không."""
        return self.point_type == PointType.CORE

    @property
    def dimension(self) -> int:
        """Trả về số chiều dữ liệu của điểm."""
        return self.coords.size

    def __repr__(self):
        return (f"DataPoint(id={self.id}, coords={self.coords.tolist()}, "
                f"cluster={self.cluster_id}, type={self.point_type.value}, "
                f"neighbor_ids={sorted(self.neighbor_ids)})")
