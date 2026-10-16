import numpy as np
from enum import Enum
from typing import Optional, List


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
        self.coords: np.ndarray = np.array(coords, dtype=float)
        self.cluster_id: Optional[int] = None  # None: Chưa duyệt, -1: Noise, >=0: Cluster ID
        self.point_type: PointType = PointType.UNCLASSIFIED
        self.neighbor_count: int = 0
        self.neighborhood: List[int] = []

    @property
    def is_core(self) -> bool:
        """Kiểm tra điểm có phải là điểm lõi (CORE Point) hay không."""
        return self.point_type == PointType.CORE

    @property
    def dimension(self) -> int:
        """Trả về số chiều dữ liệu của điểm."""
        return len(self.coords)

    def __repr__(self):
        return (f"DataPoint(id={self.id}, coords={self.coords.tolist()}, "
                f"cluster={self.cluster_id}, type={self.point_type.value}, "
                f"neighbors={self.neighbor_count})")