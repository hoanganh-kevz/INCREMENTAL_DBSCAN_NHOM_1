from enum import Enum
from typing import Optional

import numpy as np


class PointType(Enum):
    """Trạng thái của một điểm trong quá trình phân cụm."""

    UNCLASSIFIED = "unclassified"
    NOISE = "noise"
    BORDER = "border"
    CORE = "core"


class DataPoint:
    """Đại diện cho một điểm dữ liệu trong không gian N chiều."""

    def __init__(self, point_id: int, coords: np.ndarray):
        self.id: int = point_id
        self.coords: np.ndarray = np.array(coords, dtype=float)
        self.cluster_id: Optional[int] = None  # None: chưa duyệt, -1: noise, >=0: ID cụm
        self.point_type: PointType = PointType.UNCLASSIFIED
        self.neighbor_count: int = 0

    @property
    def is_core(self) -> bool:
        """Giữ thuộc tính cũ để các đoạn code đang đọc is_core không bị gãy."""
        return self.point_type is PointType.CORE

    @is_core.setter
    def is_core(self, value: bool) -> None:
        """Tương thích với code cũ; trạng thái chính được lưu trong point_type."""
        if value:
            self.point_type = PointType.CORE
        elif self.point_type is PointType.CORE:
            self.point_type = (
                PointType.BORDER
                if self.cluster_id is not None and self.cluster_id >= 0
                else PointType.UNCLASSIFIED
            )

    def __repr__(self) -> str:
        return (
            f"DataPoint(id={self.id}, coords={self.coords.tolist()}, "
            f"cluster={self.cluster_id}, type={self.point_type.value})"
        )
