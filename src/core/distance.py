import numpy as np
from src.core.data_point import DataPoint


class DistanceMetric:
    """Lớp tính khoảng cách Euclidean giữa các điểm dữ liệu trong không gian n chiều."""
    def calculate(self, p1: DataPoint, p2: DataPoint) -> float:
        """
        Tính khoảng cách Euclidean giữa hai DataPoint:
        d(p1, p2) = sqrt(sum((p1_i - p2_i)^2))
        """
        if p1.dimension != p2.dimension:
            raise ValueError(f"Kích thước vector không khớp: {p1.dimension}D vs {p2.dimension}D")

        diff = p1.coords - p2.coords
        return float(np.sqrt(np.sum(diff ** 2)))