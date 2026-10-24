from math import sqrt
from typing import Iterable

import numpy as np


class DistanceMetric:
    """Tính khoảng cách Euclidean độc lập với lớp DataPoint."""

    @staticmethod
    def euclidean(first: Iterable[float], second: Iterable[float]) -> float:
        """Tính khoảng cách Euclidean giữa hai vector cùng số chiều.

        Args:
            first: Tọa độ vector thứ nhất.
            second: Tọa độ vector thứ hai.

        Returns:
            Khoảng cách Euclidean.

        Raises:
            ValueError: Nếu hai vector có số chiều khác nhau.
        """
        first_coords = np.asarray(first, dtype=float).reshape(-1)
        second_coords = np.asarray(second, dtype=float).reshape(-1)
        if first_coords.size != second_coords.size:
            raise ValueError(
                "Hai vector phải có cùng số chiều: "
                f"{first_coords.size}D và {second_coords.size}D."
            )

        squared_distance = sum(
            (float(first_value) - float(second_value)) ** 2
            for first_value, second_value in zip(first_coords, second_coords)
        )
        return sqrt(squared_distance)
