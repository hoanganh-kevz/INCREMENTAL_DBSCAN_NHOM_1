from math import sqrt

import numpy as np


class DistanceMetric:
    """Tính khoảng cách độc lập với dữ liệu điểm và thuật toán gom cụm."""

    @staticmethod
    def euclidean(first: np.ndarray, second: np.ndarray) -> float:
        first_coords = np.asarray(first, dtype=float).reshape(-1)
        second_coords = np.asarray(second, dtype=float).reshape(-1)
        if first_coords.size != second_coords.size:
            raise ValueError("Hai điểm phải có cùng số chiều.")

        squared_distance = sum(
            (float(first_value) - float(second_value)) ** 2
            for first_value, second_value in zip(first_coords, second_coords)
        )
        return sqrt(squared_distance)
