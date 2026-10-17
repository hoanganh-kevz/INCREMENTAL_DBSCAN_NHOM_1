"""OwnRefitDBSCAN: mỗi lần insert hoặc delete thì fit lại toàn bộ bằng DBSCAN của nhóm, không dùng sklearn.

Dùng để vẽ hình khi chưa có IncrementalDBSCAN và làm mốc so tốc độ ở D09. Giao diện giống RefitDBSCAN.
"""
import time

import numpy as np

from src.clustering.dbscan import DBSCAN
from .sklearn_dbscan import RefitDBSCAN
from .static_equivalence import labels_and_types


class OwnRefitDBSCAN(RefitDBSCAN):
    """Giống RefitDBSCAN, chỉ khác là fit bằng DBSCAN của nhóm."""

    def _refit(self):
        t0 = time.perf_counter()
        if len(self._coords) == 0:
            self._labels, self._types = np.empty(0, dtype=int), np.empty(0, dtype=int)
        else:
            self._labels, self._types = labels_and_types(DBSCAN(self.eps, self.min_pts).fit(self._coords))
        self.last_update_seconds = time.perf_counter() - t0
