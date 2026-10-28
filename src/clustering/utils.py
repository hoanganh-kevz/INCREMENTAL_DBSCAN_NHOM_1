from typing import List, Set, Callable
from src.core.data_point import DataPoint, PointType


class ClusterUtils:
    """Các hàm tiện ích xử lý hợp cụm, tách cụm và đồng bộ nhãn trong Incremental DBSCAN."""

    @staticmethod
    def merge_clusters(points: List[DataPoint], source_id: int, target_id: int) -> None:
        """Hợp hai cụm bằng cách chuyển tất cả các điểm thuộc source_id sang target_id."""
        for pt in points:
            if pt.cluster_id == source_id:
                pt.cluster_id = target_id

    @staticmethod
    def relabel_unreachable_points(
        points: List[DataPoint],
        affected_indices: Set[int],
        region_query_fn: Callable[[int], List[int]]
    ) -> None:
        """Cập nhật lại nhãn BORDER hoặc NOISE cho các điểm không còn chạm tới CORE point."""
        for idx in affected_indices:
            pt = points[idx]
            if pt.is_core:
                continue

            neighbors = region_query_fn(idx)
            core_neighbors = [
                n for n in neighbors 
                if points[n].is_core and points[n].cluster_id is not None and points[n].cluster_id >= 0
            ]

            if core_neighbors:
                pt.cluster_id = points[core_neighbors[0]].cluster_id
                pt.point_type = PointType.BORDER
            else:
                pt.cluster_id = -1
                pt.point_type = PointType.NOISE

    @staticmethod
    def clean_stale_cluster_ids(points: List[DataPoint], active_cluster_ids: Set[int]) -> None:
        """Loại bỏ triệt để các tham chiếu cluster_id không còn tồn tại sau khi split/merge."""
        for pt in points:
            if pt.cluster_id is not None and pt.cluster_id >= 0 and pt.cluster_id not in active_cluster_ids:
                pt.cluster_id = -1
                pt.point_type = PointType.NOISE
"""
Module chứa các tiện ích thao tác trên cụm dữ liệu cho thuật toán Incremental DBSCAN.
"""

from typing import Callable, List, Set
from src.core.data_point import DataPoint, PointType


class ClusterUtils:
    """Tập hợp các tiện ích tĩnh xử lý gộp cụm, gắn lại nhãn và dọn dẹp ID cụm."""

    @staticmethod
    def merge_clusters(points: List[DataPoint], source_id: int, target_id: int) -> None:
        """Hợp hai cụm bằng cách gán lại nhãn cụm từ source_id sang target_id.

        Args:
            points (List[DataPoint]): Danh sách toàn bộ các điểm dữ liệu.
            source_id (int): ID cụm nguồn sẽ bị thay thế.
            target_id (int): ID cụm đích sẽ gán mới.
        """
        for pt in points:
            if pt.cluster_id == source_id:
                pt.cluster_id = target_id

    @staticmethod
    def relabel_unreachable_points(
        points: List[DataPoint],
        affected_indices: Set[int],
        region_query_fn: Callable[[int], List[int]],
    ) -> None:
        """Cập nhật lại trạng thái BORDER hoặc NOISE cho các điểm bị ảnh hưởng.

        Các điểm không còn kết nối tới bất kỳ CORE point nào sẽ chuyển thành NOISE.
        Các điểm kết nối tới ít nhất một CORE point sẽ gán nhãn BORDER theo cụm của CORE đó.

        Args:
            points (List[DataPoint]): Danh sách toàn bộ các điểm dữ liệu.
            affected_indices (Set[int]): Tập hợp chỉ số (index) các điểm bị ảnh hưởng.
            region_query_fn (Callable[[int], List[int]]): Hàm truy vấn lân cận theo index.
        """
        for idx in affected_indices:
            pt = points[idx]
            if pt.is_core:
                continue

            neighbors = region_query_fn(idx)
            core_neighbors = [
                n for n in neighbors
                if points[n].is_core
                and points[n].cluster_id is not None
                and points[n].cluster_id >= 0
            ]

            if core_neighbors:
                pt.cluster_id = points[core_neighbors[0]].cluster_id
                pt.point_type = PointType.BORDER
            else:
                pt.cluster_id = -1
                pt.point_type = PointType.NOISE

    @staticmethod
    def clean_stale_cluster_ids(points: List[DataPoint], active_cluster_ids: Set[int]) -> None:
        """Loại bỏ triệt để tham chiếu cluster_id mồ côi đã không còn tồn tại sau khi split/merge.

        Args:
            points (List[DataPoint]): Danh sách toàn bộ các điểm dữ liệu.
            active_cluster_ids (Set[int]): Tập hợp các ID cụm hợp lệ đang còn hoạt động.
        """
        for pt in points:
            if (
                pt.cluster_id is not None
                and pt.cluster_id >= 0
                and pt.cluster_id not in active_cluster_ids
            ):
                pt.cluster_id = -1
                pt.point_type = PointType.NOISE