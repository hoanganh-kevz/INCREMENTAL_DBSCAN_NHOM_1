from typing import Callable, Dict, Set
from src.core.data_point import DataPoint, PointType


class ClusterUtils:
    """Các hàm tiện ích xử lý hợp cụm, tách cụm và đồng bộ nhãn trong Incremental DBSCAN."""

    @staticmethod
    def merge_clusters(points: Dict[int, DataPoint], source_id: int, target_id: int) -> None:
        """Hợp hai cụm bằng cách chuyển tất cả các điểm thuộc source_id sang target_id."""
        for pt in points.values():
            if pt.cluster_id == source_id:
                pt.cluster_id = target_id

    @staticmethod
    def relabel_unreachable_points(
        points: Dict[int, DataPoint],
        affected_ids: Set[int],
        region_query_fn: Callable[[int], Set[int]],
    ) -> None:
        """Cập nhật lại nhãn BORDER hoặc NOISE cho các điểm không còn chạm tới CORE point."""
        for point_id in sorted(affected_ids):
            pt = points[point_id]
            if pt.is_core:
                continue

            neighbors = region_query_fn(point_id)
            core_neighbors = [
                neighbor_id
                for neighbor_id in sorted(neighbors)
                if points[neighbor_id].is_core
                and points[neighbor_id].cluster_id is not None
                and points[neighbor_id].cluster_id >= 0
            ]

            if core_neighbors:
                pt.cluster_id = points[core_neighbors[0]].cluster_id
                pt.point_type = PointType.BORDER
            else:
                pt.cluster_id = -1
                pt.point_type = PointType.NOISE

    @staticmethod
    def clean_stale_cluster_ids(points: Dict[int, DataPoint], active_cluster_ids: Set[int]) -> None:
        """Loại bỏ triệt để các tham chiếu cluster_id không còn tồn tại sau khi split/merge."""
        for pt in points.values():
            if pt.cluster_id is not None and pt.cluster_id >= 0 and pt.cluster_id not in active_cluster_ids:
                pt.cluster_id = -1
                pt.point_type = PointType.NOISE
