from typing import Callable, Dict, Set

from src.core.data_point import DataPoint, PointType


class ClusterUtils:
    """Các hàm tiện ích xử lý hợp cụm, tách cụm và đồng bộ nhãn trong Incremental DBSCAN."""

    @staticmethod
    def merge_clusters(points: Dict[int, DataPoint], source_id: int, target_id: int) -> None:
        """Đổi nhãn mọi điểm thuộc cụm nguồn sang cụm đích.

        Args:
            points: Các điểm được lập chỉ mục theo ID.
            source_id: Nhãn cụm cần gộp.
            target_id: Nhãn cụm sẽ được giữ lại.
        """
        for point in points.values():
            if point.cluster_id == source_id:
                point.cluster_id = target_id

    @staticmethod
    def relabel_unreachable_points(
        points: Dict[int, DataPoint],
        affected_ids: Set[int],
        region_query_fn: Callable[[int], Set[int]],
    ) -> None:
        """Gán lại border/noise cho điểm không còn kề core sau khi cập nhật.

        Args:
            points: Các điểm được lập chỉ mục theo ID.
            affected_ids: Các điểm cần xác định lại nhãn.
            region_query_fn: Hàm lấy lân cận theo ID điểm.
        """
        for point_id in sorted(affected_ids):
            point = points[point_id]
            if point.is_core:
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
                point.cluster_id = points[core_neighbors[0]].cluster_id
                point.point_type = PointType.BORDER
            else:
                point.cluster_id = -1
                point.point_type = PointType.NOISE

    @staticmethod
    def clean_stale_cluster_ids(points: Dict[int, DataPoint], active_cluster_ids: Set[int]) -> None:
        """Đưa điểm về noise nếu nhãn cụm không còn core nào sở hữu.

        Args:
            points: Các điểm được lập chỉ mục theo ID.
            active_cluster_ids: Nhãn cụm còn ít nhất một điểm core.
        """
        for point in points.values():
            if (
                point.cluster_id is not None
                and point.cluster_id >= 0
                and point.cluster_id not in active_cluster_ids
            ):
                point.cluster_id = -1
                point.point_type = PointType.NOISE
