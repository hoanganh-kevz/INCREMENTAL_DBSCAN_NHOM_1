from src.clustering.utils import ClusterUtils
from src.core.data_point import DataPoint, PointType


def _point(point_id, cluster_id, point_type):
    point = DataPoint(point_id, [float(point_id)])
    point.cluster_id = cluster_id
    point.point_type = point_type
    return point


def test_merge_clusters_uses_stable_id_mapping():
    points = {
        10: _point(10, 1, PointType.CORE),
        20: _point(20, 1, PointType.BORDER),
        30: _point(30, 2, PointType.CORE),
    }

    ClusterUtils.merge_clusters(points, source_id=1, target_id=2)

    assert {point.cluster_id for point in points.values()} == {2}


def test_relabel_unreachable_points_sets_border_and_noise():
    points = {
        10: _point(10, 0, PointType.CORE),
        20: _point(20, 0, PointType.BORDER),
        30: _point(30, 0, PointType.BORDER),
    }
    neighborhoods = {20: {10, 20}, 30: {30}}

    ClusterUtils.relabel_unreachable_points(points, {20, 30}, neighborhoods.__getitem__)

    assert points[20].point_type == PointType.BORDER
    assert points[20].cluster_id == 0
    assert points[30].point_type == PointType.NOISE
    assert points[30].cluster_id == -1


def test_clean_stale_cluster_ids_marks_points_as_noise():
    points = {
        10: _point(10, 0, PointType.CORE),
        20: _point(20, 4, PointType.BORDER),
    }

    ClusterUtils.clean_stale_cluster_ids(points, active_cluster_ids={0})

    assert points[10].cluster_id == 0
    assert points[20].cluster_id == -1
    assert points[20].point_type == PointType.NOISE
