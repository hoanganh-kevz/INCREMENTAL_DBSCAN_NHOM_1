"""D08: DBSCAN.fit của nhóm phải cho kết quả tương đương sklearn trên mọi bộ dữ liệu của TV4 (2, 4 và 5 chiều).
Bỏ qua nếu chưa có src/clustering/dbscan.py."""
import numpy as np
import pytest

dbscan_module = pytest.importorskip("src.clustering.dbscan", reason="chưa có DBSCAN của nhóm trong src/clustering/")

from baseline.sklearn_dbscan import RefitDBSCAN, PointType
from baseline.static_equivalence import compare_static_fit, iter_cases

CASES = list(iter_cases())


@pytest.mark.parametrize("name,coords,eps,min_pts", CASES, ids=[c[0] for c in CASES])
def test_real_dbscan_fit_matches_sklearn(name, coords, eps, min_pts):
    res = compare_static_fit(dbscan_module.DBSCAN, coords, eps, min_pts)
    assert res["types_equal"], f"{name}: loại điểm (lõi, biên, nhiễu) khác sklearn"
    assert res["cores_equal"], f"{name}: tập điểm lõi khác sklearn"
    assert res["noise_equal"], f"{name}: tập nhiễu khác sklearn"
    assert res["core_partition_ari"] == 1.0, f"{name}: phân hoạch của các điểm lõi khác sklearn"
    assert res["border_diff_all_ambiguous"], f"{name}: có điểm biên khác cụm mà không phải điểm biên mơ hồ ({res['border_diff']} điểm)"
    assert res["equivalent"]


@pytest.mark.parametrize("dims", [2, 5, 8])
def test_neighbourhoods_match_numpy_in_any_dimension(dims):
    """Lân cận Eps (cả _region_query lẫn neighbor_ids) phải khớp numpy ở 2, 5 và 8 chiều, tính cả chính điểm đó."""
    rng = np.random.default_rng(dims)
    coords = rng.normal(size=(120, dims))
    dist = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
    eps = float(np.quantile(dist[np.triu_indices(len(coords), 1)], 0.05))
    model = dbscan_module.DBSCAN(eps, 4).fit(coords)
    has_query, has_cache = hasattr(model, "_region_query"), hasattr(model.points[0], "neighbor_ids")
    if not (has_query or has_cache):
        pytest.skip("DBSCAN của nhóm không còn _region_query hay neighbor_ids")
    for i in range(len(coords)):
        expected = set(np.where(dist[i] <= eps)[0])
        if has_query:
            assert set(model._region_query(i)) == expected, f"_region_query, điểm {i}, {dims} chiều"
        if has_cache:
            assert set(model.points[i].neighbor_ids) == expected, f"neighbor_ids, điểm {i}, {dims} chiều"


def test_cases_cover_two_four_and_five_dimensions():
    dims = {np.asarray(c[1]).shape[1] for c in CASES}
    assert {2, 4, 5} <= dims, f"thiếu số chiều: {sorted(dims)}"
    assert len(CASES) >= 60


# ---------------------------------------------------------------- vài engine cố ý sai, bộ so sánh phải phát hiện được
class _CoreAsNoise(RefitDBSCAN):
    """Báo điểm lõi đầu tiên là nhiễu."""

    def snapshot(self):
        snap = super().snapshot()
        types = np.array(snap["point_types"], dtype=int)
        types[np.where(types == int(PointType.CORE))[0][0]] = int(PointType.NOISE)
        return {**snap, "point_types": types}


class _BorderUnclassified(RefitDBSCAN):
    """Để một điểm biên ở trạng thái chưa phân loại."""

    def snapshot(self):
        snap = super().snapshot()
        types = np.array(snap["point_types"], dtype=int)
        types[np.where(types == int(PointType.BORDER))[0][0]] = int(PointType.UNCLASSIFIED)
        return {**snap, "point_types": types}


class _BorderToFarCluster(RefitDBSCAN):
    """Gán một điểm biên sang cụm không có lõi nào trong Eps của nó."""

    def snapshot(self):
        snap = super().snapshot()
        labels, types, xy = snap["labels"].copy(), snap["point_types"], snap["coords"]
        core = types == int(PointType.CORE)
        for i in np.where(types == int(PointType.BORDER))[0]:
            far = sorted(set(labels[core]) - set(labels[core][np.linalg.norm(xy[core] - xy[i], axis=1) <= self.eps]))
            if far:
                labels[i] = far[0]
                break
        return {**snap, "labels": labels}


@pytest.mark.parametrize("wrong", [_CoreAsNoise, _BorderUnclassified, _BorderToFarCluster], ids=lambda c: c.__name__.strip("_"))
def test_comparison_detects_a_wrong_engine(wrong):
    name, coords, eps, min_pts = next(c for c in CASES if c[0] == "atm_txn_2d")
    assert compare_static_fit(RefitDBSCAN, coords, eps, min_pts)["equivalent"]
    assert not compare_static_fit(wrong, coords, eps, min_pts)["equivalent"]
