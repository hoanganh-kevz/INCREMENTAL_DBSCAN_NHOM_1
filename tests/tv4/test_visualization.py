"""Bộ vẽ cụm (D07): adapter snapshot, project_2d, đếm marker, mọi kịch bản toy và toy_edge, lưu file. Chạy không màn hình (Agg)."""
import ast
from enum import Enum
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from baseline.sklearn_dbscan import PointType, RefitDBSCAN, compare_partitions
from .tv4_paths import ROOT
from src.visualization import (ClusterVisualizer, align_labels, list_scenarios, load_scenario, make_snapshot, project_2d, run_scenario,
                               snapshot_from_model, type_counts)
from src.visualization.gallery import build_gallery_cases

KINDS = [("toy", n) for n in list_scenarios("toy")] + [("toy_edge", n) for n in list_scenarios("toy_edge")]
IDS = [f"{k}/{n}" for k, n in KINDS]
VIS = ClusterVisualizer()


@pytest.fixture(scope="module")
def runs():
    out = {}
    for kind, name in KINDS:
        sc = load_scenario(kind, name)
        out[(kind, name)] = (sc, run_scenario(RefitDBSCAN, sc))
    return out


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _counts_from_axes(ax):
    got = {"core": 0, "border": 0, "noise": 0, "point": 0}
    for coll in ax.collections:
        if coll.get_gid() in got:
            got[coll.get_gid()] += len(coll.get_offsets())
    return got


# ------------------------------------------------------------------ adapter
def test_adapter_with_refit_matches_engine_snapshot():
    rng = np.random.default_rng(0)
    eng = RefitDBSCAN(0.5, 4).fit(rng.normal(size=(60, 2)))
    snap, raw = snapshot_from_model(eng), eng.snapshot()
    assert set(snap) == {"ids", "coords", "labels", "point_types"}
    for k in snap:
        assert np.array_equal(snap[k], raw[k])
    assert snapshot_from_model(snap)["labels"].tolist() == raw["labels"].tolist()      # dict đã đúng dạng đi qua nguyên vẹn


def test_adapter_copies_arrays():
    eng = RefitDBSCAN(0.5, 4).fit(np.random.default_rng(1).normal(size=(20, 2)))
    snap = snapshot_from_model(eng)
    snap["labels"][:] = 99
    assert 99 not in eng.snapshot()["labels"]


def test_adapter_from_datapoint_like_model():
    class PT(Enum):
        UNCLASSIFIED = 0
        NOISE = 1
        BORDER = 2
        CORE = 3
    pts = [SimpleNamespace(id=i, coords=np.array([float(i), 0.0]), cluster_id=c, point_type=t)
           for i, (c, t) in enumerate([(0, PT.CORE), (0, PT.BORDER), (None, PT.NOISE)])]
    snap = snapshot_from_model(SimpleNamespace(points=pts))
    assert snap["labels"].tolist() == [0, 0, -1]
    assert snap["point_types"].tolist() == [int(PointType.CORE), int(PointType.BORDER), int(PointType.NOISE)]
    assert snapshot_from_model(SimpleNamespace(_points={p.id: p for p in pts}))["ids"].tolist() == [0, 1, 2]


def test_adapter_rejects_unknown_object():
    with pytest.raises(TypeError, match="snapshot"):
        snapshot_from_model(object())


def test_make_snapshot_rejects_length_mismatch():
    with pytest.raises(ValueError):
        make_snapshot([0, 1], np.zeros((3, 2)), [0, 0, 0])


# ------------------------------------------------------------------ project_2d
@pytest.mark.parametrize("n,d", [(50, 2), (80, 4), (120, 5), (30, 9)])
def test_project_2d_matches_numpy_svd(n, d):
    x = np.random.default_rng(n + d).normal(size=(n, d)) * np.arange(1, d + 1)
    proj, explained = project_2d(x)
    xc = x - x.mean(axis=0)
    _, s, vt = np.linalg.svd(xc, full_matrices=False)
    assert proj.shape == (n, 2)
    assert np.allclose(np.abs(proj), np.abs(xc @ vt[:2].T))
    assert np.allclose(explained, (s ** 2 / (s ** 2).sum())[:2])
    assert explained.sum() <= 1.0 + 1e-12
    assert explained[0] >= explained[1]


def test_project_2d_basis_projects_new_points_consistently():
    x = np.random.default_rng(3).normal(size=(40, 4))
    proj, _, mean, comps = project_2d(x, return_basis=True)
    assert np.allclose((x - mean) @ comps.T, proj)
    assert project_2d(x)[0].tolist() == proj.tolist()            # lặp lại được (dấu cố định)


def test_project_2d_one_dimensional_and_degenerate():
    proj, explained = project_2d(np.arange(5.0).reshape(-1, 1))
    assert proj.shape == (5, 2) and np.allclose(proj[:, 1], 0) and explained[0] == pytest.approx(1.0)
    proj, explained = project_2d(np.ones((4, 3)))                 # mọi điểm trùng nhau: phương sai 0, không chia cho 0
    assert np.allclose(proj, 0) and explained.sum() == 0
    assert project_2d(np.empty((0, 3)))[0].shape == (0, 2)


def test_project_2d_rejects_nan():
    with pytest.raises(ValueError, match="NaN"):
        project_2d(np.array([[1.0, np.nan], [0.0, 1.0]]))


# ------------------------------------------------------------------ align_labels
def test_align_labels_keeps_colour_through_renumbering_and_merge():
    ref = {"ids": np.arange(8), "labels": np.array([0, 0, 0, 0, 1, 1, 1, -1])}
    renum = {"ids": np.arange(8), "labels": np.array([5, 5, 5, 5, 2, 2, 2, -1])}
    assert align_labels(ref, renum).tolist() == [0, 0, 0, 0, 1, 1, 1, -1]
    merged = {"ids": np.arange(8), "labels": np.array([3, 3, 3, 3, 3, 3, 3, -1])}
    assert align_labels(ref, merged).tolist() == [0, 0, 0, 0, 0, 0, 0, -1]          # cụm lớn hơn giữ màu
    fresh = {"ids": np.arange(10), "labels": np.array([0, 0, 0, 0, 1, 1, 1, -1, 7, 7])}
    out = align_labels(ref, fresh)
    assert out[8] == out[9] and out[8] not in (0, 1)                               # cụm mới nhận nhãn mới


# ------------------------------------------------------------------ plot_2d
@pytest.mark.parametrize("kind,name", KINDS, ids=IDS)
def test_plot_2d_marker_counts_match_snapshot(runs, kind, name):
    sc, run = runs[(kind, name)]
    for snap in (run["before"], run["after"]):
        ax = VIS.plot_2d(snap, eps=sc["eps"])
        got, want = _counts_from_axes(ax), type_counts(snap)
        assert got["core"] == want["core"] and got["border"] == want["border"] and got["noise"] == want["noise"]
        assert sum(got.values()) == len(snap["labels"])


def test_plot_2d_counts_in_legend_and_cluster_colours():
    sc = load_scenario("toy", "r02_4_merge")
    snap = snapshot_from_model(RefitDBSCAN(sc["eps"], sc["min_pts"]).fit(sc["coords"]))
    ax = VIS.plot_2d(snap, eps=sc["eps"])
    texts = [t.get_text() for t in ax.get_legend().get_texts()]
    assert "Điểm lõi (core): 4" in texts and "Điểm biên (border): 8" in texts and "Số cụm: 2" in texts
    core = next(c for c in ax.collections if c.get_gid() == "core")
    assert len({tuple(np.round(f, 3)) for f in core.get_facecolor()}) == 2        # hai cụm, hai màu


def test_plot_2d_noise_is_grey_and_uses_cross_marker():
    snap = make_snapshot([0, 1, 2], [[0, 0], [5, 5], [9, 9]], [-1, -1, -1], [1, 1, 1])
    ax = VIS.plot_2d(snap, legend=False)
    assert [c.get_gid() for c in ax.collections if c.get_gid()] == ["noise"]
    assert len(ax.collections[0].get_offsets()) == 3


def test_plot_2d_into_existing_axes_and_empty_snapshot():
    fig, ax = plt.subplots()
    assert VIS.plot_2d(make_snapshot([], np.empty((0, 2)), [], []), ax=ax, eps=1.0) is ax


def test_plot_2d_highlight_adds_ring_and_eps_circle():
    sc = load_scenario("toy", "r02_4_merge")
    snap = snapshot_from_model(RefitDBSCAN(sc["eps"], sc["min_pts"]).fit(sc["coords"]))
    ax = VIS.plot_2d(snap, eps=1.1, highlight_ids=[1], fill=None)
    assert any(c.get_gid() == "highlight" for c in ax.collections)
    assert sum(1 for p in ax.patches if getattr(p, "radius", None) == 1.1) == 1


def test_unknown_ids_raise_instead_of_being_ignored():
    snap = make_snapshot([0, 1], [[0, 0], [1, 1]], [0, 0], [3, 3])
    for kw in ({"highlight_ids": ["zz"]}, {"seed_ids": [5]}):
        with pytest.raises(KeyError):
            VIS.plot_2d(snap, **kw)


@pytest.mark.parametrize("d", [1, 3, 4, 5, 9])
def test_plot_2d_n_dimensional_does_not_raise(d):
    x = np.random.default_rng(d).normal(size=(60, d))
    eng = RefitDBSCAN(1.2 * np.sqrt(d), 4).fit(x)
    ax = VIS.plot_2d(eng, eps=0.5, title="n-D")
    assert sum(_counts_from_axes(ax).values()) == 60
    if d != 2:
        assert "Thành phần chính 1" in ax.get_xlabel()
        assert not any(getattr(p, "radius", None) == 0.5 for p in ax.patches)       # không vẽ Eps trên không gian đã chiếu


# ------------------------------------------------------------------ before and after
@pytest.mark.parametrize("kind,name", KINDS, ids=IDS)
def test_plot_before_after_on_every_scenario(runs, kind, name):
    sc, run = runs[(kind, name)]
    fig = VIS.plot_before_after(run["before"], run["after"], run["update"], eps=sc["eps"], seed_ids=run["seed_ids"])
    assert isinstance(fig, plt.Figure) and len(fig.axes) == 2
    ops = {u["op"] for u in run["update"]}
    gids = {c.get_gid() for ax in fig.axes for c in ax.collections}
    if "insert" in ops:
        assert "insert" in gids and "insert_ghost" in gids
    if "delete" in ops:
        assert "delete" in gids and "delete_ghost" in gids


def test_before_after_mark_positions_and_shared_colours(runs):
    sc, run = runs[("toy", "r02_4_merge")]
    fig = VIS.plot_before_after(run["before"], run["after"], run["update"], eps=sc["eps"])
    star = next(c for c in fig.axes[1].collections if c.get_gid() == "insert")
    assert np.allclose(star.get_offsets()[0], [1.9, 0.5])
    ghost = next(c for c in fig.axes[0].collections if c.get_gid() == "insert_ghost")
    assert np.allclose(ghost.get_offsets()[0], [1.9, 0.5])
    # cụm hợp nhất giữ màu của cụm lớn hơn trước đó (hai cụm bằng nhau: khớp theo nhãn nhỏ hơn)
    col_after = {tuple(np.round(f, 3)) for c in fig.axes[1].collections if c.get_gid() in ("core", "border") for f in c.get_facecolor()}
    assert len(col_after) == 1


def test_delete_cross_drawn_where_point_was(runs):
    sc, run = runs[("toy", "r03_3a_split")]
    fig = VIS.plot_before_after(run["before"], run["after"], run["update"], eps=sc["eps"])
    cross = next(c for c in fig.axes[0].collections if c.get_gid() == "delete")
    ghost = next(c for c in fig.axes[1].collections if c.get_gid() == "delete_ghost")
    where = sc["coords"][sc["names"].index("p")]
    assert np.allclose(cross.get_offsets()[0], where) and np.allclose(ghost.get_offsets()[0], where)


def test_4d_scenario_shares_one_projection(runs):
    sc, run = runs[("toy_edge", "e18_insert_4d_merge")]
    fig = VIS.plot_before_after(run["before"], run["after"], run["update"], seed_ids=run["seed_ids"])
    shared = [(ax.get_xlim(), ax.get_ylim()) for ax in fig.axes]
    assert shared[0] == shared[1]
    assert "Thành phần chính" in fig.axes[0].get_xlabel()


# ------------------------------------------------------------------ run_scenario trên engine thật
@pytest.mark.parametrize("kind,name", KINDS, ids=IDS)
def test_run_scenario_matches_expected(runs, kind, name):
    sc, run = runs[(kind, name)]
    exp = sc["expected"]
    got = dict(zip(run["after"]["ids"].tolist(), run["after"]["labels"].tolist()))
    cmp = compare_partitions(got, exp["labels_after"], ignore_ids=exp.get("ambiguous_to_ignore", []))
    assert cmp["partition_equal"], cmp
    got0 = dict(zip(run["before"]["ids"].tolist(), run["before"]["labels"].tolist()))
    assert compare_partitions(got0, exp["labels_before"], ignore_ids=exp.get("ambiguous_to_ignore", []))["partition_equal"]
    assert set(run["seed_ids"]) <= set(run["after"]["ids"].tolist())


# ------------------------------------------------------------------ bộ ảnh và lưu file
def test_gallery_figures_saved_and_nonempty(tmp_path):
    for kind, cols, fname in (("toy", 2, "toy.png"), ("toy_edge", 3, "edge.png")):
        cases = build_gallery_cases(kind, RefitDBSCAN)
        assert len(cases) == len(list_scenarios(kind))
        fig = VIS.plot_case_gallery(cases, n_pair_cols=cols)
        assert len(fig.axes) == 2 * len(cases)
        assert fig.get_size_inches()[0] == pytest.approx(7.5)                     # khổ rộng vừa A4
        path = VIS.save(fig, tmp_path / "sub" / fname)
        assert path.exists() and path.stat().st_size > 20_000


def test_toy_gallery_has_the_seven_report_cases():
    headers = [c["header"] for c in build_gallery_cases("toy", RefitDBSCAN)]
    for case in ("Noise", "Creation", "Absorption", "Merge", "Removal", "Reduction", "Split", "Non-split"):
        assert any(case in h for h in headers)


def test_vietnamese_text_uses_a_font_with_diacritics():
    from matplotlib import font_manager
    from src.visualization.plotter import REPORT_RC
    path = font_manager.findfont(REPORT_RC["font.family"], fallback_to_default=False)
    from matplotlib.ft2font import FT2Font
    cmap = FT2Font(path).get_charmap()
    for ch in "ăâêôơưđạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữự":
        assert ord(ch) in cmap, ch


def test_src_visualization_has_no_sklearn_or_scipy():
    for f in (ROOT / "src" / "visualization").glob("*.py"):
        tree = ast.parse(f.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            mods = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""] if isinstance(node, ast.ImportFrom) else []
            assert not any(m.split(".")[0] in ("sklearn", "scipy") for m in mods), f"{f.name} import {mods}"


def test_no_dash_characters_in_new_text():
    files = list((ROOT / "src" / "visualization").glob("*.py")) + [ROOT / "tools" / "make_d07_figures.py"]
    files = [f for f in files if f.exists()]       # tools/ không nằm trong bộ commit
    assert any(f.parent.name == "visualization" for f in files)
    for f in files:
        txt = f.read_text(encoding="utf-8")
        assert "—" not in txt and "–" not in txt, f.name
