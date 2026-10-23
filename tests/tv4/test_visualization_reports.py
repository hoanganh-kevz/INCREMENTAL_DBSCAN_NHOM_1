"""Hình báo cáo (D07): tương đương theo step, thời gian, điểm bị ảnh hưởng, so thuật toán, k-distance, chuẩn hóa, stream và hoạt ảnh."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from baseline.sklearn_dbscan import RefitDBSCAN, compare_partitions, read_stream, replay_stream
from .tv4_paths import ROOT
from src.visualization import (equivalence_series, find_dips, k_distances, knee_value, label_changes, plot_affected_points,
                               plot_algorithm_contrast, plot_equivalence_timeline, plot_k_distance, plot_runtime_comparison,
                               plot_scaling_effect, stream_snapshots)

ATM = ROOT / "data" / "raw" / "bank_geo" / "atm_stream"


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def atm_replay():
    d0, upd, params, feats = read_stream(ATM)
    timing, labels = replay_stream(RefitDBSCAN, d0, upd, params, feats, repeats=1, max_steps=40)
    _, ref = replay_stream(RefitDBSCAN, d0, upd, params, feats, repeats=1, max_steps=40)   # lần chạy riêng làm tham chiếu
    return timing, labels, ref[ref["step"] <= 40]


# ------------------------------------------------------------------ tương đương
def test_equivalence_flat_then_dips_on_perturbed_labels(atm_replay):
    _, labels, ref = atm_replay
    flat = equivalence_series(labels, ref, compare_partitions)
    assert len(flat) == 41 and (flat["ari"] == 1.0).all()
    assert find_dips(flat).empty
    bad = ref.copy()
    for s in (7, 8, 25):
        idx = bad.index[bad["step"] == s][:40]
        bad.loc[idx, "label"] = np.random.default_rng(s).integers(0, 3, len(idx))
    eq = equivalence_series(labels, bad, compare_partitions)
    assert find_dips(eq)["step"].tolist() == [7, 8, 25]
    assert (eq.loc[~eq["step"].isin([7, 8, 25]), "ari"] == 1.0).all()


def test_equivalence_ignores_ambiguous_ids():
    a = pd.DataFrame({"step": [1] * 4, "point_id": [0, 1, 2, 3], "label": [0, 0, 1, 1]})
    b = pd.DataFrame({"step": [1] * 4, "point_id": [0, 1, 2, 3], "label": [0, 0, 1, 0]})        # điểm 3 là biên mơ hồ
    assert equivalence_series(a, b, compare_partitions)["ari"].iloc[0] < 1
    out = equivalence_series(a, b, compare_partitions, ambiguous={1: [3]})
    assert out["ari"].iloc[0] == 1.0 and out["n_ignored"].iloc[0] == 1


def test_plot_equivalence_timeline_marks_dips(atm_replay):
    _, labels, ref = atm_replay
    flat = equivalence_series(labels, ref, compare_partitions).assign(series="đúng")
    fig = plot_equivalence_timeline(flat)
    assert isinstance(fig, plt.Figure)
    assert not any(c.get_label().startswith("Step bị lệch") for c in fig.axes[0].collections)
    dipped = flat.assign(series="lệch")
    dipped.loc[dipped["step"].isin([3, 9]), "ari"] = 0.6
    fig = plot_equivalence_timeline(pd.concat([flat, dipped]))
    marked = [c for c in fig.axes[0].collections if c.get_label().startswith("Step bị lệch")]
    assert len(marked) == 1 and len(marked[0].get_offsets()) == 2
    assert fig.axes[0].get_ylim()[0] < 0.6


# ------------------------------------------------------------------ thời gian và điểm bị ảnh hưởng
def test_runtime_comparison_and_affected_points(atm_replay):
    timing, labels, _ = atm_replay
    slower = timing.assign(update_seconds=timing["update_seconds"] * 3)
    fig = plot_runtime_comparison(timing, slower)
    title = fig.axes[1].get_title(loc="left")
    assert isinstance(fig, plt.Figure) and "re-fit / incremental: delete 3.00, insert 3.00 lần" in title
    much_slower = timing.assign(update_seconds=timing["update_seconds"] * 300)
    title = plot_runtime_comparison(timing, much_slower, ratio_label="nhóm / sklearn").axes[1].get_title(loc="left")
    assert "nhóm / sklearn: delete 300, insert 300 lần" in title
    counts = label_changes(labels, timing)
    assert list(counts.columns) == ["step", "n_affected", "op", "n_points"] and len(counts) == 40
    assert (counts["n_affected"] >= 0).all() and (counts["n_affected"] < counts["n_points"]).all()
    assert isinstance(plot_affected_points(counts), plt.Figure)
    assert isinstance(plot_affected_points(counts[["step", "n_affected"]]), plt.Figure)


def test_label_changes_ignores_renumbering_and_counts_moves():
    rows = [(0, i, l) for i, l in enumerate([0, 0, 0, 1, 1, -1])]
    rows += [(1, i, l) for i, l in enumerate([7, 7, 7, 3, 3, -1])]                          # chỉ đánh số lại: không điểm nào đổi
    rows += [(2, i, l) for i, l in enumerate([7, 7, 7, 3, 7, -1])]                          # điểm 4 chuyển cụm
    out = label_changes(pd.DataFrame(rows, columns=["step", "point_id", "label"]))
    assert out["n_affected"].tolist() == [0, 1]


# ------------------------------------------------------------------ k-distance
def test_k_distances_match_brute_force():
    x = np.random.default_rng(0).normal(size=(90, 3))
    d = np.sqrt(((x[:, None] - x[None]) ** 2).sum(-1))
    for min_pts in (2, 4, 6):
        want = np.sort(d, axis=1)[:, min_pts - 1]           # cột 0 là chính nó, nên láng giềng thứ min_pts - 1 là cột min_pts - 1
        assert np.allclose(k_distances(x, min_pts, block=32), want)
    assert (k_distances(x, 1) == 0).all()
    with pytest.raises(ValueError):
        k_distances(x[:3], 6)


def test_knee_and_plot_k_distance():
    curve = np.r_[np.linspace(10, 3, 20), np.linspace(2.9, 2.0, 180)]
    i, v = knee_value(curve)
    assert 15 <= i <= 25 and 2.5 < v < 3.5
    pts = np.random.default_rng(1).normal(size=(200, 2))
    fig = plot_k_distance(pts, min_pts=4, eps=0.3)
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes[0].lines[0].get_ydata()) == 200


# ------------------------------------------------------------------ so thuật toán và chuẩn hóa
def test_algorithm_contrast_panels():
    rng = np.random.default_rng(2)
    xy = np.vstack([rng.normal(0, 0.3, (40, 2)), rng.normal(5, 0.3, (40, 2))])
    truth = np.r_[np.zeros(40, int), np.ones(40, int)]
    fig = plot_algorithm_contrast(xy, truth, {"A": truth, "B": np.r_[np.ones(40, int), np.zeros(40, int)], "C": np.full(80, -1)},
                                  scores={"A": 1.0})
    shown = [ax for ax in fig.axes if ax.axison]
    assert len(shown) == 4 and "ARI = 1.00" in shown[1].get_title()
    colours = lambda ax: {tuple(np.round(f, 3)) for c in ax.collections for f in c.get_facecolor()}
    assert colours(shown[1]) == colours(shown[0]) == colours(shown[2])                 # nhãn hoán đổi vẫn giữ cùng màu với nhãn thật


def test_scaling_effect_and_shape_check():
    rng = np.random.default_rng(3)
    raw = pd.DataFrame({"amount": rng.lognormal(12, 1, 100), "hour": rng.uniform(0, 24, 100), "risk": rng.uniform(0, 1, 100)})
    z = (raw - raw.mean()) / raw.std()
    assert isinstance(plot_scaling_effect(raw, z, names_after=["a_z", "h_z", "r_z"]), plt.Figure)
    with pytest.raises(ValueError):
        plot_scaling_effect(raw, z.iloc[:50])


# ------------------------------------------------------------------ stream và hoạt ảnh
def test_stream_snapshots_follow_the_stream():
    frames = stream_snapshots(RefitDBSCAN, ATM, max_steps=6)
    assert [f["step"] for f in frames] == list(range(7))
    assert frames[0]["op"] == "fit" and frames[0]["coords"] is None
    n = len(frames[0]["snapshot"]["ids"])
    for f in frames[1:]:
        n += 1 if f["op"] == "insert" else -1
        assert len(f["snapshot"]["ids"]) == n
        pid = f["point_id"]
        assert (pid in f["snapshot"]["ids"].tolist()) == (f["op"] == "insert")            # id là id của stream, không phải id engine


def test_animation_writes_small_gif(tmp_path):
    pytest.importorskip("PIL")
    from src.visualization.animation import animate_stream, persistent_labels
    frames = stream_snapshots(RefitDBSCAN, ATM, max_steps=8)
    lab = persistent_labels(frames)
    assert all(len(l) == len(f["snapshot"]["labels"]) for l, f in zip(lab, frames))
    path = animate_stream(frames, tmp_path / "t.gif", every=4)
    assert path.exists() and 20_000 < path.stat().st_size < 4_000_000
    from PIL import Image
    with Image.open(path) as im:
        assert im.size[0] == 600 and im.n_frames == 3                                          # khung 0, 4, 8
