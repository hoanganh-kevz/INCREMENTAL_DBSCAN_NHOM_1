"""IncrementalDBSCAN của nhóm (qua IncrementalEngine) trên dữ liệu của TV4: 27 kịch bản, UpdSeed, ba stream đầy đủ so với sklearn,
chèn theo lô, và bộ so sánh phải bắt được kết quả sai. Bỏ qua nếu chưa có src/clustering/incremental_dbscan.py."""
import numpy as np
import pytest

pytest.importorskip("src.clustering.incremental_dbscan", reason="chưa có IncrementalDBSCAN của nhóm")

from baseline.sklearn_dbscan import RefitDBSCAN, ambiguous_by_step, read_stream
from baseline.update_experiments import STREAMS, batch_insert, scenario_table, snapshots_equivalent
from src.experiments.engine import IncrementalEngine
from src.experiments.update_cases import changed_points, delete_scope, update_seeds
from src.visualization.scenarios import stream_snapshots
from .test_baseline_toy import EDGE, TOY, _run
from .tv4_paths import RAW


@pytest.mark.parametrize("kind,name", [("toy", n) for n in TOY] + [("toy_edge", n) for n in EDGE])
def test_incremental_solves_every_scenario(kind, name):
    _run(kind, name, engine=IncrementalEngine)


def test_scenario_table_all_steps_correct():
    t = scenario_table(repeats=1)
    assert len(t) == 31
    assert t["matches_expected"].all() and t["equivalent_sklearn"].all() and t["updseed_ok"].all()
    assert set(t["case"]) == {"Noise", "Creation", "Absorption", "Merge", "Removal", "Reduction", "PotentialSplit_Split",
                              "PotentialSplit_NonSplit", "DeleteNoise"}


@pytest.mark.parametrize("stream", list(STREAMS))
def test_incremental_matches_sklearn_on_full_stream(stream):
    sdir = RAW / STREAMS[stream] / stream
    d0, upd, params, _ = read_stream(sdir)
    eps = params.get("eps", params.get("eps_km"))
    amb = ambiguous_by_step(params, upd)
    inc, ref = stream_snapshots(IncrementalEngine, sdir), stream_snapshots(RefitDBSCAN, sdir)
    assert len(inc) == len(upd) + 1
    for a, b in zip(inc, ref):
        assert snapshots_equivalent(a["snapshot"], b["snapshot"], eps, amb.get(a["step"], [])), f"{stream} step {a['step']}"


def test_batch_insert_equivalent():
    batch, _ = batch_insert(ks=(1, 20), repeats=1)
    assert batch["equivalent_sklearn"].all() and batch["n_final"].tolist() == [1001, 1020]


def test_engine_snapshot_surface():
    eng = IncrementalEngine(1.5, 4).fit(np.array([[0, 0], [1, 0], [0, 1], [1, 1], [9, 9.0]]))
    snap = eng.snapshot()
    assert snap["ids"].tolist() == [0, 1, 2, 3, 4] and snap["labels"].tolist() == [0, 0, 0, 0, -1]
    assert snap["point_types"].tolist() == [3, 3, 3, 3, 1] and (eng.n_points, eng.n_clusters, eng.n_noise) == (5, 1, 1)
    assert eng.insert([9.5, 9.0]) == 5
    eng.delete(0)
    assert eng.snapshot()["ids"].tolist() == [1, 2, 3, 4, 5] and eng.n_clusters == 0 and eng.n_noise == 5


def _snap(ids, xy, labels, types):
    return {"ids": np.array(ids), "coords": np.array(xy, float), "labels": np.array(labels), "point_types": np.array(types)}


def test_comparator_catches_wrong_results():
    xy = [[0, 0], [0.5, 0], [1, 0], [3, 0], [3.5, 0], [4, 0], [1.6, 0]]       # điểm 6 là biên của cụm 0, cách cụm 1 quá Eps
    good = _snap(range(7), xy, [0, 0, 0, 1, 1, 1, 0], [3, 3, 3, 3, 3, 3, 2])
    assert snapshots_equivalent(good, _snap(range(7), xy, [5, 5, 5, 2, 2, 2, 5], [3, 3, 3, 3, 3, 3, 2]), eps=1.0)
    assert not snapshots_equivalent(good, _snap(range(7), xy, [0, 0, 0, 0, 0, 0, 0], [3, 3, 3, 3, 3, 3, 2]), eps=1.0)    # gộp sai
    assert not snapshots_equivalent(good, _snap(range(7), xy, [0, 0, 0, 1, 1, 1, -1], [3, 3, 3, 3, 3, 3, 1]), eps=1.0)   # biên thành nhiễu
    assert not snapshots_equivalent(good, _snap(range(7), xy, [0, 0, 0, 1, 1, 1, 1], [3, 3, 3, 3, 3, 3, 2]), eps=1.0)    # biên không mơ hồ đổi cụm
    assert not snapshots_equivalent(good, _snap(range(6), xy[:6], [0, 0, 0, 1, 1, 1], [3] * 6), eps=1.0)                 # thiếu điểm


def test_update_measures_on_split():
    # chuỗi 5 lõi, xóa lõi giữa: UpdSeed là hai lõi kề điểm xóa; nếu cả 4 điểm còn lại mất tính lõi thì UpdSeed rỗng
    xy = [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]]
    before = _snap(range(5), xy, [0] * 5, [3] * 5)
    after = _snap([0, 1, 3, 4], [xy[i] for i in (0, 1, 3, 4)], [0, 0, 1, 1], [3, 3, 3, 3])
    assert update_seeds(before, after, "delete", 1.0) == ({1, 3}, {2})
    after2 = _snap([0, 1, 3, 4], [xy[i] for i in (0, 1, 3, 4)], [0, 0, 1, 1], [2, 2, 2, 2])
    assert update_seeds(before, after2, "delete", 1.0) == (set(), {0, 1, 2, 3, 4})
    assert changed_points(before, after2) == {0, 1, 3, 4} and delete_scope(before, 2, 1.0) == 5


@pytest.mark.parametrize("which", ["moons", "rings", "blobs có nhiễu"])
def test_dataset_comparison_matches_sklearn(which):
    from baseline.dataset_comparison import DATASETS, run_dataset
    spec = next(d for d in DATASETS if d[0] == which)
    res, _ = run_dataset(*spec)
    assert res["equivalent_steps"] == res["n_updates"] > 0
    assert res["n_insert"] > res["n_delete"] > 0 and abs(res["ari_incremental_truth"] - res["ari_sklearn_truth"]) < 1e-9


def test_step_figures_follow_the_algorithm():
    import matplotlib
    matplotlib.use("Agg")
    from baseline.update_experiments import plot_delete_steps, plot_insert_steps
    ins, dele = plot_insert_steps(), plot_delete_steps()
    t_ins = [ax.get_title(loc="left") for ax in ins.axes]
    t_del = [ax.get_title(loc="left") for ax in dele.axes]
    assert len(t_ins) == len(t_del) == 4
    assert "A3, A6, B1, B4, p" in t_ins[1] and "9 điểm" in t_ins[2] and "Merge thành 1 cụm" in t_ins[3]
    assert "{A, E}" in t_del[1] and "I thành nhiễu" in t_del[3]
