"""OwnRefitDBSCAN phải giải đúng 27 kịch bản toy, toy_edge và khớp sklearn ở 20 step đầu của ba stream.
Bỏ qua nếu chưa có src/clustering/dbscan.py."""
import numpy as np
import pytest

pytest.importorskip("src.clustering.dbscan", reason="chưa có DBSCAN của nhóm trong src/clustering/")

from baseline.own_refit import OwnRefitDBSCAN
from baseline.sklearn_dbscan import RefitDBSCAN, ambiguous_by_step, compare_partitions, read_stream, replay_stream
from .test_baseline_toy import EDGE, TOY, _run
from .tv4_paths import RAW

STREAMS = ["bank_geo/atm_stream", "bank_card/card_stream", "bank_card/card_stream_hard"]
STEPS = 20      # chạy hết stream thì lâu, 20 step đầu là đủ


@pytest.mark.parametrize("kind,name", [("toy", n) for n in TOY] + [("toy_edge", n) for n in EDGE])
def test_own_refit_solves_every_scenario(kind, name):
    _run(kind, name, engine=OwnRefitDBSCAN)


@pytest.mark.parametrize("stream", STREAMS)
def test_own_refit_matches_sklearn_on_stream_prefix(stream):
    d0, upd, params, feats = read_stream(RAW / stream)
    _, own = replay_stream(OwnRefitDBSCAN, d0, upd, params, feats, repeats=1, max_steps=STEPS)
    _, ref = replay_stream(RefitDBSCAN, d0, upd, params, feats, repeats=1, max_steps=STEPS)
    amb = ambiguous_by_step(params, upd)
    assert own["step"].nunique() == STEPS + 1
    for step, g in own.groupby("step"):
        h = ref[ref["step"] == step]
        res = compare_partitions(dict(zip(g["point_id"], g["label"])), dict(zip(h["point_id"], h["label"])), ignore_ids=amb[step])
        assert res["partition_equal"], f"{stream} step {step}: {res}"


def test_same_surface_as_refit():
    eng = OwnRefitDBSCAN(1.5, 4).fit(np.array([[0, 0], [1, 0], [0, 1], [1, 1], [9, 9.0]]))
    assert eng.get_labels() == [0, 0, 0, 0, -1] and eng.snapshot()["point_types"].tolist() == [3, 3, 3, 3, 1]
    pid = eng.insert([9.5, 9.0])
    assert pid == 5 and eng.n_points == 6 and eng.last_update_seconds > 0
    eng.delete(0)
    assert eng.snapshot()["ids"].tolist() == [1, 2, 3, 4, 5] and eng.n_clusters == 0
    for i in [1, 2, 3, 4, 5]:
        eng.delete(i)
    assert eng.n_points == 0 and eng.get_labels() == []
