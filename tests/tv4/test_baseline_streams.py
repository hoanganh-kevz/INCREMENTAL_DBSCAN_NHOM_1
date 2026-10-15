"""run_stream trên ba stream: kết quả lặp lại được, khớp một DBSCAN brute-force viết riêng trong test,
và số cụm đổi đúng như verified_case ngụ ý (khi kiểm được)."""
import numpy as np
import pandas as pd
import pytest

from baseline.sklearn_dbscan import RefitDBSCAN, TIMING_COLUMNS, compare_partitions, read_stream, run_stream
from src.preprocessing.stream_scaling import apply_frozen_scaling, stream_coordinates
from .tv4_paths import RAW, ROOT

STREAMS = ["bank_geo/atm_stream", "bank_card/card_stream", "bank_card/card_stream_hard"]

# verified_case -> điều kiện trên (số cụm sau - số cụm trước)
RULE = {"Creation": lambda d: d == 1, "Removal": lambda d: d == -1,
        "Merge": lambda d: d <= -1, "PotentialSplit_Split": lambda d: d >= 1}
ZERO = lambda d: d == 0     # Noise, Absorption, DeleteNoise, Reduction, PotentialSplit_NonSplit

# Step được phép có số cụm đổi không khớp verified_case. Hiện rỗng: mọi step phải khớp (xem RULE ở trên).
KNOWN_HARD_MISMATCH = set()


@pytest.fixture(scope="module", params=STREAMS)
def stream(request, tmp_path_factory):
    name = request.param
    out = tmp_path_factory.mktemp("bl")
    timing, labels = run_stream(RefitDBSCAN, RAW / name, out, repeats=1)     # repeats=1: nhanh; đo lặp có test riêng
    return name, timing, labels, out


def test_files_written(stream):
    name, timing, labels, out = stream
    folder = out / name.split("/")[-1]
    assert (folder / "timing.csv").exists() and (folder / "labels_long.csv").exists()
    assert list(timing.columns) == TIMING_COLUMNS == ["step", "op", "n_points", "n_clusters", "n_noise", "update_seconds",
                                                      "n_repeats", "n"]
    assert (timing["n_repeats"] == 1).all() and (timing["n"] == timing["n_points"]).all()
    assert list(labels.columns) == ["step", "point_id", "label"]
    d0, upd, params, _ = read_stream(RAW / name)
    assert len(timing) == len(upd) + 1 and timing["step"].tolist() == [0] + list(range(1, len(upd) + 1))
    assert timing["n_points"].iloc[0] == len(d0) == params["n_d0"]
    assert (timing["update_seconds"] >= 0).all()
    # số điểm: insert +1, delete -1; số dòng nhãn dài đúng bằng tổng n_points
    expect = timing["n_points"].iloc[0] + np.cumsum(np.where(upd.sort_values("step")["op"] == "insert", 1, -1))
    assert timing["n_points"].iloc[1:].tolist() == expect.tolist()
    assert len(labels) == timing["n_points"].sum()
    assert timing["n_clusters"].iloc[0] == params["d0_reference_clusters"]


def test_deterministic_across_two_runs(stream, tmp_path):
    name, timing, labels, _ = stream
    t2, l2 = run_stream(RefitDBSCAN, RAW / name, tmp_path, repeats=1)
    pd.testing.assert_frame_equal(labels, l2)
    pd.testing.assert_frame_equal(timing.drop(columns="update_seconds"), t2.drop(columns="update_seconds"))
    assert (tmp_path / name.split("/")[-1] / "labels_long.csv").read_bytes() == (stream[3] / name.split("/")[-1] / "labels_long.csv").read_bytes()


def test_cluster_counts_follow_verified_case(stream):
    name, timing, _, _ = stream
    _, upd, _, _ = read_stream(RAW / name)
    upd = upd.sort_values("step").reset_index(drop=True)
    delta = timing["n_clusters"].diff().iloc[1:].astype(int).reset_index(drop=True)
    bad = {int(s) for s, c, d in zip(upd["step"], upd["verified_case"], delta) if not RULE.get(c, ZERO)(d)}
    expected_bad = KNOWN_HARD_MISMATCH if name.endswith("card_stream_hard") else set()
    assert bad == expected_bad, f"step lệch ngoài danh sách đã biết: {sorted(bad ^ expected_bad)}"


def _brute(X, eps, minpts):
    d2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    nb = d2 <= eps * eps
    core = nb.sum(1) >= minpts
    lab = -np.ones(len(X), int)
    c = 0
    for i in range(len(X)):
        if core[i] and lab[i] == -1:
            stack, lab[i] = [i], c
            while stack:
                u = stack.pop()
                for v in np.flatnonzero(nb[u]):
                    if lab[v] == -1:
                        lab[v] = c
                        if core[v]:
                            stack.append(v)
            c += 1
    return lab


def test_every_step_matches_brute_force(stream):
    """Độc lập với sklearn: DBSCAN brute-force numpy, bỏ điểm biên mơ hồ trong cột ambiguous_border_ids."""
    name, _, labels, _ = stream
    d0, upd, params, feats = read_stream(RAW / name)
    coords = {}
    for frame in (d0, upd[upd["op"] == "insert"]):
        for pid, x in zip(frame["point_id"], stream_coordinates(params, frame, feats)):
            coords[int(pid)] = x
    amb = {0: {int(v) for v in params["d0_ambiguous_border_ids"]}}
    for _, r in upd.iterrows():
        amb[int(r["step"])] = {int(v) for v in str(r["ambiguous_border_ids"]).split(";") if v and v != "nan"}
    eps = params.get("eps", params.get("eps_km"))
    for step, g in labels.groupby("step"):
        ids = g["point_id"].tolist()
        ref = _brute(np.array([coords[i] for i in ids]), eps, params["minpts"])
        r = compare_partitions(g["label"].to_numpy(), ref, ids=ids, ignore_ids=amb[step])
        assert r["partition_equal"], f"{name} step {step}: {r}"


def test_frozen_scaling_not_refitted():
    """Stream thẻ: toạ độ đi vào engine là z-score với hằng số trong params.json, không phải fit lại."""
    d0, _, params, feats = read_stream(RAW / "bank_card/card_stream")
    z = apply_frozen_scaling(params, d0)
    assert not np.allclose(z.mean(axis=0), 0, atol=1e-3)     # D0 không có trung bình 0: đúng vì scaling cố định
    assert np.allclose(np.log10(d0["amount_vnd"].to_numpy()) - params["frozen_scaling"]["mean"][0],
                       z[:, 0] * params["frozen_scaling"]["std"][0])


# ---------------------------------------------------------------- stream_coordinates (src/) giống logic gốc trong baseline/
def _old_coords(frame, params, feats):
    """Bản sao nguyên văn logic tính tọa độ stream ban đầu của baseline/sklearn_dbscan.py (_stream_scaler + _to_coords), để so từng byte."""
    from src.preprocessing.scaler import ManualScaler
    fs = params.get("frozen_scaling")
    scaler = None
    if fs is not None:
        scaler = ManualScaler.from_dict({"method": "standard", "center": fs["mean"], "scale": fs["std"],
                                         "feature_names": fs["columns"]})
    log_cols = [feats.index("amount_vnd")] if scaler is not None and "amount_vnd" in feats else []
    x = frame[feats].to_numpy(dtype=float).copy()
    for j in log_cols:
        x[:, j] = np.log10(x[:, j])
    return scaler.transform(x) if scaler is not None else x


@pytest.mark.parametrize("name", STREAMS)
def test_stream_coordinates_identical_to_old_baseline_logic(name):
    d0, upd, params, feats = read_stream(RAW / name)
    for frame in (d0, upd.sort_values("step")):
        new, old = stream_coordinates(params, frame, feats), _old_coords(frame, params, feats)
        assert new.dtype == old.dtype and new.shape == old.shape
        assert new.tobytes() == old.tobytes()          # từng byte, không chỉ allclose


@pytest.mark.parametrize("name", STREAMS)
def test_replay_is_deterministic_and_matches_saved_files_when_present(name, tmp_path):
    """Chạy lại stream hai lần phải cho đúng từng byte labels_long.csv và các cột xác định của timing.csv. File đã lưu trong
    data/processed/baseline (nếu có, không bắt buộc vì sinh lại được và không commit) phải khớp từng byte với lần chạy mới."""
    folder = name.split("/")[-1]
    keep = [c for c in TIMING_COLUMNS if c not in ("update_seconds", "n_repeats", "n")]
    t1, _ = run_stream(RefitDBSCAN, RAW / name, tmp_path / "a", repeats=1)
    t2, _ = run_stream(RefitDBSCAN, RAW / name, tmp_path / "b", repeats=1)
    first = (tmp_path / "a" / folder / "labels_long.csv").read_bytes()
    assert first == (tmp_path / "b" / folder / "labels_long.csv").read_bytes()
    pd.testing.assert_frame_equal(t1[keep], t2[keep])
    saved = ROOT / "data" / "processed" / "baseline" / folder
    if (saved / "labels_long.csv").exists():
        assert first == (saved / "labels_long.csv").read_bytes()
        pd.testing.assert_frame_equal(t1[keep], pd.read_csv(saved / "timing.csv", encoding="utf-8")[keep])
