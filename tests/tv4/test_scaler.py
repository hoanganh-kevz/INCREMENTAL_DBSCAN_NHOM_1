"""ManualScaler so với StandardScaler và MinMaxScaler của sklearn (sklearn chỉ được import trong test)."""
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src.preprocessing.scaler import ManualScaler
from .tv4_paths import RAW

CARD_COLS = ["amount_vnd", "hour_of_day", "distance_from_home_km", "txns_last_7d", "merchant_risk_score"]


def _card_matrix():
    df = pd.read_csv(RAW / "bank_card" / "card_txn_5d.csv").drop_duplicates().dropna(subset=CARD_COLS)
    x = df[CARD_COLS].to_numpy(float)
    x[:, 0] = np.log10(x[:, 0])
    return x


@pytest.mark.parametrize("method,ref", [("standard", StandardScaler), ("minmax", MinMaxScaler)])
def test_matches_sklearn_random(method, ref):
    rng = np.random.default_rng(7)
    x = rng.normal(loc=rng.uniform(-50, 50, 6), scale=rng.uniform(0.1, 30, 6), size=(400, 6))
    new = rng.normal(size=(25, 6)) * 40
    mine, theirs = ManualScaler(method).fit(x), ref().fit(x)
    assert np.abs(mine.transform(x) - theirs.transform(x)).max() < 1e-9
    assert np.abs(mine.transform(new) - theirs.transform(new)).max() < 1e-9
    assert np.abs(mine.fit_transform(x) - theirs.fit_transform(x)).max() < 1e-9


@pytest.mark.parametrize("method,ref", [("standard", StandardScaler), ("minmax", MinMaxScaler)])
def test_matches_sklearn_real_card_data(method, ref):
    x = _card_matrix()
    assert np.abs(ManualScaler(method).fit_transform(x) - ref().fit_transform(x)).max() < 1e-9


@pytest.mark.parametrize("method", ["standard", "minmax"])
def test_inverse_round_trip(method):
    x = _card_matrix()
    sc = ManualScaler(method).fit(x)
    assert np.allclose(sc.inverse_transform(sc.transform(x)), x, rtol=1e-12, atol=1e-9)


@pytest.mark.parametrize("method", ["standard", "minmax"])
def test_zero_variance_column_gets_scale_one(method):
    x = np.column_stack([np.arange(5.0), np.full(5, 3.5)])
    sc = ManualScaler(method).fit(x)
    assert sc.scale_[1] == 1.0
    z = sc.transform(x)
    assert np.all(np.isfinite(z)) and np.all(z[:, 1] == 0.0)
    # điểm mới khác hằng số: lệch đúng bằng (x - center) / 1
    assert sc.transform(np.array([[2.0, 5.5]]))[0, 1] == 2.0


def test_nan_and_inf_raise_clear_error():
    sc = ManualScaler()
    with pytest.raises(ValueError, match="NaN"):
        sc.fit(np.array([[1.0, np.nan], [2.0, 3.0]]))
    sc.fit(np.array([[1.0, 2.0], [3.0, 4.0]]))
    with pytest.raises(ValueError, match="NaN"):
        sc.transform(np.array([[np.inf, 1.0]]))
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [np.nan, 1.0]})
    with pytest.raises(ValueError, match="'b'"):
        ManualScaler().fit(df)


def test_single_new_row_reuses_d0_parameters_and_never_refits():
    d0 = np.array([[0.0, 10.0], [2.0, 20.0], [4.0, 30.0]])
    sc = ManualScaler().fit(d0)
    before = (sc.center_.copy(), sc.scale_.copy())
    row = np.array([100.0, -5.0])
    z1 = sc.transform(row)                      # vector 1-D
    z2 = sc.transform(row[None, :])             # ma trận 1 dòng
    assert z1.shape == (2,) and z2.shape == (1, 2) and np.allclose(z1, z2[0])
    assert np.allclose(z1, (row - before[0]) / before[1])
    assert np.array_equal(sc.center_, before[0]) and np.array_equal(sc.scale_, before[1])
    assert np.allclose(sc.inverse_transform(z1), row)


def test_errors_and_state():
    with pytest.raises(RuntimeError):
        ManualScaler().transform(np.zeros((1, 2)))
    with pytest.raises(ValueError):
        ManualScaler("robust")
    sc = ManualScaler().fit(np.eye(3))
    with pytest.raises(ValueError):
        sc.transform(np.zeros((2, 4)))
    with pytest.raises(ValueError):
        ManualScaler().fit(np.zeros((0, 3)))


def test_json_round_trip():
    x = _card_matrix()
    sc = ManualScaler().fit(pd.DataFrame(x, columns=CARD_COLS))
    d = json.loads(json.dumps(sc.to_dict()))
    back = ManualScaler.from_dict(d)
    assert np.array_equal(back.transform(x), sc.transform(x))
    assert back.feature_names_ == CARD_COLS and back.method == "standard"


def test_no_sklearn_or_scipy_imported_under_src():
    import ast
    from .tv4_paths import ROOT
    # chỉ các gói của TV4; src/clustering và src/evaluation do đồng đội phụ trách
    files = [f for d in ("preprocessing", "visualization") for f in (ROOT / "src" / d).rglob("*.py")]
    assert files
    for f in files:
        for node in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            names = ([a.name for a in node.names] if isinstance(node, ast.Import)
                     else [node.module or ""] if isinstance(node, ast.ImportFrom) else [])
            assert not any(n.split(".")[0] in ("sklearn", "scipy") for n in names), f
