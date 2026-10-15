"""src/preprocessing/stream_scaling.py: hàm thuần chuyển điểm thô thành tọa độ cố định, và các đường báo lỗi."""
import numpy as np
import pandas as pd
import pytest

from src.preprocessing.scaler import ManualScaler
from src.preprocessing.stream_scaling import apply_frozen_scaling, feature_columns, frozen_columns, stream_coordinates

COLS = ["log10_amount_vnd", "hour_of_day", "distance_from_home_km"]
PARAMS = {"frozen_scaling": {"columns": COLS, "mean": [4.0, 12.0, 10.0], "std": [1.0, 4.0, 5.0]}}


def _frame():
    return pd.DataFrame({"point_id": [0, 1], "amount_vnd": [1000.0, 100000.0], "hour_of_day": [8.0, 20.0],
                         "distance_from_home_km": [10.0, 0.0], "ref_label": [0, 0]})


def test_apply_frozen_scaling_exact_values_and_column_order_from_params():
    f = _frame()
    z = apply_frozen_scaling(PARAMS, f)
    assert np.allclose(z, [[-1.0, -1.0, 0.0], [1.0, 2.0, -2.0]])           # log10(1000) = 3, log10(1e5) = 5
    shuffled = f[["ref_label", "distance_from_home_km", "amount_vnd", "point_id", "hour_of_day"]]
    assert apply_frozen_scaling(PARAMS, shuffled).tobytes() == z.tobytes()  # thứ tự cột theo params, không theo frame


def test_apply_frozen_scaling_is_pure_and_matches_manual_scaler():
    f = _frame()
    before = f.copy()
    z = apply_frozen_scaling(PARAMS, f)
    pd.testing.assert_frame_equal(f, before)                                  # không sửa DataFrame đầu vào
    x = f[["amount_vnd", "hour_of_day", "distance_from_home_km"]].to_numpy(float)
    x[:, 0] = np.log10(x[:, 0])
    ref = ManualScaler.from_dict({"center": [4.0, 12.0, 10.0], "scale": [1.0, 4.0, 5.0]}).transform(x)
    assert z.tobytes() == ref.tobytes()


def test_error_paths():
    f = _frame()
    with pytest.raises(ValueError, match="không có frozen_scaling"):
        apply_frozen_scaling({}, f)
    with pytest.raises(ValueError, match="thiếu"):
        apply_frozen_scaling(PARAMS, f.drop(columns="hour_of_day"))
    bad = f.copy()
    bad.loc[0, "amount_vnd"] = 0.0
    with pytest.raises(ValueError, match="không dương"):
        apply_frozen_scaling(PARAMS, bad)


def test_stream_coordinates_checks_feature_columns_match_frozen_scaling():
    """Cột của d0 phải khớp frozen_scaling (cùng tập, cùng thứ tự); lệch thì báo lỗi chứ không chuẩn hóa nhầm cột."""
    f = _frame()
    feats = feature_columns(f)
    assert feats == ["amount_vnd", "hour_of_day", "distance_from_home_km"]
    assert frozen_columns(PARAMS) == feats and frozen_columns({}) is None
    assert stream_coordinates(PARAMS, f, feats).tobytes() == apply_frozen_scaling(PARAMS, f).tobytes()
    with pytest.raises(ValueError, match="không khớp"):
        stream_coordinates(PARAMS, f, ["hour_of_day", "amount_vnd", "distance_from_home_km"])   # đổi thứ tự
    with pytest.raises(ValueError, match="không khớp"):
        stream_coordinates(PARAMS, f, ["amount_vnd", "hour_of_day"])                              # thiếu cột


def test_stream_without_frozen_scaling_returns_raw_columns_copy():
    f = pd.DataFrame({"point_id": [0], "x_km": [1.5], "y_km": [2.5], "ref_label": [0]})
    x = stream_coordinates({}, f, feature_columns(f))
    assert x.tolist() == [[1.5, 2.5]]
    x[0, 0] = 99.0
    assert f.loc[0, "x_km"] == 1.5                                           # bản sao, không phải view
