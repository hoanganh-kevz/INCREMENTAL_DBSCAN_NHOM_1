"""data/processed/ của các bộ dữ liệu tĩnh: tái tạo từ data/make_processed.py phải trùng từng byte, và đã chuẩn hóa đúng."""
import importlib.util
import json

import numpy as np
import pandas as pd
import pytest

from .tv4_paths import RAW, ROOT

spec = importlib.util.spec_from_file_location("make_processed", ROOT / "data" / "make_processed.py")
mp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mp)


@pytest.mark.parametrize("ds", mp.DATASETS, ids=[d[0] for d in mp.DATASETS])
def test_processed_files_reproduce_byte_for_byte_and_are_standardised(ds, tmp_path):
    name, rel, feats, id_col, label_col, _ = ds
    n = mp.build(*ds, out_dir=tmp_path)
    for fn in (f"{name}_scaled.csv", f"{name}_scaler_params.json"):
        assert (tmp_path / fn).read_bytes() == (ROOT / "data" / "processed" / fn).read_bytes(), fn
    scaled = pd.read_csv(ROOT / "data" / "processed" / f"{name}_scaled.csv")
    raw = pd.read_csv(RAW / rel)
    assert len(scaled) == len(raw) == n and scaled[id_col].tolist() == raw[id_col].tolist()
    assert scaled[label_col].tolist() == raw[label_col].tolist()                  # nhãn tham chiếu giữ nguyên
    z = scaled[[c + "_z" for c in feats]].to_numpy()
    assert np.abs(z.mean(axis=0)).max() < 1e-5 and np.abs(z.std(axis=0) - 1).max() < 1e-5
    params = json.loads((ROOT / "data" / "processed" / f"{name}_scaler_params.json").read_text(encoding="utf-8"))
    back = z * np.array(params["scaler"]["scale"]) + np.array(params["scaler"]["center"])
    assert np.abs(back - raw[feats].to_numpy()).max() < 1e-4                      # đảo ngược được bằng tham số đã lưu
