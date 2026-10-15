"""Tính toàn vẹn của dữ liệu và chống rò rỉ đáp án, chạy cùng `pytest tests/`.

1. test_check_data_script_passes: chạy nguyên data/check_data.py (khoảng 30 giây). Trước đây pytest không bắt được dữ liệu bị sửa
   (nhãn, thứ tự update, một ô trong bảng thẻ...) vì chỉ check_data.py kiểm; giờ cả hai đi chung một lệnh.
2. test_cleaning_output_ignores_truth_directory: thay data/truth bằng rác trong một bản sao rồi chạy làm sạch; kết quả phải
   giống từng byte (kiểm bằng hành vi, không chỉ quét AST như test_cleaning_module_never_reads_files_or_truth).
"""
import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

from .tv4_paths import ROOT

SCRIPT = r"""
import hashlib, sys
import pandas as pd
sys.path.insert(0, ".")
from src.preprocessing.cleaning import clean_card_table
cu = pd.read_csv("data/raw/bank_card/customers.csv", encoding="utf-8")
di = pd.read_csv("data/raw/bank_geo/districts.csv", encoding="utf-8")
for t in ("card_txn_5d", "card_txn_5d_hard"):
    raw = pd.read_csv(f"data/raw/bank_card/{t}.csv", encoding="utf-8")
    for by in ("default", None):
        kw = {} if by == "default" else {"impute_by": None}
        df, med = clean_card_table(raw, cu, di, **kw)
        print(t, by, hashlib.sha256((df.to_csv(index=False) + repr(sorted(med.items()))).encode()).hexdigest())
"""


def _run(cwd):
    r = subprocess.run([sys.executable, "-c", SCRIPT], cwd=cwd, capture_output=True, text=True, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    assert r.returncode == 0, r.stderr[-800:]
    return r.stdout.strip().splitlines()


def test_check_data_script_passes():
    r = subprocess.run([sys.executable, str(ROOT / "data" / "check_data.py")], cwd=ROOT, capture_output=True, text=True,
                       env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}, timeout=300)
    bad = [l for l in r.stdout.splitlines() if not l.startswith("ok") and l.strip()]
    assert r.returncode == 0, "check_data.py báo lỗi:\n" + "\n".join(bad[:30]) + r.stderr[-500:]


def test_cleaning_output_ignores_truth_directory(tmp_path):
    real = _run(ROOT)
    assert len(real) == 4
    copy = tmp_path / "repo"
    shutil.copytree(ROOT / "src" / "preprocessing", copy / "src" / "preprocessing", ignore=shutil.ignore_patterns("__pycache__"))
    (copy / "data" / "raw" / "bank_card").mkdir(parents=True)
    (copy / "data" / "raw" / "bank_geo").mkdir(parents=True)
    for f in ("card_txn_5d.csv", "card_txn_5d_hard.csv", "customers.csv"):
        shutil.copy(ROOT / "data" / "raw" / "bank_card" / f, copy / "data" / "raw" / "bank_card" / f)
    shutil.copy(ROOT / "data" / "raw" / "bank_geo" / "districts.csv", copy / "data" / "raw" / "bank_geo" / "districts.csv")
    rng = np.random.default_rng(0)
    truth = copy / "data" / "truth"
    truth.mkdir()
    for src in (ROOT / "data" / "truth").glob("*.csv"):               # cùng tên, cùng cột, nội dung là rác ngẫu nhiên
        t = pd.read_csv(src, encoding="utf-8")
        for c in t.columns:
            t[c] = rng.permutation(t[c].to_numpy()) if t[c].dtype != float else rng.normal(size=len(t)) * 1e6
        t.to_csv(truth / src.name, index=False, encoding="utf-8")
    assert _run(copy) == real                                          # truth là rác: kết quả làm sạch không đổi
    shutil.rmtree(truth)
    assert _run(copy) == real                                          # không có thư mục truth: vẫn đổi nguyên
