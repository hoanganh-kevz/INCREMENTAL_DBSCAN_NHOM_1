"""Tạo data/processed/<tên>_scaled.csv và <tên>_scaler_params.json cho các bộ dữ liệu cố định, bằng ManualScaler (numpy thuần).

Chạy từ thư mục gốc repo:  python data/make_processed.py
Mỗi bộ: ManualScaler('standard') fit trên TOÀN BỘ file (đây là bảng tĩnh để chạy DBSCAN một lần, không có D0 hay stream), rồi
transform; file kết quả giữ id và nhãn tham chiếu (chỉ để đánh giá), thêm hậu tố _z cho cột đã chuẩn hóa, làm tròn 6 chữ số.
Bảng thẻ (card_txn_5d, card_txn_5d_hard) do notebook D02 xử lý riêng (làm sạch rồi chuẩn hóa), không nằm ở đây.
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.preprocessing.scaler import ManualScaler  # noqa: E402

RAW, OUT = ROOT / "data" / "raw", ROOT / "data" / "processed"
ATM_NOTE = ("Tọa độ ATM tính theo km và đẳng hướng; z-score từng cột có std khác nhau nên bóp méo hình học (khoảng cách Euclid trong "
            "không gian z khác khoảng cách thật). Eps 0.5 km chỉ có nghĩa trên tọa độ km thô; file này dành cho thí nghiệm cần chuẩn hóa.")
DATASETS = [  # (tên, file thô, cột đặc trưng, cột id, cột nhãn, ghi chú)
    ("moons_n500", "synthetic/moons_n500.csv", ["x", "y"], "point_id", "label", ""),
    ("rings_n500", "synthetic/rings_n500.csv", ["x", "y"], "point_id", "label", ""),
    ("blobs_noise_n500", "synthetic/blobs_noise_n500.csv", ["x", "y"], "point_id", "label", "label -1 là nhiễu cài sẵn"),
    ("blobs5d_n860", "nd/blobs5d_n860.csv", ["x1", "x2", "x3", "x4", "x5"], "point_id", "label",
     "5 chiều cùng thang nên z-score ít đổi hình học; label -1 là nhiễu cài sẵn"),
    ("atm_txn_2d", "bank_geo/atm_txn_2d.csv", ["x_km", "y_km"], "txn_id", "hotspot_id", ATM_NOTE),
    ("atm_txn_2d_hard", "bank_geo/atm_txn_2d_hard.csv", ["x_km", "y_km"], "txn_id", "hotspot_id", ATM_NOTE),
]


def build(name, rel, feats, id_col, label_col, note, out_dir=OUT):
    df = pd.read_csv(RAW / rel, encoding="utf-8")
    scaler = ManualScaler("standard").fit(df[feats])
    z = pd.DataFrame(scaler.transform(df[feats]), columns=[c + "_z" for c in feats])
    out = pd.concat([df[[id_col]], z, df[[label_col]]], axis=1)
    out.to_csv(out_dir / f"{name}_scaled.csv", index=False, float_format="%.6f", encoding="utf-8", lineterminator="\n")
    params = {"source": f"data/raw/{rel}", "n_rows": len(df), "feature_columns": feats, "id_column": id_col,
              "label_column": f"{label_col} (chỉ để đánh giá, không dùng để gom cụm)", "scaler": scaler.to_dict(),
              "fit_on": "toàn bộ file (bảng tĩnh, không có D0)", "note": note}
    (out_dir / f"{name}_scaler_params.json").write_text(json.dumps(params, indent=2, ensure_ascii=False) + "\n",
                                                    encoding="utf-8", newline="\n")
    return len(df)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for spec in DATASETS:
        print(f"{spec[0]}: {build(*spec)} dòng -> data/processed/{spec[0]}_scaled.csv + {spec[0]}_scaler_params.json")


if __name__ == "__main__":
    main()
