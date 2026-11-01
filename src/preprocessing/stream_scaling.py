"""Dùng chung phép chuẩn hóa cố định cho notebook, benchmark và kiểm tra luồng.

`apply_frozen_scaling` log10-transform amount_vnd rồi áp mean/std trong params.json. Các hằng số được tính từ toàn bộ dữ liệu
mô phỏng, gồm cả các điểm xuất hiện sau D0. Vì vậy benchmark bảo đảm mọi thuật toán nhận cùng tọa độ nhưng không mô phỏng
một hệ thống online không biết trước dữ liệu tương lai.
"""
import numpy as np

from .scaler import ManualScaler

LOG_PREFIX = "log10_"


def feature_columns(d0, params=None):
    """Tên các đặc trưng theo thứ tự chuẩn hóa trong params; hỗ trợ cả D0 chỉ có cột số."""
    if params and params.get("frozen_scaling"):
        return frozen_columns(params)
    excluded = {"point_id", "ref_label", "segment_id"}
    return [column for column in d0.columns if column not in excluded]


def frozen_columns(params):
    """Tên cột THÔ theo thứ tự của frozen_scaling (bỏ tiền tố log10_), hoặc None nếu stream không có frozen_scaling."""
    fs = params.get("frozen_scaling")
    return None if fs is None else [c[len(LOG_PREFIX):] if c.startswith(LOG_PREFIX) else c for c in fs["columns"]]


def apply_frozen_scaling(params, frame):
    """Tọa độ chuẩn hóa theo hằng số cố định của stream, không fit lại.

    params: dict đọc từ params.json (cần khóa frozen_scaling gồm columns, mean, std).
    frame: DataFrame có các cột thô (d0.csv, hoặc các dòng updates.csv). Cột có tiền tố log10_ trong frozen_scaling
    được lấy log10 từ cột thô cùng tên (amount_vnd); thứ tự cột kết quả theo params, không theo frame.
    Trả mảng float (len(frame), d). Báo lỗi rõ nếu thiếu frozen_scaling, thiếu cột, hoặc số tiền không dương.
    """
    fs = params.get("frozen_scaling")
    if fs is None:
        raise ValueError("params không có frozen_scaling (stream ATM dùng stream_coordinates)")
    raw_cols = frozen_columns(params)
    missing = [c for c in raw_cols if c not in frame.columns]
    if missing:
        raise ValueError(f"cột frozen_scaling {fs['columns']} không khớp cột stream: thiếu {missing}")
    x = frame[raw_cols].to_numpy(dtype=float).copy()
    for j, c in enumerate(fs["columns"]):
        if c.startswith(LOG_PREFIX):
            if (x[:, j] <= 0).any():
                raise ValueError(f"cột {raw_cols[j]} có giá trị không dương, không lấy log10 được")
            x[:, j] = np.log10(x[:, j])
    scaler = ManualScaler.from_dict({"method": "standard", "center": fs["mean"], "scale": fs["std"],
                                     "feature_names": fs["columns"]})
    return scaler.transform(x)


def stream_coordinates(params, frame, feature_cols):
    """Tọa độ đi vào engine cho BẤT KỲ stream nào: frozen_scaling nếu có, nếu không thì cột thô (ATM, theo km).

    feature_cols: kết quả của feature_columns(d0); với stream thẻ chúng phải khớp frozen_scaling (cùng tập, cùng thứ tự)
    để d0 và updates dùng đúng cùng một ma trận cột.
    """
    if params.get("frozen_scaling") is None:
        return frame[feature_cols].to_numpy(dtype=float).copy()
    if frozen_columns(params) != list(feature_cols):
        raise ValueError(f"cột frozen_scaling {params['frozen_scaling']['columns']} không khớp cột stream {list(feature_cols)}")
    return apply_frozen_scaling(params, frame)
