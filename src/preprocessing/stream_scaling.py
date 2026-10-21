"""Tọa độ của một stream cập nhật: dùng chung cho baseline/, src/experiments/ (sau này) và notebook.

Một hàm thuần: `apply_frozen_scaling(params, frame)` đọc các hằng số trong params.json của stream rồi trả ma trận
tọa độ (n, d). Stream thẻ: log10 của amount_vnd, rồi z = (x - mean) / std, thứ tự cột lấy từ params["frozen_scaling"]["columns"].
Stream ATM không có frozen_scaling (tọa độ đã tính theo km), dùng `stream_coordinates` để lấy cột thô.

QUAN TRỌNG, ĐÂY LÀ "CỬA SỔ THAM CHIẾU CỐ ĐỊNH" (fixed reference scaling window), KHÔNG PHẢI "FIT TRÊN D0":
mean và std trong `frozen_scaling` do generator tính trên TOÀN BỘ bảng sạch (1492 dòng cho stream thẻ, 1592 cho stream khó,
gồm cả 15 hoặc 16 giao dịch bất thường và cả các dòng sẽ chỉ xuất hiện về sau trong stream), nên chúng chứa thông tin
từ TƯƠNG LAI so với D0. Đây là quyết định có chủ ý và được ghi rõ, không phải sơ suất:
  * Lý do giữ: mọi thuật toán (re-fit, Incremental DBSCAN của TV2 và TV3) thấy đúng cùng một tọa độ, nên so sánh nhãn công bằng,
    và các hằng số này đã bị đóng băng trong data/base_hashes.json (không sinh lại).
  * Hệ quả đã đo (D0-fit so với frozen, cùng DBSCAN tham chiếu, chỉ trên D0): ARI 0.9959 ở stream thẻ dễ (474 so với 471 điểm lõi),
    0.887 ở stream thẻ khó (260 so với 240 điểm lõi). Tức là cách chuẩn hóa ảnh hưởng nhỏ ở bản dễ và rõ ở bản khó.
  * Khuyến nghị cho báo cáo: dùng hằng số cố định này để phát lại stream; chỉ dùng "fit trên D0" (ManualScaler.fit(D0)) ở notebook
    để minh họa đánh đổi rò rỉ thông tin tương lai. Khi nói với giảng viên, gọi đúng tên: cửa sổ tham chiếu cố định có thông tin tương lai.
"""
import numpy as np

from .scaler import ManualScaler

LOG_PREFIX = "log10_"


def feature_columns(d0):
    """Tên cột đặc trưng của một stream: các cột giữa point_id và ref_label trong d0.csv."""
    cols = list(d0.columns)
    return cols[1:cols.index("ref_label")]


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
