"""ManualScaler: chuẩn hóa đặc trưng bằng numpy thuần (không sklearn, không scipy).

Giao diện theo docs/oop_design.md: fit, transform, fit_transform, inverse_transform.

Hai phương pháp, cùng một công thức  z = (x - center) / scale:
    method='standard': center = mean, scale = std tổng thể (ddof=0, giống StandardScaler)
    method='minmax'  : center = min,  scale = max - min     (giống MinMaxScaler)
Cột có độ lệch bằng 0 (hằng số) nhận scale = 1 để tránh chia cho 0.

Quy tắc cho Incremental DBSCAN: tham số phải đứng yên trong suốt luồng. Điểm mới đi qua transform với đúng tham số đó; không fit lại,
nếu không tọa độ của các điểm cũ sẽ đổi và nhãn cụm đã tính không còn nhất quán. Luồng benchmark dùng hằng số chuẩn hóa cố định trong
`params.json`, tính trên toàn bộ luồng mô phỏng để mọi thuật toán nhận cùng tọa độ. Vì hằng số này có dùng dữ liệu ở các bước sau D0,
nó phục vụ so sánh có kiểm soát, không mô phỏng triển khai online không biết trước dữ liệu tương lai. Xem stream_scaling.py.
"""
import numpy as np

_METHODS = ("standard", "minmax")


class ManualScaler:
    def __init__(self, method="standard"):
        if method not in _METHODS:
            raise ValueError(f"method phải là một trong {_METHODS}, nhận được {method!r}")
        self.method = method
        self.center_ = None       # mean (standard) hoặc min (minmax), shape (d,)
        self.scale_ = None        # std (standard) hoặc range (minmax), đã thay 0 bằng 1
        self.feature_names_ = None  # tên cột nếu fit từ DataFrame
        self.n_samples_ = 0

    # ---------- kiểm tra đầu vào ----------
    def _as_2d(self, data, fitting):
        """Trả về (mảng float 2-D, có phải vector 1-D không). Báo lỗi rõ nếu có NaN/inf."""
        names = list(data.columns) if hasattr(data, "columns") else None
        arr = np.asarray(data, dtype=float)
        was_1d = arr.ndim == 1
        if was_1d:
            if fitting:
                raise ValueError("fit cần ma trận 2-D (n_mẫu, n_cột); nhận được vector 1-D")
            arr = arr[None, :]
        if arr.ndim != 2:
            raise ValueError(f"dữ liệu phải 2-D, nhận được {arr.ndim}-D")
        if fitting and arr.shape[0] == 0:
            raise ValueError("không thể fit trên 0 dòng")
        bad = ~np.isfinite(arr)
        if bad.any():
            cols = np.flatnonzero(bad.any(axis=0))
            label = [names[c] if names else int(c) for c in cols]
            raise ValueError(
                f"dữ liệu có {int(bad.sum())} ô NaN/inf ở cột {label}; "
                "hãy xử lý ở bước làm sạch (điền khuyết) trước khi chuẩn hóa")
        if fitting and names is not None:
            self.feature_names_ = names
        return arr, was_1d

    def _check_fitted(self):
        if self.center_ is None:
            raise RuntimeError("ManualScaler chưa fit; gọi fit(data) trước")

    # ---------- giao diện chính ----------
    def fit(self, data):
        x, _ = self._as_2d(data, fitting=True)
        if self.method == "standard":
            center = x.mean(axis=0)
            scale = x.std(axis=0)                  # ddof=0
        else:
            center = x.min(axis=0)
            scale = x.max(axis=0) - center
        self.center_ = center
        self.scale_ = np.where(scale == 0.0, 1.0, scale)
        self.n_samples_ = x.shape[0]
        return self

    def transform(self, data):
        """Dùng tham số đã fit. Vector 1-D (một điểm mới) trả về vector 1-D."""
        self._check_fitted()
        x, was_1d = self._as_2d(data, fitting=False)
        if x.shape[1] != self.center_.shape[0]:
            raise ValueError(f"cần {self.center_.shape[0]} cột, nhận được {x.shape[1]}")
        z = (x - self.center_) / self.scale_
        return z[0] if was_1d else z

    def fit_transform(self, data):
        return self.fit(data).transform(data)

    def inverse_transform(self, data):
        self._check_fitted()
        z, was_1d = self._as_2d(data, fitting=False)
        if z.shape[1] != self.center_.shape[0]:
            raise ValueError(f"cần {self.center_.shape[0]} cột, nhận được {z.shape[1]}")
        x = z * self.scale_ + self.center_
        return x[0] if was_1d else x

    # ---------- lưu và nạp tham số (JSON) ----------
    def to_dict(self):
        self._check_fitted()
        return {"method": self.method,
                "center": self.center_.tolist(),
                "scale": self.scale_.tolist(),
                "feature_names": self.feature_names_,
                "n_samples": int(self.n_samples_)}

    @classmethod
    def from_dict(cls, params):
        """Tạo scaler từ dict (ví dụ đọc từ JSON). Với 'standard': center = mean, scale = std."""
        obj = cls(params.get("method", "standard"))
        obj.center_ = np.asarray(params["center"], dtype=float)
        scale = np.asarray(params["scale"], dtype=float)
        if obj.center_.shape != scale.shape or obj.center_.ndim != 1:
            raise ValueError("center và scale phải là hai vector 1-D cùng độ dài")
        obj.scale_ = np.where(scale == 0.0, 1.0, scale)
        obj.feature_names_ = params.get("feature_names")
        obj.n_samples_ = int(params.get("n_samples", 0))
        return obj

    def __repr__(self):
        state = "đã fit" if self.center_ is not None else "chưa fit"
        return f"ManualScaler(method={self.method!r}, {state})"
