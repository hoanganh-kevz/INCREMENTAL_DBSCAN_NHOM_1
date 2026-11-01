# Incremental DBSCAN

## Nhóm đang làm gì?

Nhóm tự cài DBSCAN và Incremental DBSCAN theo hướng đối tượng. Khi thêm hoặc xóa một điểm, Incremental DBSCAN cập nhật vùng bị ảnh hưởng thay vì chạy lại DBSCAN trên toàn bộ dữ liệu.

Phần chính của đồ án gồm:

1. Cài DBSCAN và các thao tác insert/delete trong `src/`.
2. Minh họa một lần insert làm gộp cụm và một lần delete làm tách cụm.
3. Đánh giá chất lượng bằng Silhouette tự cài; đối chiếu tính tương đương với DBSCAN của scikit-learn.
4. So sánh thời gian với DBSCAN, HDBSCAN và OPTICS của scikit-learn trên cùng luồng dữ liệu.

## Dữ liệu chính

Toàn bộ dữ liệu là mô phỏng, không lấy từ ngân hàng thật.

| Dữ liệu | Dùng để làm gì? |
|---|---|
| `data/raw/atm_txn_2d.csv` | Demo 2D; thuật toán dùng hai cột `x_km`, `y_km`. |
| `data/raw/card_stream/` | Một luồng thẻ 5D: D0 có 500 điểm, log có 300 cập nhật. Benchmark phát lại 20 cập nhật đầu và lấy 2D/3D/4D từ cùng luồng. |
| `data/raw/toy/` | Ví dụ nhỏ để giải thích insert/Merge và delete/Split từng bước. |

Sáu ca biên phục vụ test nằm trong `tests/fixtures/toy_edge/`; chúng không được dùng như bộ dữ liệu demo.

## Bắt đầu nhanh

Từ thư mục gốc repo:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python run_demo.py
jupyter notebook notebooks/Incremental_DBSCAN_Demo.ipynb
```

### Thứ tự xem khi chuẩn bị báo cáo

1. `notebooks/Incremental_DBSCAN_Demo.ipynb`: insert/Merge, delete/Split, ví dụ ATM 2D, Silhouette và kiểm tra tương đương.
2. `notebooks/So_sanh_thuat_toan.ipynb`: benchmark 2D/3D/4D trên cùng luồng thẻ và so sánh với DBSCAN, HDBSCAN, OPTICS.

## Mã nguồn

- `src/core/`: điểm dữ liệu và khoảng cách.
- `src/clustering/`: DBSCAN và Incremental DBSCAN tự cài.
- `src/evaluation/metrics.py`: Silhouette tự cài.
- `src/preprocessing/`: chuẩn hóa thủ công và chuyển đặc trưng luồng sang tọa độ phân cụm.
- `src/visualization/`, `src/experiments/`: hình minh họa và điều phối thí nghiệm.
- `baseline/`: adapter và thuật toán scikit-learn chỉ dùng làm đối chứng.
- `tests/`: kiểm tra các thao tác cập nhật, trường hợp biên và phần hỗ trợ.

Thuật toán trong `src/clustering/` không gọi scikit-learn. NumPy và Pandas hỗ trợ dữ liệu; scikit-learn chỉ dùng ở phần đối chứng.

## Chạy kiểm tra

```bash
pytest tests/
```

Mở [`data/README.md`](data/README.md) để xem cấu trúc và cách dùng các bộ dữ liệu demo.
