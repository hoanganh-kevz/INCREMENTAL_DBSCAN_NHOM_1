# Dữ liệu demo

Toàn bộ dữ liệu dưới đây là mô phỏng. Phần demo chỉ dùng một bộ ATM 2D, một luồng thẻ và các ví dụ nhỏ cho insert/delete.

## Cấu trúc

```text
data/raw/
├── atm_txn_2d.csv       # tọa độ ATM 2D
├── card_stream/         # một luồng thẻ 5D: D0, các cập nhật, tham số
└── incremental_examples/ # 8 ví dụ nhỏ, gồm Merge và Split
```

## Dùng trong notebook

| Bộ dữ liệu | File | Mục đích |
|---|---|---|
| ATM 2D | `raw/atm_txn_2d.csv` | Hình minh họa cụm trên mặt phẳng, chỉ dùng `x_km`, `y_km`. |
| Luồng thẻ | `raw/card_stream/d0.csv`, `updates.csv`, `params.json` | Demo insert/delete và benchmark. D0 có 500 điểm, luồng có 300 bước. |
| Ví dụ cập nhật | `raw/incremental_examples/` | Theo dõi từng bước Merge, Split và các thao tác cơ bản. |

Benchmark 2D/3D/4D lấy lần lượt 2, 3, 4 đặc trưng từ cùng luồng thẻ 5D; không có các bộ CSV riêng cho từng số chiều. Notebook benchmark phát lại 20 cập nhật đầu. Các trường hợp edge chỉ nằm trong `tests/fixtures`, không thuộc dữ liệu demo.

## Các cột cần biết

- ATM: `point_id` là mã điểm; `x_km`, `y_km` là tọa độ 2D dùng để vẽ và phân cụm.
- Luồng thẻ: năm đặc trưng số là `amount_vnd`, `hour_of_day`, `distance_from_home_km`, `txns_last_7d`, `merchant_risk_score`. `step`, `op`, `point_id` mô tả thứ tự và loại cập nhật; chúng không phải đặc trưng phân cụm.
- `params.json` lưu `eps=0.4`, `minpts=6`, thứ tự đặc trưng và hằng số chuẩn hóa. Benchmark dùng `log10(amount_vnd)` và giờ cho 2D; thêm khoảng cách cho 3D; thêm số giao dịch tuần cho 4D. `merchant_risk_score` nằm trong luồng gốc nhưng không dùng ở các phép so sánh này.
- Mọi dữ liệu là mô phỏng. Toy là các ví dụ nhỏ được dựng sẵn để giải thích tình huống thuật toán, không phải quan sát thực tế.

Lưu ý: các hằng số chuẩn hóa trong `raw/card_stream/params.json` được tính từ toàn bộ luồng mô phỏng, gồm cả các bước sau D0. Cách này giữ tọa độ nhất quán giữa các thuật toán để so sánh, nhưng không mô phỏng triển khai online không biết trước dữ liệu tương lai.

## Lệnh chạy

Từ thư mục gốc repo:

```bash
jupyter notebook notebooks/Incremental_DBSCAN_Demo.ipynb
jupyter notebook notebooks/So_sanh_thuat_toan.ipynb
pytest tests/
```
