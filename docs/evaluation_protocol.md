# Quy trình đánh giá

## Dữ liệu và cách chạy

- `data/raw/incremental_examples/` minh họa từng bước insert/Merge và delete/Split.
- `data/raw/atm_txn_2d.csv` minh họa DBSCAN trên mặt phẳng 2D.
- `data/raw/card_stream/` là luồng cập nhật dùng cho benchmark 2D/3D/4D. Các chiều là lát đặc trưng từ cùng một D0 và cùng thứ tự cập nhật.
- Trong từng cấu hình, giữ nguyên dữ liệu, `Eps`, `MinPts`, metric Euclidean và thứ tự cập nhật.

## Chỉ số

| Mục tiêu | Cách đánh giá |
|---|---|
| Chất lượng cụm | Silhouette tự cài; báo cùng số cụm và tỷ lệ noise. Noise bị bỏ khỏi phép tính Silhouette. |
| Độ đúng của bản DBSCAN tự cài | ARI giữa Incremental DBSCAN và sklearn DBSCAN chạy lại trên cùng tập điểm; kiểm tra thêm tập noise. |
| So sánh thuật toán | Incremental DBSCAN OOP, sklearn DBSCAN, HDBSCAN và OPTICS nhận cùng dữ liệu và metric. Ba estimator sklearn được fit lại sau mỗi cập nhật. |
| Thời gian | 5 lượt trên prefix 20 cập nhật; báo trung vị và độ lệch chuẩn, đồng thời tách fit D0 khỏi thời gian cập nhật. |

ARI chỉ dùng để kiểm tra mức tương đương giữa hai cài đặt DBSCAN; không dùng thay Silhouette để xếp hạng HDBSCAN hoặc OPTICS.

## Chạy đối chiếu DBSCAN

`src/evaluation/equivalence.py` phát lại 8 ví dụ cập nhật và luồng thẻ, rồi so Incremental DBSCAN với sklearn DBSCAN sau từng bước. Điểm biên mơ hồ được bỏ qua theo danh sách trong dữ liệu. Chạy từ thư mục gốc:

```bash
python -m src.evaluation.equivalence
```

Kết quả được ghi vào `results/equivalence_results.csv`.
