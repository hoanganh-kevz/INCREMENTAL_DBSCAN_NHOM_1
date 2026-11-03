# Kết quả benchmark

Benchmark dùng một luồng thẻ mô phỏng. D0 có 500 điểm; mỗi cấu hình phát lại cùng 20 cập nhật đầu. 2D, 3D và 4D là các lát đặc trưng từ cùng dữ liệu. Bốn phương pháp dùng Euclidean; bản OOP cập nhật cục bộ, các estimator scikit-learn fit lại toàn bộ.

## Thời gian cập nhật

Các số dưới đây là trung vị ± độ lệch chuẩn của tổng thời gian 20 cập nhật qua 5 lượt, đơn vị ms:

| Số chiều | Incremental DBSCAN OOP | sklearn DBSCAN | sklearn HDBSCAN | sklearn OPTICS |
|---:|---:|---:|---:|---:|
| 2D | 321,08 ± 25,53 | 136,72 ± 14,81 | 198,41 ± 18,84 | 14.571,56 ± 238,01 |
| 3D | 288,78 ± 20,95 | 148,09 ± 6,57 | 194,18 ± 19,08 | 13.824,08 ± 373,53 |
| 4D | 194,56 ± 9,98 | 152,03 ± 8,23 | 208,37 ± 22,12 | 13.485,85 ± 232,03 |

Trong prefix được đo, sklearn DBSCAN fit lại nhanh hơn bản OOP ở cả ba số chiều. HDBSCAN nhanh hơn ở 2D/3D, còn bản OOP thấp hơn một chút ở 4D; OPTICS chậm nhất. Thời gian fit D0 ban đầu của bản OOP cũng cao hơn các estimator sklearn. Vì vậy kết quả này chưa cho thấy lợi thế tốc độ của bản OOP trên prefix 20 thao tác.

## Chất lượng cụm và giới hạn

Silhouette tự cài là chỉ số chất lượng chính; cần đọc cùng số cụm và tỷ lệ noise. ARI trong benchmark chỉ kiểm tra sự đồng thuận nhãn với Incremental DBSCAN, không chứng minh cấu hình HDBSCAN hoặc OPTICS là tối ưu.

Kết quả chỉ áp dụng cho dữ liệu mô phỏng và 20 cập nhật đầu của luồng này. 2D/3D/4D không phải ba bộ dữ liệu độc lập, nên không thể xem chúng như ba phép thử độc lập.

CSV theo lượt, từng bước và bảng tổng hợp nằm trong `results/density_algorithm_benchmark_*.csv`; hình tổng hợp nằm tại `report/figures/density_algorithm_benchmark.png`.
