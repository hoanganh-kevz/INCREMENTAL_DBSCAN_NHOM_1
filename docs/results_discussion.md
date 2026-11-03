# Kết quả benchmark

Benchmark dùng một luồng thẻ mô phỏng. D0 có 500 điểm; mỗi cấu hình phát lại cùng 20 cập nhật đầu. 2D, 3D và 4D là các lát đặc trưng từ cùng dữ liệu. Bốn phương pháp dùng Euclidean; bản OOP cập nhật cục bộ, các estimator scikit-learn fit lại toàn bộ.

## Thời gian cập nhật

Các số dưới đây là trung vị ± độ lệch chuẩn của tổng thời gian 20 cập nhật qua 5 lượt, đơn vị ms:

| Số chiều | Incremental DBSCAN OOP | sklearn DBSCAN | sklearn HDBSCAN | sklearn OPTICS |
|---:|---:|---:|---:|---:|
| 2D | 16,52 ± 0,50 | 54,48 ± 3,78 | 87,41 ± 1,63 | 6.189,28 ± 165,49 |
| 3D | 16,87 ± 0,14 | 64,82 ± 1,97 | 93,89 ± 1,78 | 6.150,01 ± 81,80 |
| 4D | 18,60 ± 0,23 | 71,39 ± 0,97 | 100,45 ± 2,67 | 6.093,78 ± 27,38 |

Trong prefix được đo, Incremental DBSCAN có tổng thời gian cập nhật thấp hơn các estimator sklearn fit lại. Tuy nhiên, thời gian fit D0 ban đầu của bản OOP cao hơn sklearn DBSCAN; vì vậy không kết luận bản OOP nhanh hơn từ lúc bắt đầu huấn luyện.

## Chất lượng cụm và giới hạn

Silhouette tự cài là chỉ số chất lượng chính; cần đọc cùng số cụm và tỷ lệ noise. ARI trong benchmark chỉ kiểm tra sự đồng thuận nhãn với Incremental DBSCAN, không chứng minh cấu hình HDBSCAN hoặc OPTICS là tối ưu.

Kết quả chỉ áp dụng cho dữ liệu mô phỏng và 20 cập nhật đầu của luồng này. 2D/3D/4D không phải ba bộ dữ liệu độc lập, nên không thể xem chúng như ba phép thử độc lập.

CSV theo lượt, từng bước và bảng tổng hợp nằm trong `results/density_algorithm_benchmark_*.csv`; hình tổng hợp nằm tại `report/figures/density_algorithm_benchmark.png`.
