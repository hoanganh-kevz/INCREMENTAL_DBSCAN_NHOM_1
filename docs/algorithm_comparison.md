# So sánh thuật toán phân cụm (V06)

Bảng này phục vụ lựa chọn phương pháp và diễn giải kết quả. Các đặc điểm hiệu năng phụ thuộc metric, chỉ mục không gian, số chiều và cách cài đặt; độ phức tạp bên dưới là hướng dẫn phổ biến, không phải cam kết cho mọi thư viện.

| Thuật toán | Họ phương pháp | Hình dạng cụm | Xử lý noise | Tham số tiêu biểu | Cập nhật tăng dần | Quy mô / độ phức tạp thường gặp |
|---|---|---|---|---|---|---|
| DBSCAN | Mật độ | Tìm được cụm không lồi, nối bởi các vùng đủ mật độ | Có, gán điểm không thuộc cụm là noise | `eps`, `MinPts` | Thường phải phân cụm lại toàn bộ khi dữ liệu đổi | Có chỉ mục không gian thường nhanh hơn; trường hợp xấu nhất O(n²). |
| Incremental DBSCAN | Mật độ, cập nhật cục bộ | Như DBSCAN; cụm có thể tách hoặc nhập sau cập nhật | Có; noise có thể được hấp thụ hoặc tạo ra | `eps`, `MinPts`, cùng trạng thái lân cận/cụm | Có; xử lý vùng bị ảnh hưởng khi insert/delete | Chi phí phụ thuộc kích thước vùng ảnh hưởng; trường hợp xấu nhất vẫn có thể là O(n²). |
| OPTICS | Mật độ, thứ tự hóa | Hỗ trợ cấu trúc mật độ thay đổi và cụm không lồi | Có thể xác định noise khi trích cụm | `MinPts`, `max_eps`, quy tắc trích cụm | Thông thường cần tính lại thứ tự sau thay đổi dữ liệu | Cách cài đặt phổ biến có trường hợp xấu O(n²); phù hợp phân tích mật độ trên nhiều ngưỡng hơn DBSCAN đơn ngưỡng. |
| HDBSCAN | Mật độ phân cấp | Cụm không lồi, mật độ khác nhau | Có; có thể gán điểm không ổn định thành noise | `min_cluster_size`, `min_samples` | Thường cần dựng lại phân cấp khi dữ liệu thay đổi | Phụ thuộc triển khai; dùng cấu trúc MST/phân cấp, tốn thêm thời gian và bộ nhớ để tạo, lưu cây. |
| K-Means | Tâm cụm / tối ưu khoảng cách | Hợp với cụm gần hình cầu, kích thước tương tự | Không có noise tự nhiên; mọi điểm được gán cụm | Số cụm `k`, số lần khởi tạo | Không hỗ trợ insert/delete chính xác theo chuẩn; có biến thể cập nhật tâm xấp xỉ | Mỗi vòng lặp thường O(n·k·d); thường mở rộng tốt khi k và số chiều vừa phải. |
| HAC (phân cụm phân cấp kết tụ) | Phân cấp / khoảng cách liên cụm | Phụ thuộc linkage; có thể tạo cụm không lồi nhưng nhạy với linkage | Không có noise tự nhiên; mọi điểm nằm trong cây | Linkage, metric và ngưỡng cắt cây | Thường phải cập nhật cây khi dữ liệu thay đổi | Thường cần O(n²) bộ nhớ; một số linkage cần O(n²) đến O(n³) thời gian. |

## Benchmark thực nghiệm

Benchmark cập nhật chạy cùng D0, thứ tự insert/delete, tọa độ, metric Euclidean và `MinPts` tham chiếu qua bốn phương pháp: Incremental DBSCAN OOP tự cài, DBSCAN, HDBSCAN và OPTICS của scikit-learn. Incremental DBSCAN dùng insert/delete cục bộ; ba estimator scikit-learn được fit lại toàn bộ trên tập điểm hiện tại sau từng cập nhật.

| Phương pháp | Cấu hình trong benchmark | Cách cập nhật |
|---|---|---|
| Incremental DBSCAN OOP | `eps` và `min_pts` theo `params.json`; Euclidean | Insert/delete vùng ảnh hưởng |
| scikit-learn DBSCAN | `eps` theo `params.json`, `min_samples=MinPts`, Euclidean | Fit lại toàn bộ |
| scikit-learn HDBSCAN | `min_cluster_size=MinPts`, `min_samples=MinPts`, `cluster_selection_method='eom'`, Euclidean | Fit lại toàn bộ |
| scikit-learn OPTICS | `min_samples=MinPts`, `max_eps=inf`, `cluster_method='xi'`, `xi=0.05`, Euclidean | Fit lại toàn bộ |

`MinPts` của DBSCAN được ánh xạ sang `min_samples`; với HDBSCAN và OPTICS đây là tham số gần tương ứng chứ không làm cho các thuật toán có cùng định nghĩa cụm. HDBSCAN và OPTICS không dùng `eps` DBSCAN theo cùng cách. Vì vậy benchmark so sánh thời gian cập nhật trên cùng dữ liệu và cấu hình đã công bố, còn ARI/noise là mức đồng thuận với Incremental DBSCAN, không phải khẳng định chất lượng tối ưu của HDBSCAN/OPTICS.

Mặc định benchmark chạy 5 lượt trên prefix tối đa 20 cập nhật, cùng một stream thẻ với 2D, 3D và 4D bằng cách lấy lần lượt 2, 3 và 4 đặc trưng đầu sau chuẩn hóa đóng băng. Cả ba trường hợp dùng chung D0, thứ tự cập nhật, `eps=0.4`, `MinPts=6` và seed dữ liệu; chỉ số chiều đổi. Mỗi lượt gồm một lần fit D0 được ghi riêng và tổng thời gian insert/delete. Warm-up không được tính vào số liệu. CSV theo lượt, theo từng bước, bảng tổng hợp và hình được lưu tại `results/density_algorithm_benchmark_repeats.csv`, `results/density_algorithm_benchmark_runs.csv`, `results/density_algorithm_benchmark_summary.csv` và `report/figures/density_algorithm_benchmark.png`.

Chạy lại bằng:

```bash
python -m baseline.benchmark_density_algorithms --repeats 5 --warmups 1 --max-steps 20 --seed 42
```

## Cách đọc kết quả

- Chọn DBSCAN/Incremental DBSCAN khi cần cụm theo mật độ, hình dạng tùy ý và phát hiện noise; cần kiểm tra độ nhạy của `eps` và `MinPts`.
- Chọn OPTICS khi cần xem thứ tự cấu trúc mật độ qua nhiều ngưỡng thay vì chỉ một `eps`.
- Cân nhắc HDBSCAN khi mật độ thay đổi giữa các cụm và có thể chấp nhận bước dựng phân cấp.
- Dùng K-Means khi cần chia mọi điểm thành số cụm định trước và cụm gần hình cầu.
- Dùng HAC khi cần cây phân cấp dễ diễn giải và kích thước dữ liệu phù hợp với nhu cầu bộ nhớ.
- Incremental DBSCAN có lợi khi thay đổi dữ liệu cục bộ; nếu một cập nhật ảnh hưởng phần lớn dữ liệu thì chi phí có thể gần với chạy lại DBSCAN đầy đủ.

K-Means và HAC trong bảng là phần đối chiếu lý thuyết. Benchmark thực nghiệm chỉ so sánh bốn phương pháp density-based được liệt kê ở trên.
