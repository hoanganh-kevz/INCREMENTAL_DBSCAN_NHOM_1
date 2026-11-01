# Kết quả và thảo luận (V07)

> Bản thảo để tích hợp vào chương kết quả của báo cáo PDF. Số liệu thời gian bên dưới lấy từ các bảng kết quả hiện có trong repo và phụ thuộc máy chạy.

## 1. Độ đúng của cập nhật

Ở phép đối chiếu V04, 780 trạng thái sau cập nhật đều tương đương với DBSCAN của scikit-learn khi chạy lại trên toàn bộ dữ liệu: ARI bằng 1, tập điểm noise khớp và tập điểm core khớp theo tiêu chí của dự án. Đối chiếu bỏ qua các điểm biên đã được đánh dấu là mơ hồ, vì một điểm như vậy có thể hợp lệ ở hai cụm khác nhau.

Các thí nghiệm D05/D06 bổ sung 719 lần cập nhật được tổng hợp từ ba luồng ATM 2D, thẻ 5D và thẻ 5D khó. Tất cả nhóm tình huống trong `data/processed/experiments/d05_d06/case_summary.csv` đều ghi nhận kết quả tương đương với sklearn re-fit. Kết quả này cho thấy các tình huống insert và delete đã chọn được xử lý nhất quán; nó chưa chứng minh mọi cấu hình dữ liệu, Eps và MinPts đều đúng.

## 2. Chất lượng phân cụm và chọn tham số

V05 dùng 500 điểm hai chiều dạng hai nửa trăng. Chọn Eps = 0.18 và MinPts = 4: DBSCAN tự cài và sklearn đều tìm được 2 cụm, 9 điểm noise (1,8%), Silhouette khoảng 0,390 trên các điểm không-noise và ARI = 1 giữa hai cách cài. ARI so với nhãn sinh là khoảng 0,964.

Silhouette ở mức vừa phải không đồng nghĩa với phân cụm sai: Silhouette ưu tiên cụm gọn, gần lồi, trong khi hai nửa trăng có hình dạng phi lồi. Vì vậy cần đọc chỉ số này cùng tỷ lệ noise, hình dạng cụm và phép đối chiếu với nhãn tham chiếu; không dùng riêng Silhouette để chọn kết luận.

![Đồ thị k-distance dùng hỗ trợ chọn Eps](../results/k_distance_moons.png)

## 3. Insert, delete và thời gian chạy

Bảng dưới trích các nhóm tình huống đại diện từ `case_summary.csv`. Các giá trị là trung vị thời gian mỗi update, tính bằng mili giây; cột custom re-fit là một phép đo ở lần chạy này.

| Dữ liệu và tình huống | Incremental DBSCAN | sklearn re-fit | Custom DBSCAN re-fit | Nhận xét |
|---|---:|---:|---:|---|
| ATM 2D, insert Merge | 0,72 | 0,88 | 368,14 | Cập nhật cục bộ nhanh hơn cả hai cách re-fit trong nhóm này. |
| ATM 2D, delete PotentialSplit_Split | 4,49 | 0,90 | 376,30 | Tách cụm làm tăng chi phí xử lý vùng ảnh hưởng; sklearn vẫn nhanh hơn. |
| Thẻ 5D khó, delete Reduction | 68,56 | 1,37 | 419,56 | Trường hợp vùng ảnh hưởng lớn làm update chậm hơn đáng kể, dù vẫn thấp hơn custom re-fit. |

`stream_fit.csv` ghi một phép đo fit ban đầu riêng từ thí nghiệm D05/D06 trước đó; không gộp số đo này với benchmark V06 bên dưới vì số lượt và giao thức đo khác nhau. Benchmark V06 báo riêng thời gian fit D0 theo trung vị của 5 lượt trong `density_algorithm_benchmark_summary.csv`.

![Các tình huống insert](../report/figures/d05_insert_cases.png)

![Các tình huống delete](../report/figures/d06_delete_cases.png)

## 4. Thảo luận, giới hạn và trường hợp khó

- Kết quả đúng trên các bộ dữ liệu mô phỏng và tình huống đã lưu; cần thận trọng khi khái quát sang dữ liệu thực có mật độ, nhiễu và phân phối khác.
- Delete có thể phải kiểm tra lại nhiều điểm để xác định cụm còn liên thông hay đã tách. Số điểm bị xét lại lớn làm lợi thế cập nhật cục bộ giảm, như nhóm delete Reduction trên thẻ 5D khó.
- Bản custom viết bằng Python thuần chậm hơn nhiều so với cài đặt sklearn đã tối ưu. Vì thế báo cáo phân biệt rõ so sánh với custom re-fit và với sklearn re-fit.
- Eps/MinPts đã chọn cho bộ moons chỉ là cấu hình minh họa cho bộ đó; không phải cặp tham số dùng chung cho ATM, giao dịch thẻ và mọi dữ liệu n chiều.
- Dữ liệu thí nghiệm là dữ liệu mô phỏng. ARI so với nhãn sinh phản ánh mức khớp với cách sinh dữ liệu, không thay thế đánh giá trên dữ liệu thực.

## 5. Benchmark bốn thuật toán qua nhiều lượt

Benchmark phát lại cùng D0 gồm 500 điểm và cùng 20 cập nhật đầu (10 insert, 10 delete) trong **5 lượt** cho mỗi thuật toán và số chiều. Có một lượt warm-up fit D0 không tính vào số liệu; thứ tự chạy bốn thuật toán được xáo theo seed 42. Ba trường hợp dùng cùng stream thẻ, thêm dần đặc trưng sau chuẩn hóa đóng băng:

| Số chiều | Đặc trưng dùng |
|---:|---|
| 2D | `log10_amount_vnd`, `hour_of_day` |
| 3D | 2D + `distance_from_home_km` |
| 4D | 3D + `txns_last_7d` |

Cả ba trường hợp dùng Eps = 0,4, MinPts = 6 và khoảng cách Euclidean. OPTICS tính thứ tự reachability rồi dùng `cluster_method='dbscan'` để trích cụm tại cùng Eps; đặt `max_eps=0.400001` để bao phủ ngưỡng trích. HDBSCAN giữ cách chọn cụm phân cấp riêng (`min_cluster_size=6`, `min_samples=6`, `eom`). Bảng ghi **trung vị ± độ lệch chuẩn** của tổng thời gian 20 thao tác cập nhật, đơn vị ms, qua 5 lượt:

| Số chiều | Incremental DBSCAN OOP | sklearn DBSCAN | sklearn HDBSCAN | sklearn OPTICS |
|---:|---:|---:|---:|---:|
| 2D | 31,33 ± 0,62 | 60,24 ± 4,03 | 87,52 ± 3,15 | 6.486,10 ± 116,33 |
| 3D | 34,25 ± 1,61 | 67,65 ± 1,81 | 99,35 ± 3,44 | 6.543,71 ± 112,52 |
| 4D | 54,10 ± 1,39 | 72,35 ± 4,75 | 103,87 ± 2,88 | 6.392,64 ± 26,54 |

Các tham số cụ thể trong từng chiều:

| Thuật toán | Tham số benchmark |
|---|---|
| Incremental DBSCAN OOP | `eps=0.4`, `min_pts=6`, Euclidean; cập nhật vùng ảnh hưởng |
| sklearn DBSCAN | `eps=0.4`, `min_samples=6`, Euclidean; fit lại toàn bộ sau mỗi cập nhật |
| sklearn HDBSCAN | `min_cluster_size=6`, `min_samples=6`, `cluster_selection_method='eom'`, Euclidean; fit lại toàn bộ |
| sklearn OPTICS | `min_samples=6`, `eps=0.4`, `max_eps=0.400001`, `cluster_method='dbscan'`, Euclidean; fit lại toàn bộ |

Trong 20 bước này, tổng thời gian cập nhật OOP thấp hơn cả ba cách fit lại ở cả ba số chiều. Riêng OPTICS tốn khoảng 6,4–6,5 giây cho 20 lần fit lại; cách trích cụm DBSCAN ở Eps chung giúp so sánh nhãn rõ ràng hơn nhưng không biến OPTICS thành thuật toán tăng dần. Fit D0 ban đầu của OOP có trung vị khoảng 552–659 ms, trong khi DBSCAN của sklearn khoảng 3,6–4,1 ms, HDBSCAN khoảng 4,9–5,7 ms và OPTICS khoảng 314–334 ms. Vì vậy kết quả cho thấy lợi thế về thời gian cập nhật trong prefix này, không cho thấy bản OOP nhanh hơn toàn bộ từ lúc fit ban đầu.

ARI trung bình qua 20 bước so với nhãn Incremental DBSCAN OOP là 1,000 cho cả bốn phương pháp ở cả ba số chiều; tỷ lệ khớp tập noise cũng là 1,000. Kết quả này phù hợp với cấu hình đã chọn: DBSCAN và OPTICS dùng cùng Eps/MinPts và cách trích DBSCAN; HDBSCAN tình cờ cho cùng nhãn trên stream này với tham số phân cấp đã công bố. Không suy ra rằng HDBSCAN luôn tương đương DBSCAN hoặc rằng các thuật toán sẽ cho nhãn giống nhau trên dữ liệu khác. ARI/noise đo mức đồng thuận với OOP, không phải chất lượng tối ưu riêng của mỗi thuật toán.

Kết quả từng bước, từng lượt, bảng tổng hợp, cấu hình môi trường và biểu đồ nằm tại `results/density_algorithm_benchmark_runs.csv`, `results/density_algorithm_benchmark_repeats.csv`, `results/density_algorithm_benchmark_summary.csv`, `results/density_algorithm_benchmark_README.txt` và `report/figures/density_algorithm_benchmark.png`. CSV theo lượt có một dòng cho mỗi lần lặp và chứa đầy đủ cấu hình. Các trường hợp 2D/3D/4D là các tập đặc trưng lồng nhau của cùng dữ liệu, không phải ba bộ dữ liệu độc lập; chỉ dùng prefix 20 bước nên không suy rộng kết quả này ra toàn bộ 300 bước.

## 6. Ghi chú trước khi chốt số liệu hiệu năng

Các CSV/hình D09 hiện có chưa được tạo lại sau khi mã benchmark đổi sang chạy lặp cho sweep theo số update. `data/processed/experiments/d09/README.txt` ghi rõ phần sweep theo m còn dựa trên phép đo cũ/một lần và yêu cầu chạy lại. Vì vậy không dùng các đường cong D09 hiện tại để đưa ra kết luận định lượng cuối cùng về độ phức tạp hoặc xu hướng theo kích thước dữ liệu; cần cập nhật chúng sau khi benchmark D09 được chạy lại.
