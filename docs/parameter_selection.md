# Chọn Eps và MinPts (V05)

## Dữ liệu và cách chọn

Thử nghiệm dùng `data/processed/moons_n500_scaled.csv` (500 điểm, 2 đặc trưng sau z-score). Đây là hai dải cong giao nhau, phù hợp để kiểm tra cụm không lồi của DBSCAN. Nhãn sinh chỉ dùng để báo cáo ARI sau khi chọn tham số, không dùng làm mục tiêu tối ưu.

Với MinPts = 4 (tính cả chính điểm theo quy ước dự án), đồ thị k-distance dùng khoảng cách tới láng giềng thứ ba. Điểm khuỷu ước lượng Eps ≈ 0.135. Grid cục bộ thử Eps `{0.14, 0.16, 0.18, 0.20, 0.22}` và MinPts `{4, 6, 8}`. Mỗi cặp được chạy bằng DBSCAN tự cài và sklearn DBSCAN với cùng dữ liệu, Eps, MinPts và Euclidean.

## Kết quả và lựa chọn

Chọn **Eps = 0.18, MinPts = 4**. Trên grid, đây là Eps nhỏ nhất tạo đúng hai cụm lớn và liên tục; Eps 0.14 và 0.16 còn chia nhỏ cấu trúc. Với Eps 0.20–0.22, số cụm không đổi và số noise giảm thêm, nhưng Eps 0.18 giữ lựa chọn bảo thủ ngay phía trên khuỷu k-distance để tránh nới vùng mật độ quá mức. MinPts = 4 phù hợp dữ liệu 2D và giữ lại các điểm biên thưa hơn so với MinPts 6 hoặc 8.

Ở cặp được chọn, Silhouette là khoảng 0.390 khi bỏ noise khỏi phép tính, tỷ lệ noise là 1.8%, và ARI so với nhãn sinh là khoảng 0.964. Silhouette không cao vì các cụm dạng trăng lưỡi liềm không lồi; không dùng điểm này một mình để đánh giá. ARI chỉ là kiểm tra tham khảo sau khi chọn, không phải tiêu chí chọn tham số.

Các con số đầy đủ theo grid nằm trong `results/parameter_selection.csv`; profile k-distance nằm trong `results/k_distance_moons.csv` và hình tại `results/k_distance_moons.png`. Có thể tạo lại từ thư mục gốc:

```bash
python -m src.evaluation.parameter_selection
```

Kết quả DBSCAN tăng dần so với DBSCAN sklearn chạy lại sau từng update được lưu riêng tại `results/equivalence_results.csv` (V04).
