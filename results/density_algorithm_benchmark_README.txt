Benchmark chuỗi cập nhật: Incremental DBSCAN OOP so với DBSCAN, HDBSCAN và OPTICS của sklearn chạy lại toàn bộ.
Máy: Windows-11-10.0.26300-SP0, AMD64 Family 25 Model 68 Stepping 1, AuthenticAMD, Python 3.14.2, scikit-learn 1.9.1, numpy 2.4.3, pandas 3.0.3.
Mỗi stream: 1 warm-up fit D0 và 5 lượt phát lại prefix tối đa 20 update; seed xáo thứ tự thuật toán: 42.
Bốn phương pháp nhận cùng D0, thứ tự update, tọa độ, metric Euclidean và MinPts tham chiếu. Incremental DBSCAN chỉ cập nhật vùng ảnh hưởng;
ba estimator sklearn không có insert/delete nên adapter cập nhật tập điểm rồi fit lại toàn bộ ở từng step.
DBSCAN dùng Eps/MinPts trong params.json; HDBSCAN dùng min_cluster_size=min_samples=MinPts và EOM;
OPTICS dùng min_samples=MinPts, xi=0.05 và cluster_method='xi'. OPTICS/HDBSCAN có ngữ nghĩa tham số riêng, không dùng Eps DBSCAN.
CSV repeats có một dòng cho mỗi stream/phương pháp/lượt chạy; CSV runs lưu từng step/lượt; CSV summary tổng hợp median/mean/std/min/max thời gian qua các lượt.
Các CSV có cả cấu hình tham số, số cụm/noise, ARI trung bình/cuối và tỷ lệ noise khớp Incremental DBSCAN OOP.
Thời gian phụ thuộc máy; kết quả trên prefix stream không đại diện toàn bộ 119/300 bước.
