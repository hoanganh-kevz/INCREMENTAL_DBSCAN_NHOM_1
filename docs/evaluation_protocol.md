# Quy trình đánh giá (V01)

## Mục tiêu

Đánh giá riêng ba khía cạnh: **độ đúng của cập nhật**, **chất lượng phân cụm** và **chi phí chạy**. Không dùng Silhouette làm tiêu chí duy nhất để kết luận một thuật toán DBSCAN tốt hay kém.

## Dữ liệu và cách chạy

- Dùng bộ dữ liệu nhỏ có nhãn kỳ vọng để kiểm tra từng tình huống insert/delete; dùng thêm dữ liệu tổng hợp 2D có dạng cầu, hai nửa trăng và vòng tròn, cùng ít nhất một bộ nhiều chiều.
- Giữ cố định dữ liệu, seed, `eps`, `MinPts`, metric khoảng cách và phiên bản môi trường khi so sánh các lần chạy.
- Với mỗi phép cập nhật, lưu dữ liệu trước/sau, thao tác, nhãn trước/sau, số cụm, số điểm noise và thời gian cập nhật.
- So sánh kết quả incremental với DBSCAN chạy lại từ đầu trên cùng dữ liệu và cùng tham số.

## Chỉ số

| Khía cạnh | Chỉ số / phép kiểm tra | Cách diễn giải |
|---|---|---|
| Độ đúng cập nhật | ARI giữa nhãn incremental và nhãn DBSCAN chạy lại từ đầu | ARI = 1 nghĩa là hai phép chia cụm tương đương, không phụ thuộc số ID cụm. Với điểm biên có thể thuộc hơn một cụm hợp lệ, cần kiểm tra thêm noise/core và kịch bản kỳ vọng. |
| Chất lượng khi có nhãn chuẩn | ARI và AMI giữa nhãn dự đoán và nhãn chuẩn | Báo cáo cùng tỷ lệ noise; noise được giữ như một nhãn dự đoán để lỗi nhận diện noise không bị che đi. |
| Chất lượng không nhãn | Silhouette Euclidean tự cài đặt | Bỏ noise khỏi phép tính và ghi rõ số lượng/tỷ lệ điểm bị bỏ. Silhouette cao không đảm bảo cụm mật độ có hình dạng đúng. |
| Mô tả đầu ra | Số cụm (không tính noise), số điểm noise, tỷ lệ noise | Cần có các số này cạnh Silhouette để thể hiện thuật toán đã loại bao nhiêu điểm. |
| Hiệu năng | Thời gian fit đầy đủ, insert, delete; số điểm bị xét lại nếu có | Chạy lặp nhiều lần cùng môi trường và báo cáo trung vị; không so sánh thời gian đo ở máy/môi trường khác nhau. |

## Quy ước Silhouette và noise

Silhouette chỉ tính trên các điểm có nhãn cụm khác `noise_label` (mặc định `-1`). Hàm `silhouette_samples` vẫn giữ nguyên thứ tự đầu vào và trả `None` ở vị trí noise; `silhouette_score` lấy trung bình các điểm còn lại. Cần ít nhất hai cụm không-noise và nhiều điểm hơn số cụm. Điểm thuộc cụm đơn lẻ nhận giá trị 0.

Silhouette đo mức gần nhau trong cụm và xa nhau giữa các cụm bằng khoảng cách trung bình. Chỉ số này thiên về cụm gọn, gần lồi; có thể đánh giá thấp các cụm dạng vòng/nửa trăng, vốn là dạng DBSCAN thường tìm được. Bỏ noise có thể làm điểm số trông tốt hơn, vì vậy luôn báo cáo tỷ lệ noise bên cạnh.

## Đối chiếu triển khai thư viện

`src/evaluation/metrics.py` là bản tự cài, chỉ dùng thư viện chuẩn. `src/evaluation/evaluation_library.py` gọi scikit-learn để đối chiếu trên cùng các điểm không-noise và cùng metric Euclidean. Bản thư viện không được import vào `src/clustering` hay dùng trong logic phân cụm.

Khi có nhãn chuẩn, ARI/AMI của bản thư viện tính trên toàn bộ điểm; nhãn noise là một nhãn dự đoán bình thường. Điều này giữ lại cả lỗi dự đoán điểm thuộc cụm thành noise và ngược lại. Kết quả đối chiếu Silhouette trên fixture mẫu được lưu tại `results/evaluation_metrics_crosscheck.csv` và được kiểm tra trong `tests/test_evaluation_metrics.py`.

## Tương đương với DBSCAN chạy lại (V04)

`src/evaluation/equivalence.py` phát lại từng kịch bản toy/toy_edge và ba stream dữ liệu trên Incremental DBSCAN tự cài, rồi so với `RefitDBSCAN` của sklearn sau mỗi bước. ARI không phụ thuộc ID cụm; so sánh thêm tập noise và tập core. Chỉ các điểm biên mơ hồ đã được đánh dấu trong dữ liệu mới bị bỏ qua. Chạy `python -m src.evaluation.equivalence` để tạo `results/equivalence_results.csv`; cột `equivalent` chỉ đạt khi ARI = 1, noise khớp và tập core khớp.
