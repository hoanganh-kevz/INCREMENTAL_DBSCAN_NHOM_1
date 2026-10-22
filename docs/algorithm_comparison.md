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

## Cách đọc kết quả

- Chọn DBSCAN/Incremental DBSCAN khi cần cụm theo mật độ, hình dạng tùy ý và phát hiện noise; cần kiểm tra độ nhạy của `eps` và `MinPts`.
- Chọn OPTICS khi cần xem thứ tự cấu trúc mật độ qua nhiều ngưỡng thay vì chỉ một `eps`.
- Cân nhắc HDBSCAN khi mật độ thay đổi giữa các cụm và có thể chấp nhận bước dựng phân cấp.
- Dùng K-Means khi cần chia mọi điểm thành số cụm định trước và cụm gần hình cầu.
- Dùng HAC khi cần cây phân cấp dễ diễn giải và kích thước dữ liệu phù hợp với nhu cầu bộ nhớ.
- Incremental DBSCAN có lợi khi thay đổi dữ liệu cục bộ; nếu một cập nhật ảnh hưởng phần lớn dữ liệu thì chi phí có thể gần với chạy lại DBSCAN đầy đủ.

Các thuật toán trong bảng ngoài DBSCAN/Incremental DBSCAN là đối tượng đối chiếu lý thuyết; bảng không tuyên bố rằng chúng đã được cài đặt trong dự án.
