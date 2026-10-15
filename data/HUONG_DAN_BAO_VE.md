# Hướng dẫn bảo vệ phần dữ liệu, tiền xử lý và baseline (TV4)

Mọi con số lấy từ repo; chỗ nào là lựa chọn thiết kế chứ không phải tối ưu đo được thì nói thẳng như vậy.

## Một trang tóm tắt

**Dữ liệu.** Toàn bộ là mô phỏng một ngân hàng số hư cấu (seed cố định, không có dữ liệu thật hay Timo). Có bản đồ ATM 2D theo km (1200 và 2000 giao dịch), bảng thẻ 5D cố tình bẩn (1500 dòng, bản khó 1600), 27 kịch bản toy phủ đủ 9 case của thuật toán gia tăng, ba stream cập nhật (ATM 119 update, thẻ 300, thẻ khó 300) và các bảng khách hàng, quận, ATM, nhóm hàng. Nhãn case (`verified_case`) do chạy lại DBSCAN đầy đủ trước và sau mỗi update, không do người đặt.

**Tiền xử lý (D02).** Bảng thẻ có 8 loại lỗi cài sẵn có kiểm đếm (ô thiếu khoảng 1,5%, 8 dòng trùng, kênh viết 16 kiểu, hai định dạng timestamp và 6 giá trị vô lý, khách lạ, mã quận sai). Sửa theo quy tắc, điền ô số thiếu bằng trung vị theo (kênh x nhóm hàng) học trên D0, gắn cờ thay vì xóa, KHÔNG loại ngoại lai. D0 = 500 giao dịch sớm nhất. Chuẩn hóa bằng `ManualScaler` tự viết (z-score); stream thẻ dùng hằng số cố định trong `params.json`.

**Baseline (D04).** `RefitDBSCAN` dùng sklearn, fit lại toàn bộ sau mỗi insert hoặc delete. Đây là lớp THAY THẾ TẠM cho thuật toán của TV2 và TV3, chưa phải incremental. Nó khớp một DBSCAN brute-force numpy viết riêng ở cả 719 bước của ba stream và 27 kịch bản toy; mỗi update đo 5 lần, lấy trung vị.

**Kiểm chứng.** `pytest tests/` (gồm cả `data/check_data.py`) và sha256 các cột nền trong `data/base_hashes.json`.

## 12 câu hỏi khó và câu trả lời ngắn

1. **Vì sao Eps, MinPts khác nhau giữa file thẻ (0.65, 10), stream thẻ (0.4, 6) và stream khó (0.30, 6)?** Mỗi bộ chọn riêng bằng đồ thị k-distance và quét lưới. File thẻ có vùng Eps ổn định rộng từ 0.26 đến 1.04 với MinPts 6 đến 14, nên chọn giữa vùng. Cửa sổ stream chỉ 500 điểm, ít điểm hơn nên MinPts nhỏ hơn; Eps 0.4 cho 474 trên 500 điểm đủ điều kiện lõi. Bản khó chồng lấn và có dải cầu nối, Eps phải nhỏ (0.30) để không nối cụm, chỉ 260 trên 500 điểm đủ điều kiện lõi. Đây là lựa chọn thiết kế theo từng bộ, không phải tối ưu đo được; ATM không chuẩn hóa, dùng 0.5 km và MinPts 4.
2. **Vì sao log10 số tiền?** Số tiền từ 1.000 đến 500 triệu, độ lệch (skewness) 14,05, sau log10 còn 0,61. Nhưng đo thực tế: file thẻ dễ đạt ARI 1.0 cả khi không log10; chỉ file khó được lợi (ARI 0.646 so với 0.552). Vậy log10 là lựa chọn hợp lý có lợi ở bản khó, không bắt buộc.
3. **Vì sao z-score?** Các cột khác đơn vị (đồng, giờ, km, số lần, điểm 0 đến 1) nên một Eps duy nhất chỉ có nghĩa sau chuẩn hóa. Không z-score, Eps 0.65 chỉ cho ARI 0.079 ở file dễ. Đây là bước quyết định.
4. **Vì sao giữ ngoại lai, không dùng IQR?** Ngoại lai chính là nhiễu mà DBSCAN phải tự tìm. IQR sẽ xóa 251 dòng (16,8%), gồm cả 15 trên 15 giao dịch bất thường cài sẵn. Bài mẫu K47 xóa 33% dòng bằng IQR.
5. **Vì sao điền theo kênh và nhóm hàng, có rò rỉ nhãn không?** Sai số điền nhỏ hơn trung vị chung (tiền gấp 3,3 lần, số giao dịch 7 ngày gấp 4 lần). Nhưng kênh cộng nhóm hàng đoán đúng phân khúc khoảng 87,7% (bình chọn đa số), nên giá trị điền mang ít thông tin về cụm: rò rỉ nhẹ, có thật. Giảm nhẹ: không dùng `segment_id`, học trên D0, chỉ khoảng 1,5% ô bị điền.
6. **Vì sao `hour_of_day` không dùng dạng vòng (sin, cos)?** Không phân khúc nào vắt qua nửa đêm (online từ 17,5 đến 24, rút tiền đêm từ 0 đến 4,1), nên dạng tuyến tính đủ. Đo thử sin và cos: ARI tốt nhất 1.0 ở bản dễ (không đổi), 0.662 so với 0.646 ở bản khó, nhưng thêm một chiều nên phải chỉnh lại Eps (giữ Eps 0.30 thì tụt còn 0.39). Lựa chọn đơn giản, không phải tối ưu.
7. **Vì sao thời gian baseline chỉ vài mili giây, và n = 4000 thì sao?** Ở n khoảng 500 mỗi lần fit lại mất 0,9 ms (ATM) và 1,4 đến 2,1 ms (thẻ). Đo theo n (`--bench`): ATM 0,9 ms ở n = 500 lên 7,9 ms ở n = 4000; thẻ 5D 1,6 ms lên khoảng 53 ms (nhanh hơn tuyến tính vì 5 chiều). Số phụ thuộc máy, và chỉ có nghĩa khi so với thuật toán incremental thật, hiện chưa có: chưa được nói "nhanh hơn".
8. **Điểm biên mơ hồ là gì?** Điểm biên nằm trong Eps của lõi của hai cụm khác nhau; hai cài đặt đúng có thể gán khác cụm (DBSCAN phụ thuộc thứ tự). ATM và stream thẻ không có; stream thẻ khó có 983 lượt (tối đa 7 mỗi step). Khi so nhãn bỏ các id này (cột `ambiguous_border_ids`), không phải lỗi.
9. **Vì sao D0 là 500 giao dịch sớm nhất, không dùng `d0.csv`?** `d0.csv` của stream là điểm sinh riêng, không dòng nào trùng bảng thẻ và không có `txn_id`, nên không chọn được D0 từ bảng. Đây là quyết định của nhóm, có ghi trong notebook D02.
10. **Chuẩn hóa của stream có dùng thông tin từ tương lai không?** Có. Hằng số `frozen_scaling` tính trên toàn bộ bảng sạch (cả dòng về sau và 15 bất thường): cửa sổ tham chiếu cố định, không phải fit trên D0. Giữ vì mọi thuật toán thấy cùng tọa độ. Đo trên D0, fit D0 so với hằng số cố định: ARI 0.9959 (dễ), 0.887 (khó, lõi 260 so với 240). Lựa chọn có chủ ý, không phải tối ưu.
11. **Sao biết nhãn case và baseline đúng?** Sklearn và một DBSCAN numpy viết độc lập cho cùng phân hoạch ở mọi bước (bỏ điểm biên mơ hồ); 19 kịch bản toy_edge tính bằng số học phân số chính xác; `check_data.py` kiểm bất biến số cụm và nó chạy trong pytest.
12. **Dữ liệu mô phỏng có giống thực tế không?** Không khẳng định. Số tiền ATM trong bảng thẻ có trung vị 42.000 VND, 73% dưới 50.000, chỉ 1,8% là bội của 50.000, trong khi bảng ATM 2D toàn bội của 50.000; hai bảng sinh độc lập và cột nền đã đóng băng nên không sửa. Đừng dùng số tiền ATM của bảng thẻ làm lập luận về độ thực tế.
