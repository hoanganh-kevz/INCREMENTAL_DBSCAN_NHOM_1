# Tổng quan dữ liệu (đọc trước `README_data.md`)

Tài liệu này trả lời ba câu: dữ liệu để làm gì, mỗi bộ dùng để hỏi điều gì, và dữ liệu trông ra sao. Chi tiết từng cột, số liệu và cách sinh nằm trong `README_data.md`.

**Toàn bộ là dữ liệu mô phỏng** của một ngân hàng số hư cấu, tự sinh bằng seed cố định, không lấy từ nguồn thật.

## 1. Dữ liệu để làm gì

Đồ án cài DBSCAN và Incremental DBSCAN. Để biết bản tự cài có đúng, nhanh và hợp lý hay không, nhóm cần dữ liệu có bốn tính chất mà dữ liệu công khai thường thiếu:

1. **Có nhãn thật**, để tính ARI và biết cụm nào là nhiễu.
2. **Có luồng cập nhật** (chèn và xóa điểm theo thứ tự) với nhãn case của từng bước, để thử đủ 4 case insert và 3 case delete.
3. **Có ranh giới và n chiều**, để bắt các lỗi hay gặp: khoảng cách đúng bằng Eps, điểm trùng, MinPts nhỏ, 4 và 5 chiều.
4. **Có dữ liệu bẩn kèm bản sạch**, để làm và chấm bước tiền xử lý.

## 2. Mỗi bộ dùng để hỏi điều gì

| Bộ | Câu hỏi nó giúp trả lời | Quy mô |
|---|---|---|
| `toy/` | Bản tự cài có cho đúng kết quả trên chính các ví dụ trong báo cáo (R02, R03) không? | 8 kịch bản, 3 đến 13 điểm |
| `toy_edge/` | Có đúng ở ranh giới không: cách đúng Eps, trùng tọa độ, merge 3 cụm, split thành 3, xóa rồi chèn lại, 4 chiều, MinPts = 2? | 19 kịch bản |
| ATM 2D | DBSCAN có tìm được cụm hình tùy ý (vòng, lưỡi liềm, dải) mà KMeans và HAC không làm được không? | 1.200 điểm |
| Thẻ 5D | Gom cụm nhiều chiều ra sao khi thang đo lệch, dữ liệu bẩn và có giao dịch bất thường? | 1.500 dòng |
| Bản khó (ATM, thẻ) | Khi cụm chồng lấn và mật độ không đều, DBSCAN còn hơn KMeans, HDBSCAN, OPTICS không? | 2.000 điểm, 1.600 dòng |
| Luồng cập nhật | Sau mỗi insert hoặc delete, kết quả tăng dần có giống re-fit toàn bộ không, và nhanh hơn bao nhiêu? | 119 và 300 step |
| `truth/` | Làm sạch dữ liệu đúng bao nhiêu phần trăm? | bản sạch, nhãn sinh, danh mục lỗi |

## 3. Dữ liệu trông ra sao

### 3.1 Bản đồ ATM 2D (`raw/bank_geo/atm_txn_2d.csv`)

Giao dịch ATM trong một thành phố giả 32 x 24 km. Mỗi dòng là một giao dịch, hai cột `x_km`, `y_km` là đặc trưng, `hotspot_id` là nhãn thật, các cột còn lại (thời điểm, mã máy, mã khách, số tiền, quận) chỉ để mô tả.

```
txn_id,x_km,y_km,hotspot_id,timestamp,atm_id,customer_id,amount_vnd,atm_network,district_id,atm_distance_km
ATM00001,16.031,6.190,4,2025-03-13 20:22:03,A23,C0055,2050000,own,D06,0.181
```

| hotspot_id | Hình dạng | Số điểm | Ý nghĩa |
|---|---|---|---|
| 0 | cụm tròn dày | 204 | rút tiền khu văn phòng trung tâm |
| 1 | cụm tròn gọn | 150 | khu đại học |
| 2 | vòng quanh chợ | 222 | rút tiền trước khi vào chợ |
| 3 | cụm nhỏ nằm trong vòng | 90 | tiểu thương trong chợ |
| 4 | lưỡi liềm | 228 | dọc bờ sông |
| 5 | dải dài | 168 | dọc quốc lộ |
| -1 | rải rác | 138 | giao dịch lẻ loi, là nhiễu |

Với Eps = 0.5 km và MinPts = 4, DBSCAN tìm đúng 6 cụm (ARI 0.99). KMeans với k = 6 chỉ đạt 0.58 vì cắt vỡ vòng và lưỡi liềm.

### 3.2 Giao dịch thẻ 5D (`raw/bank_card/card_txn_5d.csv`)

Năm đặc trưng số có thang đo rất khác nhau (số tiền từ vài nghìn đến hàng trăm triệu, giờ từ 0 đến 24, khoảng cách theo km, số lần, điểm 0 đến 1), cộng nhãn `segment_id` và các cột mô tả (thời điểm, khách, kênh, nhóm hàng, quận).

```
txn_id,amount_vnd,hour_of_day,distance_from_home_km,txns_last_7d,merchant_risk_score,segment_id,...
TXN001280,173000,13.23,6.29,8,0.201,0,2025-03-06 13:13:48,C0260,CONTACTLESS,ENT,D08,0
```

| segment_id | Hành vi | Số tiền điển hình | Giờ điển hình |
|---|---|---|---|
| 0 | bán lẻ hằng ngày | khoảng 200 nghìn | giữa ngày |
| 1 | mua sắm online | khoảng 800 nghìn | buổi tối |
| 2 | chi tiêu lớn khi đi xa | khoảng 10 triệu, cách nhà khoảng 60 km | giữa chiều |
| 3 | rút tiền đêm | khoảng 40 nghìn, nhiều lần mỗi tuần | rạng sáng |
| -1 | bất thường (15 giao dịch) | hàng trăm triệu, rất xa nhà | tùy |

Bảng này cố ý có lỗi để làm tiền xử lý: ô thiếu (khoảng 1.5%), 8 dòng trùng hệt, kênh viết nhiều kiểu (`POS`, `pos `, `Pos`), hai định dạng ngày giờ, mã khách và mã quận lạ. Bản sạch của mọi cột bị làm bẩn nằm trong `truth/`.

### 3.3 Luồng cập nhật (`atm_stream/`, `card_stream/`, `card_stream_hard/`)

Mỗi luồng có `d0.csv` (dữ liệu ban đầu), `updates.csv` (từng bước chèn hoặc xóa một điểm) và `params.json` (Eps, MinPts, số case).

```
step,timestamp,op,point_id,x_km,y_km,planted_case,verified_case,ambiguous_border_ids,district_id
1,2025-03-01,insert,529,14.000,2.000,Merge,Merge,,D04
2,2025-03-02,delete,437,26.500,18.000,Reduction,Reduction,,D02
```

- `verified_case` do chạy lại DBSCAN toàn bộ trước và sau mỗi bước, so theo phân hoạch của các điểm lõi, không do người đặt.
- `atm_stream` (119 step) là kịch bản dựng sẵn, phủ đủ các case insert và delete (Noise 12, Creation 13, Absorption 32, Merge 12, Removal 6, Reduction 25, Split 12, Non-split 7).
- `card_stream` (300 step) là cửa sổ trượt tự nhiên (chèn giao dịch mới, xóa giao dịch cũ nhất), chủ yếu Absorption và Reduction.
- `card_stream_hard` (300 step) có cầu nối mỏng giữa các phân khúc nên Merge (5) và Split (7) xảy ra thật.

### 3.4 Kịch bản chuẩn và biên

Mỗi thư mục có `points.csv`, `update.csv` và `expected.json` (nhãn trước và sau, tập lõi, UpdSeed). Ví dụ `toy/r02_4_merge`: 12 điểm thuộc hai cụm tách nhau 1.8 (Eps = 1.1, MinPts = 4); chèn `p = (1.9, 0.5)` thì hai cụm hợp nhất thành một. Kết quả của mọi kịch bản được tính bằng số học phân số chính xác, độc lập với sklearn.

## 4. Các lựa chọn thiết kế và lý do

- **Eps và MinPts khác nhau giữa các bộ.** Chọn riêng cho từng bộ bằng đồ thị k-distance và quét lưới: thẻ 0.65 và 10 (Eps dùng được trong khoảng 0.26 đến 1.04 với MinPts 6 đến 14), luồng thẻ 0.4 và 6, luồng thẻ khó 0.30 và 6 (Eps nhỏ để không nối các cụm chồng lấn), ATM 0.5 km và 4. Đây là lựa chọn thiết kế theo từng bộ, không phải tối ưu đo được.
- **z-score là bước bắt buộc.** Các cột khác đơn vị nên một Eps duy nhất chỉ có nghĩa sau khi chuẩn hóa: không z-score, Eps 0.65 chỉ cho ARI 0.08 ở file thẻ dễ.
- **log10 cho số tiền là lựa chọn hợp lý nhưng không bắt buộc.** Độ lệch của số tiền khoảng 14, sau log10 còn khoảng 0.6. File thẻ dễ đạt ARI 1.0 cả khi không log10; chỉ file khó được lợi (0.646 so với 0.552).
- **Giữ ngoại lai.** Ngoại lai chính là nhiễu mà DBSCAN phải tự tìm. Loại bằng IQR sẽ xóa 251 dòng (16.8%), gồm cả 15 trên 15 giao dịch bất thường.
- **Điền ô thiếu theo (kênh, nhóm hàng)** thay vì trung vị chung, vì sai số nhỏ hơn rõ rệt (số tiền gấp 3.3 lần, số giao dịch 7 ngày gấp 4 lần). Hai cột này đoán đúng phân khúc khoảng 88%, nên giá trị điền mang một ít thông tin về cụm; chỉ khoảng 1.5% ô bị điền và `segment_id` không được dùng.
- **D0 của quy trình tiền xử lý là 500 giao dịch sớm nhất theo thời gian.** `d0.csv` của luồng là điểm sinh riêng, không trùng dòng nào của bảng thẻ.
- **Scale của luồng là hằng số cố định** (`frozen_scaling`), tính trên toàn bộ bảng sạch nên có thông tin từ tương lai so với D0. Bù lại mọi thuật toán thấy cùng tọa độ. Fit lại bằng D0 chỉ lệch nhẹ: ARI 0.9959 (luồng thẻ) và 0.887 (luồng thẻ khó).
- **`hour_of_day` giữ dạng tuyến tính.** Không phân khúc nào vắt qua nửa đêm. Thử sin và cos cho cùng ARI ở bản dễ (1.0) và nhỉnh hơn một chút ở bản khó (0.662 so với 0.646), nhưng thêm một chiều nên phải chỉnh lại Eps.
- **Điểm biên mơ hồ** (nằm trong Eps của lõi hai cụm khác nhau) có thể được hai cài đặt đúng gán khác nhau. Cột `ambiguous_border_ids` liệt kê chúng; khi so ARI với re-fit, bỏ các id này.
- **Thời gian của baseline chỉ có nghĩa khi so với thuật toán tăng dần thật.** Re-fit bằng sklearn mất khoảng 0.9 ms (ATM, n = 500) đến khoảng 53 ms (thẻ, n = 4000); số phụ thuộc máy.

## 5. Giới hạn cần nói rõ khi dùng

- Dữ liệu mô phỏng nên kết luận nghiệp vụ không có giá trị thực tế; hồ sơ cụm chỉ để minh họa cách diễn giải.
- ARI trong bảng độ khó chọn tham số bằng nhãn thật nên lạc quan hơn thực tế.
- Phân khúc thẻ sinh theo phân phối Gauss nên bản dễ sạch hơn dữ liệu thật; bản khó giảm bớt nhưng vẫn là mô phỏng.
- Số tiền ATM trong bảng thẻ (trung vị 42.000 đồng, 73% dưới 50.000) không giống bảng ATM 2D (luôn là bội của 50.000); đừng dùng nó để lập luận về độ thực tế.

Danh sách đầy đủ ở mục 14 của `README_data.md`.

## 6. Ai dùng phần nào

| Thành viên | Dùng | Để làm gì |
|---|---|---|
| TV2 (insert) | `toy/`, `toy_edge/e01` đến `e10`, `e17` đến `e19` | test 4 case insert và các ranh giới |
| TV3 (delete) | `toy/`, `toy_edge/e11` đến `e17` | test 3 case delete, split, non-split |
| TV5 (đánh giá) | luồng cập nhật, `compare_partitions`, `difficulty_report.json` | kiểm tính tương đương với re-fit, so các thuật toán |
| TV4 (dữ liệu, thí nghiệm) | tất cả | tiền xử lý, baseline, hình, benchmark, notebook demo |
