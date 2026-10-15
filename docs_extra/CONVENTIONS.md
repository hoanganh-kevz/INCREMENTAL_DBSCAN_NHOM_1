# Quy ước chung cho code của nhóm (phần dữ liệu và baseline)

Các quy ước dưới đây có thể chép sang `README.md` hoặc `docs/oop_design.md` khi nhóm thống nhất.

## 1. Lân cận Eps tính cả chính điểm đó

- `N_eps(p) = { q : dist(p, q) <= eps }` gồm cả `p` (khoảng cách 0). Điểm `p` là **lõi** khi `|N_eps(p)| >= MinPts`, nghĩa là cần `MinPts - 1` điểm khác trong bán kính.
- Biên Eps là **bao hàm** (`<=`): hai điểm cách nhau đúng Eps là láng giềng của nhau. Dữ liệu toy_edge `e01_insert_at_eps_inclusive` kiểm đúng điều này.
- Khoảng cách Euclid, không đổi sang bình phương trừ khi so với `eps * eps`.
- `MinPts` của nhóm bằng `min_samples` của scikit-learn (sklearn cũng đếm cả chính điểm). Nếu một tài liệu khác không đếm chính điểm thì MinPts của họ nhỏ hơn 1 so với của nhóm; đừng trộn.
- Điểm trùng toạ độ (`e04_insert_duplicate_coordinate`) là hai điểm khác nhau, mỗi điểm đếm điểm kia.
- Điểm biên nằm trong Eps của lõi của hai cụm khác nhau là **điểm biên mơ hồ**: hai cài đặt đúng có thể gán khác cụm. Cột `ambiguous_border_ids` của `updates.csv` liệt kê chúng; khi so nhãn, bỏ các id đó (`compare_partitions(..., ignore_ids=...)`).

## 2. Nhãn, id và cách so hai kết quả

- Nhãn `-1` là nhiễu, cụm đánh số từ 0 (id cụm không có nghĩa, chỉ cách chia mới có nghĩa).
- So hai kết quả bằng `baseline.sklearn_dbscan.compare_partitions`: ARI (nhiễu tính một lớp), tập nhiễu phải giống nhau, tập lõi nếu có kiểu điểm. `partition_equal` đòi cả ARI bằng 1 lẫn tập nhiễu giống nhau.
- **id stream khác id engine.** `point_id` của `d0.csv` và `updates.csv` là id của stream; engine tự cấp id (RefitDBSCAN: 0..n-1 cho `fit`, rồi tăng dần cho `insert`, không bao giờ dùng lại id đã xóa). Baseline giữ ánh xạ rõ ràng trong lớp `IdMap` và `labels_long.csv` luôn ghi id của stream. Engine thật của nhóm không cần cấp id giống stream; chỉ cần `insert` trả id và `snapshot()["ids"]` dùng cùng id đó.
- `insert` nhận tọa độ trần (list, ndarray) **hoặc** một đối tượng có thuộc tính `.coords` (như `DataPoint` trong `docs/oop_design.md`); `fit` nhận ma trận hoặc dãy các đối tượng như vậy (`as_coords`, `as_matrix` trong `baseline/sklearn_dbscan.py`).

## 3. Ba cách đặt tên phương thức đang cùng tồn tại (CHƯA CHỐT)

| Nguồn | Khởi tạo | Chèn | Xóa |
|---|---|---|---|
| Bảng task của nhóm | `fit_initial` | `insert` | `delete` |
| Tài liệu R04 | `fit` | `insert_point` | `delete_point` |
| `docs/oop_design.md` | `fit` | `insert` | `delete` |

`RefitDBSCAN` có đủ cả sáu tên (bản chính `fit`, `insert`, `delete`; `fit_initial`, `insert_point`, `delete_point` là bí danh tạm thời) để D05 đến D10 chạy với bất kỳ cách nào. Khi nhóm chốt, xóa các bí danh còn lại và đổi nơi gọi. Đề xuất: theo `docs/oop_design.md` (`fit`, `insert`, `delete`) vì đó là tài liệu thiết kế và `IncrementalDBSCAN` kế thừa `fit` từ `DBSCAN`.

## 4. Chuẩn hóa và stream

- Stream thẻ dùng `frozen_scaling` trong `params.json`: log10 của số tiền rồi z-score với hằng số cố định. Hằng số này do generator tính trên **toàn bộ bảng sạch** (gồm dòng sẽ chỉ xuất hiện về sau và các giao dịch bất thường), nên đây là **cửa sổ tham chiếu cố định có chứa thông tin từ tương lai**, không phải fit trên D0. Giữ lại vì mọi thuật toán thấy cùng tọa độ và vì hằng số đã bị đóng băng trong `data/base_hashes.json`.
- Số liệu đo (cùng DBSCAN tham chiếu, chỉ trên D0): fit trên D0 so với hằng số cố định cho ARI 0.9959 ở stream thẻ dễ (474 so với 471 điểm lõi) và 0.887 ở stream thẻ khó (260 so với 240 điểm lõi).
- Khuyến nghị cho báo cáo: phát lại stream bằng hằng số cố định (mọi thuật toán cùng tọa độ); chỉ dùng fit trên D0 trong notebook để minh họa đánh đổi rò rỉ.
- Mã dùng chung: `src/preprocessing/stream_scaling.py` (`apply_frozen_scaling`, `stream_coordinates`). Baseline và `src/experiments/` sau này cùng gọi hàm này, không viết lại.

## 5. Đo thời gian

- `run_stream` đo mỗi update 5 lần trên bản sao sâu (`copy.deepcopy`) của engine, ghi **trung vị** vào `update_seconds`, kèm `n_repeats` và `n` (số điểm engine xử lý) trong `timing.csv`. Update chính thức chỉ áp dụng một lần và không tính giờ, nên nhãn và số cụm không phụ thuộc số lần đo. Engine của nhóm phải `deepcopy` được; nếu không, đặt `repeats=1`.
- `python -m baseline.sklearn_dbscan --bench` chạy RefitDBSCAN trên các bảng `--n 500 1000 2000 4000` (tự sinh `data/raw/bench/` bằng `generate_data.write_bench` nếu chưa có) và ghi `data/processed/baseline_bench/`. **File thời gian phụ thuộc máy (CPU, hệ điều hành, phiên bản sklearn), không đưa lên git (có trong `data/.gitignore`).** `README.txt` trong thư mục đó ghi cấu hình máy lúc đo.

## 6. Tệp văn bản và Windows

- Mọi lệnh đọc/ghi file văn bản trong repo ghi rõ `encoding="utf-8"` (JSON và README có tiếng Việt; Windows mặc định cp1252 hoặc cp1258). `tests/test_encoding.py` quét AST để không ai thêm lệnh thiếu encoding. Ghi file dùng `newline="\n"` (hoặc `newline=""` với csv) để kết quả giống từng byte trên mọi máy.
- Gợi ý `.gitattributes` ở `data/GITATTRIBUTES_PROPOSED.txt`: nếu Windows checkout đổi LF thành CRLF, `check_data.py` báo `file bytes changed` ở các `params.json` và `expected.json`, vì chúng được băm và so từng byte.
