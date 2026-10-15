# Data card: bộ dữ liệu thí nghiệm cho DBSCAN và Incremental DBSCAN (task D01)

**Toàn bộ dữ liệu là DỮ LIỆU MÔ PHỎNG (simulated), tự sinh bằng seed cố định. Không có một dòng nào lấy từ ngân hàng thật, không có dữ liệu Timo, không dùng lại dữ liệu IBFT của môn khác. Khách hàng, quận, máy ATM, merchant đều là hư cấu.**
*All data is simulated with fixed seeds. No real bank or Timo data is used.*

Tái tạo và kiểm tra (chạy từ thư mục gốc của repo, mất khoảng 35 giây và 30 giây):

```bash
python data/generate_data.py
```

```bash
python data/check_data.py
```

Các tùy chọn: `--preview` vẽ PNG vào `data/preview/`; `--difficulty medium` để sinh thêm bản vừa (chỉ tái tạo, không cần commit; bản dễ và bản khó luôn được ghi); `--n 500 1000 2000 4000` sinh bản benchmark vào `data/raw/bench/` (không commit). `check_data.py` thoát mã 1 nếu có sai lệch.

## 0. Tóm tắt

| Phần | Nội dung | Dùng cho |
|---|---|---|
| Toy (8 thư mục) | 8 ví dụ đúng theo tài liệu R02/R03 của nhóm, kết quả tính tay bằng Fraction | TV2, TV3, TV5, kiểm thử đơn vị |
| `toy_edge` (19 thư mục) | 19 kịch bản biên, nhiều update có thứ tự, 4D, MinPts=2, khoảng cách đúng bằng Eps | C12, TV2, TV3, TV5 |
| Synthetic, nd | moons, rings, blobs, blobs 5D | D04, so sánh với sklearn |
| ATM map 2D (dễ, khó) | 1200 và 2000 giao dịch ATM trên thành phố 32 x 24 km | D05 đến D10, DBSCAN 2D |
| Card 5D (dễ, khó) | 1500 và 1600 giao dịch thẻ, cố tình bẩn, chưa chuẩn hóa | D02 làm sạch, ManualScaler |
| Streams (ATM, card, card_stream_hard) | chuỗi chèn và xóa có nhãn case đã kiểm chứng | D03, D05, D06, D09, D10, TV2, TV3, TV5 |
| Bảng mô tả | 600 khách, 8 quận, 40 ATM, 14 nhóm merchant, cột timestamp, kênh, nhóm hàng | EDA, diễn giải cụm, báo cáo |
| `truth/` | bản sạch của mọi cột bị làm bẩn, nhãn sinh dữ liệu, danh mục lỗi | chấm điểm tiền xử lý |
| `DATA_DICTIONARY.csv` | từ điển dữ liệu song ngữ cho mọi bảng và cột | báo cáo |

So với bài mẫu K47 (một bảng Kaggle 167 dòng, 10 thuộc tính, không thiếu giá trị, chỉ có histogram), bộ này có: 6 bảng chiều liên kết bằng khóa ngoại với các bảng giao dịch, 8 loại lỗi dữ liệu có kiểm đếm và đáp án sạch, 3 mức độ khó có bảng ARI của 5 thuật toán, 3 stream với nhãn case đã kiểm chứng, 19 kịch bản biên tính bằng số học chính xác, hồ sơ cụm bằng ngôn ngữ nghiệp vụ và 17 hình tiếng Việt.

## 1. Mục đích, giả định, hạn chế, đạo đức

**Mục đích.** (1) Kiểm thử đúng đắn của DBSCAN và Incremental DBSCAN tự cài (so với re-fit toàn bộ), (2) cung cấp bài toán gom cụm có diễn giải nghiệp vụ để viết báo cáo sâu hơn một bảng Kaggle, (3) có dữ liệu bẩn có đáp án để chấm bước tiền xử lý, (4) có dữ liệu khó để so DBSCAN với KMeans, Ward, OPTICS, HDBSCAN.

**Cách sinh.** Dữ liệu 2D sinh theo lưới nguyên 1 m (để loại trừ cặp điểm cách nhau đúng bằng Eps); dữ liệu thẻ sinh từ 4 phân khúc Gauss trong không gian log10(số tiền), giờ, khoảng cách, số giao dịch 7 ngày, điểm rủi ro, cộng bất thường cài sẵn, rồi mới đục ô thiếu và nhân bản dòng. Các cột mô tả (timestamp, khách, kênh, nhóm hàng, quận) được sinh SAU, từ nhãn phân khúc và giờ trong ngày, nên KHÔNG ảnh hưởng đặc trưng gom cụm. Nhãn `-1` là bất thường hoặc nhiễu cài sẵn. DBSCAN của sklearn chỉ dùng trong generator và checker để xác minh dữ liệu (không bao giờ nằm trong `src/`).

**Giả định.**
- `x_km, y_km` của bảng ATM là vị trí ghi nhận giao dịch (lưới 1 m), `atm_id` là máy gần nhất trong cùng hotspot (điểm cô lập: máy gần nhất toàn thành phố); `atm_distance_km` là khoảng cách giữa hai vị trí.
- Quận là ô Voronoi quanh tâm quận (điểm thuộc quận có tâm gần nhất), nên suy ra được từ tọa độ một cách xác định.
- Hồ sơ khách: nghề, hạng, thu nhập, tuổi tương quan (sinh viên trẻ và thu nhập thấp, tiểu thương gần chợ, hạng cao thu nhập cao); phần lớn khách bị chi phối bởi một phân khúc, một số ít hỗn hợp (`truth/customer_segment_mix.csv`).
- Lịch: 8 tuần từ 2025-03-01 đến 2025-04-25; cuối tuần đông hơn với bán lẻ và online; đỉnh quanh ngày 5 và 3 ngày cuối tháng; ba ngày nghỉ lễ Giỗ Tổ (2025-04-05 đến 07) làm quốc lộ đông hơn.

**Hạn chế (biết rõ, không che giấu).**
- Phân khúc là Gauss trong không gian đặc trưng nên bản dễ sạch hơn dữ liệu thật; bản khó giảm bớt điều này nhưng vẫn là mô phỏng.
- `txns_last_7d`, `distance_from_home_km`, `merchant_risk_score` và `amount_vnd` được sinh theo phân khúc, KHÔNG tính lại từ `timestamp` hay từ thu nhập khách. Khách chỉ được chọn sao cho hợp phân khúc và hạng (khách hạng cao hay có chi tiêu du lịch lớn), số tiền không ràng buộc với thu nhập.
- Chỉ 549 trong 600 khách có giao dịch thẻ ở bảng chính (đếm trên bảng sạch 1492 dòng; bảng thô có 551 giá trị customer_id khác nhau nhưng 5 trong số đó là mã bẩn như `UNKNOWN`, và chỉ 546 mã hợp lệ còn xuất hiện sau khi bẩn hóa); giới tính nhị phân là đơn giản hóa của mô phỏng.
- Giờ trong ngày (`hour_of_day`) là đặc trưng không tuần hoàn (23.9 và 0.1 cách xa nhau); đây là bẫy có chủ ý cho bước tiền xử lý.
- Cột `distance_from_home_km` có std bị 15 điểm bất thường (150 đến 900 km) làm phình ra, nên sau z-score cột này gần như không phân biệt được các nhóm thường: đây là hiện tượng thật cho báo cáo (cân nhắc chuẩn hóa bền vững hơn), không phải lỗi.
- Bản khó của bản đồ ATM là một bố cục khác (tên hotspot giữ nguyên để so nhãn, vị trí khác), nên file đó chỉ có 4 cột nền và không có quận hay ATM.
- Mọi ARI trong bảng độ khó chọn tham số bằng nhãn thật (oracle), nên lạc quan hơn thực tế.

**Đạo đức (ethical note).** Không có dữ liệu cá nhân thật; tên khách là mã `C0001`..`C0600`, tên quận hư cấu. Mô tả phân khúc "nghi ngờ" chỉ là nhãn mô phỏng, không dùng để suy ra gì về người thật. Các gợi ý hành động trong `hotspots.csv` và `segments.csv` chỉ minh họa cách diễn giải cụm.

**Seed.** Seed gốc 42 (toy, synthetic, nd, bản đồ ATM), 43 (thẻ), 44 (stream ATM), 45 (stream thẻ). Mọi phần mới dùng bộ sinh riêng với seed 42 cộng độ lệch từ 10 trở lên, nên không làm xáo trộn các lần rút cũ: khách 52, mô tả thẻ 53, mô tả ATM 54, vị trí ATM 55, lỗi bẩn 56, mô tả stream 57, thẻ khó 62, thẻ vừa 63, ATM khó 64, ATM vừa 65, stream khó 66, mô tả bản khó 73, lỗi bẩn bản khó 74, mô tả bản vừa 75, lỗi bẩn bản vừa 76, mô tả stream khó 78.

**Bất biến bước 0.** `data/base_hashes.json` lưu sha256 của mọi cột nền, id, nhãn, hằng số `frozen_scaling` và cột `verified_case` của dữ liệu gốc (tạo bằng `tools/make_base_hashes.py` trước khi làm giàu dữ liệu). `check_data.py` thất bại nếu có cột nền nào đổi dù chỉ một byte. Mọi cột mới được thêm vào CUỐI file.

## 2. Bản đồ file và kích thước

<!-- FILEMAP:START -->
| File | Dòng | Dung lượng |
|---|---|---|
| `raw/DATA_DICTIONARY.csv` | 307 | 67.3 KB |
| `raw/bank_card/card_stream/d0.csv` | 500 | 46.4 KB |
| `raw/bank_card/card_stream/params.json` |  | 2.4 KB |
| `raw/bank_card/card_stream/updates.csv` | 300 | 38.9 KB |
| `raw/bank_card/card_stream_hard/d0.csv` | 500 | 46.3 KB |
| `raw/bank_card/card_stream_hard/params.json` |  | 3.0 KB |
| `raw/bank_card/card_stream_hard/updates.csv` | 300 | 42.5 KB |
| `raw/bank_card/card_txn_5d.csv` | 1500 | 119.6 KB |
| `raw/bank_card/card_txn_5d_hard.csv` | 1600 | 127.5 KB |
| `raw/bank_card/customers.csv` | 600 | 31.9 KB |
| `raw/bank_card/merchant_categories.csv` | 14 | 0.7 KB |
| `raw/bank_card/params.json` |  | 1.6 KB |
| `raw/bank_card/segments.csv` | 5 | 1.5 KB |
| `raw/bank_geo/atm_stream/d0.csv` | 529 | 12.2 KB |
| `raw/bank_geo/atm_stream/params.json` |  | 1.7 KB |
| `raw/bank_geo/atm_stream/updates.csv` | 119 | 7.7 KB |
| `raw/bank_geo/atm_txn_2d.csv` | 1200 | 92.5 KB |
| `raw/bank_geo/atm_txn_2d_hard.csv` | 2000 | 49.7 KB |
| `raw/bank_geo/atms.csv` | 40 | 1.2 KB |
| `raw/bank_geo/districts.csv` | 8 | 0.5 KB |
| `raw/bank_geo/hotspots.csv` | 7 | 1.8 KB |
| `raw/bank_geo/params.json` |  | 0.3 KB |
| `raw/nd/blobs5d_n860.csv` | 860 | 46.0 KB |
| `raw/synthetic/blobs_noise_n500.csv` | 500 | 12.7 KB |
| `raw/synthetic/moons_n500.csv` | 500 | 12.4 KB |
| `raw/synthetic/rings_n500.csv` | 500 | 12.6 KB |
| `truth/atm_txn_2d_hard_generation_flags.csv` | 2000 | 27.8 KB |
| `truth/card_txn_5d_clean_base.csv` | 1492 | 57.6 KB |
| `truth/card_txn_5d_clean_descriptive.csv` | 1492 | 77.8 KB |
| `truth/card_txn_5d_hard_clean_base.csv` | 1592 | 61.3 KB |
| `truth/card_txn_5d_hard_clean_descriptive.csv` | 1592 | 83.2 KB |
| `truth/card_txn_5d_hard_generation_flags.csv` | 1592 | 29.6 KB |
| `truth/customer_segment_mix.csv` | 600 | 23.7 KB |
| `truth/defect_catalogue_card.csv` | 24 | 1.4 KB |
| `processed/atm_txn_2d_hard_scaled.csv` | 2000 | 59.1 KB |
| `processed/atm_txn_2d_hard_scaler_params.json` |  | 0.8 KB |
| `processed/atm_txn_2d_scaled.csv` | 1200 | 35.3 KB |
| `processed/atm_txn_2d_scaler_params.json` |  | 0.9 KB |
| `processed/baseline/atm_stream/labels_long.csv` | 64890 | 596.5 KB |
| `processed/baseline/atm_stream/timing.csv` | 120 | 5.7 KB |
| `processed/baseline/card_stream/labels_long.csv` | 150650 | 1412.1 KB |
| `processed/baseline/card_stream/timing.csv` | 301 | 13.9 KB |
| `processed/baseline/card_stream_hard/labels_long.csv` | 150650 | 1450.9 KB |
| `processed/baseline/card_stream_hard/timing.csv` | 301 | 14.3 KB |
| `processed/blobs5d_n860_scaled.csv` | 860 | 45.0 KB |
| `processed/blobs5d_n860_scaler_params.json` |  | 0.9 KB |
| `processed/blobs_noise_n500_scaled.csv` | 500 | 12.3 KB |
| `processed/blobs_noise_n500_scaler_params.json` |  | 0.6 KB |
| `processed/card_txn_5d_clean.csv` | 1492 | 242.5 KB |
| `processed/card_txn_5d_hard_clean.csv` | 1592 | 258.7 KB |
| `processed/card_txn_5d_hard_scaled.csv` | 1592 | 181.9 KB |
| `processed/card_txn_5d_scaled.csv` | 1492 | 170.0 KB |
| `processed/cleaning_scorecard.csv` | 38 | 6.1 KB |
| `processed/moons_n500_scaled.csv` | 500 | 12.1 KB |
| `processed/moons_n500_scaler_params.json` |  | 0.5 KB |
| `processed/rings_n500_scaled.csv` | 500 | 12.1 KB |
| `processed/rings_n500_scaler_params.json` |  | 0.6 KB |
| `processed/scaler_params.json` |  | 2.0 KB |
| `raw/toy/<scenario>/` (8 thư mục x 3 file) | | 6.3 KB |
| `raw/toy_edge/<scenario>/` (19 thư mục x 3 file) | | 38.1 KB |
| **Toàn bộ `data/`** | | **7.85 MB** |
<!-- FILEMAP:END -->

**Thô và đã xử lý (`raw/` và `processed/`).** `raw/` là dữ liệu sinh ra, không sửa tay; `processed/` là kết quả của tiền xử lý, luôn tái tạo được:
- Bảng thẻ (`card_txn_5d`, `card_txn_5d_hard`): notebook `notebooks/D02_tien_xu_ly_giao_dich_the.ipynb` ghi `*_clean.csv` (làm sạch, giữ mọi dòng trừ bản sao y hệt), `*_scaled.csv` (z-score bằng `ManualScaler`), `scaler_params.json` và `cleaning_scorecard.csv`.
- Các bộ còn lại (moons, rings, blobs, blobs5d, `atm_txn_2d`, `atm_txn_2d_hard`): `python data/make_processed.py` ghi `<tên>_scaled.csv` (id, tọa độ chuẩn hóa hậu tố `_z`, nhãn tham chiếu) và `<tên>_scaler_params.json` (center, scale, cột, file nguồn). Các bảng này tĩnh nên `ManualScaler` fit trên toàn bộ file. Với ATM, z-score bóp méo hình học km (std x khác y) nên Eps 0.5 km chỉ đúng trên tọa độ thô; file chuẩn hóa chỉ để thí nghiệm cần chuẩn hóa.
- `processed/baseline/<stream>/` là kết quả của baseline sklearn (`python -m baseline.sklearn_dbscan`): `labels_long.csv` xác định (kiểm từng byte trong test), `timing.csv` có cột thời gian phụ thuộc máy.
- `processed/baseline_bench/` (do `python -m baseline.sklearn_dbscan --bench` ghi) và `raw/bench/` (do `generate_data.py --n 500 1000 2000 4000` ghi) là file đo thời gian THEO MÁY, sinh lại được, KHÔNG nên commit (xem `GITIGNORE_PROPOSED.txt`); chúng không được tính vào giới hạn 8 MB.

**Tài liệu đi kèm.** `data/HUONG_DAN_BAO_VE.md` (tóm tắt và 12 câu hỏi khó cho người bảo vệ), `docs_extra/CONVENTIONS.md` (quy ước chung: Eps tính cả chính điểm, id, tên phương thức, đo thời gian), và ba file ĐỀ XUẤT cho trưởng nhóm vì file thật đang được Git theo dõi: `data/GITATTRIBUTES_PROPOSED.txt`, `GITIGNORE_PROPOSED.txt`, `REQUIREMENTS_PROPOSED.txt` (thư mục gốc repo).

## 3. Toy scenarios: `data/raw/toy/<scenario>/`

Eps = 1.1, MinPts = 4, khoảng cách Euclid, lân cận Eps tính cả chính điểm đó.

| Thư mục | Nguồn trong doc | Update | UpdSeed |
|---|---|---|---|
| r02_1_noise | R02 3.1 | insert p=(5,5) | {} |
| r02_2_creation | R02 Case 2 | insert p=(1,0.5) | {B} |
| r02_3_absorption | R02 Case 3 | insert p=(2.2,0.5) | {B,D,E,F} |
| r02_4_merge | R02 Case 4 | insert p=(1.9,0.5) | {A2,A3,A5,A6,B1,B2,B4,B5,p} |
| r03_1_removal | R03 Case 1 | delete p | {} |
| r03_2_reduction | R03 Case 2 | delete p | {B,D} |
| r03_3a_split | R03 Case 3A | delete p | {A,E} |
| r03_3b_nonsplit | R03 Case 3B (3A + J,K,L) | delete p | {A,E,I} |

(Doc có 7 loại, riêng Potential Split có 2 ví dụ A và B nên có 8 thư mục.)

Mỗi thư mục gồm `points.csv` (`point_id,x,y`, id đúng tên trong doc), `update.csv` (`op,point_id,x,y`; `op` là `insert` hoặc `delete`) và `expected.json` (`labels_before`, `labels_after` số cụm 1,2,.. với `-1` là nhiễu, so sánh theo phân hoạch; `updseed`, `core_before`, `core_after`).

Định nghĩa UpdSeed (Ester et al. 1998, khớp mọi ví dụ trong doc): với insert, các điểm lõi của D' nằm trong lân cận Eps của một điểm vừa chuyển thành lõi (kể cả p); với delete, các điểm lõi của D' nằm trong lân cận Eps của một điểm vừa mất tính lõi (kể cả p nếu p từng là lõi).

Điểm lệch nhỏ trong doc (không ảnh hưởng kết quả): bảng Absorption ghi nhầm `N_Eps(E)` ở hàng F và p; doc Case 3B chỉ nêu A,E,I,J,K là lõi nhưng C và G cũng là lõi; doc Case 3A ghi UpdSeed = {A,E} dù không điểm nào mất tính lõi ngoài p (khớp định nghĩa vì p được tính).

## 4. Kịch bản biên: `data/raw/toy_edge/<scenario>/` (task C12)

Cùng cấu trúc với toy nhưng `update.csv` có thể gồm NHIỀU dòng theo thứ tự, toạ độ có thể 4 chiều (`x1..x4`), Eps và MinPts ghi trong `expected.json`. Kết quả được tính bằng một bộ tham chiếu Fraction chính xác (không sklearn, không numpy; `gen_edge.py`) và tính lại bằng một cài đặt Fraction độc lập thứ hai trong `check_data.py`. `expected.json` gồm: `labels_before`, `labels_after` (sau toàn bộ update), `core_before`, `core_after`, `updseed` (một update) hoặc `updseed_by_step`, `steps` (với mỗi dòng update: `case`, `labels_after`, `core_after`, `updseed`, `ambiguous_border_ids`, `max_core_hops_between_seeds` khi xóa), `ambiguous_to_ignore` (điểm biên có thể mang nhãn của một trong hai cụm kề: bỏ khi so với re-fit), `ties_at_exactly_eps` (cặp cách nhau đúng bằng Eps, có chủ ý), `tests` và `why_vi`.

Ở các kịch bản có khoảng cách đúng bằng Eps, toạ độ là bội của 1/16 nên chính xác trong số thực nhị phân: cài đặt dùng float vẫn so sánh đúng, và biên Eps là **bao hàm** (`<=`). Ở các kịch bản khác, checker đảm bảo không có cặp nào nằm trong 1e-4 của Eps bình phương.

Bảng ánh xạ kịch bản sang mã của từng thành viên và task (TV2 = chèn, TV3 = xóa, TV5 = tương đương với re-fit sklearn; C12 là task biên của nhóm):

<!-- EDGE:START -->
| Kịch bản | Case (theo bước) | Eps / MinPts | Kiểm tra mã của | Hỗ trợ task | Ghi chú |
|---|---|---|---|---|---|
| `e01_insert_at_eps_inclusive` | Absorption | 1 / 4 | TV2 (chèn), TV5 (tương đương với re-fit) | C12 | Insert at exactly Eps from a core (inclusive boundary) (khoảng cách đúng bằng Eps) |
| `e02_insert_just_outside_eps` | Noise | 1 / 4 | TV2 (chèn), TV5 (tương đương với re-fit) | C12 | Insert just outside Eps (1/16 beyond the boundary) |
| `e03_insert_tie_makes_core` | Creation | 1 / 4 | TV2 (chèn), TV5 (tương đương với re-fit) | C12 | Insert at exactly Eps turns a noise point into a core (khoảng cách đúng bằng Eps) |
| `e04_insert_duplicate_coordinate` | Creation | 1 / 4 | TV2 (chèn), TV5 (tương đương với re-fit) | C12 | Insert a duplicate of an existing coordinate |
| `e05_insert_border_to_core_cascade` | Absorption | 1 / 4 | TV2 (chèn), TV5 (tương đương với re-fit) | C12 | Insert promotes a border point to core and a noise trio to core (cascade) |
| `e06_insert_bridges_three_clusters` | Merge | 1 / 4 | TV2 (chèn), TV5 (tương đương với re-fit) | C12 | One insert merges three clusters |
| `e07_insert_merge_absorbs_noise` | Merge | 1 / 4 | TV2 (chèn), TV5 (tương đương với re-fit) | C12 | Merge that also absorbs a noise point (and leaves another noise) |
| `e08_insert_creation_exactly_minpts` | Noise → Noise → Creation | 1 / 4 | TV2 (chèn), TV5 (tương đương với re-fit) | C12 | Cluster is created exactly when MinPts points have accumulated |
| `e09_insert_noise_near_border` | Noise | 1 / 4 | TV2 (chèn), TV5 (tương đương với re-fit) | C12 | Noise inserted next to a border point but outside every core Eps |
| `e10_insert_border_of_two_clusters` | Absorption | 1 / 4 | TV2 (chèn), TV5 (tương đương với re-fit) | C12 | Inserted point is a border of two clusters (ambiguous) (có điểm biên mơ hồ: p) |
| `e11_delete_border_point` | Reduction | 1 / 4 | TV3 (xóa), TV5 (tương đương với re-fit) | C12 | Delete a border point (no core changes status) |
| `e12_delete_noise_point` | DeleteNoise | 1 / 4 | TV3 (xóa), TV5 (tương đương với re-fit) | C12 | Delete a noise point |
| `e13_delete_splits_into_three` | PotentialSplit_Split | 1 / 4 | TV3 (xóa), TV5 (tương đương với re-fit) | C12 | Delete the hub core: one cluster splits into three |
| `e14_delete_nonsplit_long_detour` | PotentialSplit_NonSplit | 1 / 3 | TV3 (xóa), TV5 (tương đương với re-fit) | C12 | Delete a bridge core; the alternative path is long (BFS depth above 2) (độ sâu BFS giữa các hạt giống = 9) |
| `e15_delete_shared_border_to_noise` | Removal | 1 / 4 | TV3 (xóa), TV5 (tương đương với re-fit) | C12 | Delete a core; a border shared by two cores becomes noise |
| `e16_delete_bridge_leaves_ambiguous_border` | PotentialSplit_Split | 1 / 4 | TV3 (xóa), TV5 (tương đương với re-fit) | C12 | Delete a bridging core; a border point is left shared by two clusters (có điểm biên mơ hồ: b) |
| `e17_delete_then_reinsert_round_trip` | PotentialSplit_Split → Merge | 1 / 4 | TV2 (chèn), TV3 (xóa), TV5 (tương đương với re-fit) | C12; kiểm tra id mới và trạng thái nội bộ sau Split | Delete a bridge then re-insert the same coordinates (new id) |
| `e18_insert_4d_merge` | Merge | 1 / 4 | TV2 (chèn), TV5 (tương đương với re-fit) | C12; chứng minh khoảng cách không viết cứng 2D | Insert merges two clusters in 4-D (distances use all 4 coordinates) (4 chiều) |
| `e19_minpts2_chain` | Merge → Absorption | 1 / 2 | TV2 (chèn), TV5 (tương đương với re-fit) | C12; MinPts nhỏ nhất, không có điểm biên | MinPts = 2: every non-noise point is core, chains link by single steps |
<!-- EDGE:END -->

Hình độ phủ: `data/preview/toy_edge_coverage.png`. Cả 9 case của thuật toán gia tăng (Noise, Creation, Absorption, Merge, Removal, Reduction, PotentialSplit_NonSplit, PotentialSplit_Split, DeleteNoise) đều xuất hiện; checker thất bại nếu thiếu.

## 5. Synthetic 2D và nD

| File | Sinh bằng | Ghi chú |
|---|---|---|
| `synthetic/moons_n500.csv` | make_moons(500, noise=0.08, seed 42) | label 0/1 |
| `synthetic/rings_n500.csv` | make_circles(500, factor=0.4, noise=0.05, seed 42) | label 0 (ngoài), 1 (trong) |
| `synthetic/blobs_noise_n500.csv` | make_blobs(450, tâm (-4,-3),(0,4),(5,-2), std 0.8/0.9/0.7) + 50 điểm đều | label 0,1,2; `-1` là nhiễu |
| `nd/blobs5d_n860.csv` | make_blobs(800, 4 tâm, 5 chiều) + 60 điểm nhiễu | `x1..x5`, label 0..3, `-1` nhiễu; cần chuẩn hóa và chọn Eps |

`sklearn.datasets` chỉ dùng để SINH dữ liệu. Nhãn là nhãn sinh, không phải đầu ra DBSCAN.

## 6. Bản đồ ATM 2D: `bank_geo/atm_txn_2d.csv` (1200 dòng) và bảng liên quan

Thành phố giả 32 x 24 km, 6 hotspot và điểm cô lập. Gợi ý: **Eps = 0.5 km, MinPts = 4** (`bank_geo/params.json`: DBSCAN tìm đúng 6 cụm, ARI 0.99; KMeans k=6 chỉ đạt 0.58 vì cắt vòng tròn và lưỡi liềm).

Cột nền (không đổi): `txn_id`, `x_km`, `y_km`, `hotspot_id` (0 downtown blob, 1 university blob, 2 market ring, 3 market plaza nằm trong vòng, 4 river crescent, 5 highway dải dài, `-1` cô lập). Điểm cô lập cách mọi hotspot hơn 1.3 Eps và cách nhau hơn 2 Eps; không cặp nào cách nhau đúng Eps.

Cột mô tả thêm vào cuối: `timestamp` (đường cong theo giờ riêng từng hotspot: trưa và cuối giờ chiều ở trung tâm, buổi tối ở trường, sáng sớm ở chợ, ngày lễ ở quốc lộ), `atm_id` (khóa ngoại `atms.csv`), `customer_id` (khóa ngoại `customers.csv`, nghề khớp hotspot), `amount_vnd` (bội của 50 nghìn, theo hotspot), `atm_network` (own hoặc partner), `district_id` (suy từ tọa độ), `atm_distance_km`.

Bảng chiều: `districts.csv` (8 quận, loại khu vực CBD, university, market, riverside, residential, suburb, highway; tâm và diện tích), `atms.csv` (40 máy: 35 đặt tại tâm k-means của hotspot, 5 máy đứng riêng cho điểm cô lập), `hotspots.csv` (tên nghiệp vụ, ý nghĩa, hành động đề xuất).

Hồ sơ hotspot (tính từ `atm_txn_2d.csv`):

| hotspot_id | tên | n | x_km (mean / median) | y_km (mean / median) |
|---|---|---|---|---|
| 0 | downtown_office_cash | 204 | 7.97 / 8.00 | 15.99 / 15.98 |
| 1 | student_campus | 150 | 26.00 / 26.02 | 18.00 / 17.99 |
| 2 | market_ring | 222 | 22.06 / 22.07 | 7.04 / 6.90 |
| 3 | market_plaza | 90 | 22.07 / 22.13 | 7.00 / 6.99 |
| 4 | riverside_cash | 228 | 11.02 / 11.03 | 6.74 / 7.02 |
| 5 | highway_travellers | 168 | 21.95 / 21.96 | 16.53 / 16.55 |
| -1 | isolated | 138 | 15.73 / 15.28 | 12.16 / 12.57 |

## 7. Giao dịch thẻ 5D: `bank_card/card_txn_5d.csv` (1500 dòng, cố tình bẩn)

Cột nền (không đổi): `txn_id` (dòng trùng lặp là bản sao y hệt, cùng id), `amount_vnd` (làm tròn nghìn, từ ~1.000 đến ~500 triệu), `hour_of_day` (0 đến 23.99, không tuần hoàn), `distance_from_home_km`, `txns_last_7d`, `merchant_risk_score` (0 đến 1), `segment_id` (0 daily_retail, 1 online_shopping, 2 travel_big_ticket, 3 night_cash_out, `-1` bất thường cài sẵn, 15 điểm).

Cột mô tả thêm vào cuối (KHÔNG dùng để gom cụm): `timestamp` (giờ trong ngày khớp `hour_of_day` trong 0.01 giờ; nếu `hour_of_day` thiếu thì `timestamp` vẫn mang giờ thật, có thể dùng để điền), `customer_id` (khóa ngoại `customers.csv`), `channel` (POS, ONLINE, ATM, CONTACTLESS), `merchant_category` (khóa ngoại `merchant_categories.csv`), `txn_district_id` (khóa ngoại `districts.csv`), `is_foreign` (1 thì `txn_district_id` để trống: máy ngoài thành phố).

**Lưu ý trung thực về số tiền ATM trong bảng thẻ.** Ở kênh `ATM` của bảng này, `amount_vnd` có trung vị 42.000 VND, 73% (204 trong 281 dòng ATM của bảng sạch) dưới 50.000 VND và chỉ 1,8% là bội của 50.000, trong khi `amount_vnd` của `atm_txn_2d.csv` luôn là bội của 50.000 (100%). Hai bảng được sinh độc lập nên không khớp nhau về mệnh giá rút. Cột nền bị đóng băng (base_hashes.json) nên giá trị này KHÔNG được sửa. Báo cáo không nên dùng số tiền ATM của bảng thẻ làm lập luận về độ giống thực tế; tính chân thực của mệnh giá rút chỉ nói được cho bảng `atm_txn_2d.csv`.

Hành vi nhất quán: rút tiền đêm hầu hết qua ATM và nhóm `ATM`; mua online qua kênh ONLINE và nhóm `ECO`; chi tiêu lớn đi xa thuộc nhóm du lịch `TRV`; bất thường lẫn lộn và hay ở nước ngoài; phân khúc tại chỗ (0, 1) xảy ra ở quận nhà, đi xa (2) và bất thường ở quận khác; khách không dùng ngân hàng số không mua online.

**Tiền xử lý tham chiếu** (`bank_card/params.json`, tính trong generator, KHÔNG phải ManualScaler): bỏ dòng thiếu và dòng trùng, `log10(amount_vnd)`, z-score mọi cột (std tổng thể). Khi đó **Eps = 0.65 với MinPts = 10** tách đúng 4 cụm (ARI 1.0, 15 bất thường đều là nhiễu); Eps có tác dụng trong khoảng 0.26 đến 1.04 với MinPts 6 đến 14. Số tiền lệch phải mạnh (từ 1.000 đến 500 triệu), nên log10 là lựa chọn thiết kế hợp lý, nhưng **đo thực tế cho thấy nó không bắt buộc** (chọn Eps, MinPts tốt nhất trên lưới với nhãn thật; ARI, nhiễu tính là một lớp): file thẻ dễ đạt ARI 1.0 cả khi KHÔNG log10 (z-score thẳng trên số tiền, Eps 0.65 và MinPts 10 vẫn cho 1.0), chỉ file thẻ khó được lợi (ARI 0.646 với log10 so với 0.552 không log10). Bước quyết định là **z-score**: log10 mà không z-score thì Eps 0.65 chỉ cho ARI 0.08 ở file dễ, vì các cột khác đơn vị nên một Eps duy nhất vô nghĩa (phải chỉnh lại Eps thành 2.8 mới đạt 0.997). Bảng đo đầy đủ (ba cách chuẩn hóa, hai file) nằm trong notebook D02, mục 3b.

Hồ sơ phân khúc (sau khi bỏ dòng thiếu và trùng):

| segment_id | tên | n | amount_vnd (mean / median) | hour_of_day (mean / median) | distance_from_home_km (mean / median) | txns_last_7d (mean / median) | merchant_risk_score (mean / median) |
|---|---|---|---|---|---|---|---|
| 0 | daily_retail | 523 | 205,261.95 / 201,000.00 | 12.45 / 12.39 | 4.01 / 4.03 | 9.02 / 9.00 | 0.12 / 0.12 |
| 1 | online_shopping | 377 | 818,779.84 / 787,000.00 | 21.07 / 21.10 | 0.98 / 0.93 | 3.99 / 4.00 | 0.35 / 0.35 |
| 2 | travel_big_ticket | 218 | 10,364,623.85 / 10,071,500.00 | 14.95 / 14.98 | 59.50 / 59.48 | 1.92 / 2.00 | 0.45 / 0.45 |
| 3 | night_cash_out | 250 | 40,872.00 / 40,000.00 | 1.55 / 1.52 | 7.71 / 7.84 | 13.77 / 14.00 | 0.65 / 0.65 |
| -1 | suspicious_far_high_value | 15 | 132,982,133.33 / 123,826,000.00 | 13.05 / 14.71 | 471.57 / 441.20 | 30.13 / 30.00 | 0.87 / 0.86 |

Bảng chiều: `customers.csv` (600 khách, 11 thuộc tính), `merchant_categories.csv` (14 nhóm, mức rủi ro cơ sở), `segments.csv` (tên nghiệp vụ và hành động đề xuất).

### 7a. Danh mục lỗi dữ liệu (defect catalogue)

Lỗi ở cột nền (thiếu ô, dòng trùng) giữ nguyên từ bản gốc để D02 xử lý; lỗi ở cột mô tả là loại mới, chỉ nằm ở cột mô tả. Bản sạch của mọi cột bị làm bẩn nằm trong `data/truth/card_txn_5d_clean_descriptive.csv` (và `card_txn_5d_clean_base.csv` cho giá trị nền trước khi đục ô thiếu); danh mục số liệu đầy đủ ở `data/truth/defect_catalogue_card.csv`. Số dòng tính theo file (dòng trùng tính hai lần).

<!-- DEFECTS:START -->
| Phạm vi | Bảng | Loại lỗi | Cột | Số dòng bị ảnh hưởng |
|---|---|---|---|---|
| base | card_txn_5d | missing_value | amount_vnd | 23 |
| base | card_txn_5d | missing_value | hour_of_day | 22 |
| base | card_txn_5d | missing_value | distance_from_home_km | 18 |
| base | card_txn_5d | missing_value | txns_last_7d | 31 |
| base | card_txn_5d | missing_value | merchant_risk_score | 18 |
| base | card_txn_5d | duplicate_row_exact_copy | (whole row) | 8 |
| descriptive | card_txn_5d | inconsistent_spelling_or_case | channel | 104 |
| descriptive | card_txn_5d | unknown_customer_id | customer_id | 18 |
| descriptive | card_txn_5d | missing_value | merchant_category | 18 |
| descriptive | card_txn_5d | mixed_timestamp_format_dd_mm_yyyy | timestamp | 30 |
| descriptive | card_txn_5d | impossible_timestamp | timestamp | 6 |
| descriptive | card_txn_5d | invalid_district_id | txn_district_id | 5 |
| base | card_txn_5d_hard | missing_value | amount_vnd | 26 |
| base | card_txn_5d_hard | missing_value | hour_of_day | 28 |
| base | card_txn_5d_hard | missing_value | distance_from_home_km | 16 |
| base | card_txn_5d_hard | missing_value | txns_last_7d | 27 |
| base | card_txn_5d_hard | missing_value | merchant_risk_score | 22 |
| base | card_txn_5d_hard | duplicate_row_exact_copy | (whole row) | 8 |
| descriptive | card_txn_5d_hard | inconsistent_spelling_or_case | channel | 113 |
| descriptive | card_txn_5d_hard | unknown_customer_id | customer_id | 19 |
| descriptive | card_txn_5d_hard | missing_value | merchant_category | 19 |
| descriptive | card_txn_5d_hard | mixed_timestamp_format_dd_mm_yyyy | timestamp | 32 |
| descriptive | card_txn_5d_hard | impossible_timestamp | timestamp | 6 |
| descriptive | card_txn_5d_hard | invalid_district_id | txn_district_id | 5 |
<!-- DEFECTS:END -->

Bảng trên gồm cả bảng khó (`card_txn_5d_hard`). Mọi lỗi mô tả đều khôi phục được từ `data/truth/` (kênh bằng cách `strip().upper()`, khách lạ không có bản sạch trong bảng khách, `timestamp` dd/mm/yyyy bằng `dayfirst=True`).

## 8. Stream cập nhật (D03, D05, D06, D09, D10)

Mỗi stream gồm `d0.csv` (dữ liệu ban đầu, `ref_label` chỉ để tham khảo, so sánh theo phân hoạch), `updates.csv` và `params.json` (Eps, MinPts, định nghĩa case, số lượng case, ghi chú).

`updates.csv`: `step,timestamp,op,point_id,<đặc trưng>,planted_case,verified_case,ambiguous_border_ids` rồi các cột mô tả thêm vào cuối. `op` là `insert` hoặc `delete`; `point_id` không bao giờ dùng lại (điểm chèn lại nhận id mới); dòng `delete` vẫn ghi toạ độ điểm bị xóa. `planted_case` là case kịch bản cố ý tạo (trống nếu tự nhiên). `verified_case` là case suy ra bằng re-fit toàn bộ trước và sau update, so sánh phân hoạch (không so id cụm); checker thất bại nếu hai cột khác nhau.

Case: insert gồm Noise, Creation, Absorption, Merge; delete gồm Removal, Reduction, PotentialSplit_NonSplit, PotentialSplit_Split, và `DeleteNoise` (xóa điểm nhiễu, chỉ ở stream tự nhiên). Reduction nghĩa là các điểm UpdSeed đã nối trực tiếp với nhau; PotentialSplit nghĩa là UpdSeed chưa nối trực tiếp nên phải kiểm tra liên thông, kết quả là Split hoặc NonSplit.

**Cách xác định `verified_case` (sửa ngày 2025-10: dựa trên cấu trúc điểm lõi).** Case được suy ra từ phân hoạch của các điểm LÕI trước và sau update (chạy lại DBSCAN toàn bộ), KHÔNG từ nhãn của điểm biên, vì một điểm biên nằm giữa hai cụm có thể đổi cụm khi re-fit mà cấu trúc không đổi. Cụ thể: Split khi các lõi còn lại của một cụm cũ rơi vào từ 2 cụm trở lên; Removal khi một cụm cũ không còn lõi nào (các điểm biên cũ của nó có thể được cụm khác nhận, vẫn là Removal); Merge khi một cụm mới chứa lõi cũ của từ 2 cụm cũ trở lên; Creation khi một cụm mới không có lõi cũ nào. Bất biến được `check_data.py` kiểm cho cả 3 stream, với C là số cụm: Merge cho C giảm, Creation cho C tăng, Removal cho C giảm, Split cho C tăng, Noise, Absorption, Reduction, NonSplit và DeleteNoise giữ C nguyên. Ngoại lệ có tài liệu: một bước Split mà đồng thời có cụm khác biến mất (chỉ xảy ra khi xóa điểm biên mơ hồ) được phép có C không tăng; hiện có 0 ngoại lệ. Bản trước sửa gán sai 10 bước của `card_stream_hard` (7 Split, 1 Merge, 2 Reduction); hai stream còn lại không đổi.

**Điểm biên mơ hồ.** Điểm biên nằm trong Eps của core thuộc từ 2 cụm khác nhau có thể bị hai cài đặt đúng gán nhãn khác nhau. `ambiguous_border_ids` liệt kê các id đó SAU mỗi step (`d0_ambiguous_border_ids` trong params cho D0). Hãy bỏ chúng khi so ARI. Sự kiện planted không bao giờ rơi vào điểm như vậy.

### 8a. `bank_geo/atm_stream/` (kịch bản sẵn, 119 step, 1 step = 1 ngày từ 2025-03-01)
Đặc trưng `x_km,y_km`, Eps = 0.5 km, MinPts = 4, d0 có 529 điểm: 2 hotspot nền và 43 "mảnh ghép" lấy từ 8 toy scenario (nhân tỉ lệ 5/11, xoay 90 độ) đặt ở các ô cách nhau 4 km. Mỗi case được tái hiện nhiều lần, có cả chuỗi hoàn tác. Cột mô tả thêm: `district_id`.

### 8b. `bank_card/card_stream/` (cửa sổ trượt tự nhiên, 300 step, 1 step = 1 giờ)
Đặc trưng là 5 cột thô như bảng thẻ. d0 = 500 giao dịch đầu; step lẻ chèn giao dịch mới, step chẵn xóa giao dịch cũ nhất. Cơ cấu trôi dần về night_cash_out. **Scaling cố định** (`params.json` > `frozen_scaling`): `log10(amount)` rồi `(x - mean) / std` bằng hằng số đã ghi. **Đây là cửa sổ tham chiếu cố định, KHÔNG phải fit trên D0:** mean và std do generator tính trên toàn bộ bảng sạch (1492 dòng, gồm cả dòng chỉ xuất hiện về sau và 15 giao dịch bất thường), nên chứa thông tin từ tương lai so với D0. Giữ có chủ ý để mọi thuật toán thấy cùng tọa độ; hằng số đã đóng băng nên không sinh lại. Đo trên D0 (cùng DBSCAN tham chiếu): fit trên D0 so với hằng số cố định cho ARI 0.9959 ở stream thẻ (474 so với 471 điểm lõi) và 0.887 ở stream thẻ khó (260 so với 240 điểm lõi). Khuyến nghị cho báo cáo: phát lại stream bằng hằng số cố định (mã dùng chung `src/preprocessing/stream_scaling.py`), chỉ dùng fit trên D0 trong notebook D02 để minh họa đánh đổi rò rỉ. Eps = 0.4, MinPts = 6. Chủ yếu Absorption và Reduction, không có Merge hay Split.

**Hai đồng hồ trong stream thẻ.** Cột có sẵn `timestamp` là giờ ĐẾN: thời điểm bản ghi vào cửa sổ, tính theo bước (bước 1 = 2025-03-01 00:00, mỗi bước 1 giờ). Đặc trưng `hour_of_day` là giờ SỰ KIỆN: giờ giao dịch thực sự xảy ra. Hai giá trị độc lập nên một giao dịch đến lúc 00:00 có thể có `hour_of_day = 14.03`. Để không đổi tên gì, các cột mới được thêm vào cuối: `point_arrival_timestamp` (lúc điểm vào cửa sổ; với D0 là giả lập, cách nhau 1 giờ trước 2025-03-01; với dòng delete là lúc điểm bị xóa vào cửa sổ chứ không phải thời điểm xóa) và `event_timestamp` (lúc giao dịch xảy ra: giờ trong ngày bằng `hour_of_day` trong 0.01 giờ và sớm hơn lúc đến dưới 24 giờ). Thêm `segment_id` (nhãn thật), `customer_id`, `channel`, `merchant_category`, `txn_district_id`, `is_foreign`; mọi dòng của cùng một point_id mang cùng các giá trị mô tả (checker kiểm tra).

## 9. Biến thể khó

`--difficulty easy|medium|hard` (easy và hard luôn được ghi; thêm medium để sinh bản vừa). **easy = đúng các file gốc** (byte-identical ở mọi cột nền, kiểm bằng `base_hashes.json`). **hard** được commit; **medium** chỉ tái tạo (`python data/generate_data.py --difficulty medium` ghi `card_txn_5d_medium.csv`, `atm_txn_2d_medium.csv` và bản sạch tương ứng; báo cáo độ khó vẫn tính medium trong bộ nhớ). Medium là phép nội suy 65 % từ easy đến hard theo mọi tham số sinh.

**`bank_card/card_txn_5d_hard.csv`** (1600 dòng, cùng cột với bảng thẻ chính, id `TXH`): bán lẻ (rất loãng) chồng lấn một phần với mua online buổi tối cả về giờ lẫn số tiền; 12% mua online có giỏ hàng lớn nằm giữa online và du lịch; một dải cầu nối mỏng 12 điểm "khách lai" nối bán lẻ và online (rủi ro chaining khi Eps đơn); mật độ không đều (rút tiền đêm rất chặt, bán lẻ và du lịch rất loãng, tỉ số khoảng cách láng giềng thứ 5 trên 2); 16 bất thường đặt sát nhóm đặc cùng 16 bất thường xa; vẫn có ô thiếu, dòng trùng và lỗi cột mô tả như bản chính. Cờ sinh dữ liệu (`is_bridge`, `is_near_anomaly`, `is_large_online`) nằm trong `data/truth/card_txn_5d_hard_generation_flags.csv`; điểm cầu nối mang nhãn của đầu mút gần hơn.

**`bank_geo/atm_txn_2d_hard.csv`** (2000 điểm, chỉ 4 cột nền, id `ATH`): trung tâm kề sát quảng trường chợ với dải mật độ nối giữa chúng (40 điểm), vòng quanh chợ tiếp tuyến với quốc lộ, hotspot khu đại học rất loãng (độ lệch chuẩn gấp 2 lần bản dễ), nhiễu 20 %, mật độ thay đổi mạnh giữa các hotspot.

**`bank_card/card_stream_hard/`**: cửa sổ trượt tự nhiên 300 step trên phân khúc khó. Ba dải cầu nối mỏng nối cặp phân khúc (0,1), (1,2), (0,3): mỗi dải đủ trong d0, mất dần điểm khi cửa sổ trượt (điểm cũ ra trước, đầu tiên vài điểm giữa dải, rồi 4 điểm liền nhau làm đứt dải), rồi các điểm bị mất được chèn lại về sau. Merge, Split và NonSplit vì vậy xảy ra thật; thêm vào đó phần tự nhiên từ các phân khúc loãng cũng tạo ra nhiều sự kiện. `planted_case` chỉ điền ở sự kiện đầu tiên của mỗi nhóm cài sẵn đã thật sự tạo ra case đó (3 Split và 3 Merge); mọi `verified_case` đến từ re-fit tham chiếu. Eps = 0.30, MinPts = 6, scaling cố định tính từ toàn bộ bảng sạch `card_txn_5d_hard` (1592 dòng, cửa sổ tham chiếu cố định có thông tin từ tương lai, xem stream thẻ ở trên; trong `params.json`). Cửa sổ có nhiều điểm nhiễu (khoảng 29 %) vì dữ liệu khó và Eps nhỏ; số cụm của cửa sổ dao động từ 3 (d0) đến 9 (đỉnh quanh step 88, có 13 step với 9 cụm; step 150 có 7) rồi 4 (step 300). Số case:

<!-- STREAMHARD:START -->
| verified_case | Số lần | Trong đó cài sẵn (planted_case) |
|---|---|---|
| Absorption | 108 | 0 |
| Creation | 2 | 0 |
| DeleteNoise | 43 | 0 |
| Merge | 5 | 3 |
| Noise | 35 | 0 |
| PotentialSplit_NonSplit | 12 | 0 |
| PotentialSplit_Split | 7 | 3 |
| Reduction | 85 | 0 |
| Removal | 3 | 0 |
| **Tổng** | **300** | **6** |
<!-- STREAMHARD:END -->

### 9a. Báo cáo độ khó (`data/difficulty_report.json`, `data/preview/difficulty_*.png`)

Với mỗi họ dữ liệu và mỗi mức, tìm lưới Eps và MinPts của DBSCAN trên đặc trưng đã chuẩn hóa, cộng KMeans và Ward với k thật, OPTICS và HDBSCAN, rồi lấy ARI tốt nhất so với nhãn thật. Chọn tham số bằng nhãn thật nên đây là cận trên (oracle). ATM tính theo mét (lưới Eps 200 đến 1300 m, MinPts 3 đến 12); thẻ sau bỏ dòng thiếu và trùng, log10 amount và z-score (lưới Eps 0.2 đến 1.5, MinPts 4 đến 15). Nhiễu (-1) là một lớp riêng khi tính ARI.

<!-- DIFFICULTY:START -->
**ATM 2D (gom theo mét, nhãn thật = hotspot_id)**

| Thuật toán | Dễ | Vừa | Khó | Tham số tốt nhất (mức khó) |
|---|---|---|---|---|
| DBSCAN | 0.997 | 0.737 | 0.509 | easy: eps=500.0, MinPts=3; medium: eps=600.0, MinPts=10; hard: eps=400.0, MinPts=4 |
| HDBSCAN | 0.962 | 0.763 | 0.689 | easy: min_cluster_size=15, min_samples=12, selection=eom; medium: min_cluster_size=30, min_samples=12, selection=eom; hard: min_cluster_size=30, min_samples=6, selection=eom |
| OPTICS | 0.589 | 0.254 | 0.137 | easy: min_samples=10, xi=0.08; medium: min_samples=5, xi=0.08; hard: min_samples=5, xi=0.08 |
| KMeans (true k) | 0.583 | 0.419 | 0.404 | easy: k=6; medium: k=6; hard: k=6 |
| Ward HAC (true k) | 0.544 | 0.436 | 0.412 | easy: k=6; medium: k=6; hard: k=6 |
| (số điểm / nhiễu thật) | 1200 / 12 % | 2000 / 17 % | 2000 / 20 % | |

- Mức easy: DBSCAN tốt nhất hoặc hòa nhất.
- Mức medium: thuật toán đánh bại DBSCAN: HDBSCAN (0.76 so với DBSCAN 0.74)
- Mức hard: thuật toán đánh bại DBSCAN: HDBSCAN (0.69 so với DBSCAN 0.51)

**Thẻ 5D (log10 amount + z-score, nhãn thật = segment_id)**

| Thuật toán | Dễ | Vừa | Khó | Tham số tốt nhất (mức khó) |
|---|---|---|---|---|
| DBSCAN | 1.000 | 0.894 | 0.646 | easy: eps=0.45, MinPts=4; medium: eps=0.35, MinPts=12; hard: eps=0.3, MinPts=15 |
| HDBSCAN | 1.000 | 0.774 | 0.528 | easy: min_cluster_size=15, min_samples=3, selection=eom; medium: min_cluster_size=60, min_samples=12, selection=leaf; hard: min_cluster_size=60, min_samples=12, selection=leaf |
| OPTICS | 1.000 | 0.294 | 0.250 | easy: min_samples=10, xi=0.08; medium: min_samples=10, xi=0.08; hard: min_samples=10, xi=0.08 |
| KMeans (true k) | 0.800 | 0.785 | 0.546 | easy: k=4; medium: k=4; hard: k=4 |
| Ward HAC (true k) | 0.801 | 0.789 | 0.548 | easy: k=4; medium: k=4; hard: k=4 |
| (số điểm / nhiễu thật) | 1383 / 1 % | 1476 / 2 % | 1475 / 2 % | |

- Mức easy: DBSCAN tốt nhất hoặc hòa nhất.
- Mức medium: DBSCAN tốt nhất hoặc hòa nhất.
- Mức hard: DBSCAN tốt nhất hoặc hòa nhất.
<!-- DIFFICULTY:END -->

Kiểm tra trong `check_data.py`: tính lại ARI DBSCAN tốt nhất của cả 3 mức (medium sinh lại trong bộ nhớ) và bắt buộc dễ > vừa > khó với **biên tối thiểu 0.08** ở mỗi bước, cho cả hai họ. Trao đổi cần bàn trong nhóm (đọc từ bảng trên, không phải kết luận chung): (1) KMeans và Ward với k thật thua DBSCAN ở mọi mức của cả hai họ; ở ATM vì cụm là vòng, lưỡi liềm và dải, ở thẻ vì bất thường kéo tâm cụm và các phân khúc chồng lấn; (2) khoảng cách thu hẹp ở mức khó của ATM (KMeans 0.40 so với DBSCAN 0.51) vì cả hai đều khó; (3) HDBSCAN thắng DBSCAN ở ATM mức vừa (nhẹ) và mức khó (rõ), vì mật độ giữa các hotspot khác nhau nên một Eps không đủ; trên dữ liệu thẻ DBSCAN thắng ở mọi mức (hòa ở mức dễ); (4) OPTICS trong lưới nhỏ này (4 tổ hợp, trích xuất xi) luôn thua, nên con số chỉ cho thấy nó cần chỉnh nhiều hơn, không nói OPTICS kém.

## 10. Từ điển dữ liệu

`data/raw/DATA_DICTIONARY.csv`: mỗi dòng là một cột của một bảng với `table, column, role` (key, feature, descriptive, label, truth), `type, unit, allowed_range, missing_allowed, description_en, description_vi`. Checker bắt buộc từ điển phủ mọi cột của mọi bảng (hai chiều); các bảng dưới `toy/` và `toy_edge/` dùng tên chung `<scenario>`.

## 11. Hình xem trước (`python data/generate_data.py --preview`)

Tiêu đề, nhãn trục và chú thích đều bằng tiếng Việt (đã kiểm tra dấu hiển thị đúng).

| Hình | Nội dung |
|---|---|
| `eda_hour_dow_channel.png` | hoạt động theo giờ và thứ, từng kênh thanh toán |
| `eda_payday_weekend.png` | đỉnh ngày 5 và cuối tháng, tỉ lệ cuối tuần theo phân khúc |
| `eda_category_mix_by_segment.png` | cơ cấu nhóm merchant theo phân khúc |
| `eda_correlation.png` | tương quan Pearson và Spearman của 5 đặc trưng |
| `eda_missing_duplicates.png` | ô thiếu, dòng trùng, danh mục lỗi cột mô tả |
| `eda_customers.png` | nhân khẩu học khách hàng |
| `eda_atm_map_districts.png` | bản đồ ATM với ranh giới quận và máy ATM |
| `eda_scaling_before_after.png` | phân phối trước và sau log10 + z-score |
| `difficulty_atm.png`, `difficulty_card.png` | dễ, vừa, khó cạnh nhau, nhãn thật và DBSCAN tốt nhất |
| `difficulty_ari_bars.png` | ARI tốt nhất của từng thuật toán theo mức khó |
| `toy_edge_coverage.png` | kịch bản biên phủ những case nào và kiểm tra mã của ai |
| `stream_hard_timeline.png` | case theo bước của stream thẻ khó |
| `atm_map.png`, `card_scatter_before.png`, `card_scatter_after.png`, `stream_timeline.png` | hình của bản đầu tiên (nhãn thật, KMeans, DBSCAN; ma trận scatter; case của hai stream gốc) |

## 12. Benchmark `--n`

`python data/generate_data.py --n 500 1000 2000 4000` ghi `data/raw/bench/atm_txn_2d_n<N>.csv`, `card_txn_5d_n<N>.csv` và `params_n<N>.json` (Eps gợi ý cho từng cỡ). Không được checker kiểm tra và không nên commit.

## 13. Tên nghiệp vụ và hồ sơ cụm

Tên, ý nghĩa và hành động đề xuất nằm trong `bank_geo/hotspots.csv` và `bank_card/segments.csv`. Nhãn `-1` luôn mô tả là giao dịch cô lập hoặc bất thường, kèm hành động đề xuất. Các tên này mô tả dữ liệu giả lập, dùng để diễn giải cụm bằng ngôn ngữ nghiệp vụ trong báo cáo.

<!-- WEAK:START -->
## 14. Điểm yếu đã biết (tự đánh giá, chưa sửa được)

1. **ARI độ khó là cận trên.** Tham số mọi thuật toán chọn bằng nhãn thật; trong thực tế không biết nhãn nên con số thấp hơn. HDBSCAN và OPTICS chỉ thử lưới nhỏ (HDBSCAN 18 tổ hợp, OPTICS 4), nên thứ hạng giữa chúng và DBSCAN là gợi ý, không phải kết luận.
2. **Mức khó của thẻ vẫn do các phân khúc Gauss chồng lấn tạo ra**, không phải cấu trúc thật như dữ liệu giao dịch ngoài đời; ARI DBSCAN tốt nhất 0.65 nhưng KMeans (0.55) cũng không quá xa vì bất thường kéo tâm cụm.
3. **Stream thẻ khó có khoảng 29 % nhiễu và d0 gần như một cụm lớn** (ba dải cầu nối đều đủ lúc đầu). Merge và Split xảy ra thật nhưng một phần do cài sẵn (3 Merge, 3 Split planted), một phần tự nhiên; tỉ lệ tự nhiên so với cài sẵn phụ thuộc Eps = 0.30. Cửa sổ 500 điểm trên dữ liệu loãng nên kết quả nhạy với Eps.
4. **Cột mô tả không ràng buộc ngược với cột nền.** Số tiền, `txns_last_7d`, khoảng cách và điểm rủi ro không được tính từ thu nhập khách hay từ timestamp; chỉ có việc chọn khách, kênh, nhóm hàng, quận, ngày giờ là nhất quán với phân khúc, nghề và hạng.
5. **`hour_of_day` không tuần hoàn** nên 23.9 giờ và 0.1 giờ bị coi là xa nhau (có chủ ý, nhưng báo cáo nên nêu); `timestamp` cho phép điền giá trị thiếu của cột này, điều có thể làm bài tiền xử lý dễ hơn dự kiến.
6. **Quận và hotspot không trùng khít:** quốc lộ cắt qua nhiều quận; file ATM khó có bố cục khác nên không có quận hay máy ATM.
7. **Loại trừ cặp cách nhau đúng Eps chỉ áp dụng cho Eps = 0.5 km** của các file ATM. Với Eps khác một cặp có thể rơi đúng biên (xác suất rất nhỏ nhưng khác 0); dữ liệu thẻ liên tục nên không có vấn đề này.
8. **Kịch bản biên thiết kế tay**, không sinh ngẫu nhiên; chưa có kiểm thử so sánh ngẫu nhiên (fuzzing) giữa bản tăng dần và re-fit, việc đó thuộc TV5.
9. **Bản vừa không được commit** nên checker phải sinh lại trong bộ nhớ (thêm khoảng 10 giây). Bản benchmark `--n` chưa được làm giàu cột mô tả.
10. **Task ngoài C12 chưa được ánh xạ chi tiết:** repo không chứa bảng task của nhóm nên bảng kịch bản biên chỉ ghi C12 và mã của TV2, TV3, TV5; nhóm cần tự gắn thêm mã task nếu muốn.
<!-- WEAK:END -->
