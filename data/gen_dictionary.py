"""Part 4: data/raw/DATA_DICTIONARY.csv, one row per column of every table (bilingual)."""
import csv

COLS = ["table", "column", "role", "type", "unit", "allowed_range", "missing_allowed", "description_en", "description_vi"]
R = []


def d(table, col, role, typ, unit, rng, miss, en, vi):
    R.append([table, col, role, typ, unit, rng, miss, en, vi])


def block(table, rows):
    for r in rows:
        d(table, *r)


FEAT_CARD = [
    ("amount_vnd", "feature", "int", "VND", "1000 to about 5e8", "yes (about 1.5 %)", "Transaction amount, rounded to 1000 VND; spans orders of magnitude, use log10 before scaling.", "Số tiền giao dịch, làm tròn nghìn đồng; chênh nhiều bậc độ lớn nên cần log10 trước khi chuẩn hóa."),
    ("hour_of_day", "feature", "float", "hour", "0 to 23.99", "yes (about 1.5 %)", "Local hour of the transaction (not cyclic). In the table files it matches the time of day of timestamp within 0.01 h.", "Giờ xảy ra giao dịch (không tuần hoàn). Trong file bảng, khớp giờ trong timestamp với sai số 0.01 giờ."),
    ("distance_from_home_km", "feature", "float", "km", "0.05 to about 900", "yes (about 1.5 %)", "Distance between the transaction location and the customer's home.", "Khoảng cách từ nơi giao dịch tới nơi ở của khách."),
    ("txns_last_7d", "feature", "int", "count", "0 to about 60", "yes (about 1.5 %)", "Number of transactions of the customer in the previous 7 days (generated per row, not recomputed from timestamp).", "Số giao dịch của khách trong 7 ngày trước (sinh theo từng dòng, không tính lại từ timestamp)."),
    ("merchant_risk_score", "feature", "float", "score", "0.001 to 0.999", "yes (about 1.5 %)", "Risk score of the merchant, higher is riskier.", "Điểm rủi ro của merchant, càng cao càng rủi ro."),
]
CARD_DESC = [
    ("timestamp", "descriptive", "string", "datetime", "2025-03-01 to 2025-04-25", "no", "Transaction time YYYY-MM-DD HH:MM:SS. Defects: some rows use dd/mm/yyyy, 6 rows are impossible (hour 24, 25, minute 75, invalid dates).", "Thời điểm giao dịch dạng YYYY-MM-DD HH:MM:SS. Lỗi: một số dòng dùng dd/mm/yyyy, 6 dòng không hợp lệ (giờ 24, 25, phút 75, ngày không tồn tại)."),
    ("customer_id", "key", "string", "id", "C0001 to C0600", "no", "Foreign key to customers.csv. Defect: UNKNOWN or ids not in the table (orphans).", "Khóa ngoại tới customers.csv. Lỗi: UNKNOWN hoặc mã không có trong bảng (mồ côi)."),
    ("channel", "descriptive", "string", "category", "POS, ONLINE, ATM, CONTACTLESS", "no", "Payment channel. Defect: inconsistent case and spaces (pos, Pos, ' POS', contact-less).", "Kênh thanh toán. Lỗi: sai chữ hoa thường và khoảng trắng (pos, Pos, ' POS', contact-less)."),
    ("merchant_category", "key", "string", "code", "3-letter code of merchant_categories.csv", "yes (about 1.2 % defect)", "Foreign key to merchant_categories.csv. Defect: blank in about 1.2 % of rows.", "Khóa ngoại tới merchant_categories.csv. Lỗi: để trống ở khoảng 1.2 % dòng."),
    ("txn_district_id", "key", "string", "id", "D01 to D08", "yes (blank iff is_foreign = 1)", "District where the transaction took place (FK to districts.csv); blank for foreign terminals. Defect: 5 rows hold the invalid code D99.", "Quận nơi diễn ra giao dịch (khóa ngoại tới districts.csv); để trống nếu giao dịch ở nước ngoài. Lỗi: 5 dòng mang mã không tồn tại D99."),
    ("is_foreign", "descriptive", "int", "flag", "0 or 1", "no", "1 if the terminal is outside the fictional city.", "1 nếu máy giao dịch nằm ngoài thành phố giả lập."),
]
SEG_NOTE = ("0 daily_retail, 1 online_shopping, 2 travel_big_ticket, 3 night_cash_out, -1 planted anomaly",
            "0 bán lẻ hằng ngày, 1 mua sắm online, 2 chi tiêu lớn khi đi xa, 3 rút tiền đêm, -1 bất thường cài sẵn")


def card_table(table, label_extra=""):
    block(table, [("txn_id", "key", "string", "id", "TXN000001.. (hard: TXH, medium: TXI)", "no", "Transaction id; exact duplicate rows repeat the same id.", "Mã giao dịch; dòng trùng lặp lặp lại cùng mã.")] + FEAT_CARD +
          [("segment_id", "label", "int", "class", SEG_NOTE[0], "no", "Ground-truth segment used by the generator (not a DBSCAN output)." + label_extra, "Nhãn phân khúc thật của bộ sinh (không phải đầu ra DBSCAN). " + SEG_NOTE[1])] + CARD_DESC)


def stream_card(table, upd):
    head = [("step", "key", "int", "count", "1..300", "no", "Step number, 1 step = 1 hour.", "Số thứ tự bước, 1 bước = 1 giờ."),
            ("timestamp", "descriptive", "string", "datetime", "2025-03-01 00:00 onward", "no", "ARRIVAL clock: when the update reaches the window (step based). Not the time of the transaction itself.", "Đồng hồ ĐẾN: lúc bản cập nhật vào cửa sổ (theo bước). Không phải giờ xảy ra giao dịch."),
            ("op", "descriptive", "string", "category", "insert, delete", "no", "Operation.", "Thao tác.")] if upd else [("point_id", "key", "int", "id", "0..499", "no", "Point id of the initial data D0.", "Mã điểm của dữ liệu ban đầu D0.")]
    if upd:
        head.append(("point_id", "key", "int", "id", "never reused", "no", "Point id; a re-inserted point gets a new id.", "Mã điểm; điểm chèn lại nhận mã mới."))
    feats = [(c, "feature", t, u, r, "no", e, v) for c, _, t, u, r, _, e, v in FEAT_CARD]
    tail = []
    if upd:
        tail = [("planted_case", "label", "string", "category", "case name or blank", "yes", "Case the scenario meant to create; blank for natural events.", "Case kịch bản cố ý tạo; để trống nếu tự nhiên."),
                ("verified_case", "label", "string", "category", "Noise, Creation, Absorption, Merge, Removal, Reduction, PotentialSplit_NonSplit, PotentialSplit_Split, DeleteNoise", "no", "Case derived from reference full re-fits compared as partitions.", "Case suy ra từ re-fit toàn bộ chạy lại, so sánh theo phân hoạch."),
                ("ambiguous_border_ids", "descriptive", "string", "id list", "ids separated by ;", "yes", "Border points shared by two clusters AFTER the step; ignore when comparing labels.", "Điểm biên mơ hồ SAU bước; bỏ qua khi so nhãn.")]
    else:
        tail = [("ref_label", "label", "int", "cluster id", "-1, 1..", "no", "Reference DBSCAN label of D0 (compare as a partition).", "Nhãn DBSCAN tham chiếu của D0 (so sánh theo phân hoạch).")]
    app = [("point_arrival_timestamp", "descriptive", "string", "datetime", "", "no", "When this point entered the window (arrival clock). For d0 points it is virtual, one hour apart before 2025-03-01.", "Lúc điểm này vào cửa sổ (đồng hồ đến). Với d0 là giả lập, cách nhau 1 giờ trước 2025-03-01."),
           ("event_timestamp", "descriptive", "string", "datetime", "", "no", "EVENT clock: when the transaction happened; its time of day equals hour_of_day within 0.01 h and it is at most 24 h before the arrival.", "Đồng hồ SỰ KIỆN: lúc giao dịch xảy ra; giờ trong ngày bằng hour_of_day (sai số 0.01 giờ), sớm hơn lúc đến tối đa 24 giờ."),
           ("segment_id", "label", "int", "class", SEG_NOTE[0], "no", "Ground-truth segment of the point.", "Nhãn phân khúc thật của điểm. " + SEG_NOTE[1]),
           ("customer_id", "key", "string", "id", "C0001 to C0600", "no", "FK to customers.csv.", "Khóa ngoại tới customers.csv."),
           ("channel", "descriptive", "string", "category", "POS, ONLINE, ATM, CONTACTLESS", "no", "Payment channel (clean in streams).", "Kênh thanh toán (sạch trong stream)."),
           ("merchant_category", "key", "string", "code", "see merchant_categories.csv", "no", "FK to merchant_categories.csv.", "Khóa ngoại tới merchant_categories.csv."),
           ("txn_district_id", "key", "string", "id", "D01 to D08", "yes (blank iff is_foreign = 1)", "FK to districts.csv.", "Khóa ngoại tới districts.csv."),
           ("is_foreign", "descriptive", "int", "flag", "0 or 1", "no", "1 if foreign terminal.", "1 nếu máy ở nước ngoài.")]
    block(table, head + feats + tail + app)


def stream_geo(table, upd):
    head = [("step", "key", "int", "count", "1..119", "no", "Step number, 1 step = 1 day.", "Số thứ tự bước, 1 bước = 1 ngày."),
            ("timestamp", "descriptive", "string", "date", "2025-03-01 onward", "no", "Date of the update.", "Ngày của bản cập nhật."),
            ("op", "descriptive", "string", "category", "insert, delete", "no", "Operation.", "Thao tác.")] if upd else []
    key = ("point_id", "key", "int", "id", "never reused", "no", "Point id; a re-inserted point gets a new id.", "Mã điểm; điểm chèn lại nhận mã mới.")
    xy = [("x_km", "feature", "float", "km", "0 to 32", "no", "East-west coordinate.", "Toạ độ đông tây."),
          ("y_km", "feature", "float", "km", "0 to 24", "no", "North-south coordinate.", "Toạ độ bắc nam.")]
    if upd:
        tail = [("planted_case", "label", "string", "category", "case name or blank", "yes", "Case the scenario meant to create.", "Case kịch bản cố ý tạo."),
                ("verified_case", "label", "string", "category", "see card_stream", "no", "Case derived from reference full re-fits.", "Case suy ra từ re-fit toàn bộ."),
                ("ambiguous_border_ids", "descriptive", "string", "id list", "ids separated by ;", "yes", "Ambiguous border points after the step.", "Điểm biên mơ hồ sau bước.")]
    else:
        tail = [("ref_label", "label", "int", "cluster id", "-1, 1..", "no", "Reference DBSCAN label of D0.", "Nhãn DBSCAN tham chiếu của D0.")]
    block(table, head + [key] + xy + tail + [("district_id", "key", "string", "id", "D01 to D08", "no", "District containing (x_km, y_km): nearest centroid of districts.csv.", "Quận chứa (x_km, y_km): tâm gần nhất trong districts.csv.")])


def build():
    R.clear()
    atm_base = [("txn_id", "key", "string", "id", "ATM00001..", "no", "ATM transaction id.", "Mã giao dịch ATM."),
                ("x_km", "feature", "float", "km", "0 to 32, 1 m grid", "no", "Location where the transaction was recorded (east-west).", "Vị trí ghi nhận giao dịch (đông tây)."),
                ("y_km", "feature", "float", "km", "0 to 24, 1 m grid", "no", "Location where the transaction was recorded (north-south).", "Vị trí ghi nhận giao dịch (bắc nam)."),
                ("hotspot_id", "label", "int", "class", "-1, 0..5", "no", "Ground-truth hotspot used by the generator; -1 isolated or noise (not a DBSCAN output).", "Nhãn hotspot thật của bộ sinh; -1 cô lập hoặc nhiễu (không phải đầu ra DBSCAN).")]
    block("raw/bank_geo/atm_txn_2d.csv", atm_base + [
        ("timestamp", "descriptive", "string", "datetime", "2025-03-01 to 2025-04-25", "no", "Time of the withdrawal, hour curve depends on the hotspot.", "Thời điểm rút tiền, đường cong theo giờ phụ thuộc hotspot."),
        ("atm_id", "key", "string", "id", "A01..A40", "no", "FK to atms.csv: nearest ATM of the same hotspot (isolated points: nearest ATM overall).", "Khóa ngoại tới atms.csv: ATM gần nhất cùng hotspot (điểm cô lập: ATM gần nhất toàn thành phố)."),
        ("customer_id", "key", "string", "id", "C0001 to C0600", "no", "FK to customers.csv; occupation consistent with the hotspot.", "Khóa ngoại tới customers.csv; nghề nghiệp khớp với hotspot."),
        ("amount_vnd", "descriptive", "int", "VND", "50000 to 5000000, multiple of 50000", "no", "Withdrawn amount.", "Số tiền rút."),
        ("atm_network", "descriptive", "string", "category", "own, partner", "no", "Network of the ATM used.", "Mạng của máy ATM."),
        ("district_id", "key", "string", "id", "D01 to D08", "no", "District containing the location (nearest centroid).", "Quận chứa vị trí (tâm gần nhất)."),
        ("atm_distance_km", "descriptive", "float", "km", "0 to about 7", "no", "Distance from the location to atm_id.", "Khoảng cách từ vị trí tới atm_id.")])
    for lvl, tag in (("hard", ""), ("medium", " (optional, regenerate with --difficulty medium)")):
        block(f"raw/bank_geo/atm_txn_2d_{lvl}.csv", [(c, r, t, u, g, m, e + tag, v) for c, r, t, u, g, m, e, v in atm_base])
    block("raw/bank_geo/hotspots.csv", [
        ("hotspot_id", "key", "int", "id", "-1, 0..5", "no", "Hotspot id.", "Mã hotspot."), ("code", "descriptive", "string", "code", "", "no", "Short code.", "Mã ngắn."),
        ("name_en", "descriptive", "string", "text", "", "no", "Business name (English).", "Tên nghiệp vụ (tiếng Anh)."), ("name_vi", "descriptive", "string", "text", "", "no", "Business name (Vietnamese).", "Tên nghiệp vụ (tiếng Việt)."),
        ("meaning_vi", "descriptive", "string", "text", "", "no", "Meaning of the cluster.", "Ý nghĩa của cụm."), ("suggested_action_vi", "descriptive", "string", "text", "", "no", "Suggested business action.", "Hành động đề xuất.")])
    block("raw/bank_geo/districts.csv", [
        ("district_id", "key", "string", "id", "D01..D08", "no", "District id.", "Mã quận."), ("name_vi", "descriptive", "string", "text", "", "no", "Fictional name (Vietnamese).", "Tên giả lập (tiếng Việt)."),
        ("name_en", "descriptive", "string", "text", "", "no", "Fictional name (English).", "Tên giả lập (tiếng Anh)."),
        ("area_type", "descriptive", "string", "category", "CBD, university, market, riverside, highway, residential, suburb", "no", "Type of area.", "Loại khu vực."),
        ("centroid_x_km", "descriptive", "float", "km", "0 to 32", "no", "Centroid; a point belongs to the district of the nearest centroid (Voronoi tiling of the 32 x 24 km city).", "Tâm quận; điểm thuộc quận có tâm gần nhất (lát Voronoi thành phố 32 x 24 km)."),
        ("centroid_y_km", "descriptive", "float", "km", "0 to 24", "no", "Centroid y.", "Tung độ tâm."), ("area_km2", "descriptive", "float", "km2", "sum = 768", "no", "Area of the Voronoi cell (0.1 km grid estimate).", "Diện tích ô Voronoi (ước lượng lưới 0.1 km).")])
    block("raw/bank_geo/atms.csv", [
        ("atm_id", "key", "string", "id", "A01..A40", "no", "ATM id.", "Mã máy ATM."), ("x_km", "descriptive", "float", "km", "0 to 32", "no", "Fixed location.", "Vị trí cố định."),
        ("y_km", "descriptive", "float", "km", "0 to 24", "no", "Fixed location.", "Vị trí cố định."),
        ("hotspot_id", "descriptive", "int", "id", "-1, 0..5", "no", "Hotspot the ATM serves (-1 = stand-alone ATM for isolated points).", "Hotspot mà ATM phục vụ (-1 = ATM đứng riêng cho điểm cô lập)."),
        ("district_id", "key", "string", "id", "D01..D08", "no", "District of the ATM.", "Quận của ATM."), ("atm_network", "descriptive", "string", "category", "own, partner", "no", "Own or partner network.", "Mạng riêng hoặc đối tác.")])
    stream_geo("raw/bank_geo/atm_stream/d0.csv", False)
    stream_geo("raw/bank_geo/atm_stream/updates.csv", True)
    card_table("raw/bank_card/card_txn_5d.csv")
    card_table("raw/bank_card/card_txn_5d_hard.csv", " Hard variant: overlapping, unequal density, bridge, near anomalies.")
    card_table("raw/bank_card/card_txn_5d_medium.csv", " (optional, regenerate with --difficulty medium)")
    block("raw/bank_card/segments.csv", [
        ("segment_id", "key", "int", "id", "-1, 0..3", "no", "Segment id.", "Mã phân khúc."), ("code", "descriptive", "string", "code", "", "no", "Short code.", "Mã ngắn."),
        ("name_en", "descriptive", "string", "text", "", "no", "Business name (English).", "Tên nghiệp vụ (tiếng Anh)."), ("name_vi", "descriptive", "string", "text", "", "no", "Business name (Vietnamese).", "Tên nghiệp vụ (tiếng Việt)."),
        ("meaning_vi", "descriptive", "string", "text", "", "no", "Meaning.", "Ý nghĩa."), ("suggested_action_vi", "descriptive", "string", "text", "", "no", "Suggested action.", "Hành động đề xuất.")])
    block("raw/bank_card/customers.csv", [
        ("customer_id", "key", "string", "id", "C0001..C0600", "no", "Customer id.", "Mã khách hàng."),
        ("age", "descriptive", "int", "year", "18 to 85", "no", "Age; students 18 to 26, retired 56 to 85.", "Tuổi; sinh viên 18 đến 26, hưu trí 56 đến 85."),
        ("gender", "descriptive", "string", "category", "F, M", "no", "Gender (binary in the simulation, a simplification).", "Giới tính (nhị phân trong mô phỏng, là một đơn giản hóa)."),
        ("occupation_group", "descriptive", "string", "category", "office, student, trader, freelancer, retired, other", "no", "Occupation group.", "Nhóm nghề nghiệp."),
        ("tier", "descriptive", "string", "category", "mass, affluent, priority", "no", "Customer tier; correlates with income and occupation.", "Hạng khách hàng; tương quan với thu nhập và nghề."),
        ("tenure_months", "descriptive", "int", "month", "1 to 96", "no", "Months since opening the account (the fictional bank is 8 years old).", "Số tháng từ lúc mở tài khoản (ngân hàng giả lập 8 năm tuổi)."),
        ("monthly_income_vnd", "descriptive", "int", "VND", "about 1e6 to 2e8", "no", "Monthly income, rounded to 100000.", "Thu nhập tháng, làm tròn 100000."),
        ("home_district_id", "key", "string", "id", "D01..D08", "no", "FK to districts.csv.", "Khóa ngoại tới districts.csv."),
        ("digital_active", "descriptive", "int", "flag", "0 or 1", "no", "Uses mobile or web banking.", "Dùng ngân hàng số."),
        ("has_credit_card", "descriptive", "int", "flag", "0 or 1", "no", "Holds a credit card.", "Có thẻ tín dụng."),
        ("preferred_channel", "descriptive", "string", "category", "mobile_app, web, atm, branch", "no", "Preferred banking channel.", "Kênh giao dịch ưa thích.")])
    block("raw/bank_card/merchant_categories.csv", [
        ("category_code", "key", "string", "code", "3 letters", "no", "Merchant category code used in card tables.", "Mã nhóm merchant dùng trong bảng thẻ."),
        ("name_en", "descriptive", "string", "text", "", "no", "Name (English).", "Tên (tiếng Anh)."), ("name_vi", "descriptive", "string", "text", "", "no", "Name (Vietnamese).", "Tên (tiếng Việt)."),
        ("base_risk_level", "descriptive", "string", "category", "low, medium, high", "no", "Typical risk of the category.", "Mức rủi ro điển hình của nhóm.")])
    stream_card("raw/bank_card/card_stream/d0.csv", False)
    stream_card("raw/bank_card/card_stream/updates.csv", True)
    stream_card("raw/bank_card/card_stream_hard/d0.csv", False)
    stream_card("raw/bank_card/card_stream_hard/updates.csv", True)
    xy = lambda t: [("point_id", "key", "string", "id", "", "no", "Point name used in the group doc.", "Tên điểm dùng trong tài liệu nhóm.")]
    block("raw/toy/<scenario>/points.csv", xy(0) + [("x", "feature", "float", "unit", "", "no", "First coordinate.", "Toạ độ thứ nhất."), ("y", "feature", "float", "unit", "", "no", "Second coordinate.", "Toạ độ thứ hai.")])
    block("raw/toy/<scenario>/update.csv", [("op", "descriptive", "string", "category", "insert, delete", "no", "Operation.", "Thao tác."), ("point_id", "key", "string", "id", "", "no", "Point id.", "Mã điểm."),
                                          ("x", "feature", "float", "unit", "", "yes (delete)", "First coordinate.", "Toạ độ thứ nhất."), ("y", "feature", "float", "unit", "", "yes (delete)", "Second coordinate.", "Toạ độ thứ hai.")])
    for tb in ("raw/toy_edge/<scenario>/points.csv", "raw/toy_edge/<scenario>/update.csv"):
        pre = [("op", "descriptive", "string", "category", "insert, delete", "no", "Operation (several ordered rows allowed).", "Thao tác (có thể nhiều dòng, theo thứ tự).")] if "update" in tb else []
        block(tb, pre + [("point_id", "key", "string", "id", "", "no", "Point id (ids are not reused).", "Mã điểm (không dùng lại mã).")] +
              [(c, "feature", "float", "unit", "multiples of 1/16 where a distance equals Eps", "yes (delete rows)" if "update" in tb else "no", "Coordinate (2-D scenarios use x, y; the 4-D scenario uses x1..x4).", "Toạ độ (kịch bản 2D dùng x, y; kịch bản 4D dùng x1..x4).")
                for c in ("x", "y", "x1", "x2", "x3", "x4")])
    for tb in ("raw/toy/<scenario>/expected.json", "raw/toy_edge/<scenario>/expected.json"):
        for k, en, vi in (("labels_before / labels_after", "Labels before and after (-1 noise); compare as partitions.", "Nhãn trước và sau (-1 là nhiễu); so sánh theo phân hoạch."),
                          ("core_before / core_after", "Core point sets.", "Tập điểm lõi."), ("updseed", "UpdSeed of the update (toy_edge multi-step: updseed_by_step).", "UpdSeed của update (toy_edge nhiều bước: updseed_by_step)."),
                          ("steps (toy_edge)", "Per update row: case, labels, cores, UpdSeed, ambiguous_border_ids, max_core_hops_between_seeds.", "Mỗi dòng update: case, nhãn, lõi, UpdSeed, ambiguous_border_ids, max_core_hops_between_seeds."),
                          ("ambiguous_to_ignore (toy_edge)", "Points whose label may be either adjacent cluster.", "Điểm có thể mang nhãn một trong hai cụm kề."),
                          ("ties_at_exactly_eps (toy_edge)", "Pairs at distance exactly Eps (boundary tests).", "Cặp điểm cách nhau đúng Eps (kiểm tra biên)."),
                          ("why_vi (toy_edge)", "Short Vietnamese note on why the scenario matters.", "Ghi chú ngắn vì sao kịch bản quan trọng.")):
            d(tb, k, "truth", "json", "", "", "no", en, vi)
    for fn, extra in (("moons_n500", "0 or 1"), ("rings_n500", "0 or 1"), ("blobs_noise_n500", "-1, 0..2")):
        block(f"raw/synthetic/{fn}.csv", [("point_id", "key", "int", "id", "", "no", "Row id.", "Mã dòng."), ("x", "feature", "float", "unit", "", "no", "Coordinate.", "Toạ độ."),
                                           ("y", "feature", "float", "unit", "", "no", "Coordinate.", "Toạ độ."), ("label", "label", "int", "class", extra, "no", "Generating label (not DBSCAN output).", "Nhãn lúc sinh (không phải đầu ra DBSCAN).")])
    block("raw/nd/blobs5d_n860.csv", [("point_id", "key", "int", "id", "", "no", "Row id.", "Mã dòng.")] + [(f"x{i}", "feature", "float", "unit", "", "no", "Coordinate.", "Toạ độ.") for i in range(1, 6)] +
          [("label", "label", "int", "class", "-1, 0..3", "no", "Generating label.", "Nhãn lúc sinh.")])
    # truth tables
    for lvl, pre in (("", "TXN"), ("_hard", "TXH"), ("_medium", "TXI")):
        t = f"truth/card_txn_5d{lvl}_clean_descriptive.csv"
        opt = " (optional)" if lvl == "_medium" else ""
        block(t, [("txn_id", "key", "string", "id", pre + "000001..", "no", "Transaction id (unique).", "Mã giao dịch (duy nhất)." + opt)] +
              [(c, "truth", ty, u, r, "no", "Clean version of the dirtied column in the card table. " + e, "Phiên bản sạch của cột bị làm bẩn trong bảng thẻ. " + v)
               for c, _, ty, u, r, _, e, v in CARD_DESC])
        t = f"truth/card_txn_5d{lvl}_clean_base.csv"
        block(t, [("txn_id", "key", "string", "id", pre + "000001..", "no", "Transaction id (unique).", "Mã giao dịch (duy nhất).")] +
              [(c, "truth", ty, u, r, "no", "Clean value before missing cells were punched in. " + e, "Giá trị sạch trước khi tạo ô thiếu. " + v) for c, _, ty, u, r, _, e, v in FEAT_CARD] +
              [("segment_id", "truth", "int", "class", SEG_NOTE[0], "no", "Ground-truth segment.", SEG_NOTE[1])])
    for lvl in ("hard", "medium"):
        block(f"truth/card_txn_5d_{lvl}_generation_flags.csv", [
            ("txn_id", "key", "string", "id", "", "no", "Transaction id.", "Mã giao dịch."), ("segment_id", "truth", "int", "class", SEG_NOTE[0], "no", "Ground-truth segment (bridge points take the nearer endpoint).", "Nhãn phân khúc thật (điểm cầu nối lấy nhãn đầu mút gần hơn)."),
            ("is_bridge", "truth", "int", "flag", "0 or 1", "no", "Part of the thin bridge between retail and online.", "Thuộc dải cầu nối mỏng giữa bán lẻ và online."),
            ("is_near_anomaly", "truth", "int", "flag", "0 or 1", "no", "Anomaly placed close to a dense group.", "Bất thường đặt sát một nhóm đặc."),
            ("is_large_online", "truth", "int", "flag", "0 or 1", "no", "Online purchase with a large basket, near travel.", "Mua online giỏ hàng lớn, gần nhóm du lịch.")])
        block(f"truth/atm_txn_2d_{lvl}_generation_flags.csv", [
            ("txn_id", "key", "string", "id", "", "no", "Transaction id.", "Mã giao dịch."), ("hotspot_id", "truth", "int", "class", "-1, 0..5", "no", "Ground-truth hotspot.", "Nhãn hotspot thật."),
            ("is_bridge", "truth", "int", "flag", "0 or 1", "no", "Part of the density bridge downtown to market plaza.", "Thuộc dải cầu nối mật độ giữa trung tâm và quảng trường chợ.")])
    block("truth/customer_segment_mix.csv", [
        ("customer_id", "key", "string", "id", "", "no", "Customer id.", "Mã khách hàng."), ("kind", "truth", "string", "category", "dominated, mixed", "no", "Whether one segment dominates the customer's behaviour.", "Một phân khúc chi phối hành vi hay hành vi hỗn hợp."),
        ("p_seg0", "truth", "float", "prob", "0..1", "no", "Probability of segment 0 for this customer.", "Xác suất phân khúc 0 của khách."), ("p_seg1", "truth", "float", "prob", "0..1", "no", "Segment 1.", "Phân khúc 1."),
        ("p_seg2", "truth", "float", "prob", "0..1", "no", "Segment 2.", "Phân khúc 2."), ("p_seg3", "truth", "float", "prob", "0..1", "no", "Segment 3.", "Phân khúc 3.")])
    block("truth/defect_catalogue_card.csv", [
        ("scope", "truth", "string", "category", "base, descriptive", "no", "Whether the defect sits in a base (clustering) column or a descriptive column.", "Lỗi nằm ở cột nền (dùng để gom cụm) hay cột mô tả."),
        ("table", "truth", "string", "name", "", "no", "Table name.", "Tên bảng."), ("defect_type", "truth", "string", "category", "", "no", "Defect type.", "Loại lỗi."),
        ("column", "truth", "string", "name", "", "no", "Column.", "Cột."), ("n_rows", "truth", "int", "count", "", "no", "Number of rows of the file affected (duplicates counted).", "Số dòng của file bị ảnh hưởng (tính cả dòng trùng).")])
    return R


def write(path):
    rows = build()
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\r\n")
        w.writerow(COLS)
        w.writerows(rows)
    return len(rows)
