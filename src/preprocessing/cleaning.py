"""Hàm làm sạch bảng giao dịch thẻ (pandas + numpy), mỗi hàm một việc, không dùng lớp.

QUAN TRỌNG, KHÔNG LOẠI NGOẠI LAI BẰNG IQR: với DBSCAN, điểm ngoại lai chính là tín hiệu
nhiễu (noise, nhãn -1) mà thuật toán phải tự tìm ra. Bảng thẻ cố tình có 15 giao dịch
bất thường (xa nhà, giá trị rất lớn); xóa chúng bằng quy tắc IQR sẽ làm mất đúng thứ cần
phát hiện. Ở đây chỉ SỬA lỗi dữ liệu (thiếu ô, trùng
dòng, sai chính tả, định dạng giờ) và GẮN CỜ, không xóa dòng nào ngoài bản sao y hệt.

Quy tắc: tham số học từ dữ liệu (trung vị) chỉ học trên D0; các hàm ở đây không đọc
thư mục data/truth (đáp án chỉ dùng trong scorecard.py và notebook để chấm điểm).
"""
import numpy as np
import pandas as pd

NUMERIC_COLS = ["amount_vnd", "hour_of_day", "distance_from_home_km",
                "txns_last_7d", "merchant_risk_score"]
FEATURE_COLS = ["log10_amount_vnd", "hour_of_day", "distance_from_home_km",
                "txns_last_7d", "merchant_risk_score"]
CHANNELS = ("POS", "ONLINE", "ATM", "CONTACTLESS")
UNKNOWN_CATEGORY = "UNKNOWN"
# Thứ bậc điền khuyết: (kênh x nhóm hàng) -> kênh -> trung vị chung; nhóm dưới MIN_GROUP dòng D0 bị bỏ
IMPUTE_LEVELS = (("channel", "merchant_category"), ("channel",))
MIN_GROUP = 5
_ISO = r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$"
_DMY = r"^\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2}$"


# ---------------------------------------------------------------- kênh, giờ
def normalise_channel(channel):
    """Bỏ khoảng trắng, đổi chữ hoa, bỏ gạch nối ('contact-less'), rồi ánh xạ về 4 giá trị chuẩn.
    Giá trị không nhận ra được trả NaN (không đoán)."""
    s = channel.astype("string").str.strip().str.upper().str.replace("-", "", regex=False)
    s = s.str.replace(" ", "", regex=False)
    return s.where(s.isin(CHANNELS)).astype(object)


def parse_timestamp(timestamp):
    """Đọc timestamp có hai định dạng: 'YYYY-MM-DD hh:mm:ss' và 'dd/mm/YYYY hh:mm:ss'.

    Trả về DataFrame hai cột: `ts` (datetime64, NaT nếu không đọc được) và `ts_status`
    trong {'iso', 'dmy', 'impossible', 'missing'}. 'impossible' là chuỗi có hình dạng
    ngày giờ nhưng giá trị vô lý (tháng 13, ngày 31/4, giờ 25, phút 75): gắn cờ, không đoán.
    Chuỗi có dấu '/' luôn đọc theo dd/mm (dayfirst), vì định dạng ISO không dùng '/'.
    """
    s = timestamp.astype("string").str.strip()
    is_iso, is_dmy = s.str.match(_ISO).fillna(False), s.str.match(_DMY).fillna(False)
    ts = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    ts[is_iso] = pd.to_datetime(s[is_iso], format="%Y-%m-%d %H:%M:%S", errors="coerce")
    ts[is_dmy] = pd.to_datetime(s[is_dmy], format="%d/%m/%Y %H:%M:%S", errors="coerce")
    status = np.where(s.isna(), "missing", np.where(ts.isna(), "impossible", np.where(is_dmy, "dmy", "iso")))
    return pd.DataFrame({"ts": ts, "ts_status": status}, index=s.index)


def hour_from_timestamp(ts):
    """Giờ trong ngày dạng thập phân (13:13:48 -> 13.23), làm tròn 2 chữ số như hour_of_day."""
    return (ts.dt.hour + ts.dt.minute / 60 + ts.dt.second / 3600).round(2)


def repair_hour_of_day(df, ts_col="ts", hour_col="hour_of_day"):
    """Điền hour_of_day thiếu bằng giờ của timestamp (trong dữ liệu sạch hai giá trị khớp trong 0.01).
    Thêm cột cờ `hour_from_ts`. Ô còn thiếu (timestamp cũng hỏng) để cho bước điền trung vị."""
    out = df.copy()
    fill = out[hour_col].isna() & out[ts_col].notna()
    out.loc[fill, hour_col] = hour_from_timestamp(out.loc[fill, ts_col])
    out["hour_from_ts"] = fill
    return out


# ---------------------------------------------------------------- trùng, thiếu
def drop_exact_duplicates(df):
    """Bỏ dòng trùng y hệt (mọi cột), giữ dòng đầu. Dòng chỉ trùng một phần KHÔNG bị bỏ."""
    return df.drop_duplicates(keep="first").reset_index(drop=True)


def select_d0(df, n=500, ts_col="ts"):
    """D0 = n giao dịch sớm nhất theo timestamp; dòng không có timestamp xếp cuối. Trả mặt nạ bool."""
    order = df[ts_col].sort_values(na_position="last", kind="stable").index[:n]
    return df.index.isin(order)


def fit_medians(df, cols=NUMERIC_COLS):
    """Trung vị từng cột, chỉ tính trên các dòng đưa vào (truyền D0, không truyền cả bảng)."""
    return {c: float(df[c].median()) for c in cols}


def impute_median(df, medians):
    """Điền ô số thiếu bằng trung vị đã học; thêm cột cờ `imp_<cột>` (True nếu ô được điền)."""
    out = df.copy()
    for c, m in medians.items():
        out["imp_" + c] = out[c].isna()
        out[c] = out[c].fillna(m)
    return out


def _level_keys(df, level):
    """(khóa nhóm, mặt nạ khóa hợp lệ) theo các cột của một tầng; 'UNKNOWN' (nhóm hàng chưa biết) coi như
    thiếu khóa, không phải một nhóm, nên dòng đó rơi xuống tầng thấp hơn."""
    keys = df[list(level)]
    keys = keys.mask(keys == UNKNOWN_CATEGORY)
    valid = keys.notna().all(axis=1).to_numpy()
    idx = pd.Index(keys.iloc[:, 0]) if len(level) == 1 else pd.MultiIndex.from_frame(keys)
    return idx, valid


def fit_group_medians(df, levels=IMPUTE_LEVELS, cols=NUMERIC_COLS, min_group=MIN_GROUP):
    """Trung vị từng cột theo nhóm, một bảng cho mỗi tầng (kênh x nhóm hàng, rồi kênh); chỉ truyền D0.
    Nhóm có ít hơn min_group giá trị quan sát được bị bỏ (trung vị của vài dòng quá nhiễu).
    Chỉ dùng cột mô tả (channel, merchant_category), không dùng segment_id và không đọc data/truth."""
    tables = []
    for level in levels:
        idx, valid = _level_keys(df, level)
        g = df.loc[valid, list(cols)].groupby(idx[valid])
        tables.append(g.median().where(g.count() >= min_group))
    return tables


def impute_group_median(df, group_medians, medians, levels=IMPUTE_LEVELS):
    """Điền ô số thiếu theo thứ bậc: trung vị (kênh x nhóm hàng), nếu nhóm đó không đủ thì trung vị kênh,
    nếu vẫn không có thì trung vị chung của D0. Tất cả học trên D0. Thêm cờ `imp_<cột>`."""
    out = df.copy()
    keys = [_level_keys(out, level) for level in levels]
    for c, m in medians.items():
        guess = pd.Series(np.nan, index=out.index)
        for tbl, (k, valid) in zip(group_medians, keys):
            found = pd.Series(tbl[c].reindex(k).to_numpy(), index=out.index).where(valid)
            guess = guess.fillna(found)
        out["imp_" + c] = out[c].isna()
        out[c] = out[c].fillna(guess.fillna(m))
    return out


def fill_missing_category(df, col="merchant_category"):
    """merchant_category thiếu không suy ra được: điền 'UNKNOWN' và gắn cờ `category_missing`."""
    out = df.copy()
    out["category_missing"] = out[col].isna()
    out[col] = out[col].fillna(UNKNOWN_CATEGORY)
    return out


# ---------------------------------------------------------------- khóa ngoại
def flag_unknown_customers(df, customers, col="customer_id"):
    """Cờ True nếu customer_id không có trong bảng khách (gồm 'UNKNOWN'). Không xóa dòng."""
    return ~df[col].isin(customers["customer_id"])


def flag_invalid_districts(df, districts, col="txn_district_id"):
    """Cờ True nếu có mã quận nhưng mã đó không có trong bảng quận (vd D99).
    Quận trống khi is_foreign = 1 là hợp lệ (máy ngoài thành phố), không bị cờ."""
    return df[col].notna() & ~df[col].isin(districts["district_id"])


def add_log_amount(df, col="amount_vnd"):
    """Thêm cột log10_amount_vnd. Số tiền lệch phải mạnh (1.000 đến 500 triệu), nên log10 là lựa chọn thiết kế hợp lý.
    Đo thực tế (notebook D02, mục 3b): ở file thẻ dễ, bỏ log10 vẫn đạt ARI 1.0 (Eps 0.65, MinPts 10); chỉ file thẻ khó được lợi
    (ARI 0.646 có log10, 0.552 không có). Bước quyết định là z-score, không phải log10."""
    out = df.copy()
    out["log10_amount_vnd"] = np.log10(out[col].where(out[col] > 0))
    return out


# ---------------------------------------------------------------- báo cáo chất lượng
def quality_report(df, customers=None, districts=None, numeric_cols=NUMERIC_COLS):
    """Bảng tóm tắt lỗi của bảng thô: mỗi dòng (check, column, n_rows, share)."""
    n = len(df)
    rows = []

    def add(check, column, count):
        rows.append((check, column, int(count), round(float(count) / n, 4)))

    for c in numeric_cols:
        add("missing_value", c, df[c].isna().sum())
    add("duplicate_row_exact_copy", "(whole row)", df.duplicated(keep="first").sum())
    add("duplicate_txn_id", "txn_id", df["txn_id"].duplicated(keep="first").sum())
    canon = df["channel"].isin(CHANNELS)
    add("inconsistent_spelling_or_case", "channel", (~canon).sum())
    st = parse_timestamp(df["timestamp"])["ts_status"]
    add("mixed_timestamp_format_dd_mm_yyyy", "timestamp", (st == "dmy").sum())
    add("impossible_timestamp", "timestamp", (st == "impossible").sum())
    add("missing_value", "merchant_category", df["merchant_category"].isna().sum())
    if customers is not None:
        add("unknown_customer_id", "customer_id", flag_unknown_customers(df, customers).sum())
    if districts is not None:
        add("invalid_district_id", "txn_district_id", flag_invalid_districts(df, districts).sum())
    foreign = df["is_foreign"].astype(float) == 1
    add("missing_district_not_foreign", "txn_district_id", (df["txn_district_id"].isna() & ~foreign).sum())
    return pd.DataFrame(rows, columns=["check", "column", "n_rows", "share"])


# ---------------------------------------------------------------- toàn bộ quy trình
def clean_card_table(raw, customers, districts, n_d0=500, impute_by=IMPUTE_LEVELS):
    """Chạy các bước trên theo thứ tự cố định và trả (bảng sạch, trung vị chung của D0).

    impute_by=IMPUTE_LEVELS (mặc định): điền trung vị theo nhóm (kênh x nhóm hàng, rồi kênh, rồi chung);
    impute_by=None: chỉ điền trung vị chung của D0 (cách đơn giản, sai số lớn hơn nhiều, xem scorecard).
    Mọi tham số học chỉ từ D0 = n_d0 giao dịch sớm nhất theo timestamp.
    Bảng sạch giữ mọi dòng sau khi bỏ bản sao y hệt, thêm: ts, ts_status, hour_from_ts, is_d0,
    imp_<cột số>, category_missing, unknown_customer, invalid_district, log10_amount_vnd.
    """
    df = drop_exact_duplicates(raw)
    df["channel"] = normalise_channel(df["channel"])
    df = pd.concat([df, parse_timestamp(df["timestamp"])], axis=1)
    df = repair_hour_of_day(df)
    df = fill_missing_category(df)
    df["is_d0"] = select_d0(df, n_d0)
    d0 = df[df["is_d0"]]
    medians = fit_medians(d0)
    if impute_by is None:
        df = impute_median(df, medians)
    else:
        df = impute_group_median(df, fit_group_medians(d0, impute_by), medians, impute_by)
    df["unknown_customer"] = flag_unknown_customers(df, customers)
    df["invalid_district"] = flag_invalid_districts(df, districts)
    df = add_log_amount(df)
    return df, medians
