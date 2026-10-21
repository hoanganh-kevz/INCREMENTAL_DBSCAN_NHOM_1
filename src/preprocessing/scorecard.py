"""Chấm điểm bước làm sạch so với đáp án (data/truth). CHỈ dùng để đánh giá, không dùng để sửa.

Hàm nhận DataFrame đã đọc sẵn (bảng thô, bảng sạch, đáp án nền, đáp án mô tả) nên cleaning.py
không phải đọc thư mục truth. Mọi so khớp theo txn_id.
"""
import numpy as np
import pandas as pd

from .cleaning import CHANNELS, NUMERIC_COLS, parse_timestamp

COLUMNS = ["table", "defect", "column", "n", "metric", "value", "global_median_error", "gain_x", "note"]


def _row(table, defect, column, n, metric, value, ref=np.nan, note=""):
    value = float(value) if value == value else np.nan
    ref = float(ref) if ref == ref else np.nan
    gain = ref / value if value == value and ref == ref and value > 0 else np.nan
    return dict(table=table, defect=defect, column=column, n=int(n), metric=metric, value=value,
                global_median_error=ref, gain_x=gain, note=note)


def _recall_precision(flag, truly):
    tp = int((flag & truly).sum())
    rec = tp / truly.sum() if truly.sum() else np.nan
    prec = tp / flag.sum() if flag.sum() else np.nan
    return rec, prec


def score_cleaning(table, raw, clean, truth_base, truth_desc, clean_global, medians_global):
    """Trả về bảng điểm theo từng loại lỗi (DataFrame, cột COLUMNS).

    raw: bảng thô đã đọc (có dòng trùng); clean: kết quả clean_card_table (mặc định, điền trung vị theo nhóm);
    clean_global: kết quả clean_card_table(impute_by=None) và medians_global: trung vị chung của nó. Hai bảng này cho
    cột global_median_error (cùng thước đo, cách điền chỉ bằng trung vị chung) và gain_x = global / mặc định.
    truth_base / truth_desc: file *_clean_base.csv và *_clean_descriptive.csv.
    """
    raw_u = raw.drop_duplicates().set_index("txn_id")
    cl = clean.set_index("txn_id")
    tb, td = truth_base.set_index("txn_id"), truth_desc.set_index("txn_id")
    ids = cl.index
    out = []

    # 1) dòng trùng
    n_dup = len(raw) - len(raw_u)
    out.append(_row(table, "duplicate_row_exact_copy", "(whole row)", n_dup, "share_removed",
                    (len(raw) - len(cl)) / n_dup if n_dup else np.nan,
                    note=f"còn {len(cl)} dòng, đáp án có {len(tb)}; txn_id duy nhất: {cl.index.is_unique}"))

    # 2) ô số thiếu: sai số so với giá trị gốc trước khi đục ô, cạnh cách điền trung vị chung
    cg = clean_global.set_index("txn_id")
    for c in NUMERIC_COLS:
        miss = raw_u[c].isna().reindex(ids).fillna(False).to_numpy()
        n = int(miss.sum())
        if n == 0:
            continue
        truth_v = tb.loc[ids, c].to_numpy()[miss]
        err = np.abs(cl.loc[ids, c].to_numpy()[miss] - truth_v)
        if c == "hour_of_day":   # cách "chung" cho giờ = điền trung vị mà không dùng timestamp
            err_g = np.abs(medians_global[c] - truth_v)
            note = "sửa từ timestamp; cột chung = điền trung vị, không dùng timestamp"
        else:
            err_g = np.abs(cg.loc[ids, c].to_numpy()[miss] - truth_v)
            note = "trung vị theo (kênh x nhóm hàng) -> kênh -> chung, học trên D0"
        out.append(_row(table, "missing_value", c, n, "median_abs_error", np.median(err), np.median(err_g), note))
        out.append(_row(table, "missing_value", c, n, "mean_abs_error", err.mean(), err_g.mean(), note))
    hr = cl["hour_from_ts"].to_numpy()
    if hr.any():
        e = np.abs(cl.loc[ids, "hour_of_day"].to_numpy()[hr] - tb.loc[ids, "hour_of_day"].to_numpy()[hr])
        left = int((~hr & raw_u["hour_of_day"].isna().reindex(ids).to_numpy()).sum())
        out.append(_row(table, "missing_value", "hour_of_day (từ timestamp)", hr.sum(), "max_abs_error", e.max(),
                        np.nan, note=f"{int(hr.sum())} ô sửa từ timestamp, {left} ô phải điền trung vị"))

    # 3) kênh sai chính tả
    dirty = (raw_u.loc[ids, "channel"] != td.loc[ids, "channel"]).to_numpy()
    ok = (cl.loc[ids, "channel"] == td.loc[ids, "channel"]).to_numpy()
    out.append(_row(table, "inconsistent_spelling_or_case", "channel", dirty.sum(), "accuracy_on_dirty_rows",
                    ok[dirty].mean(), np.nan,
                    note=f"độ chính xác trên mọi dòng {ok.mean():.4f}; để nguyên bản thô thì đúng 0 trên các dòng này"))

    # 4) timestamp
    status = cl["ts_status"]
    true_ts = pd.to_datetime(td.loc[ids, "timestamp"])
    got_ts = cl.loc[ids, "ts"]
    dmy = (status == "dmy").to_numpy()
    out.append(_row(table, "mixed_timestamp_format_dd_mm_yyyy", "timestamp", dmy.sum(), "share_recovered_exactly",
                    (got_ts[dmy] == true_ts[dmy]).mean() if dmy.any() else np.nan,
                    note="so khớp đúng từng giây với đáp án"))
    raw_ts_bad = (raw_u.loc[ids, "timestamp"] != td.loc[ids, "timestamp"]).to_numpy() & ~dmy
    imp = (status == "impossible").to_numpy()
    rec, prec = _recall_precision(pd.Series(imp), pd.Series(raw_ts_bad))
    out.append(_row(table, "impossible_timestamp", "timestamp", raw_ts_bad.sum(), "flag_recall", rec,
                    note=f"precision {prec:.4f}; ngày giờ đúng không khôi phục được, để NaT"))
    iso = (status == "iso").to_numpy()
    out.append(_row(table, "valid_timestamp_untouched", "timestamp", iso.sum(), "share_exact",
                    (got_ts[iso] == true_ts[iso]).mean(), note="timestamp ISO hợp lệ phải giữ nguyên"))

    # 5) khóa ngoại và mã thiếu
    unk_true = (raw_u.loc[ids, "customer_id"] != td.loc[ids, "customer_id"]).to_numpy()
    rec, prec = _recall_precision(cl["unknown_customer"].reset_index(drop=True), pd.Series(unk_true))
    out.append(_row(table, "unknown_customer_id", "customer_id", unk_true.sum(), "flag_recall", rec,
                    note=f"precision {prec:.4f}; không sửa, chỉ gắn cờ"))
    dist_true = ((raw_u.loc[ids, "txn_district_id"] != td.loc[ids, "txn_district_id"]) &
                 raw_u.loc[ids, "txn_district_id"].notna()).to_numpy()
    rec, prec = _recall_precision(cl["invalid_district"].reset_index(drop=True), pd.Series(dist_true))
    out.append(_row(table, "invalid_district_id", "txn_district_id", dist_true.sum(), "flag_recall", rec,
                    note=f"precision {prec:.4f}; không sửa, chỉ gắn cờ"))
    cat_true = raw_u.loc[ids, "merchant_category"].isna().to_numpy()
    out.append(_row(table, "missing_value", "merchant_category", cat_true.sum(), "flag_recall",
                    cl["category_missing"].to_numpy()[cat_true].mean() if cat_true.any() else np.nan,
                    note="không suy ra được, điền UNKNOWN và gắn cờ; độ chính xác khôi phục: không áp dụng"))
    return pd.DataFrame(out, columns=COLUMNS)
