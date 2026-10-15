"""Hàm làm sạch trên bảng nhỏ tự tạo, rồi điểm số (scorecard) trên dữ liệu thật.

Ngưỡng scorecard dưới đây là ngưỡng em coi là trung thực cho từng loại lỗi, đặt TRƯỚC khi nhìn kết quả
theo nguyên tắc "sửa được thì phải đúng gần như tuyệt đối, đoán thì chỉ đòi hỏi tốt hơn trung vị chung".
Không chỉnh hàm làm sạch để vừa đáp án.
"""
import ast
import json
import pathlib

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import cleaning as cl
from src.preprocessing.scaler import ManualScaler
from src.preprocessing.scorecard import score_cleaning
from .tv4_paths import RAW, TRUTH

CARD = RAW / "bank_card"
FILES = {"card_txn_5d": "card_stream", "card_txn_5d_hard": "card_stream_hard"}


# ---------------------------------------------------------------- bảng nhỏ tự tạo
def test_normalise_channel():
    s = pd.Series([" pos", "Contact-less", "ATM ", "online", "Online", "xyz", None, "CONTACTLESS"])
    out = cl.normalise_channel(s)
    assert out.tolist()[:5] == ["POS", "CONTACTLESS", "ATM", "ONLINE", "ONLINE"]
    assert pd.isna(out[5]) and pd.isna(out[6]) and out[7] == "CONTACTLESS"


def test_parse_timestamp_formats_and_impossible():
    s = pd.Series(["2025-03-06 13:13:48", "03/04/2025 15:37:48", "31/04/2025 10:00:00", "2025-13-02 09:00:00",
                   "2025-03-14 24:30:00", "2025-03-09 25:07:00", "2025-03-22 12:75:00", None, "garbage"])
    r = cl.parse_timestamp(s)
    assert r["ts_status"].tolist() == ["iso", "dmy", "impossible", "impossible", "impossible", "impossible",
                                       "impossible", "missing", "impossible"]
    assert r["ts"][0] == pd.Timestamp("2025-03-06 13:13:48")
    assert r["ts"][1] == pd.Timestamp("2025-04-03 15:37:48")        # dd/mm, không phải mm/dd
    assert r["ts"][[2, 3, 4, 5, 6, 7, 8]].isna().all()


def test_hour_from_timestamp_and_repair():
    ts = pd.Series(pd.to_datetime(["2025-03-06 13:13:48", "2025-03-06 01:30:00", None]))
    assert cl.hour_from_timestamp(ts).iloc[:2].tolist() == [13.23, 1.5]
    df = pd.DataFrame({"hour_of_day": [np.nan, 5.0, np.nan], "ts": ts})
    out = cl.repair_hour_of_day(df)
    assert out["hour_of_day"].iloc[0] == 13.23 and out["hour_of_day"].iloc[1] == 5.0     # không ghi đè ô có sẵn
    assert np.isnan(out["hour_of_day"].iloc[2])                                        # timestamp hỏng: để cho trung vị
    assert out["hour_from_ts"].tolist() == [True, False, False]
    assert df["hour_of_day"].isna().sum() == 2                                         # không sửa bảng gốc


def test_drop_exact_duplicates_only_exact():
    df = pd.DataFrame({"txn_id": ["a", "a", "b", "b"], "x": [1, 1, 2, 3]})
    out = cl.drop_exact_duplicates(df)
    assert out["txn_id"].tolist() == ["a", "b", "b"] and out["x"].tolist() == [1, 2, 3]


def test_medians_learned_on_d0_only_and_flags():
    df = pd.DataFrame({"amount_vnd": [10.0, 20.0, 30.0, 1e9, np.nan, np.nan],
                       "hour_of_day": [1.0, 2.0, 3.0, 20.0, 21.0, np.nan],
                       "distance_from_home_km": 1.0, "txns_last_7d": 2.0, "merchant_risk_score": 0.5,
                       "ts": pd.to_datetime(["2025-03-01", "2025-03-02", "2025-03-03", "2025-03-04", "2025-03-05", None])})
    mask = cl.select_d0(df, n=3)
    assert mask.tolist() == [True, True, True, False, False, False]
    med = cl.fit_medians(df[mask])
    assert med["amount_vnd"] == 20.0 and med["hour_of_day"] == 2.0           # dòng 1e9 ngoài D0 không ảnh hưởng
    out = cl.impute_median(df, med)
    assert out["amount_vnd"].tolist()[4:] == [20.0, 20.0] and out["imp_amount_vnd"].tolist()[3:] == [False, True, True]
    assert out["hour_of_day"].iloc[5] == 2.0 and not out[cl.NUMERIC_COLS].isna().any().any()


def test_select_d0_orders_by_time_and_puts_nat_last():
    df = pd.DataFrame({"ts": pd.to_datetime(["2025-03-05", None, "2025-03-01", "2025-03-03"])})
    assert cl.select_d0(df, n=2).tolist() == [False, False, True, True]
    assert cl.select_d0(df, n=3).tolist() == [True, False, True, True]


def test_group_median_hierarchy_and_min_group():
    """(kênh x nhóm hàng) -> kênh -> chung; nhóm dưới min_group dòng bị bỏ; UNKNOWN không là một nhóm."""
    n = 6
    d0 = pd.DataFrame({"channel": ["POS"] * n + ["ATM"] * n + ["ONLINE"] * 2,
                       "merchant_category": ["GRO"] * n + ["ATM"] * 3 + ["UNKNOWN"] * 3 + ["ECO"] * 2,
                       "v": [1.0] * n + [10.0] * 3 + [30.0] * 3 + [50.0, 50.0]})
    levels = (("channel", "merchant_category"), ("channel",))
    gm = cl.fit_group_medians(d0, levels, cols=["v"], min_group=3)
    assert gm[0]["v"].dropna().to_dict() == {("POS", "GRO"): 1.0, ("ATM", "ATM"): 10.0}     # ATM|UNKNOWN và ONLINE|ECO bỏ
    assert gm[1]["v"].dropna().to_dict() == {"POS": 1.0, "ATM": 20.0}                       # ONLINE chỉ có 2 dòng: bị bỏ
    new = pd.DataFrame({"channel": ["POS", "ATM", "ATM", "ONLINE", "ONLINE"],
                        "merchant_category": ["GRO", "ATM", "UNKNOWN", "ECO", "ECO"], "v": [np.nan] * 5})
    out = cl.impute_group_median(new, gm, {"v": 99.0}, levels)
    assert out["v"].tolist() == [1.0, 10.0, 20.0, 99.0, 99.0]
    assert out["imp_v"].all()


def test_flags_unknown_customer_and_invalid_district():
    customers = pd.DataFrame({"customer_id": ["C1", "C2"]})
    districts = pd.DataFrame({"district_id": ["D01", "D02"]})
    df = pd.DataFrame({"customer_id": ["C1", "UNKNOWN", "C9137"], "txn_district_id": ["D01", "D99", None],
                       "is_foreign": [0, 0, 1]})
    assert cl.flag_unknown_customers(df, customers).tolist() == [False, True, True]
    assert cl.flag_invalid_districts(df, districts).tolist() == [False, True, False]   # trống + foreign là hợp lệ


def test_fill_category_and_log_amount():
    df = pd.DataFrame({"merchant_category": ["GRO", None], "amount_vnd": [1000.0, np.nan]})
    out = cl.add_log_amount(cl.fill_missing_category(df))
    assert out["merchant_category"].tolist() == ["GRO", "UNKNOWN"] and out["category_missing"].tolist() == [False, True]
    assert out["log10_amount_vnd"].iloc[0] == 3.0 and np.isnan(out["log10_amount_vnd"].iloc[1])


def test_cleaning_module_never_reads_files_or_truth():
    """cleaning.py chỉ nhận DataFrame: không gọi hàm đọc file và chỉ import numpy, pandas."""
    tree = ast.parse(pathlib.Path(cl.__file__).read_text(encoding="utf-8"))
    forbidden = {"read_csv", "open", "read_json", "read_parquet", "read_excel", "Path"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            name = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else ""
            assert name not in forbidden, f"cleaning.py gọi {name}"
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mods = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module]
            assert all(m.split(".")[0] in ("numpy", "pandas") for m in mods), mods


# ---------------------------------------------------------------- dữ liệu thật
@pytest.fixture(scope="module", params=list(FILES))
def card(request):
    t = request.param
    raw = pd.read_csv(CARD / f"{t}.csv")
    customers = pd.read_csv(CARD / "customers.csv")
    districts = pd.read_csv(RAW / "bank_geo" / "districts.csv")
    clean, med = cl.clean_card_table(raw, customers, districts)                    # mặc định: trung vị theo nhóm
    clean_g, med_g = cl.clean_card_table(raw, customers, districts, impute_by=None)  # chỉ trung vị chung
    tb = pd.read_csv(TRUTH / f"{t}_clean_base.csv")
    td = pd.read_csv(TRUTH / f"{t}_clean_descriptive.csv")
    return dict(name=t, raw=raw, customers=customers, districts=districts, clean=clean, med=med,
                clean_g=clean_g, med_g=med_g, tb=tb, td=td)


def test_quality_report_matches_defect_catalogue(card):
    cat = pd.read_csv(TRUTH / "defect_catalogue_card.csv")
    cat = cat[cat["table"] == card["name"]].set_index(["defect_type", "column"])["n_rows"]
    rep = cl.quality_report(card["raw"], card["customers"], card["districts"]).set_index(["check", "column"])["n_rows"]
    for key, n in cat.to_dict().items():
        defect, col = key
        assert rep[(defect, col)] == n, key
    assert rep[("missing_district_not_foreign", "txn_district_id")] == 0


def test_no_rows_dropped_beyond_exact_duplicates_and_no_iqr(card):
    clean, tb = card["clean"], card["tb"]
    assert len(clean) == len(tb) == card["raw"].drop_duplicates().shape[0]
    assert clean["txn_id"].is_unique and set(clean["txn_id"]) == set(tb["txn_id"])
    far = tb.set_index("txn_id")["distance_from_home_km"].nlargest(15).index           # các điểm xa nhà (bất thường)
    assert set(far) <= set(clean["txn_id"])
    assert clean[cl.FEATURE_COLS].notna().all().all()


def test_scorecard_thresholds(card):
    sc = score_cleaning(card["name"], card["raw"], card["clean"], card["tb"], card["td"], card["clean_g"], card["med_g"])
    got = sc.set_index(["defect", "column", "metric"])["value"]
    assert got[("duplicate_row_exact_copy", "(whole row)", "share_removed")] == 1.0
    assert got[("inconsistent_spelling_or_case", "channel", "accuracy_on_dirty_rows")] >= 0.99
    assert got[("mixed_timestamp_format_dd_mm_yyyy", "timestamp", "share_recovered_exactly")] >= 0.99
    assert got[("impossible_timestamp", "timestamp", "flag_recall")] == 1.0
    assert got[("valid_timestamp_untouched", "timestamp", "share_exact")] == 1.0
    assert got[("missing_value", "hour_of_day (từ timestamp)", "max_abs_error")] <= 0.01
    assert got[("unknown_customer_id", "customer_id", "flag_recall")] == 1.0
    assert got[("invalid_district_id", "txn_district_id", "flag_recall")] == 1.0
    assert got[("missing_value", "merchant_category", "flag_recall")] == 1.0
    # giờ thiếu: sửa từ timestamp phải tốt hơn hẳn điền trung vị
    hr = sc[(sc["column"] == "hour_of_day") & (sc["metric"] == "median_abs_error")].iloc[0]
    assert hr["value"] <= 0.01 < hr["global_median_error"]


def test_precision_of_flags(card):
    clean = card["clean"].set_index("txn_id")
    raw = card["raw"].drop_duplicates().set_index("txn_id").loc[clean.index]
    td = card["td"].set_index("txn_id").loc[clean.index]
    assert (clean["unknown_customer"] == (raw["customer_id"] != td["customer_id"])).all()
    bad_dist = raw["txn_district_id"].notna() & (raw["txn_district_id"] != td["txn_district_id"])
    assert (clean["invalid_district"] == bad_dist).all()
    assert (clean["ts_status"] == "impossible").sum() == 6


def test_global_option_still_fills_d0_median(card):
    clean, tb = card["clean_g"].set_index("txn_id"), card["tb"].set_index("txn_id")
    for c, m in card["med_g"].items():
        imp = clean["imp_" + c]
        if c == "hour_of_day":
            imp = imp & ~clean["hour_from_ts"]
        assert (clean.loc[imp, c] == m).all()
    assert clean[cl.NUMERIC_COLS].notna().all().all()


def test_default_imputation_uses_d0_only(card):
    """Đổi giá trị ngoài D0 không được làm đổi giá trị điền (mọi tham số học trên D0)."""
    raw = card["raw"].copy()
    base, _ = cl.clean_card_table(raw, card["customers"], card["districts"])
    late = base.loc[~base["is_d0"], "txn_id"]
    raw.loc[raw["txn_id"].isin(late), "txns_last_7d"] = raw.loc[raw["txn_id"].isin(late), "txns_last_7d"].where(
        raw["txns_last_7d"].isna(), 999.0)
    shifted, _ = cl.clean_card_table(raw, card["customers"], card["districts"])
    miss = base["imp_txns_last_7d"]
    assert (base.loc[miss, "txns_last_7d"].to_numpy() == shifted.loc[miss, "txns_last_7d"].to_numpy()).all()


def test_group_median_gain_over_global_is_clear_and_errors_bounded(card):
    """Trung vị theo nhóm phải tốt hơn rõ trung vị chung (>= 1.5 lần) ở bốn cột, và sai số tuyệt đối đủ nhỏ.
    Ngưỡng đặt trước khi chạy: 1.5 lần là mức 'rõ' thấp nhất em chấp nhận, không chỉnh theo kết quả."""
    sc = score_cleaning(card["name"], card["raw"], card["clean"], card["tb"], card["td"], card["clean_g"], card["med_g"])
    sc = sc[(sc["defect"] == "missing_value") & sc["metric"].isin(["median_abs_error", "mean_abs_error"])]
    for col in ["amount_vnd", "distance_from_home_km", "txns_last_7d", "merchant_risk_score"]:
        for metric in ["median_abs_error", "mean_abs_error"]:
            r = sc[(sc["column"] == col) & (sc["metric"] == metric)].iloc[0]
            assert r["gain_x"] >= 1.5, (col, metric, r["value"], r["global_median_error"])
    sd = card["tb"].std(numeric_only=True)
    for col, bound in [("txns_last_7d", 0.30), ("merchant_risk_score", 0.30), ("distance_from_home_km", 0.10)]:
        r = sc[(sc["column"] == col) & (sc["metric"] == "median_abs_error")].iloc[0]
        assert r["value"] / sd[col] <= bound, col


@pytest.mark.parametrize("name", list(FILES))
def test_scaler_reproduces_frozen_scaling_from_truth(name):
    """Ngoài phạm vi cleaning: ManualScaler fit trên bảng thật chưa bị đục ô cho đúng hằng số của stream."""
    tb = pd.read_csv(TRUTH / f"{name}_clean_base.csv")
    x = tb[cl.NUMERIC_COLS].to_numpy(float)
    x[:, 0] = np.log10(x[:, 0])
    fs = json.loads((CARD / FILES[name] / "params.json").read_text(encoding="utf-8"))["frozen_scaling"]
    sc = ManualScaler().fit(x)
    assert np.abs(sc.center_ - fs["mean"]).max() < 1e-6 and np.abs(sc.scale_ - fs["std"]).max() < 1e-6


# ---------------------------------------------------------------- khoảng trống mà review độc lập chỉ ra
def test_normalise_channel_removes_inner_spaces():
    """Docstring hứa 'bỏ khoảng trắng': khoảng trắng GIỮA chữ cũng phải bị bỏ (dữ liệu hiện tại không có, nên cần test riêng)."""
    out = cl.normalise_channel(pd.Series(["Contact less", " on line ", "P O S", "A TM"]))
    assert out.tolist() == ["CONTACTLESS", "ONLINE", "POS", "ATM"]


def test_log_amount_of_nonpositive_is_nan_not_minus_inf():
    out = cl.add_log_amount(pd.DataFrame({"amount_vnd": [0.0, -5.0, 1000.0]}))["log10_amount_vnd"]
    assert out.isna().tolist() == [True, True, False] and not np.isinf(out).any()


def _mini_table(n_d0=6):
    """12 giao dịch theo giờ: 6 dòng đầu (D0) khoảng cách 1..6 km, 6 dòng sau 100..600 km; vài ô thiếu ở phần SAU D0."""
    ts = pd.date_range("2025-03-01 00:00:00", periods=12, freq="h")
    raw = pd.DataFrame({
        "txn_id": [f"T{i:02d}" for i in range(12)], "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
        "amount_vnd": [10000.0 * (i + 1) for i in range(12)], "hour_of_day": [float(t.hour) for t in ts],
        "distance_from_home_km": [1.0, 2, 3, 4, 5, 6, 100, 200, 300, 400, 500, 600],
        "txns_last_7d": [1.0] * 12, "merchant_risk_score": [0.1] * 12,
        "customer_id": ["C1"] * 12, "channel": ["POS"] * 10 + ["ATM", "ATM"], "merchant_category": ["GRO"] * 10 + ["ATM", "ATM"],
        "txn_district_id": ["D01"] * 12, "is_foreign": [0] * 12, "segment_id": [0] * 12})
    raw.loc[8, "distance_from_home_km"] = np.nan                  # POS + GRO, nhóm có 6 dòng D0
    raw.loc[10, "distance_from_home_km"] = np.nan                 # ATM: không có dòng D0 nào, phải rơi xuống trung vị chung
    return raw, pd.DataFrame({"customer_id": ["C1"]}), pd.DataFrame({"district_id": ["D01"]})


@pytest.mark.parametrize("impute_by", [cl.IMPUTE_LEVELS, None], ids=["group", "global"])
def test_all_medians_are_learned_on_d0_rows_only(impute_by):
    """D0 = 6 giao dịch sớm nhất, khoảng cách 1..6 (trung vị 3.5). Phần sau D0 có giá trị 100 đến 600: nếu tham số học trên cả bảng
    thì ô thiếu bị điền khoảng 250 đến 350 thay vì 3.5."""
    raw, customers, districts = _mini_table()
    clean, medians = cl.clean_card_table(raw, customers, districts, n_d0=6, impute_by=impute_by)
    assert medians["distance_from_home_km"] == 3.5 and medians["amount_vnd"] == 35000.0
    assert clean.loc[clean["txn_id"].isin(["T08", "T10"]), "distance_from_home_km"].tolist() == [3.5, 3.5]
    assert clean["is_d0"].sum() == 6 and clean.loc[clean["is_d0"], "txn_id"].tolist() == [f"T{i:02d}" for i in range(6)]


def test_group_median_drops_groups_below_min_group():
    """MIN_GROUP = 5: nhóm có 4 giá trị quan sát được thì bỏ (NaN), nhóm có 5 thì giữ; giá trị mặc định lấy từ hằng số của module."""
    assert cl.MIN_GROUP == 5
    df = pd.DataFrame({"channel": ["A"] * 4 + ["B"] * 5, "x": [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0, 50.0]})
    (tbl,) = cl.fit_group_medians(df, levels=(("channel",),), cols=["x"])        # min_group mặc định
    assert np.isnan(tbl.loc["A", "x"]) and tbl.loc["B", "x"] == 30.0
    fallback = cl.impute_group_median(pd.DataFrame({"channel": ["A", "B"], "x": [np.nan, np.nan]}), [tbl], {"x": -1.0},
                                      levels=(("channel",),))
    assert fallback["x"].tolist() == [-1.0, 30.0]                                  # A rơi xuống trung vị chung


def _precision(note):
    return float(note.split("precision ")[1].split(";")[0])


def test_scorecard_metrics_are_computed_on_the_right_rows(card):
    """Tính độc lập từ bảng thô, đáp án và bảng sạch; thủ phạm trước đây: độ chính xác kênh tính trên mọi dòng thay vì dòng bẩn,
    precision của cờ khách lạ không dùng cờ thật. Cố ý làm hỏng bảng sạch để hai thước đo khác nhau."""
    raw, tb, td = card["raw"], card["tb"], card["td"]
    clean = card["clean"].copy()
    ids = clean["txn_id"].to_numpy()
    raw_u = raw.drop_duplicates().set_index("txn_id").loc[ids]
    truth = td.set_index("txn_id").loc[ids]
    dirty = (raw_u["channel"] != truth["channel"]).to_numpy()
    clean_rows = np.flatnonzero(~dirty)[:7]
    dirty_rows = np.flatnonzero(dirty)[:3]
    clean.loc[clean.index[clean_rows], "channel"] = "WRONG"           # hỏng 7 dòng vốn sạch
    clean.loc[clean.index[dirty_rows], "channel"] = "WRONG"           # và 3 dòng bẩn
    clean.loc[clean.index[[1, 2]], "unknown_customer"] = True         # hai cờ khách lạ sai (dương tính giả)
    sc = score_cleaning(card["name"], card["raw"], clean, tb, td, card["clean_g"], card["med_g"]).set_index(["defect", "column", "metric"])
    ok = (clean["channel"].to_numpy() == truth["channel"].to_numpy())
    expect_dirty_acc = ok[dirty].mean()
    got = sc.loc[("inconsistent_spelling_or_case", "channel", "accuracy_on_dirty_rows")]
    assert got["value"] == pytest.approx(expect_dirty_acc) and expect_dirty_acc != pytest.approx(ok.mean())
    assert f"độ chính xác trên mọi dòng {ok.mean():.4f}" in got["note"]
    flag, true_unknown = clean["unknown_customer"].to_numpy(), (raw_u["customer_id"] != truth["customer_id"]).to_numpy()
    rec = sc.loc[("unknown_customer_id", "customer_id", "flag_recall")]
    assert rec["value"] == pytest.approx((flag & true_unknown).sum() / true_unknown.sum())
    assert _precision(rec["note"]) == pytest.approx((flag & true_unknown).sum() / flag.sum(), abs=5e-5)
    assert _precision(rec["note"]) < 1.0                                # có dương tính giả thật nên precision phải < 1
