"""Part 4: EDA and difficulty figures with Vietnamese titles (data/preview/*.png). Called by --preview."""
import json

import numpy as np

PAL = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756", "#72B7B2", "#9D755D", "#BAB0AC"]
GRAY = "#9AA0A6"
SEGN = {0: "Bán lẻ hằng ngày", 1: "Mua sắm online", 2: "Chi tiêu lớn khi đi xa", 3: "Rút tiền đêm", -1: "Bất thường"}
SEGC = {0: PAL[0], 1: PAL[1], 2: PAL[2], 3: PAL[3], -1: "#D62728"}
HOTN = {0: "Trung tâm", 1: "Khu đại học", 2: "Vòng quanh chợ", 3: "Quảng trường chợ", 4: "Ven sông", 5: "Quốc lộ", -1: "Cô lập / nhiễu"}
DOW = ["T2", "T3", "T4", "T5", "T6", "T7", "CN"]
CHN = {"POS": "POS (quẹt thẻ)", "ONLINE": "Online", "ATM": "ATM", "CONTACTLESS": "Chạm không tiếp xúc"}


def setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "figure.dpi": 110, "savefig.dpi": 110, "axes.titlesize": 10.5})
    return plt


def save(plt, fig, name, PREVIEW):
    fig.savefig(PREVIEW / name, bbox_inches="tight")
    plt.close(fig)


def card_clean_df():
    """Card table merged with the clean descriptive truth (EDA after the cleaning of descriptive columns)."""
    import pandas as pd
    import generate_data as gen
    c = pd.read_csv(gen.CARD / "card_txn_5d.csv")
    t = pd.read_csv(gen.RAW.parent / "truth" / "card_txn_5d_clean_descriptive.csv")
    c = c.drop(columns=[x for x in t.columns if x != "txn_id"]).drop_duplicates("txn_id").merge(t, on="txn_id")
    c["ts"] = pd.to_datetime(c.timestamp)
    return c


def fig_hour_dow(plt, PREVIEW):
    c = card_clean_df()
    fig, ax = plt.subplots(1, 4, figsize=(17, 4.2), sharey=True)
    for a, ch in zip(ax, ["POS", "CONTACTLESS", "ONLINE", "ATM"]):
        g = c[c.channel == ch]
        M = np.zeros((7, 24))
        for d, h in zip(g.ts.dt.dayofweek, g.ts.dt.hour):
            M[d, h] += 1
        im = a.imshow(M, aspect="auto", cmap="YlOrRd")
        a.set_title(f"{CHN[ch]} (n = {len(g)})")
        a.set_xlabel("Giờ trong ngày")
        a.set_xticks(range(0, 24, 3))
        a.set_yticks(range(7)); a.set_yticklabels(DOW)
        plt.colorbar(im, ax=a, fraction=.046, pad=.02, label="Số giao dịch")
    ax[0].set_ylabel("Thứ trong tuần")
    fig.suptitle("Hoạt động theo giờ và thứ trong tuần, từng kênh thanh toán (dữ liệu giả lập, 8 tuần từ 2025-03-01)", fontweight="bold")
    save(plt, fig, "eda_hour_dow_channel.png", PREVIEW)


def fig_payday(plt, PREVIEW):
    c = card_clean_df()
    fig, ax = plt.subplots(1, 2, figsize=(14, 3.8), gridspec_kw={"width_ratios": [2.2, 1]})
    d = c.groupby(c.ts.dt.normalize()).size()
    ax[0].bar(d.index, d.values, color=PAL[0], width=.9)
    for day in d.index:
        if day.day == 5:
            ax[0].axvline(day, color=PAL[4], lw=1, ls="--")
            ax[0].text(day, d.max() * 1.03, "ngày 5", color=PAL[4], fontsize=8, ha="center")
        if (day + __import__("datetime").timedelta(days=3)).month != day.month and day.day >= 28:
            ax[0].axvline(day, color=PAL[2], lw=1, ls=":")
    import matplotlib.dates as mdates
    ax[0].xaxis.set_major_locator(mdates.DayLocator(interval=7))
    ax[0].xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
    ax[0].set_ylim(0, d.max() * 1.12)
    ax[0].set_ylabel("Số giao dịch mỗi ngày"); ax[0].set_xlabel("Ngày (dd/mm, năm 2025)")
    ax[0].set_title("Giao dịch theo ngày: đỉnh quanh ngày 5 và những ngày cuối tháng (đường chấm xanh)")
    seg = c.assign(we=c.ts.dt.dayofweek >= 5).groupby("segment_id").we.mean()
    seg = seg.reindex([0, 1, 2, 3, -1])
    ax[1].bar([SEGN[s] for s in seg.index], seg.values * 100, color=[SEGC[s] for s in seg.index])
    ax[1].axhline(2 / 7 * 100, color="k", lw=1, ls="--"); ax[1].text(4.4, 2 / 7 * 100 + 1, "đều (28.6 %)", ha="right", fontsize=8)
    ax[1].set_ylabel("% giao dịch rơi vào cuối tuần"); ax[1].set_title("Cuối tuần theo phân khúc")
    plt.setp(ax[1].get_xticklabels(), rotation=25, ha="right", fontsize=8)
    save(plt, fig, "eda_payday_weekend.png", PREVIEW)


def fig_category(plt, PREVIEW):
    c = card_clean_df()
    import pandas as pd
    cats = pd.read_csv(__import__("generate_data").CARD / "merchant_categories.csv")
    name = dict(zip(cats.category_code, cats.name_vi))
    tab = pd.crosstab(c.segment_id, c.merchant_category, normalize="index").reindex([0, 1, 2, 3, -1])
    order = tab.sum().sort_values(ascending=False).index
    tab = tab[order]
    fig, ax = plt.subplots(figsize=(12, 4.6))
    left = np.zeros(len(tab))
    cm = __import__("matplotlib").colormaps["tab20"]
    for i, k in enumerate(order):
        ax.barh([SEGN[s] for s in tab.index], tab[k].values, left=left, color=cm(i / 20), label=f"{k}: {name[k]}")
        left += tab[k].values
    ax.invert_yaxis(); ax.set_xlabel("Tỷ trọng giao dịch trong phân khúc")
    ax.set_title("Cơ cấu nhóm merchant theo phân khúc: rút tiền đêm gần như chỉ là ATM, mua online là thương mại điện tử")
    ax.legend(ncol=1, fontsize=7.5, frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left")
    save(plt, fig, "eda_category_mix_by_segment.png", PREVIEW)


def fig_corr(plt, PREVIEW):
    import pandas as pd
    import generate_data as gen
    c = pd.read_csv(gen.CARD / "card_txn_5d.csv").drop_duplicates("txn_id").dropna(subset=gen.CARD_COLS)
    X = c[gen.CARD_COLS].copy()
    X["amount_vnd"] = np.log10(X.amount_vnd)
    lab = ["log10(số tiền)", "giờ", "khoảng cách\ntới nhà", "GD 7 ngày", "điểm rủi ro"]
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.8))
    for a, m, t in zip(ax, ("pearson", "spearman"), ("Pearson", "Spearman")):
        C = X.corr(method=m).values
        im = a.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
        a.set_xticks(range(5)); a.set_xticklabels(lab, fontsize=8); a.set_yticks(range(5)); a.set_yticklabels(lab, fontsize=8)
        for i in range(5):
            for j in range(5):
                a.text(j, i, f"{C[i, j]:.2f}", ha="center", va="center", fontsize=8, color="w" if abs(C[i, j]) > .55 else "k")
        a.set_title(f"Tương quan {t}"); a.spines[:].set_visible(False)
    plt.colorbar(im, ax=ax, fraction=.025, pad=.02)
    fig.suptitle("Tương quan giữa 5 đặc trưng gom cụm (sau khi bỏ dòng thiếu và dòng trùng)", fontweight="bold")
    save(plt, fig, "eda_correlation.png", PREVIEW)


def fig_missing_defects(plt, PREVIEW):
    import pandas as pd
    import generate_data as gen
    cat = pd.read_csv(gen.RAW.parent / "truth" / "defect_catalogue_card.csv")
    cat = cat[cat.table == "card_txn_5d"]
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.4), gridspec_kw={"width_ratios": [1, 1.25]})
    b = cat[(cat.scope == "base") & (cat.defect_type == "missing_value")]
    n = 1500
    ax[0].bar(b.column, b.n_rows / n * 100, color=PAL[0])
    for x, v in zip(range(len(b)), b.n_rows):
        ax[0].text(x, v / n * 100 + .03, str(v), ha="center", fontsize=8)
    ax[0].set_ylabel("% ô thiếu"); ax[0].set_title("Ô thiếu ở 5 cột đặc trưng (tổng khoảng 1.5 %)")
    plt.setp(ax[0].get_xticklabels(), rotation=25, ha="right", fontsize=8)
    d = cat[~((cat.scope == "base") & (cat.defect_type == "missing_value"))].copy()
    vi = {"duplicate_row_exact_copy": "Dòng trùng y hệt (cột nền)", "inconsistent_spelling_or_case": "Kênh sai chính tả / chữ hoa",
          "unknown_customer_id": "customer_id không tồn tại", "missing_value": "Thiếu merchant_category",
          "mixed_timestamp_format_dd_mm_yyyy": "timestamp dạng dd/mm/yyyy", "impossible_timestamp": "timestamp không hợp lệ",
          "invalid_district_id": "Mã quận D99 không hợp lệ"}
    d["lab"] = [f"{vi[t]}" for t in d.defect_type]
    d = d.sort_values("n_rows")
    ax[1].barh(d.lab, d.n_rows, color=[PAL[0] if s == "base" else PAL[1] for s in d.scope])
    for y, v in enumerate(d.n_rows):
        ax[1].text(v + 1, y, str(v), va="center", fontsize=8)
    ax[1].set_xlabel("Số dòng bị ảnh hưởng (xanh: cột nền, cam: cột mô tả)")
    ax[1].set_title("Danh mục lỗi dữ liệu còn lại")
    fig.subplots_adjust(wspace=.75)
    fig.suptitle("Chất lượng dữ liệu bảng thẻ: thiếu, trùng và lỗi cột mô tả (bản sạch nằm trong data/truth/)", fontweight="bold")
    save(plt, fig, "eda_missing_duplicates.png", PREVIEW)


def fig_customers(plt, PREVIEW):
    import pandas as pd
    import generate_data as gen
    cu = pd.read_csv(gen.CARD / "customers.csv")
    occ = ["office", "student", "trader", "freelancer", "retired", "other"]
    ovi = dict(office="Văn phòng", student="Sinh viên", trader="Tiểu thương", freelancer="Tự do", retired="Hưu trí", other="Khác")
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    bins = np.arange(18, 90, 4)
    ax[0, 0].hist([cu[cu.occupation_group == o].age for o in occ], bins=bins, stacked=True, color=PAL[:6], label=[ovi[o] for o in occ])
    ax[0, 0].set_xlabel("Tuổi"); ax[0, 0].set_ylabel("Số khách"); ax[0, 0].set_title("Tuổi theo nhóm nghề (sinh viên trẻ, hưu trí lớn tuổi)")
    ax[0, 0].legend(fontsize=8, frameon=False)
    tiers = ["mass", "affluent", "priority"]
    tvi = dict(mass="Phổ thông", affluent="Khá giả", priority="Ưu tiên")
    ax[0, 1].boxplot([np.log10(cu[cu.tier == t].monthly_income_vnd) for t in tiers], tick_labels=[tvi[t] for t in tiers], showfliers=True)
    ax[0, 1].set_ylabel("log10(thu nhập tháng, VND)"); ax[0, 1].set_title("Thu nhập tăng theo hạng khách hàng")
    tab = (cu.groupby(["occupation_group", "tier"]).size().unstack(fill_value=0).reindex(occ)[tiers])
    bottom = np.zeros(6)
    for i, t in enumerate(tiers):
        ax[1, 0].bar([ovi[o] for o in occ], tab[t].values, bottom=bottom, color=PAL[i], label=tvi[t])
        bottom += tab[t].values
    ax[1, 0].legend(fontsize=8, frameon=False); ax[1, 0].set_ylabel("Số khách"); ax[1, 0].set_title("Hạng khách hàng theo nhóm nghề")
    ax[1, 0].tick_params(axis="x", labelsize=8)
    r = cu.groupby("occupation_group")[["digital_active", "has_credit_card"]].mean().reindex(occ) * 100
    x = np.arange(6)
    ax[1, 1].bar(x - .2, r.digital_active, .4, color=PAL[0], label="Dùng ngân hàng số")
    ax[1, 1].bar(x + .2, r.has_credit_card, .4, color=PAL[1], label="Có thẻ tín dụng")
    ax[1, 1].set_xticks(x); ax[1, 1].set_xticklabels([ovi[o] for o in occ], fontsize=8); ax[1, 1].set_ylabel("%")
    ax[1, 1].legend(fontsize=8, frameon=False); ax[1, 1].set_title("Mức dùng số và thẻ tín dụng theo nhóm nghề")
    fig.suptitle(f"Khách hàng giả lập ({len(cu)} khách): nhân khẩu học và tương quan nghề, hạng, thu nhập", fontweight="bold")
    fig.tight_layout()
    save(plt, fig, "eda_customers.png", PREVIEW)


def fig_atm_districts(plt, PREVIEW):
    import pandas as pd
    import generate_data as gen
    import gen_enrich as ge
    a = pd.read_csv(gen.GEO / "atm_txn_2d.csv")
    atms = pd.read_csv(gen.GEO / "atms.csv")
    dis = pd.read_csv(gen.GEO / "districts.csv")
    gx, gy = np.meshgrid(np.arange(0, 32.01, .1), np.arange(0, 24.01, .1))
    idx = ge.district_index(gx.ravel(), gy.ravel()).reshape(gx.shape)
    fig, ax = plt.subplots(figsize=(13.5, 8))
    ax.imshow(idx, origin="lower", extent=(0, 32, 0, 24), cmap=__import__("matplotlib").colors.ListedColormap(
        ["#F3F1EA", "#E7EEF4", "#F4EBE4", "#E8F1EA", "#F2E9F1", "#EDEFE4", "#E4EFEF", "#F1ECE0"]), alpha=.9, aspect="equal", interpolation="nearest")
    ax.contour(gx, gy, idx, levels=np.arange(.5, 7.5, 1), colors="#7A7A7A", linewidths=1, linestyles="--")
    for h in (0, 1, 2, 3, 4, 5, -1):
        g = a[a.hotspot_id == h]
        ax.scatter(g.x_km, g.y_km, s=7 if h >= 0 else 12, c=GRAY if h < 0 else PAL[h], marker="x" if h < 0 else "o",
                   lw=.8, alpha=.8, label=f"{HOTN[h]} ({len(g)})")
    own = atms[atms.atm_network == "own"]; par = atms[atms.atm_network == "partner"]
    ax.scatter(own.x_km, own.y_km, s=70, marker="^", c="k", edgecolors="w", lw=.8, label=f"ATM của ngân hàng ({len(own)})", zorder=5)
    ax.scatter(par.x_km, par.y_km, s=70, marker="^", c="#FFFFFF", edgecolors="k", lw=1.1, label=f"ATM đối tác ({len(par)})", zorder=5)
    pts = a[["x_km", "y_km"]].values
    cell = np.column_stack([gx.ravel(), gy.ravel()])
    for k, (_, r) in enumerate(dis.iterrows()):
        inside = cell[(idx.ravel() == k) & (cell[:, 0] > 1.6) & (cell[:, 0] < 30.4) & (cell[:, 1] > .8) & (cell[:, 1] < 23.2)]
        dmin = np.sqrt(((inside[::3, None, :] - pts[None, ::2, :]) ** 2).sum(-1)).min(1)
        lx, ly = inside[::3][dmin.argmax()]
        ax.text(lx, ly, f"{r.district_id}\n{r['name_vi']}\n({r.area_type})", ha="center", va="center",
                fontsize=8, color="#444", fontweight="bold",
                bbox=dict(boxstyle="round,pad=.2", fc="white", ec="none", alpha=.6))
    ax.set_xlim(0, 32); ax.set_ylim(0, 24); ax.set_xlabel("x (km)"); ax.set_ylabel("y (km)")
    ax.set_title("Thành phố giả lập 32 x 24 km: 8 quận (ô Voronoi quanh tâm quận), 40 máy ATM và 1200 giao dịch")
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.07), ncol=5, fontsize=8, frameon=False)
    save(plt, fig, "eda_atm_map_districts.png", PREVIEW)


def fig_scaling(plt, PREVIEW):
    import pandas as pd
    import generate_data as gen
    c = pd.read_csv(gen.CARD / "card_txn_5d.csv").drop_duplicates("txn_id").dropna(subset=gen.CARD_COLS)
    A = c[gen.CARD_COLS].values
    Z, _, _ = gen.card_z(A)
    lab = ["số tiền (VND)", "giờ trong ngày", "khoảng cách tới nhà (km)", "GD trong 7 ngày", "điểm rủi ro"]
    lab2 = ["log10(số tiền), z-score", "giờ, z-score", "khoảng cách, z-score", "GD 7 ngày, z-score", "điểm rủi ro, z-score"]
    fig, ax = plt.subplots(2, 5, figsize=(18, 6))
    for j in range(5):
        if j == 0:
            ax[0, j].hist(A[:, j], bins=np.logspace(np.log10(A[:, j].min()), np.log10(A[:, j].max()), 41), color=PAL[0])
            ax[0, j].set_xscale("log"); ax[0, j].set_title("số tiền (VND, thang log)", fontsize=9)
        else:
            ax[0, j].hist(A[:, j], bins=40, color=PAL[0])
            ax[0, j].set_title(lab[j], fontsize=9)
        ax[1, j].hist(Z[:, j], bins=40, color=PAL[2])
        ax[1, j].set_title(lab2[j], fontsize=9)
    ax[0, 0].set_ylabel("TRƯỚC: đơn vị gốc"); ax[1, 0].set_ylabel("SAU: log10 + z-score")
    fig.suptitle("Trước và sau tiền xử lý (cột khoảng cách: 15 điểm bất thường 150 đến 900 km làm phình độ lệch chuẩn, các nhóm bình thường bị nén sau z-score)",
                 fontweight="bold", fontsize=10.5)
    fig.tight_layout()
    save(plt, fig, "eda_scaling_before_after.png", PREVIEW)


def _pca2(Z):
    Zc = Z - Z.mean(0)
    u, s, vt = np.linalg.svd(Zc, full_matrices=False)
    return Zc @ vt[:2].T, (s[:2] ** 2 / (s ** 2).sum())


def _best_labels(Z, y, grid, params):
    from sklearn.cluster import DBSCAN
    e = float(params["eps"]); mp = int(params["mp"])
    return DBSCAN(eps=e, min_samples=mp).fit_predict(Z)


def _parse(p):
    out = {}
    for kv in p.split(","):
        k, v = kv.strip().split("=")
        out[k] = v
    return dict(eps=out["eps"], mp=out["MinPts"])


def fig_difficulty(plt, PREVIEW, rep, data):
    """data[fam][lvl] = (Z, y). Two files: difficulty_atm.png, difficulty_card.png."""
    from sklearn.metrics import adjusted_rand_score as ari
    for fam, ttl in (("atm", "Bản đồ ATM"), ("card", "Giao dịch thẻ (chiếu PCA 2D của 5 đặc trưng đã chuẩn hóa)")):
        fig, ax = plt.subplots(2, 3, figsize=(17, 9.2))
        for j, lvl in enumerate(("easy", "medium", "hard")):
            Z, y = data[fam][lvl]
            r = rep[fam][lvl]["results"]["DBSCAN"]
            prm = _parse(r[1])
            lab = _best_labels(Z, y, None, prm)
            if fam == "atm":
                XY = Z / 1000.0; xl, yl = "x (km)", "y (km)"
            else:
                XY, ev = _pca2(Z); xl, yl = f"PC1 ({ev[0] * 100:.0f} % phương sai)", f"PC2 ({ev[1] * 100:.0f} % phương sai)"
            for row, labs, name in ((0, y, "Nhãn thật"), (1, lab, f"DBSCAN tốt nhất ({r[1]}), ARI {ari(y, lab):.2f}")):
                a = ax[row, j]
                for l in sorted(set(labs.tolist())):
                    m = labs == l
                    if l < 0:
                        a.scatter(XY[m, 0], XY[m, 1], s=7, c=GRAY, marker="x", lw=.6, alpha=.7, label="nhiễu / -1")
                    else:
                        a.scatter(XY[m, 0], XY[m, 1], s=5, c=PAL[l % 8], alpha=.8)
                a.set_title(f"{ {'easy': 'Dễ', 'medium': 'Vừa', 'hard': 'Khó'}[lvl] }: {name}", fontsize=9.5)
                if fam == "atm":
                    a.set_aspect("equal")
                if j == 0:
                    a.set_ylabel(yl)
                if row == 1:
                    a.set_xlabel(xl)
        fig.suptitle(f"Ba mức độ khó của dữ liệu: {ttl} (hàng trên: nhãn thật, hàng dưới: DBSCAN với tham số tốt nhất theo ARI)", fontweight="bold")
        fig.tight_layout()
        save(plt, fig, f"difficulty_{fam}.png", PREVIEW)


def fig_difficulty_bars(plt, PREVIEW, rep):
    fig, ax = plt.subplots(1, 2, figsize=(15, 4.6), sharey=True)
    algs = ["DBSCAN", "HDBSCAN", "OPTICS", "KMeans (true k)", "Ward HAC (true k)"]
    avi = {"DBSCAN": "DBSCAN", "HDBSCAN": "HDBSCAN", "OPTICS": "OPTICS", "KMeans (true k)": "KMeans (k thật)", "Ward HAC (true k)": "Ward (k thật)"}
    for a, fam, t in zip(ax, ("atm", "card"), ("Bản đồ ATM (2D)", "Giao dịch thẻ (5D)")):
        w = .27
        for i, lvl in enumerate(("easy", "medium", "hard")):
            v = [rep[fam][lvl]["results"][k][0] for k in algs]
            b = a.bar(np.arange(len(algs)) + (i - 1) * w, v, w, color=[PAL[2], PAL[1], PAL[4]][i], label={"easy": "Dễ", "medium": "Vừa", "hard": "Khó"}[lvl])
            for x, y in zip(np.arange(len(algs)) + (i - 1) * w, v):
                a.text(x, max(y, 0) + .01, f"{y:.2f}", ha="center", fontsize=6.5, rotation=90)
        a.set_xticks(range(len(algs))); a.set_xticklabels([avi[k] for k in algs], fontsize=8.5)
        a.set_title(t); a.set_ylim(0, 1.12)
    ax[0].set_ylabel("ARI tốt nhất so với nhãn thật (oracle)"); ax[0].legend(frameon=False, ncol=3)
    fig.suptitle("Dữ liệu càng khó thì ARI tốt nhất càng thấp; HDBSCAN thắng DBSCAN ở bản đồ ATM mật độ không đều", fontweight="bold")
    save(plt, fig, "difficulty_ari_bars.png", PREVIEW)


def fig_edge_coverage(plt, PREVIEW):
    import generate_data as gen
    base = gen.RAW / "toy_edge"
    cases = ["Noise", "Creation", "Absorption", "Merge", "Removal", "Reduction", "PotentialSplit_NonSplit", "PotentialSplit_Split", "DeleteNoise"]
    ids, M, tests = [], [], []
    for d in sorted(p for p in base.iterdir() if p.is_dir()):
        e = json.loads((d / "expected.json").read_text(encoding="utf-8"))
        ids.append(d.name)
        M.append([sum(s["case"] == c for s in e["steps"]) for c in cases])
        tests.append([int(e["tests"]["insert"]), int(e["tests"]["delete"]), 1, int(len(e["ambiguous_to_ignore"]) > 0), int(len(e["ties_at_exactly_eps"]) > 0),
                      int(e["dims"] > 2), int(e["n_updates"] > 1)])
    M, tests = np.array(M), np.array(tests)
    fig, ax = plt.subplots(1, 2, figsize=(15, 7.4), gridspec_kw={"width_ratios": [1.15, 1]}, sharey=True)
    ax[0].imshow((M > 0).astype(int), cmap="Blues", vmin=0, vmax=1.6, aspect="auto")
    for i in range(len(ids)):
        for j in range(len(cases)):
            if M[i, j]:
                ax[0].text(j, i, str(M[i, j]), ha="center", va="center", color="w", fontsize=9)
    ax[0].set_xticks(range(len(cases))); ax[0].set_xticklabels(cases, rotation=40, ha="right", fontsize=8)
    ax[0].set_yticks(range(len(ids))); ax[0].set_yticklabels(ids, fontsize=8)
    ax[0].set_title("Mỗi kịch bản tạo ra case nào (số bước)")
    tl = ["TV2 chèn", "TV3 xóa", "TV5 tương đương", "điểm biên mơ hồ", "khoảng cách = Eps", "4 chiều", "nhiều bước"]
    ax[1].imshow(tests, cmap="Greens", vmin=0, vmax=1.6, aspect="auto")
    for i in range(len(ids)):
        for j in range(len(tl)):
            if tests[i, j]:
                ax[1].text(j, i, "x", ha="center", va="center", color="w", fontsize=9)
    ax[1].set_xticks(range(len(tl))); ax[1].set_xticklabels(tl, rotation=40, ha="right", fontsize=8)
    ax[1].set_title("Kịch bản kiểm tra mã của ai và tính chất đặc biệt")
    fig.suptitle(f"Độ phủ của {len(ids)} kịch bản biên toy_edge: đủ 9 loại case của thuật toán gia tăng", fontweight="bold")
    fig.tight_layout()
    save(plt, fig, "toy_edge_coverage.png", PREVIEW)


def fig_hard_stream(plt, PREVIEW):
    import pandas as pd
    import generate_data as gen
    u = pd.read_csv(gen.CARD / "card_stream_hard" / "updates.csv")
    cases = ["Noise", "Creation", "Absorption", "Merge", "Removal", "Reduction", "PotentialSplit_NonSplit", "PotentialSplit_Split", "DeleteNoise"]
    col = dict(zip(cases, ["#9AA0A6", "#54A24B", "#4C78A8", "#B279A2", "#E45756", "#F58518", "#72B7B2", "#8C564B", "#C8C8C8"]))
    fig, ax = plt.subplots(figsize=(13, 4.8))
    for k, c in enumerate(cases):
        m = u.verified_case == c
        ax.scatter(u.step[m], [k] * int(m.sum()), s=34, c=col[c], marker="s" if k < 4 else "o")
        pm = m & u.planted_case.notna()
        ax.scatter(u.step[pm], [k] * int(pm.sum()), s=130, facecolors="none", edgecolors="k", lw=1.4)
    ax.set_yticks(range(len(cases))); ax.set_yticklabels([f"{c} ({int((u.verified_case == c).sum())})" for c in cases], fontsize=8)
    ax.invert_yaxis(); ax.axhline(3.5, color="#DDD", lw=1)
    ax.set_xlabel("bước (1 bước = 1 giờ); vòng tròn đen: sự kiện cài sẵn")
    ax.set_title("Stream thẻ khó: Merge và Split xuất hiện khi các dải cầu nối vào rồi rời cửa sổ trượt")
    save(plt, fig, "stream_hard_timeline.png", PREVIEW)


def write_all(PREVIEW, rep, data):
    plt = setup()
    PREVIEW.mkdir(exist_ok=True)
    for f in (fig_hour_dow, fig_payday, fig_category, fig_corr, fig_missing_defects, fig_customers, fig_atm_districts,
              fig_scaling, fig_edge_coverage, fig_hard_stream):
        f(plt, PREVIEW)
    fig_difficulty(plt, PREVIEW, rep, data)
    fig_difficulty_bars(plt, PREVIEW, rep)
