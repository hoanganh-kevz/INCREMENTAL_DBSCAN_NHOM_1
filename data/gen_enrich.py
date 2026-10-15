"""Part 1 enrichment: dimension tables and DESCRIPTIVE columns (never used for clustering).

All randomness comes from separate generators seeded SEED + offset, so the base draws of
generate_data.py (seeds 42..45) are never reshuffled. Everything here is SIMULATED.
"""
import csv
from datetime import datetime, timedelta

import numpy as np

SEED = 42
OFF = dict(customers=10, card_desc=11, atm_desc=12, atms=13, dirt=14, stream_desc=15, truth_mix=16,
           card_desc_hard=31, dirt_hard=32, card_desc_medium=33, dirt_medium=34, stream_desc_hard=36)


def rng_for(name, extra=0):
    return np.random.default_rng(SEED + OFF[name] + extra)


T0 = datetime(2025, 3, 1)
N_DAYS = 56                                           # 2025-03-01 .. 2025-04-25 (8 weeks)
DATES = [T0 + timedelta(days=i) for i in range(N_DAYS)]
HOLIDAY = {datetime(2025, 4, 5), datetime(2025, 4, 6), datetime(2025, 4, 7)}   # Hung Kings long weekend

# ------------------------------------------------------------------ districts (Voronoi tiling)
DISTRICTS = [   # id, name_vi, name_en, area_type, centroid x, y (km)
    ("D01", "Trung Tâm", "Central", "CBD", 8.0, 16.0),
    ("D02", "Đại Học", "University", "university", 26.0, 18.0),
    ("D03", "Chợ Đông", "East Market", "market", 22.0, 7.0),
    ("D04", "Ven Sông", "Riverside", "riverside", 11.0, 5.0),
    ("D05", "Quốc Lộ", "Highway", "highway", 20.0, 16.5),
    ("D06", "Dân Cư Hòa Bình", "Hoa Binh Residential", "residential", 15.0, 11.0),
    ("D07", "Ngoại Ô Đông Nam", "South-East Suburb", "suburb", 29.0, 4.0),
    ("D08", "Ngoại Ô Tây Bắc", "North-West Suburb", "suburb", 3.0, 22.0),
]
DIST_IDS = [d[0] for d in DISTRICTS]
SITES = np.array([[d[4], d[5]] for d in DISTRICTS])
CITY_W, CITY_H = 32.0, 24.0


def district_index(x, y):
    """Index (0..7) of the nearest centroid; the tiling is the Voronoi diagram of the centroids."""
    P = np.column_stack([np.atleast_1d(x), np.atleast_1d(y)]).astype(float)
    d2 = ((P[:, None, :] - SITES[None, :, :]) ** 2).sum(-1)
    return d2.argmin(1)


def district_ids(x, y):
    return [DIST_IDS[i] for i in district_index(x, y)]


def district_rows():
    g = np.stack(np.meshgrid(np.arange(0.05, CITY_W, 0.1), np.arange(0.05, CITY_H, 0.1)), -1).reshape(-1, 2)
    idx = district_index(g[:, 0], g[:, 1])
    rows = []
    for k, d in enumerate(DISTRICTS):
        rows.append([d[0], d[1], d[2], d[3], f"{d[4]:.1f}", f"{d[5]:.1f}", f"{(idx == k).sum() * 0.01:.2f}"])
    return rows


DISTRICT_COLS = ["district_id", "name_vi", "name_en", "area_type", "centroid_x_km", "centroid_y_km", "area_km2"]

# ------------------------------------------------------------------ merchant categories
CATEGORIES = [   # code, name_en, name_vi, base_risk_level
    ("GRO", "grocery", "Tạp hóa và siêu thị", "low"),
    ("DIN", "dining", "Ăn uống", "low"),
    ("FUE", "fuel", "Xăng dầu", "low"),
    ("UTL", "utilities", "Điện nước và viễn thông", "low"),
    ("HLT", "health", "Y tế và nhà thuốc", "low"),
    ("EDU", "education", "Giáo dục", "low"),
    ("TRN", "ride_hailing_transport", "Di chuyển và gọi xe", "low"),
    ("ENT", "entertainment", "Giải trí", "medium"),
    ("ECO", "ecommerce", "Thương mại điện tử", "medium"),
    ("FSH", "fashion_retail", "Thời trang và bán lẻ", "medium"),
    ("TRV", "travel_lodging", "Du lịch và lưu trú", "medium"),
    ("GAM", "gaming_digital_goods", "Game và hàng hóa số", "high"),
    ("ATM", "cash_atm", "Rút tiền mặt tại ATM", "high"),
    ("LUX", "jewelry_luxury", "Trang sức và hàng xa xỉ", "high"),
]
CAT_CODES = [c[0] for c in CATEGORIES]
CATEGORY_COLS = ["category_code", "name_en", "name_vi", "base_risk_level"]
CHANNELS = ["POS", "ONLINE", "ATM", "CONTACTLESS"]
CHAN_OK = {"POS": {"GRO", "DIN", "FUE", "HLT", "TRN", "FSH", "ENT", "TRV", "LUX", "EDU"},
           "CONTACTLESS": {"GRO", "DIN", "FUE", "HLT", "TRN", "FSH", "ENT", "TRV", "LUX", "EDU"},
           "ONLINE": {"ECO", "GAM", "FSH", "ENT", "DIN", "TRV", "EDU", "UTL", "GRO", "LUX", "TRN", "HLT"},
           "ATM": {"ATM"}}
SEG_CHANNEL = {0: [.50, .05, .05, .40], 1: [.03, .94, .00, .03], 2: [.45, .30, .05, .20],
               3: [.08, .00, .88, .04], -1: [.25, .40, .20, .15]}
SEG_CATEGORY = {   # weights over CAT_CODES order
    0: dict(GRO=.30, DIN=.18, FUE=.10, UTL=.08, HLT=.06, TRN=.08, FSH=.08, ENT=.04, EDU=.02, ECO=.04),
    1: dict(ECO=.55, FSH=.12, ENT=.08, GAM=.08, DIN=.06, EDU=.05, UTL=.03, GRO=.03),
    2: dict(TRV=.70, LUX=.06, DIN=.08, ECO=.06, FSH=.05, ENT=.05),
    3: dict(ATM=.90, GAM=.04, ENT=.03, DIN=.03),
    -1: dict(LUX=.18, GAM=.18, ATM=.15, TRV=.12, ECO=.12, ENT=.08, FSH=.07, DIN=.05, GRO=.05),
}
OCC_TILT = {"student": dict(DIN=1.5, ENT=2, EDU=4, FUE=.3, HLT=.3),
            "retired": dict(HLT=4, GRO=1.5, UTL=1.5, ENT=.5, FUE=.5),
            "trader": dict(GRO=1.6, FUE=1.8, UTL=1.2),
            "office": dict(DIN=1.5, TRN=1.8),
            "freelancer": dict(ECO=1.3, ENT=1.3), "other": {}}
CHAN_VARIANTS = {"POS": ["pos", "Pos", "pos ", " POS"], "ONLINE": ["Online", "online", "ONLINE "],
                 "ATM": ["atm", "Atm"], "CONTACTLESS": ["Contactless", "contactless", "contact-less"]}

# ------------------------------------------------------------------ customers
OCCS = ["office", "student", "trader", "freelancer", "retired", "other"]
OCC_P = [.30, .15, .15, .15, .08, .17]
OCC_DIST = {"office": [.30, .03, .04, .10, .05, .28, .10, .10], "student": [.08, .55, .03, .05, .02, .17, .05, .05],
            "trader": [.08, .02, .50, .06, .04, .18, .06, .06], "freelancer": [.15, .08, .08, .12, .06, .30, .11, .10],
            "retired": [.10, .02, .08, .12, .05, .35, .14, .14], "other": [.10, .04, .12, .10, .12, .25, .13, .14]}
OCC_TIER = {"office": [.60, .30, .10], "student": [.97, .03, .00], "trader": [.55, .33, .12],
            "freelancer": [.65, .28, .07], "retired": [.70, .25, .05], "other": [.85, .13, .02]}
OCC_INCOME = {"office": 16e6, "student": 3e6, "trader": 22e6, "freelancer": 14e6, "retired": 7e6, "other": 9e6}
TIER_MULT = [1.0, 2.2, 5.0]
TIERS = ["mass", "affluent", "priority"]
OCC_MIX = {"office": [.55, .25, .15, .05], "student": [.40, .45, .02, .13], "trader": [.45, .10, .10, .35],
           "freelancer": [.30, .30, .12, .28], "retired": [.75, .05, .12, .08], "other": [.45, .15, .10, .30]}
N_CUST = 600
CUSTOMER_COLS = ["customer_id", "age", "gender", "occupation_group", "tier", "tenure_months", "monthly_income_vnd",
                 "home_district_id", "digital_active", "has_credit_card", "preferred_channel"]


def _age(rng, occ):
    if occ == "office": return rng.normal(34, 7), 22, 58
    if occ == "student": return rng.normal(20.5, 1.8), 18, 26
    if occ == "trader": return rng.normal(42, 10), 24, 70
    if occ == "freelancer": return rng.normal(32, 7), 21, 55
    if occ == "retired": return rng.normal(66, 5.5), 56, 85
    return rng.normal(40, 12), 19, 75


def make_customers(n=N_CUST):
    """Returns (rows as lists in CUSTOMER_COLS order, mix matrix (n,4), kind list)."""
    rng = rng_for("customers")
    rows, mixes, kinds = [], [], []
    fem = dict(office=.52, student=.55, trader=.55, freelancer=.45, retired=.55, other=.45)
    for i in range(n):
        occ = OCCS[rng.choice(6, p=OCC_P)]
        a, lo, hi = _age(rng, occ)
        age = int(np.clip(round(a), lo, hi))
        tier_i = int(rng.choice(3, p=OCC_TIER[occ]))
        inc = OCC_INCOME[occ] * TIER_MULT[tier_i] * float(np.exp(rng.normal(0, .25)))
        if occ == "student":
            inc = min(inc, 12e6)
        inc = int(round(inc, -5))
        tenure = int(max(1, round(min(max(age - 18, 1), 8) * 12 * rng.uniform(.05, 1)))) if occ != "student" else int(rng.integers(1, 25))
        dist = DIST_IDS[rng.choice(8, p=np.array(OCC_DIST[occ]) / sum(OCC_DIST[occ]))]
        p_dig = float(np.clip(.96 - max(age - 30, 0) * .012 + (.03 if tier_i == 2 else 0), .3, .98))
        digital = int(rng.random() < p_dig)
        p_cc = [.15, .55, .90][tier_i] * (.3 if occ == "student" else .6 if occ == "retired" else 1.0)
        cc = int(rng.random() < p_cc)
        if not digital:
            ch = ["atm", "branch"][int(rng.random() < .45)]
        elif age > 60:
            ch = ["mobile_app", "atm", "branch", "web"][rng.choice(4, p=[.5, .2, .2, .1])]
        else:
            ch = ["mobile_app", "web", "atm"][rng.choice(3, p=[.8, .1, .1])]
        gender = "F" if rng.random() < fem[occ] else "M"
        rows.append([f"C{i + 1:04d}", age, gender, occ, TIERS[tier_i], tenure, inc, dist, digital, cc, ch])
        base = np.array(OCC_MIX[occ], float)
        base[2] *= [.5, 1.8, 3.0][tier_i]
        if not digital:
            base[1] *= .1
        base /= base.sum()
        if rng.random() < .85:
            dom = int(rng.choice(4, p=base))
            m = .12 * base; m[dom] = 0; m = m / m.sum() * .12 if m.sum() > 0 else m
            m[dom] = .88
            kinds.append("dominated")
        else:
            m = rng.dirichlet(base * 6 + .3)
            kinds.append("mixed")
        mixes.append(m)
    return rows, np.array(mixes), kinds


# ------------------------------------------------------------------ dates and clocks
def date_probs(dow_w, pay_mult):
    """pay_mult = (day5, days 4 and 6, last three days of the month)."""
    w = []
    for d in DATES:
        x = dow_w[d.weekday()]
        last = (d + timedelta(days=3)).month != d.month
        if d.day == 5: x *= pay_mult[0]
        elif d.day in (4, 6): x *= pay_mult[1]
        elif last: x *= pay_mult[2]
        w.append(x)
    w = np.array(w, float)
    return w / w.sum()


def fmt_ts(day, seconds):
    return (day + timedelta(seconds=int(seconds))).strftime("%Y-%m-%d %H:%M:%S")


def hour_to_seconds(h):
    return int(min(round(float(h) * 3600), 86399))


def pick(rng, names, w):
    w = np.array(w, float)
    return names[int(rng.choice(len(names), p=w / w.sum()))]


# ------------------------------------------------------------------ card enrichment
SEG_DOW = {0: [1, 1, 1, 1, 1.15, 1.35, 1.3], 1: [.95, .95, 1, 1, 1.1, 1.4, 1.45], 2: [.9, .8, .8, .9, 1.3, 1.5, 1.4],
           3: [1, 1, 1, 1, 1.2, 1.3, 1.1], -1: [1] * 7}
SEG_PAY = {0: (1.8, 1.3, 1.6), 1: (1.5, 1.2, 1.4), 2: (1.0, 1.0, 1.0), 3: (1.3, 1.0, 1.5), -1: (1, 1, 1)}


def card_descriptive(seg, hours, customers, mix, rng):
    """Clean descriptive columns for the unique transactions. seg, hours: arrays over unique txns.
    Returns dict of lists: timestamp, customer_id, channel, merchant_category, txn_district_id, is_foreign."""
    n = len(seg)
    occ = np.array([c[3] for c in customers])
    tier = np.array([c[4] for c in customers])
    home = [c[7] for c in customers]
    digital = np.array([c[8] for c in customers])
    cid = [c[0] for c in customers]
    dprobs = {s: date_probs(SEG_DOW[s], SEG_PAY[s]) for s in SEG_DOW}
    act = .6 + .4 * digital
    out = {k: [] for k in ("timestamp", "customer_id", "channel", "merchant_category", "txn_district_id", "is_foreign")}
    for i in range(n):
        s = int(seg[i])
        day = DATES[int(rng.choice(N_DAYS, p=dprobs[s]))]
        ts = fmt_ts(day, hour_to_seconds(hours[i]))
        w = act if s == -1 else mix[:, s] * act
        c = int(rng.choice(len(customers), p=w / w.sum()))
        ch = pick(rng, CHANNELS, SEG_CHANNEL[s])
        if ch == "ONLINE" and not digital[c] and s != -1:
            ch = "POS"                                  # a customer without digital access does not shop online
        cw = np.array([SEG_CATEGORY[s].get(k, 0.0) * OCC_TILT[occ[c]].get(k, 1.0) * (k in CHAN_OK[ch]) for k in CAT_CODES])
        if cw.sum() == 0:
            cw = np.array([float(k in CHAN_OK[ch]) for k in CAT_CODES])
        cat = pick(rng, CAT_CODES, cw)
        p_for = {-1: .6, 2: .12}.get(s, .01)
        foreign = int(rng.random() < p_for)
        h = home[c]
        if foreign:
            td = ""                                      # foreign terminal: no district of the fictional city
        elif s in (0, 1) or (s == 3 and rng.random() < .5):
            td = h
        else:
            td = pick(rng, [d for d in DIST_IDS if d != h], [1] * 7)
        for k, v in zip(out, (ts, cid[c], ch, cat, td, foreign)):
            out[k].append(v)
    return out


def dirty_card_descriptive(clean, rng):
    """Apply defects to descriptive columns only. Returns dirty dict and a defect log
    (type, column, list of positions)."""
    n = len(clean["timestamp"])
    dirty = {k: list(v) for k, v in clean.items()}
    log = []
    used = set()

    def take(frac_or_n, forbid=()):
        m = int(round(frac_or_n * n)) if frac_or_n < 1 else int(frac_or_n)
        pool = [i for i in range(n) if i not in used and i not in forbid]
        idx = [int(i) for i in rng.choice(pool, m, replace=False)]
        used.update(idx)
        return idx
    idx = take(.07)                                                    # channel spelling and case
    for i in idx:
        dirty["channel"][i] = pick(rng, CHAN_VARIANTS[clean["channel"][i]], [1] * len(CHAN_VARIANTS[clean["channel"][i]]))
    log.append(("inconsistent_spelling_or_case", "channel", idx))
    idx = take(.012)                                                   # unknown customer
    for i in idx:
        dirty["customer_id"][i] = "UNKNOWN" if rng.random() < .5 else f"C{int(rng.integers(9001, 9999)):04d}"
    log.append(("unknown_customer_id", "customer_id", idx))
    idx = take(.012)                                                   # missing category
    for i in idx:
        dirty["merchant_category"][i] = ""
    log.append(("missing_value", "merchant_category", idx))
    idx = take(.02)                                                    # alternative timestamp format
    for i in idx:
        d = datetime.strptime(clean["timestamp"][i], "%Y-%m-%d %H:%M:%S")
        dirty["timestamp"][i] = d.strftime("%d/%m/%Y %H:%M:%S")
    log.append(("mixed_timestamp_format_dd_mm_yyyy", "timestamp", idx))
    idx = take(6)                                                      # impossible timestamps
    bad = ["2025-03-14 24:30:00", "2025-02-30 10:15:00", "2025-04-31 08:00:00", "2025-03-09 25:07:00",
           "2025-03-22 12:75:00", "2025-13-02 09:00:00"]
    for i, b in zip(idx, bad):
        dirty["timestamp"][i] = b
    log.append(("impossible_timestamp", "timestamp", idx))
    idx = take(5, forbid={i for i in range(n) if clean["txn_district_id"][i] == ""})
    for i in idx:
        dirty["txn_district_id"][i] = "D99"
    log.append(("invalid_district_id", "txn_district_id", idx))
    return dirty, log


# ------------------------------------------------------------------ ATMs and ATM transactions
ATM_PER_HOTSPOT = {0: 6, 1: 4, 2: 8, 3: 3, 4: 8, 5: 6}
N_STANDALONE = 5
ATM_COLS = ["atm_id", "x_km", "y_km", "hotspot_id", "district_id", "atm_network"]
HS_HOUR = {0: [(.45, 12.5, .8), (.40, 18.0, 1.0), (.15, 9.0, 1.2)], 1: [(.20, 12.5, 1.2), (.65, 21.5, 1.5), (.15, 16.0, 2.0)],
           2: [(.7, 6.5, 1.2), (.2, 15.5, 2), (.1, 19, 1.5)], 3: [(.8, 5.5, 1.0), (.2, 16.5, 1.5)],
           4: [(.3, 8.0, 1.5), (.5, 18.5, 1.5), (.2, 13, 2)], 5: [(.25, 9.0, 1.5), (.5, 13.0, 2.5), (.25, 17.5, 1.5)],
           -1: [(1.0, 14.0, 5.0)]}
HS_DOW = {0: [1.2, 1.2, 1.2, 1.2, 1.3, .5, .3], 1: [1, 1, 1, 1, 1.1, .8, .6], 2: [1, .9, .9, 1, 1.1, 1.6, 1.7],
          3: [1, .9, .9, 1, 1.1, 1.6, 1.7], 4: [.8, .8, .8, .9, 1.2, 1.6, 1.5], 5: [.9, .9, .9, 1, 1.5, 1.7, 1.6],
          -1: [1] * 7}
HS_PAY = {0: (1.8, 1.3, 1.7), 1: (1.4, 1.2, 1.3), 2: (1.3, 1.1, 1.3), 3: (1.3, 1.1, 1.3), 4: (1.3, 1.1, 1.3),
          5: (1.0, 1.0, 1.0), -1: (1.1, 1.0, 1.1)}
HS_OCC = {0: dict(office=.55, freelancer=.12, other=.10, trader=.08, student=.05, retired=.10),
          1: dict(student=.75, office=.08, freelancer=.07, other=.10),
          2: dict(trader=.30, other=.20, retired=.12, office=.12, freelancer=.13, student=.05, ),
          3: dict(trader=.65, other=.15, retired=.05, freelancer=.10, office=.05),
          4: dict(retired=.20, other=.20, office=.20, freelancer=.20, trader=.10, student=.10),
          5: dict(other=.35, trader=.25, freelancer=.25, office=.10, retired=.05),
          -1: dict(office=.30, student=.15, trader=.15, freelancer=.15, retired=.08, other=.17)}
HS_AMOUNT = {0: (2.0e6, .55), 1: (.5e6, .5), 2: (1.5e6, .5), 3: (2.5e6, .6), 4: (1.0e6, .55), 5: (2.0e6, .5), -1: (1.5e6, .7)}
ATM_P_OWN = {0: .85, 1: .85, 2: .5, 3: .5, 4: .5, 5: .25, -1: .2}


def make_atms(P_km, hs):
    """40 fictional ATMs: k-means centres inside each hotspot + standalone ATMs for isolated points."""
    from sklearn.cluster import KMeans
    rng = rng_for("atms")
    atms = []
    for h, k in ATM_PER_HOTSPOT.items():
        km = KMeans(n_clusters=k, n_init=5, random_state=SEED + OFF["atms"]).fit(P_km[hs == h])
        atms += [(h, c) for c in km.cluster_centers_]
    iso = P_km[hs == -1]
    km = KMeans(n_clusters=N_STANDALONE, n_init=5, random_state=SEED + OFF["atms"]).fit(iso)
    atms += [(-1, c) for c in km.cluster_centers_]
    rows = []
    for j, (h, c) in enumerate(atms):
        x, y = round(float(c[0]), 3), round(float(c[1]), 3)
        net = "own" if rng.random() < ATM_P_OWN[h] else "partner"
        rows.append([f"A{j + 1:02d}", f"{x:.3f}", f"{y:.3f}", h, district_ids(x, y)[0], net])
    return rows


def atm_descriptive(P_km, hs, atm_rows, customers):
    rng = rng_for("atm_desc")
    n = len(P_km)
    A = np.array([[float(r[1]), float(r[2])] for r in atm_rows])
    ah = np.array([r[3] for r in atm_rows])
    anet = [r[5] for r in atm_rows]
    aid = [r[0] for r in atm_rows]
    occ = np.array([c[3] for c in customers])
    homes = np.array([c[7] for c in customers])
    dprobs = {h: date_probs(HS_DOW[h], HS_PAY[h]) for h in HS_DOW}
    # highway: Hung Kings long weekend travel days
    hw = np.array([2.2 if d in HOLIDAY else 1.0 for d in DATES])
    dprobs[5] = dprobs[5] * hw / (dprobs[5] * hw).sum()
    dist_ids = district_ids(P_km[:, 0], P_km[:, 1])
    out = {k: [] for k in ("timestamp", "atm_id", "customer_id", "amount_vnd", "atm_network", "district_id", "atm_distance_km")}
    for i in range(n):
        h = int(hs[i])
        day = DATES[int(rng.choice(N_DAYS, p=dprobs[h]))]
        mix = HS_HOUR[h]
        w, mu, sd = mix[int(rng.choice(len(mix), p=np.array([m[0] for m in mix]) / sum(m[0] for m in mix)))]
        hour = float(np.clip(rng.normal(mu, sd), 0, 23.99)) if h != -1 else float(rng.uniform(6, 23.5))
        ts = fmt_ts(day, hour_to_seconds(hour))
        cand = np.where(ah == h)[0] if h != -1 else np.arange(len(A))
        d = np.sqrt(((A[cand] - P_km[i]) ** 2).sum(1))
        j = int(cand[d.argmin()])
        ow = HS_OCC[h]
        oc = pick(rng, list(ow), list(ow.values()))
        pool = np.where(occ == oc)[0]
        wt = np.where(homes[pool] == dist_ids[i], 3.0, 1.0)
        c = int(pool[int(rng.choice(len(pool), p=wt / wt.sum()))])
        med, sg = HS_AMOUNT[h]
        amt = int(np.clip(round(med * float(np.exp(rng.normal(0, sg))) / 50_000) * 50_000, 50_000, 5_000_000))
        for k, v in zip(out, (ts, aid[j], customers[c][0], amt, anet[j], dist_ids[i], f"{float(d.min()):.3f}")):
            out[k].append(v)
    return out


# ------------------------------------------------------------------ writers
def write_rows(path, cols, rows, eol="\n"):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator=eol)
        w.writerow(cols)
        w.writerows(rows)


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        r = list(csv.reader(f))
    return r[0], r[1:]


def append_columns(path, new_cols, new_values_by_row):
    """Append columns to an existing CSV (existing text of base columns is re-emitted unchanged)."""
    eol = "\r\n" if "\r\n" in open(path, newline="", encoding="utf-8").read(300) else "\n"
    head, rows = read_rows(path)
    out = [r + list(v) for r, v in zip(rows, new_values_by_row)]
    write_rows(path, head + list(new_cols), out, eol)


# ------------------------------------------------------------------ pipeline (called by generate_data.main)
TRUTH = None


def enrich_all(card, seg_log, hard=None):
    """Writes dimension tables, enriched base files, enriched streams and data/truth/*.
    card: dict from gen.make_card (same one that wrote card_txn_5d.csv); seg_log: true segment of
    the card stream points (ids 0..649)."""
    import generate_data as gen
    global TRUTH
    TRUTH = gen.RAW.parent / "truth"
    TRUTH.mkdir(parents=True, exist_ok=True)
    GEO, CARD = gen.GEO, gen.CARD
    customers, mix, kinds = make_customers()
    write_rows(CARD / "customers.csv", CUSTOMER_COLS, customers, "\r\n")
    write_rows(TRUTH / "customer_segment_mix.csv", ["customer_id", "kind", "p_seg0", "p_seg1", "p_seg2", "p_seg3"],
               [[c[0], k] + [f"{v:.3f}" for v in m] for c, k, m in zip(customers, kinds, mix)], "\r\n")
    write_rows(GEO / "districts.csv", DISTRICT_COLS, district_rows(), "\r\n")
    write_rows(CARD / "merchant_categories.csv", CATEGORY_COLS, [list(c) for c in CATEGORIES], "\r\n")

    # ---- card_txn_5d.csv
    defects = enrich_card_table(gen, CARD / "card_txn_5d.csv", card, customers, mix, "card_desc", "dirt", "card_txn_5d")

    # ---- atm_txn_2d.csv
    gpath = GEO / "atm_txn_2d.csv"
    head, grows = read_rows(gpath)
    P_km = np.array([[float(r[1]), float(r[2])] for r in grows])
    hs = np.array([int(r[3]) for r in grows])
    atm_rows = make_atms(P_km, hs)
    write_rows(GEO / "atms.csv", ATM_COLS, atm_rows, "\r\n")
    ad = atm_descriptive(P_km, hs, atm_rows, customers)
    acols = ["timestamp", "atm_id", "customer_id", "amount_vnd", "atm_network", "district_id", "atm_distance_km"]
    append_columns(gpath, acols, list(zip(*[ad[c] for c in acols])))

    # ---- streams
    gp = GEO / "atm_stream"
    for fn in ("d0.csv", "updates.csv"):
        h, rr = read_rows(gp / fn)
        di = district_ids([float(r[h.index("x_km")]) for r in rr], [float(r[h.index("y_km")]) for r in rr])
        append_columns(gp / fn, ["district_id"], [[d] for d in di])
    enrich_card_stream(CARD / "card_stream", seg_log, customers, mix)
    if hard:                                   # hard (and optional medium) card tables + hard stream
        for name, c, fn, dk, tk in hard["tables"]:
            defects += enrich_card_table(gen, CARD / fn, c, customers, mix, dk[0], dk[1], name)
        enrich_card_stream(CARD / "card_stream_hard", hard["stream_segs"], customers, mix, "stream_desc_hard")
    write_rows(TRUTH / "defect_catalogue_card.csv", ["scope", "table", "defect_type", "column", "n_rows"], defects, "\r\n")
    return dict(customers=customers, defects=defects)


def enrich_card_table(gen, path, card, customers, mix, desc_key, dirt_key, truth_name):
    """Append descriptive columns to a card CSV (card = dict from make_card / make_card_level) and write
    the clean versions to data/truth/. Returns the defect rows for the catalogue."""
    TR = gen.RAW.parent / "truth"
    TR.mkdir(parents=True, exist_ok=True)
    n_u = len(card["clean"])
    clean_d = card_descriptive(card["yc"], card["clean"][:, 1], customers, mix, rng_for(desc_key))
    dirty_d, log = dirty_card_descriptive(clean_d, rng_for(dirt_key))
    head, rows = read_rows(path)
    uid = [int(r[0][3:]) - 1 for r in rows]
    cols = ["timestamp", "customer_id", "channel", "merchant_category", "txn_district_id", "is_foreign"]
    append_columns(path, cols, [[dirty_d[c][u] for c in cols] for u in uid])
    mult = np.bincount(uid, minlength=n_u)
    pre = rows[0][0][:3]
    write_rows(TR / f"{truth_name}_clean_descriptive.csv", ["txn_id"] + cols,
               [[f"{pre}{u + 1:06d}"] + [clean_d[c][u] for c in cols] for u in range(n_u)], "\r\n")
    write_rows(TR / f"{truth_name}_clean_base.csv", ["txn_id"] + gen.CARD_COLS + ["segment_id"],
               [[f"{pre}{u + 1:06d}"] + [str(int(v)) if j in (0, 3) else (f"{v:.3f}" if j == 4 else f"{v:.2f}")
                                         for j, v in enumerate(card["clean"][u])] + [int(card["yc"][u])]
                for u in range(n_u)], "\r\n")
    defects = []
    base = [r[1:6] for r in rows]
    for j, name in enumerate(gen.CARD_COLS):
        defects.append(["base", truth_name, "missing_value", name, sum(r[j] == "" for r in base)])
    seen, dup = set(), 0
    for r in rows:
        k = tuple(r[:7])
        dup += k in seen
        seen.add(k)
    defects.append(["base", truth_name, "duplicate_row_exact_copy", "(whole row)", dup])
    for typ, col, pos in log:
        defects.append(["descriptive", truth_name, typ, col, int(sum(mult[p] for p in pos))])
    return defects


def enrich_card_stream(base, seg_log, customers, mix, rng_key="stream_desc"):
    import generate_data as gen
    n = len(seg_log)
    d0h, d0 = read_rows(base / "d0.csv")
    uph, up = read_rows(base / "updates.csv")
    hours = np.zeros(n)
    for r in d0:
        hours[int(r[0])] = float(r[d0h.index("hour_of_day")])
    for r in up:
        if r[uph.index("op")] == "insert":
            hours[int(r[uph.index("point_id")])] = float(r[uph.index("hour_of_day")])
    d = card_descriptive(np.array(seg_log), hours, customers, mix, rng_for(rng_key))
    W = len(d0)
    t0 = T0
    arrival = {}
    for r in d0:
        arrival[int(r[0])] = t0 - timedelta(hours=W - int(r[0]))
    for r in up:
        if r[uph.index("op")] == "insert":
            arrival[int(r[uph.index("point_id")])] = datetime.strptime(r[uph.index("timestamp")], "%Y-%m-%d %H:%M")

    def event_ts(p):
        a = arrival[p]
        e = datetime(a.year, a.month, a.day) + timedelta(seconds=hour_to_seconds(hours[p]))
        return (e - timedelta(days=1) if e > a else e).strftime("%Y-%m-%d %H:%M:%S")

    def extra(p):
        return [arrival[p].strftime("%Y-%m-%d %H:%M"), event_ts(p), int(seg_log[p]), d["customer_id"][p],
                d["channel"][p], d["merchant_category"][p], d["txn_district_id"][p], d["is_foreign"][p]]
    cols = ["point_arrival_timestamp", "event_timestamp", "segment_id", "customer_id", "channel",
            "merchant_category", "txn_district_id", "is_foreign"]
    append_columns(base / "d0.csv", cols, [extra(int(r[0])) for r in d0])
    append_columns(base / "updates.csv", cols, [extra(int(r[uph.index("point_id")])) for r in up])
