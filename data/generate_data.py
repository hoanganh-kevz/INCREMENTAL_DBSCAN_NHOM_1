"""Generate the raw datasets for task D01 (run from anywhere: python data/generate_data.py).

Toy scenarios: coordinates and doc claims are typed by hand from group doc R02/R03.
Synthetic sets: sklearn.datasets is used for DATA GENERATION ONLY, never for clustering.
"""
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.datasets import make_blobs, make_circles, make_moons

RAW = Path(__file__).resolve().parent / "raw"
SEED = 42
EPS, MINPTS = 1.1, 4

# name -> (doc section, points, update, expected from the doc)
# labels: int cluster id (1,2,..) or -1 for noise; ids are the doc's object names
TOY = {
    "r02_1_noise": dict(
        doc="R02 3.1 Noise",
        pts={"A": "0 0", "B": "1 0", "C": "2 0", "D": "0 1", "E": "1 1", "F": "2 1"},
        upd=("insert", "p", "5 5"),
        before={k: 1 for k in "ABCDEF"},
        after={**{k: 1 for k in "ABCDEF"}, "p": -1},
        updseed=[], cores_before=["B", "E"], cores_after=["B", "E"],
        note="p is isolated; clusters unchanged"),
    "r02_2_creation": dict(
        doc="R02 Case 2 Creation",
        pts={"A": "0 0", "B": "1 0", "C": "2 0"},
        upd=("insert", "p", "1 0.5"),
        before={"A": -1, "B": -1, "C": -1},
        after={"A": 1, "B": 1, "C": 1, "p": 1},
        updseed=["B"], cores_before=[], cores_after=["B"],
        note="B turns core, new cluster {A,B,C,p}"),
    "r02_3_absorption": dict(
        doc="R02 Case 3 Absorption",
        pts={"A": "0 0", "B": "1 0", "C": "0 1", "D": "1 1", "E": "2 0", "F": "2 1"},
        upd=("insert", "p", "2.2 0.5"),
        before={k: 1 for k in "ABCDEF"},
        after={**{k: 1 for k in "ABCDEF"}, "p": 1},
        updseed=["B", "D", "E", "F"], cores_before=["B", "D"],
        cores_after=["B", "D", "E", "F"],
        note="p joins the existing cluster; E, F become core"),
    "r02_4_merge": dict(
        doc="R02 Case 4 Merge",
        pts={"A1": "-1 0", "A2": "0 0", "A3": "1 0", "A4": "-1 1", "A5": "0 1",
             "A6": "1 1", "B1": "2.8 0", "B2": "3.8 0", "B3": "4.8 0",
             "B4": "2.8 1", "B5": "3.8 1", "B6": "4.8 1"},
        upd=("insert", "p", "1.9 0.5"),
        before={**{f"A{i}": 1 for i in range(1, 7)}, **{f"B{i}": 2 for i in range(1, 7)}},
        after={**{f"A{i}": 1 for i in range(1, 7)}, **{f"B{i}": 1 for i in range(1, 7)}, "p": 1},
        updseed=["A2", "A3", "A5", "A6", "B1", "B2", "B4", "B5", "p"],
        cores_before=["A2", "A5", "B2", "B5"],
        cores_after=["A2", "A3", "A5", "A6", "B1", "B2", "B4", "B5", "p"],
        note="two clusters merge into 13 points"),
    "r03_1_removal": dict(
        doc="R03 Case 1 Removal",
        pts={"A": "-1 0", "B": "1 0", "C": "0 -1", "D": "0 1", "p": "0 0"},
        upd=("delete", "p", None),
        before={k: 1 for k in ["A", "B", "C", "D", "p"]},
        after={k: -1 for k in "ABCD"},
        updseed=[], cores_before=["p"], cores_after=[],
        note="only core deleted, cluster disappears, 4 points become noise"),
    "r03_2_reduction": dict(
        doc="R03 Case 2 Reduction",
        pts={"A": "0 0", "B": "1 0", "C": "0 1", "D": "1 1", "E": "2 0", "F": "2 1",
             "p": "2.2 0.5"},
        upd=("delete", "p", None),
        before={k: 1 for k in ["A", "B", "C", "D", "E", "F", "p"]},
        after={k: 1 for k in "ABCDEF"},
        updseed=["B", "D"], cores_before=["B", "D", "E", "F"], cores_after=["B", "D"],
        note="E, F lose core status; B and D directly reachable, no split"),
    "r03_3a_split": dict(
        doc="R03 Case 3A Potential Split -> Split",
        pts={"A": "1 0", "B": "0.2 0", "C": "0.6 0.6", "D": "0.6 -0.6", "E": "3 0",
             "F": "3.8 0", "G": "3.4 0.6", "H": "3.4 -0.6", "I": "2 1", "p": "2 0"},
        upd=("delete", "p", None),
        before={k: 1 for k in "ABCDEFGHI"} | {"p": 1},
        after={"A": 1, "B": 1, "C": 1, "D": 1, "E": 2, "F": 2, "G": 2, "H": 2, "I": -1},
        updseed=["A", "E"], cores_before=["A", "B", "E", "F", "p"],
        cores_after=["A", "B", "E", "F"],
        note="p was the only bridge; cluster splits in two, I becomes noise"),
    "r03_3b_nonsplit": dict(
        doc="R03 Case 3B Potential Split -> Non-split (3A plus J, K, L)",
        pts={"A": "1 0", "B": "0.2 0", "C": "0.6 0.6", "D": "0.6 -0.6", "E": "3 0",
             "F": "3.8 0", "G": "3.4 0.6", "H": "3.4 -0.6", "I": "2 1", "p": "2 0",
             "J": "1 1", "K": "3 1", "L": "2 2"},
        upd=("delete", "p", None),
        before={k: 1 for k in "ABCDEFGHIJKLp"},
        after={k: 1 for k in "ABCDEFGHIJKL"},
        updseed=["A", "E", "I"], cores_before=["A", "B", "C", "E", "F", "G", "I", "J", "K", "p"],
        cores_after=["A", "B", "C", "E", "F", "G", "I", "J", "K"],
        note="chain A-J-I-K-E of cores keeps one cluster; C and G also become core after adding J, K (doc silent)"),
}
# The doc lists 4+3 scenario types but R03 case 3 has two sub-cases: 8 folders in total.


def write_toy():
    for name, s in TOY.items():
        d = RAW / "toy" / name
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "points.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["point_id", "x", "y"])
            for k, xy in s["pts"].items():
                w.writerow([k, *xy.split()])
        op, pid, xy = s["upd"]
        with open(d / "update.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["op", "point_id", "x", "y"])
            w.writerow([op, pid, *(xy.split() if xy else ["", ""])])
        exp = dict(doc_section=s["doc"], eps=EPS, minpts=MINPTS,
                   neighbourhood_includes_self=True, metric="euclidean",
                   label_convention="int cluster id (1,2,..), -1 = noise; ids only identify partition",
                   labels_before=s["before"], labels_after=s["after"],
                   updseed=s["updseed"], core_before=s["cores_before"],
                   core_after=s["cores_after"], note=s["note"])
        (d / "expected.json").write_text(json.dumps(exp, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


def dump(path, cols, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        w.writerows(rows)


def write_synthetic():
    syn = RAW / "synthetic"
    syn.mkdir(parents=True, exist_ok=True)
    X, y = make_moons(n_samples=500, noise=0.08, random_state=SEED)
    dump(syn / "moons_n500.csv", ["point_id", "x", "y", "label"],
         [[i, f"{a:.6f}", f"{b:.6f}", int(c)] for i, ((a, b), c) in enumerate(zip(X, y))])
    X, y = make_circles(n_samples=500, factor=0.4, noise=0.05, random_state=SEED)
    dump(syn / "rings_n500.csv", ["point_id", "x", "y", "label"],
         [[i, f"{a:.6f}", f"{b:.6f}", int(c)] for i, ((a, b), c) in enumerate(zip(X, y))])
    # blobs with noise: 3 gaussian blobs (450) + 50 uniform noise points, label -1
    rng = np.random.default_rng(SEED)
    X, y = make_blobs(n_samples=450, centers=[(-4, -3), (0, 4), (5, -2)],
                      cluster_std=[0.8, 0.9, 0.7], random_state=SEED)
    lo, hi = X.min(0) - 1.5, X.max(0) + 1.5
    N = rng.uniform(lo, hi, size=(50, 2))
    X, y = np.vstack([X, N]), np.concatenate([y, -np.ones(50, dtype=int)])
    dump(syn / "blobs_noise_n500.csv", ["point_id", "x", "y", "label"],
         [[i, f"{a:.6f}", f"{b:.6f}", int(c)] for i, ((a, b), c) in enumerate(zip(X, y))])


def write_nd():
    nd = RAW / "nd"
    nd.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    X, y = make_blobs(n_samples=800, n_features=5, centers=4, cluster_std=1.0,
                      center_box=(-8, 8), random_state=SEED)
    lo, hi = X.min(0) - 2, X.max(0) + 2
    N = rng.uniform(lo, hi, size=(60, 5))
    X, y = np.vstack([X, N]), np.concatenate([y, -np.ones(60, dtype=int)])
    cols = ["point_id"] + [f"x{i}" for i in range(1, 6)] + ["label"]
    dump(nd / "blobs5d_n860.csv", cols,
         [[i, *[f"{v:.6f}" for v in row], int(c)] for i, (row, c) in enumerate(zip(X, y))])


# =====================================================================================
# Fictional digital-bank suite (all SIMULATED, seed 42, no real bank / Timo data).
#   B  bank_geo/atm_txn_2d.csv      ATM transactions on a fictional city map (2D, km)
#   C  bank_card/card_txn_5d.csv    card transactions, 5 unscaled features + dirty cells
#   D  bank_geo/atm_stream/, bank_card/card_stream/   update streams with verified cases
# The reference DBSCAN (sklearn) is used HERE ONLY to verify/label the data, never as the
# group's algorithm. 2D geometry is generated in integer metres so ties (distance == eps)
# can be excluded exactly.
# =====================================================================================
import argparse
import io
from collections import defaultdict
from datetime import datetime, timedelta
from fractions import Fraction

from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import adjusted_rand_score

GEO, CARD = RAW / "bank_geo", RAW / "bank_card"
PREVIEW = Path(__file__).resolve().parent / "preview"
GEO_EPS_KM, GEO_MINPTS, GEO_N = 0.5, 4, 1200
CARD_N = 1500
INSERT_CASES = ["Noise", "Creation", "Absorption", "Merge"]
DELETE_CASES = ["Removal", "Reduction", "PotentialSplit_NonSplit", "PotentialSplit_Split"]
CARD_COLS = ["amount_vnd", "hour_of_day", "distance_from_home_km", "txns_last_7d",
             "merchant_risk_score"]
CASE_DEFS = {
    "Noise": "insert; p stays noise, all old labels unchanged",
    "Creation": "insert; a new cluster appears made only of p and former noise points",
    "Absorption": "insert; p joins exactly one existing cluster (former noise may join too), no merge",
    "Merge": "insert; one new cluster contains >= 2 old clusters",
    "Removal": "delete; an old cluster loses all its remaining members to noise (cluster vanishes)",
    "Reduction": "delete; cluster shrinks or loses cores, but the UpdSeed cores are already directly eps-linked (no connectivity search)",
    "PotentialSplit_NonSplit": "delete; UpdSeed cores of an old cluster are not directly linked (search needed) but stay connected",
    "PotentialSplit_Split": "delete; an old cluster falls apart into >= 2 clusters",
    "DeleteNoise": "delete of a point that was noise (natural streams only)"}


# ---------- reference DBSCAN, partitions, case classification (shared with check_data.py) ----------
def ref_fit(X, eps, minpts):
    """sklearn reference: labels (-1 noise) and core mask. Neighbourhood includes the point."""
    db = DBSCAN(eps=eps, min_samples=minpts).fit(X)
    core = np.zeros(len(X), bool)
    core[db.core_sample_indices_] = True
    return db.labels_.copy(), core


def ambiguous_border(X, labels, core, eps):
    """Positions of border points within eps of cores from >= 2 different clusters."""
    ci = np.where(core)[0]
    bi = np.where((~core) & (labels >= 0))[0]
    if len(ci) == 0 or len(bi) == 0:
        return []
    d = np.sqrt(((X[bi][:, None, :] - X[ci][None, :, :]) ** 2).sum(-1))
    return [int(b) for b, row in zip(bi, d) if len(set(labels[ci[row <= eps]])) > 1]


def classify_insert(ids_a, la, p, lab_before, core_b, core_a):
    """Insert case from the STRUCTURE of the core points (core partition), not from border labels.
    la / lab_before: labels after / before (dict before); core_b / core_a: dict id -> is core.
    A border point shared by two clusters can flip between them under a re-fit, so it never counts."""
    members = defaultdict(list)
    for k, l in zip(ids_a, la):
        if l >= 0 and core_a[k]:
            members[l].append(k)
    olds = {c: {lab_before[k] for k in m if k != p and core_b.get(k, False)} for c, m in members.items()}
    if any(len(o) >= 2 for o in olds.values()):
        return "Merge"
    if any(len(o) == 0 for o in olds.values()):
        return "Creation"
    if la[ids_a.index(p)] == -1:
        return "Noise"
    return "Absorption"


def classify_delete(ids_b, Xb, lb, cb, ids_a, Xa, la, ca, p, eps):
    """Delete case from the core partition before (b) and after (a); p already removed from a.
    Split: the remaining cores of an old cluster fall into >= 2 clusters. Removal: an old cluster has
    no core left (its former border points may be re-assigned to other clusters, that does not matter).
    Border points never decide the case (they are ambiguous between clusters)."""
    pb = ids_b.index(p)
    if lb[pb] == -1:
        return "DeleteNoise"
    lab_b, lab_a = dict(zip(ids_b, lb)), dict(zip(ids_a, la))
    core_b, core_a = dict(zip(ids_b, cb)), dict(zip(ids_a, ca))
    old = {lab_b[k] for k in ids_b if core_b[k]}
    pieces = {c: {lab_a[k] for k in ids_a if core_a[k] and lab_b[k] == c} for c in old}
    if any(len(v) >= 2 for v in pieces.values()):
        return "PotentialSplit_Split"
    if any(len(v) == 0 for v in pieces.values()):
        return "Removal"
    demoted = [i for i, k in enumerate(ids_b) if cb[i] and not core_a.get(k, False)]
    seeds = set()
    for i in demoted:                      # UpdSeed: cores of D' within eps of a demoted point
        near = np.sqrt(((Xa - Xb[i]) ** 2).sum(1)) <= eps
        seeds |= {ids_a[j] for j in np.where(near & ca)[0]}
    pos_a = {k: i for i, k in enumerate(ids_a)}
    for c in set(lab_b[k] for k in seeds):
        S = sorted(k for k in seeds if lab_b[k] == c)
        XS = Xa[[pos_a[k] for k in S]]
        adj = np.sqrt(((XS[:, None, :] - XS[None, :, :]) ** 2).sum(-1)) <= eps
        comp, stack = {0}, [0]             # seeds linked by direct eps-reachability
        while stack:
            u = stack.pop()
            for v in np.where(adj[u])[0]:
                if int(v) not in comp:
                    comp.add(int(v)); stack.append(int(v))
        if len(comp) < len(S):
            return "PotentialSplit_NonSplit"
    return "Reduction"


def tie_pairs(P, e2):
    """Index pairs of an integer point array at squared distance exactly e2 (ties with eps)."""
    out = []
    for s in range(0, len(P), 400):
        d2 = ((P[s:s + 400, None, :] - P[None, :, :]) ** 2).sum(-1)
        r, c = np.where(d2 == e2)
        out += [(s + a, int(b)) for a, b in zip(r, c) if s + a < b]
    return out


def fix_ties(P, e2):
    """Nudge one point of a tied pair by 1 unit until no tie remains."""
    while True:
        t = tie_pairs(P, e2)
        if not t:
            return P
        P[t[0][1], 0] += 1


# ---------- B. ATM 2D map ----------
def trunc_gauss(rng, m, sd, t=2.3):
    out = []
    while len(out) < m:
        v = rng.normal(0, 1, 2)
        if np.hypot(*v) <= t:
            out.append(v * sd)
    return np.array(out)


def geo_hotspots(rng, counts):
    """Hotspot points in km (float). Ids 0..5."""
    H = {}
    H[0] = np.array([8.0, 16.0]) + trunc_gauss(rng, counts[0], 0.9)                  # downtown blob
    H[1] = np.array([26.0, 18.0]) + trunc_gauss(rng, counts[1], 0.75)                # university blob
    t = rng.uniform(0, 2 * np.pi, counts[2])                                         # market ring
    r = 2.4 + rng.normal(0, 0.17, counts[2]).clip(-0.4, 0.4)
    H[2] = np.column_stack([22 + r * np.cos(t), 7 + r * np.sin(t)])
    H[3] = np.array([22.0, 7.0]) + trunc_gauss(rng, counts[3], 0.45, 2.1)            # market plaza (inside ring)
    u = (np.arange(counts[4]) + rng.uniform(-0.3, 0.3, counts[4])) / counts[4]       # river crescent
    th = np.deg2rad(20 + 140 * u)
    w = 0.1 + 0.3 * np.sin(np.pi * u)
    rr = 7.5 + rng.normal(0, 1, counts[4]).clip(-1.8, 1.8) * w
    H[4] = np.column_stack([11 + rr * np.cos(th), 1 + rr * np.sin(th)])
    u = (np.arange(counts[5]) + rng.uniform(-0.3, 0.3, counts[5])) / counts[5]       # highway
    a, b = np.array([13.0, 22.0]), np.array([31.0, 11.0])
    nrm = np.array([-(b - a)[1], (b - a)[0]]) / np.linalg.norm(b - a)
    H[5] = a + u[:, None] * (b - a) + rng.normal(0, 0.15, counts[5])[:, None] * nrm
    return H


def make_geo(n, eps_km=GEO_EPS_KM):
    """Returns (P int metres (n,2), hotspot ids (n,), eps_m). Shuffled, ties removed."""
    rng = np.random.default_rng(SEED)
    w = [0.17, 0.125, 0.185, 0.075, 0.19, 0.14]
    counts = [int(round(n * x)) for x in w]
    iso_n = n - sum(counts)
    H = geo_hotspots(rng, counts)
    eps_m = int(round(eps_km * 1000))
    pts = [np.round(H[k] * 1000).astype(np.int64) for k in range(6)]
    allh = np.vstack(pts)
    iso = []
    while len(iso) < iso_n:                                    # isolated: far from everything
        q = np.round(rng.uniform([500, 500], [31500, 23500])).astype(np.int64)
        if np.sqrt(((allh - q) ** 2).sum(1)).min() <= 1.3 * eps_m:
            continue
        if iso and np.sqrt(((np.array(iso) - q) ** 2).sum(1)).min() <= 2 * eps_m:
            continue
        iso.append(q)
    P = np.vstack(pts + [np.array(iso)])
    y = np.concatenate([np.full(len(p), k) for k, p in enumerate(pts)] + [np.full(iso_n, -1)])
    P = fix_ties(P, eps_m ** 2)
    perm = rng.permutation(len(P))
    return P[perm], y[perm], eps_m


def write_geo_csv(path, P, y, prefix="ATM"):
    dump(path, ["txn_id", "x_km", "y_km", "hotspot_id"],
         [[f"{prefix}{i + 1:05d}", f"{a / 1000:.3f}", f"{b / 1000:.3f}", int(h)]
          for i, ((a, b), h) in enumerate(zip(P, y))])


def geo_report(P, y, eps_m, minpts):
    X = P.astype(float)
    lab, _ = ref_fit(X, eps_m, minpts)
    km = KMeans(n_clusters=6, n_init=10, random_state=SEED).fit_predict(X)
    return dict(eps_km=eps_m / 1000, minpts=minpts,
                dbscan_clusters=int(len(set(lab)) - (-1 in lab)),
                dbscan_noise_fraction=round(float((lab == -1).mean()), 4),
                ari_dbscan_vs_truth=round(float(adjusted_rand_score(y, lab)), 4),
                ari_kmeans6_vs_truth=round(float(adjusted_rand_score(y, km)), 4)), lab, km


def write_geo():
    GEO.mkdir(parents=True, exist_ok=True)
    P, y, eps_m = make_geo(GEO_N)
    write_geo_csv(GEO / "atm_txn_2d.csv", P, y)
    rep, _, _ = geo_report(P, y, eps_m, GEO_MINPTS)
    rep.update(n=len(P), seed=SEED, units="km", simulated=True,
               note="reference run by sklearn inside the generator only; truth = hotspot_id")
    (GEO / "params.json").write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8", newline="\n")
    print("atm_txn_2d", rep)
    return rep


# ---------- C. card 5D ----------
# (mu, sd) of: log10 amount, hour, distance km, txns/7d, risk; p = proportion of the 4 segments.
CARD_SEGS = {
    0: dict(name="daily_retail", p=0.38, f=[(5.3, .10), (12.5, 1.6), (4.0, 1.5), (9, 1.5), (.12, .035)]),
    1: dict(name="online_shopping", p=0.28, f=[(5.9, .10), (21.0, 1.2), (1.0, .6), (4, 1.0), (.35, .04)]),
    2: dict(name="travel_big_ticket", p=0.16, f=[(7.0, .11), (15.0, 2.0), (60, 6), (2, .8), (.45, .05)]),
    3: dict(name="night_cash_out", p=0.18, f=[(4.6, .08), (1.5, .9), (8, 2.5), (14, 2), (.65, .05)]),
}


def card_round(a):
    out = a.copy()
    out[:, 0] = np.maximum(np.round(a[:, 0], -3), 1000)
    out[:, 1] = np.round(a[:, 1].clip(0, 23.99), 2)
    out[:, 2] = np.round(a[:, 2].clip(0.05, None), 2)
    out[:, 3] = np.round(a[:, 3].clip(0, None))
    out[:, 4] = np.round(a[:, 4].clip(0.001, 0.999), 3)
    return out


def card_sample(rng, seg, m):
    f = CARD_SEGS[seg]["f"]
    la = rng.normal(*f[0], m)
    return card_round(np.column_stack([10 ** la] + [rng.normal(*f[i], m) for i in range(1, 5)]))


def card_z(A, mean=None, std=None):
    """Tooling-only scaling: log10(amount) then z-score (population std). Returns Z, mean, std."""
    T = A.copy()
    T[:, 0] = np.log10(T[:, 0])
    if mean is None:
        mean, std = T.mean(0), T.std(0)
    return (T - mean) / std, mean, std


def card_anomalies(rng, m, A_clean, min_d=1.6):
    """Planted anomalies far (in scaled space) from every clean point and from each other."""
    out, zs = [], []
    Zc, mean, std = card_z(A_clean)
    while len(out) < m:
        la = rng.uniform(3.2, 3.8) if rng.random() < 0.3 else rng.uniform(8.0, 8.7)
        a = card_round(np.array([[10 ** la, rng.uniform(0, 24), rng.uniform(150, 900),
                                  rng.choice([0, 1, 30, 45, 60]), rng.uniform(.8, .99)]]))
        z = card_z(a, mean, std)[0][0]
        if np.sqrt(((Zc - z) ** 2).sum(1)).min() < min_d:
            continue
        if zs and np.sqrt(((np.array(zs) - z) ** 2).sum(1)).min() < 1.0:
            continue
        out.append(a[0]); zs.append(z)
    return np.array(out)


def eps_run(Z, y, minpts, grid, min_ari=0.9):
    """Longest run of grid eps values where the reference DBSCAN reproduces the truth (ARI >= min_ari)."""
    best, cur = (None, None), None
    for e in grid:
        lab = DBSCAN(eps=float(e), min_samples=minpts).fit_predict(Z)
        if adjusted_rand_score(y, lab) >= min_ari:
            cur = (cur[0] if cur else float(e), float(e))
            if best[0] is None or cur[1] - cur[0] > best[1] - best[0]:
                best = cur
        else:
            cur = None
    return best


def kdist(Z, k):
    d = np.sqrt(((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1))
    d.sort(1)
    return d[:, k - 1]            # k-th nearest incl. self (matches MinPts counting)


def make_card(n, minpts_list=(6, 8, 10, 12, 14)):
    """Dirty rows (NaN = missing, duplicates), truth, clean latent, tooling params."""
    rng = np.random.default_rng(SEED + 1)
    n_dup, n_anom = int(round(0.005 * n)), max(int(round(0.01 * n)), 6)
    n_u = n - n_dup
    n_seg = n_u - n_anom
    cnt = [int(round(n_seg * CARD_SEGS[s]["p"])) for s in range(4)]
    cnt[0] += n_seg - sum(cnt)
    A = np.vstack([card_sample(rng, s, cnt[s]) for s in range(4)])
    y = np.concatenate([np.full(cnt[s], s) for s in range(4)])
    A, y = np.vstack([A, card_anomalies(rng, n_anom, A)]), np.concatenate([y, np.full(n_anom, -1)])
    perm = rng.permutation(len(A)); A, y = A[perm], y[perm]
    clean = A.copy()
    dirty = A.copy()
    cells = rng.choice(dirty.size, int(round(0.015 * dirty.size)), replace=False)
    dirty.reshape(-1)[cells] = np.nan
    dup_idx = rng.choice(len(dirty), n_dup, replace=False)
    uidx = np.concatenate([np.arange(len(dirty)), dup_idx])
    perm = rng.permutation(len(uidx))
    uidx = uidx[perm]
    rows, ys = dirty[uidx], y[uidx]
    # reference (generator tooling only): single eps after log10(amount)+z-score
    Zc, mean, std = card_z(clean)
    grid = np.round(np.arange(0.10, 1.60, 0.02), 2)
    cc = ~np.isnan(dirty).any(1)
    Zd = card_z(dirty[cc], mean, std)[0]
    pl = {mp: eps_run(Zc, y, mp, grid) for mp in minpts_list}
    pc = {mp: eps_run(Zd, y[cc], mp, grid) for mp in minpts_list}
    mp0 = 10 if 10 in minpts_list else minpts_list[len(minpts_list) // 2]
    eps_rec = round((max(pl[mp0][0], pc[mp0][0]) + min(pl[mp0][1], pc[mp0][1])) / 2, 2)
    okmp = [mp for mp in minpts_list if pl[mp][0] is not None and pc[mp][0] is not None
            and max(pl[mp][0], pc[mp][0]) <= eps_rec <= min(pl[mp][1], pc[mp][1])]
    lab = DBSCAN(eps=eps_rec, min_samples=mp0).fit_predict(Zc)
    kd = kdist(Zc, mp0)
    params = dict(
        n_rows=int(len(rows)), n_unique=int(n_u), n_duplicate_rows=int(n_dup),
        missing_cell_fraction=round(float(np.isnan(rows).mean()), 4),
        n_anomalies=int(n_anom), seed=SEED, simulated=True,
        reference_preprocessing=("drop rows with a missing cell and duplicate rows; log10(amount_vnd); "
                                 "z-score every column (population std)"),
        recommended_eps=eps_rec, recommended_minpts=mp0,
        minpts_range_with_same_eps=[min(okmp), max(okmp)] if okmp else None,
        eps_plateau_clean_latent={str(k): v for k, v in pl.items() if v[0] is not None},
        eps_plateau_complete_cases={str(k): v for k, v in pc.items() if v[0] is not None},
        kdist_minpts=mp0,
        kdist_percentiles={str(q): round(float(np.percentile(kd, q)), 3) for q in (50, 90, 95, 99)},
        reference_clusters=int(len(set(lab)) - (-1 in lab)),
        reference_ari_vs_segments=round(float(adjusted_rand_score(y, lab)), 4),
        reference_anomalies_labelled_noise=int(((lab == -1) & (y == -1)).sum()),
        segments={str(s): CARD_SEGS[s]["name"] for s in range(4)},
        scaling_reference=dict(columns=["log10_amount_vnd"] + CARD_COLS[1:],
                               mean=[round(float(v), 4) for v in mean], std=[round(float(v), 4) for v in std]))
    return dict(rows=rows, y=ys, uidx=uidx, clean=clean, yc=y, params=params, mean=mean, std=std)


def write_card_csv(path, c, prefix="TXN"):
    def cell(j, v):
        return "" if np.isnan(v) else [f"{int(v)}", f"{v:.2f}", f"{v:.2f}", f"{int(v)}", f"{v:.3f}"][j]
    dump(path, ["txn_id"] + CARD_COLS + ["segment_id"],
         [[f"{prefix}{u + 1:06d}"] + [cell(j, v) for j, v in enumerate(r)] + [int(s)]
          for u, r, s in zip(c["uidx"], c["rows"], c["y"])])


def write_card():
    CARD.mkdir(parents=True, exist_ok=True)
    c = make_card(CARD_N)
    write_card_csv(CARD / "card_txn_5d.csv", c)
    (CARD / "params.json").write_text(json.dumps(c["params"], indent=2) + "\n", encoding="utf-8", newline="\n")
    print("card_txn_5d", {k: c["params"][k] for k in (
        "n_rows", "missing_cell_fraction", "recommended_eps", "recommended_minpts",
        "minpts_range_with_same_eps", "reference_clusters", "reference_ari_vs_segments")})
    return c


# ---------- D. update streams ----------
def csv_text(cols, rows):
    s = io.StringIO()
    w = csv.writer(s, lineterminator="\n")
    w.writerow(cols); w.writerows(rows)
    return s.getvalue()


def canon(labels):
    """Relabel clusters 1.. by order of first appearance (-1 stays); ids only identify the partition."""
    m, out = {}, []
    for v in labels:
        out.append(-1 if v < 0 else m.setdefault(int(v), len(m) + 1))
    return out


def run_stream(X0, ids0, events, eps, minpts, to_row):
    """events: dicts(op, key|x, planted, tag). Returns csv rows and ambiguous ids of d0."""
    ids, X = list(ids0), np.array(X0, float)
    lab, core = ref_fit(X, eps, minpts)
    amb_ids = lambda I, XX, l, c: [I[i] for i in ambiguous_border(XX, l, c, eps)]
    amb = amb_ids(ids, X, lab, core)
    amb0 = list(amb)
    rows, nid = [], max(ids) + 1
    for step, ev in enumerate(events, 1):
        lab_b = dict(zip(ids, lab))
        if ev["op"] == "insert":
            p = nid; nid += 1; x = np.array(ev["x"], float)
            ids2, X2 = ids + [p], np.vstack([X, x])
            lab2, core2 = ref_fit(X2, eps, minpts)
            case = classify_insert(ids2, lab2, p, lab_b, dict(zip(ids, core)), dict(zip(ids2, core2)))
        else:
            p = ev["key"]; j = ids.index(p); x = X[j]
            ids2, X2 = ids[:j] + ids[j + 1:], np.delete(X, j, 0)
            lab2, core2 = ref_fit(X2, eps, minpts)
            case = classify_delete(ids, X, lab, core, ids2, X2, lab2, core2, p, eps)
        amb2 = amb_ids(ids2, X2, lab2, core2)
        planted = ev.get("planted", "")
        if planted and planted != case:
            raise RuntimeError(f"step {step}: planted {planted} but verified {case} ({ev.get('tag')})")
        if planted and (p in amb or p in amb2):
            raise RuntimeError(f"step {step}: planted event on an ambiguous border point {p}")
        rows.append(to_row(step, ev["op"], p, x, planted, case, amb2))
        ids, X, lab, core, amb = ids2, X2, lab2, core2, amb2
    return rows, amb0


GADGETS = {   # toy coordinates in Eps=1.1 units; scaled by 5/11 so Eps becomes 0.5 km, rotated per instance
    "create": dict(pts=dict(A=(0, 0), B=(1, 0), C=(2, 0)), p0=False,
                   ev=[("insert", "p", (1, .5), "Creation"), ("insert", "q", (1, -.5), "Absorption"),
                       ("delete", "p", None, "Reduction")]),
    "absorb": dict(pts=dict(A=(0, 0), B=(1, 0), C=(0, 1), D=(1, 1), E=(2, 0), F=(2, 1)), p0=False,
                   ev=[("insert", "p", (2.2, .5), "Absorption"), ("delete", "p", None, "Reduction")]),
    "merge": dict(pts={"A1": (-1, 0), "A2": (0, 0), "A3": (1, 0), "A4": (-1, 1), "A5": (0, 1), "A6": (1, 1),
                       "B1": (2.8, 0), "B2": (3.8, 0), "B3": (4.8, 0), "B4": (2.8, 1), "B5": (3.8, 1),
                       "B6": (4.8, 1)}, p0=False,
                  ev=[("insert", "p", (1.9, .5), "Merge"), ("delete", "p", None, "PotentialSplit_Split")]),
    "removal": dict(pts=dict(A=(-1, 0), B=(1, 0), C=(0, -1), D=(0, 1)), p0=(0, 0),
                    ev=[("delete", "p", None, "Removal"), ("insert", "p", (0, 0), "Creation")]),
    "reduce": dict(pts=dict(A=(0, 0), B=(1, 0), C=(0, 1), D=(1, 1), E=(2, 0), F=(2, 1)), p0=(2.2, .5),
                   ev=[("delete", "p", None, "Reduction"), ("insert", "p", (2.2, .5), "Absorption")]),
    "split": dict(pts=dict(A=(1, 0), B=(.2, 0), C=(.6, .6), D=(.6, -.6), E=(3, 0), F=(3.8, 0), G=(3.4, .6),
                           H=(3.4, -.6), I=(2, 1)), p0=(2, 0),
                  ev=[("delete", "p", None, "PotentialSplit_Split"), ("insert", "p", (2, 0), "Merge")]),
    "nonsplit": dict(pts=dict(A=(1, 0), B=(.2, 0), C=(.6, .6), D=(.6, -.6), E=(3, 0), F=(3.8, 0), G=(3.4, .6),
                              H=(3.4, -.6), I=(2, 1), J=(1, 1), K=(3, 1), L=(2, 2)), p0=(2, 0),
                     ev=[("delete", "p", None, "PotentialSplit_NonSplit"), ("insert", "p", (2, 0), "Absorption")]),
}
CELL_M = 4000


def toy_to_m(xy, rot, bb):
    x, y = Fraction(str(xy[0])) - bb[0], Fraction(str(xy[1])) - bb[1]
    for _ in range(rot):
        x, y = -y, x
    return int(round(x * Fraction(5, 11) * 1000)), int(round(y * Fraction(5, 11) * 1000))


def build_atm_stream():
    """Scripted ~120-step stream. Returns {filename: text}. Pure function of SEED."""
    rng = np.random.default_rng(SEED + 2)
    eps_m, minpts = 500, GEO_MINPTS
    cells = [(8000 + CELL_M * i + CELL_M // 2, CELL_M * j + CELL_M // 2) for j in range(7) for i in range(7)]
    cells = [cells[i] for i in rng.permutation(len(cells))]
    d0, seqs = [], []                       # d0: (key, x_m, y_m)
    plan = [("create", 7)] + [(k, 6) for k in ("absorb", "merge", "removal", "reduce", "split", "nonsplit")]
    ci = 0
    for kind, m in plan:
        g = GADGETS[kind]
        for inst in range(m):
            cx, cy = cells[ci]; ci += 1
            rot = int(rng.integers(0, 4))
            allxy = list(g["pts"].values()) + ([g["p0"]] if g["p0"] else [])
            F = lambda v: Fraction(str(v))
            bb = ((min(F(a) for a, _ in allxy) + max(F(a) for a, _ in allxy)) / 2,
                  (min(F(b) for _, b in allxy) + max(F(b) for _, b in allxy)) / 2)
            tag = f"{kind}{inst}"
            for nm, xy in g["pts"].items():
                a, b = toy_to_m(xy, rot, bb); d0.append((f"{tag}:{nm}", cx + a, cy + b))
            if g["p0"]:
                a, b = toy_to_m(g["p0"], rot, bb); d0.append((f"{tag}:p0", cx + a, cy + b))
            seq = []
            for op, who, xy, case in g["ev"]:
                if op == "insert":
                    a, b = toy_to_m(xy, rot, bb)
                    seq.append(dict(op="insert", x=(cx + a, cy + b), planted=case, tag=tag, who=who))
                else:
                    seq.append(dict(op="delete", planted=case, tag=tag))
            seqs.append(seq)
    blobs = [("blobA", (3500, 21500), 800, 110), ("blobB", (4500, 7500), 700, 80)]
    for nm, cen, sd, m in blobs:            # background hotspots in the left strip
        B = np.array(cen) + trunc_gauss(rng, m, sd).round().astype(int)
        d0 += [(f"{nm}:{i}", int(a), int(b)) for i, (a, b) in enumerate(B)]
    d0 = [d0[i] for i in rng.permutation(len(d0))]
    P0 = fix_ties(np.array([[x, y] for _, x, y in d0]), eps_m ** 2)
    d0 = [(k, int(P0[i, 0]), int(P0[i, 1])) for i, (k, _, _) in enumerate(d0)]
    for _ in range(7):                       # planted Absorption inside the background blobs
        _, cen, sd, _ = blobs[int(rng.integers(0, 2))]
        q = np.array(cen) + np.round(rng.normal(0, 1, 2) * sd * 0.5).astype(int)
        seqs.append([dict(op="insert", x=(int(q[0]), int(q[1])), planted="Absorption", tag="blob")])
    blob_keys = [k for k, _, _ in d0 if k.startswith("blob")]
    for k in rng.choice(blob_keys, 7, replace=False):    # natural deletes inside blobs (not planted)
        seqs.append([dict(op="delete", direct=str(k), planted="", tag="blob")])
    for i in range(12):                      # Noise inserts in the leftover cells
        cx, cy = cells[ci + i // 4]
        seqs.append([dict(op="insert", x=(cx + (-1000, 1000)[i % 2], cy + (-1000, 1000)[(i // 2) % 2]),
                          planted="Noise", tag="noise")])
    order = []                               # random interleave, per-sequence order kept
    pool = [list(s) for s in seqs]
    while pool:
        w = np.array([len(s) for s in pool], float)
        i = int(rng.choice(len(pool), p=w / w.sum()))
        order.append(pool[i].pop(0))
        if not pool[i]:
            pool.pop(i)
    ids0 = list(range(len(d0)))
    key2id = {k: i for i, (k, _, _) in enumerate(d0)}
    cur = {k.split(":")[0]: i for k, i in key2id.items() if k.endswith(":p0")}
    events, nid = [], len(d0)
    for ev in order:                         # resolve delete targets to point ids
        if ev["op"] == "insert":
            if ev.get("who") == "p":
                cur[ev["tag"]] = nid
            events.append(dict(op="insert", x=ev["x"], planted=ev["planted"], tag=ev["tag"])); nid += 1
        else:
            pid = key2id[ev["direct"]] if "direct" in ev else cur.pop(ev["tag"])
            events.append(dict(op="delete", key=pid, planted=ev["planted"], tag=ev["tag"]))
    X0 = np.array([[x, y] for _, x, y in d0], float)
    allp = np.vstack([P0] + [np.array([e["x"]]) for e in events if e["op"] == "insert"])
    if tie_pairs(allp.astype(np.int64), eps_m ** 2):
        raise RuntimeError("atm stream has a tie at distance exactly eps")
    t0 = datetime(2025, 3, 1)

    def to_row(step, op, p, x, planted, case, amb):
        return [step, (t0 + timedelta(days=step - 1)).strftime("%Y-%m-%d"), op, p,
                f"{x[0] / 1000:.3f}", f"{x[1] / 1000:.3f}", planted, case, ";".join(map(str, amb))]

    rows, amb0 = run_stream(X0, ids0, events, eps_m, minpts, to_row)
    lab0, _ = ref_fit(X0, eps_m, minpts)
    d0_rows = [[i, f"{x / 1000:.3f}", f"{y / 1000:.3f}", l] for i, (x, y), l in zip(
        ids0, X0.astype(int), canon(lab0))]
    cnt = defaultdict(int)
    for r in rows:
        cnt[r[7]] += 1
    params = dict(
        simulated=True, seed=SEED, eps_km=0.5, minpts=minpts, metric="euclidean", units="km",
        neighbourhood_includes_self=True, n_d0=len(d0), n_steps=len(rows),
        timestamp="1 step = 1 day from 2025-03-01",
        d0_reference_clusters=int(max(canon(lab0))), d0_ambiguous_border_ids=sorted(amb0),
        verified_case_counts=dict(sorted(cnt.items())), case_definitions=CASE_DEFS,
        id_note="point_id is never reused; a re-inserted point gets a new id. Delete rows carry the deleted point's coordinates for convenience.",
        ambiguous_note="ambiguous_border_ids lists, AFTER the step, border points within eps of cores of >= 2 clusters; planted events never fall on such points; exclude them when comparing labels (ARI) with a full re-fit.")
    return {"d0.csv": csv_text(["point_id", "x_km", "y_km", "ref_label"], d0_rows),
            "updates.csv": csv_text(["step", "timestamp", "op", "point_id", "x_km", "y_km", "planted_case",
                                     "verified_case", "ambiguous_border_ids"], rows),
            "params.json": json.dumps(params, indent=2) + "\n"}


def build_card_stream(card=None, W=500, steps=300, extra=None):
    """Natural sliding window: insert a new txn, then delete the oldest, alternately.
    extra: optional dict that receives 'segs' (true segment of every point id; side channel only)."""
    card = card or make_card(CARD_N)
    mean, std = card["mean"], card["std"]
    rng = np.random.default_rng(SEED + 3)
    n_ins = steps // 2

    segs_log = []

    def draw(mix, anom):
        if rng.random() < anom:
            segs_log.append(-1)
            return card_anomalies(rng, 1, card["clean"])[0]
        s = int(rng.choice(4, p=mix))
        segs_log.append(s)
        return card_sample(rng, s, 1)[0]

    mix0, mix1 = np.array([.45, .33, .22, 0.0]), np.array([.30, .22, .03, .45])
    A0 = np.array([draw(mix0, 0.01) for _ in range(W)])
    arr = np.array([draw((1 - t / (n_ins - 1)) * mix0 + t / (n_ins - 1) * mix1, 0.02) for t in range(n_ins)])
    # eps for a window of W points: smallest eps giving >= 3 clusters and < 6 % noise in the first
    # and the last window, plus a 0.05 safety margin
    minpts, last = 6, np.vstack([A0[n_ins:], arr])
    eps = None
    for e in np.round(np.arange(0.30, 1.01, 0.05), 2):
        ok = True
        for Aw in (A0, last):
            lab = DBSCAN(eps=float(e), min_samples=minpts).fit_predict(card_z(Aw, mean, std)[0])
            ok &= (len(set(lab)) - (-1 in lab)) >= 3 and (lab == -1).mean() < 0.06
        if ok:
            eps = round(float(e) + 0.05, 2); break
    t0 = datetime(2025, 3, 1)
    feats = {i: A0[i] for i in range(W)}
    feats.update({W + k: arr[k] for k in range(n_ins)})
    events, nxt, queue = [], W, list(range(W))
    for s in range(1, steps + 1):
        if s % 2 == 1:
            events.append(dict(op="insert", x=card_z(arr[(s - 1) // 2][None, :], mean, std)[0][0], planted=""))
            queue.append(nxt); nxt += 1
        else:
            events.append(dict(op="delete", key=queue.pop(0), planted=""))

    def to_row(step, op, p, x, planted, case, amb):
        r = feats[p]
        return [step, (t0 + timedelta(hours=step - 1)).strftime("%Y-%m-%d %H:%M"), op, p,
                f"{int(r[0])}", f"{r[1]:.2f}", f"{r[2]:.2f}", f"{int(r[3])}", f"{r[4]:.3f}",
                planted, case, ";".join(map(str, amb))]

    Z0 = card_z(A0, mean, std)[0]
    rows, amb0 = run_stream(Z0, list(range(W)), events, eps, minpts, to_row)
    lab0, _ = ref_fit(Z0, eps, minpts)
    d0_rows = [[i, f"{int(a[0])}", f"{a[1]:.2f}", f"{a[2]:.2f}", f"{int(a[3])}", f"{a[4]:.3f}", l]
               for i, (a, l) in enumerate(zip(A0, canon(lab0)))]
    cnt = defaultdict(int)
    for r in rows:
        cnt[r[10]] += 1
    params = dict(
        simulated=True, seed=SEED, eps=eps, minpts=minpts, metric="euclidean (after frozen scaling)",
        neighbourhood_includes_self=True, n_d0=W, n_steps=len(rows),
        timestamp="1 step = 1 hour from 2025-03-01 00:00; odd steps insert a new txn, even steps delete the oldest",
        frozen_scaling=dict(columns=["log10_amount_vnd"] + CARD_COLS[1:],
                            mean=[float(v) for v in mean], std=[float(v) for v in std],
                            note="apply log10 to amount_vnd then (x - mean) / std with these constants; do not re-fit on the stream"),
        drift="arrival mix drifts toward night_cash_out (new behaviour appears) while travel_big_ticket fades; about 2% anomalies",
        d0_reference_clusters=int(max(canon(lab0))), d0_ambiguous_border_ids=sorted(amb0),
        verified_case_counts=dict(sorted(cnt.items())), case_definitions=CASE_DEFS,
        id_note="point_id never reused. The stream is clean (no missing cells, no duplicate rows).",
        ambiguous_note="ambiguous_border_ids lists, AFTER the step, border points within eps of cores of >= 2 clusters; exclude them when comparing labels with a full re-fit. planted_case is empty: this stream is natural.")
    if extra is not None:
        extra["segs"] = segs_log
    return {"d0.csv": csv_text(["point_id"] + CARD_COLS + ["ref_label"], d0_rows),
            "updates.csv": csv_text(["step", "timestamp", "op", "point_id"] + CARD_COLS +
                                    ["planted_case", "verified_case", "ambiguous_border_ids"], rows),
            "params.json": json.dumps(params, indent=2) + "\n"}


def write_streams(card):
    extra = {}
    for sub, files in (("atm_stream", build_atm_stream()), ("card_stream", build_card_stream(card, extra=extra))):
        base = (GEO if sub == "atm_stream" else CARD) / sub
        base.mkdir(parents=True, exist_ok=True)
        for fn, txt in files.items():
            (base / fn).write_text(txt, encoding="utf-8", newline="\n")
        p = json.loads(files["params.json"])
        print(sub, "steps", p["n_steps"], "d0", p["n_d0"], p["verified_case_counts"],
              "ambiguous in d0:", len(p["d0_ambiguous_border_ids"]))
    return extra["segs"]


# ---------- business names for hotspots / segments ----------
HOTSPOTS = [
    (0, "downtown_office_cash", "Downtown office cash", "Rút tiền khu văn phòng trung tâm",
     "Cụm dày ở trung tâm, nhân viên văn phòng rút tiền giờ nghỉ trưa và cuối ngày.",
     "Đặt thêm máy ATM và chuẩn bị tiền mặt đủ vào giờ cao điểm."),
    (1, "student_campus", "Student campus withdrawals", "Rút tiền khu đại học",
     "Cụm gọn quanh trường đại học, giao dịch nhỏ và đều của sinh viên.",
     "Chạy ưu đãi tài khoản sinh viên, ưu tiên máy ATM nạp tiền mệnh giá nhỏ."),
    (2, "market_ring", "Market-edge cash withdrawals", "Rút tiền quanh chợ",
     "Các máy ATM xếp thành vòng quanh khu chợ, khách rút tiền trước khi vào chợ.",
     "Kiểm tra hạn mức tiền trong các máy quanh chợ vào buổi sáng và cuối tuần."),
    (3, "market_plaza", "Market plaza traders", "Tiểu thương trong khu chợ",
     "Cụm nhỏ rất dày ở trung tâm chợ, tiểu thương rút và gửi tiền lẻ liên tục.",
     "Đề xuất gói thu tiền QR cho tiểu thương, đặt máy nạp tiền mặt."),
    (4, "riverside_cash", "Riverside cash withdrawals", "Rút tiền dọc bờ sông",
     "Dải cong dọc bờ sông (lưỡi liềm), dân cư và quán xá ven sông.",
     "Rà soát độ phủ ATM dọc tuyến ven sông, nhất là hai đầu dải."),
    (5, "highway_travellers", "Highway travellers", "Người đi đường trên quốc lộ",
     "Dải dài thẳng dọc quốc lộ, lái xe và khách đi đường rút tiền khi dừng chân.",
     "Đặt ATM ở các trạm dừng chân, theo dõi tiền mặt và an ninh ban đêm."),
    (-1, "isolated", "Isolated transactions", "Giao dịch cô lập hoặc bất thường",
     "Điểm nằm riêng lẻ ngoài mọi hotspot, DBSCAN coi là nhiễu.",
     "Không mở rộng ATM theo từng điểm lẻ; xem lại nếu cụm lẻ lặp lại nhiều tuần."),
]
SEGMENTS = [
    (0, "daily_retail", "Daily retail spending", "Chi tiêu bán lẻ hằng ngày",
     "Giao dịch nhỏ quanh 200 nghìn đồng, giữa ngày, gần nhà, rủi ro thấp, khoảng 9 giao dịch mỗi tuần.",
     "Khách cốt lõi: thưởng hoàn tiền theo tần suất, đề xuất thẻ tín dụng hạn mức nhỏ."),
    (1, "online_shopping", "Night online shopping", "Mua sắm online buổi tối",
     "Giao dịch khoảng 800 nghìn đồng vào buổi tối, ngay tại nhà, rủi ro trung bình thấp.",
     "Ưu đãi thanh toán trả góp và ví điện tử vào khung 20 đến 22 giờ."),
    (2, "travel_big_ticket", "Far-from-home big ticket travel", "Chi tiêu lớn khi đi xa",
     "Giao dịch khoảng 10 triệu đồng, cách nhà khoảng 60 km, ít giao dịch, rủi ro trung bình.",
     "Cung cấp bảo hiểm du lịch, nâng hạn mức tạm thời; theo dõi xác thực thêm khi đi xa."),
    (3, "night_cash_out", "Night cash out", "Rút tiền mặt ban đêm",
     "Giao dịch nhỏ khoảng 40 nghìn đồng lúc rạng sáng, rất nhiều lần mỗi tuần, rủi ro cao.",
     "Theo dõi sát: giới hạn số lần rút ban đêm, gửi cảnh báo và hỏi xác nhận khi bất thường."),
    (-1, "suspicious_far_high_value", "Suspicious far-from-home high-value", "Giao dịch nghi ngờ giá trị lớn ở xa nhà",
     "Số tiền cực lớn hoặc cực nhỏ, rất xa nhà, rủi ro merchant rất cao: giao dịch cô lập hoặc bất thường.",
     "Chặn tạm thời hoặc gọi xác nhận khách hàng, chuyển bộ phận chống gian lận."),
]
META_COLS = ["id", "name_en", "name_vi", "meaning_vi", "suggested_action_vi"]


def write_meta():
    dump(GEO / "hotspots.csv", ["hotspot_id", "code"] + META_COLS[1:], [[h[0], h[1], *h[2:]] for h in HOTSPOTS])
    dump(CARD / "segments.csv", ["segment_id", "code"] + META_COLS[1:], [[h[0], h[1], *h[2:]] for h in SEGMENTS])


def profile_tables():
    """Markdown profile tables (mean / median per feature) from the generated files."""
    import pandas as pd
    cd = pd.read_csv(CARD / "card_txn_5d.csv").drop_duplicates().dropna()
    out = ["| segment_id | tên | n | " + " | ".join(f"{c} (mean / median)" for c in CARD_COLS) + " |",
           "|---|---|---|" + "---|" * len(CARD_COLS)]
    for sid, nm, *_ in SEGMENTS:
        g = cd[cd.segment_id == sid]
        cells = [f"{g[c].mean():,.2f} / {g[c].median():,.2f}" for c in CARD_COLS]
        out.append(f"| {sid} | {nm} | {len(g)} | " + " | ".join(cells) + " |")
    gd = pd.read_csv(GEO / "atm_txn_2d.csv")
    geo = ["| hotspot_id | tên | n | x_km (mean / median) | y_km (mean / median) |", "|---|---|---|---|---|"]
    for hid, nm, *_ in HOTSPOTS:
        g = gd[gd.hotspot_id == hid]
        geo.append(f"| {hid} | {nm} | {len(g)} | {g.x_km.mean():.2f} / {g.x_km.median():.2f} | "
                   f"{g.y_km.mean():.2f} / {g.y_km.median():.2f} |")
    return "\n".join(out), "\n".join(geo)


# ---------- benchmark variants (--n) ----------
def write_bench(ns):
    B = RAW / "bench"
    B.mkdir(parents=True, exist_ok=True)
    for n in ns:
        eps_km = round(GEO_EPS_KM * (GEO_N / n) ** 0.5, 3)   # keep eps matched to the point density
        P, y, eps_m = make_geo(n, eps_km)
        write_geo_csv(B / f"atm_txn_2d_n{n}.csv", P, y)
        rep, _, _ = geo_report(P, y, eps_m, GEO_MINPTS)
        c = make_card(n, minpts_list=(10,))
        write_card_csv(B / f"card_txn_5d_n{n}.csv", c)
        (B / f"params_n{n}.json").write_text(json.dumps(dict(atm=rep, card={k: c["params"][k] for k in (
            "n_rows", "recommended_eps", "recommended_minpts", "reference_clusters",
            "reference_ari_vs_segments")}), indent=2) + "\n", encoding="utf-8", newline="\n")
        print("bench", n, rep, "card eps", c["params"]["recommended_eps"], "ARI", c["params"]["reference_ari_vs_segments"])


# ---------- preview PNGs ----------
def write_preview():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    PREVIEW.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "figure.dpi": 140})
    pal = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756", "#72B7B2"]
    gray = "#9AA0A6"
    # 1. ATM map: truth vs KMeans vs DBSCAN
    df = pd.read_csv(GEO / "atm_txn_2d.csv")
    P = (df[["x_km", "y_km"]].values * 1000).round().astype(np.int64)
    rep, lab, km = geo_report(P, df["hotspot_id"].values, 500, GEO_MINPTS)
    names = {0: "trung tâm", 1: "khu đại học", 2: "vòng quanh chợ", 3: "quảng trường chợ", 4: "ven sông", 5: "quốc lộ"}
    fig, ax = plt.subplots(1, 3, figsize=(19, 5.8), sharex=True, sharey=True)

    def draw(a, labels, title, legend=False):
        for l in sorted(set(labels)):
            m = labels == l
            if l < 0:
                a.scatter(df.x_km[m], df.y_km[m], s=14, c=gray, marker="x", lw=.8, label="cô lập / nhiễu")
            else:
                a.scatter(df.x_km[m], df.y_km[m], s=9, c=pal[l % 6], alpha=.85, label=names[l] if legend else None)
        a.set_title(title, fontsize=10.5); a.set_aspect("equal"); a.set_xlabel("x (km)")
    draw(ax[0], df.hotspot_id.values, "Nhãn thật: 6 hotspot + điểm cô lập", True)
    ax[0].set_ylabel("y (km)")
    ax[0].legend(loc="upper center", ncol=3, fontsize=8, frameon=False, bbox_to_anchor=(.5, -.12))
    draw(ax[1], km, f"KMeans k=6, ARI {rep['ari_kmeans6_vs_truth']:.2f}: cắt đôi vòng và lưỡi liềm")
    draw(ax[2], lab, f"DBSCAN Eps=0.5 km, MinPts=4, ARI {rep['ari_dbscan_vs_truth']:.2f}")
    fig.suptitle("Thành phố giả lập: giao dịch ATM (dữ liệu mô phỏng)", fontsize=13, fontweight="bold")
    fig.tight_layout(); fig.savefig(PREVIEW / "atm_map.png", bbox_inches="tight"); plt.close(fig)
    # 2. card scatter matrices, before and after scaling
    cd = pd.read_csv(CARD / "card_txn_5d.csv").drop_duplicates().dropna()
    _, mean, std = card_z(cd[CARD_COLS].values)
    sub = cd.sample(450, random_state=SEED)
    seg = sub.segment_id.values
    cpal = {0: pal[0], 1: pal[1], 2: pal[2], 3: pal[3], -1: "#D62728"}
    sname = {0: "bán lẻ hằng ngày", 1: "mua sắm online", 2: "chi tiêu lớn khi đi xa", 3: "rút tiền đêm", -1: "bất thường"}
    for tag, M, labs, title in (
            ("before", sub[CARD_COLS].values, CARD_COLS, "Giao dịch thẻ TRƯỚC khi chuẩn hóa: đơn vị gốc, số tiền chênh nhiều bậc độ lớn"),
            ("after", card_z(sub[CARD_COLS].values, mean, std)[0], ["log10(amount)"] + CARD_COLS[1:],
             "Giao dịch thẻ SAU log10(số tiền) + z-score")):
        fig, axs = plt.subplots(5, 5, figsize=(13, 12))
        for i in range(5):
            for j in range(5):
                a = axs[i, j]
                if i == j:
                    a.hist(M[:, i], bins=30, color=gray)
                else:
                    for s in (0, 1, 2, 3, -1):
                        m = seg == s
                        a.scatter(M[m, j], M[m, i], s=7 if s >= 0 else 34, c=cpal[s], alpha=.8,
                                  marker="o" if s >= 0 else "x", label=sname[s] if (i, j) == (1, 0) else None)
                a.tick_params(labelsize=7)
                if i == 4: a.set_xlabel(labs[j], fontsize=8)
                if j == 0: a.set_ylabel(labs[i], fontsize=8)
        fig.legend(*axs[1, 0].get_legend_handles_labels(), loc="upper center", ncol=5, frameon=False,
                   bbox_to_anchor=(.5, .965))
        fig.suptitle(title + " (dữ liệu mô phỏng)", fontsize=12, fontweight="bold", y=.995)
        fig.tight_layout(rect=(0, 0, 1, .95)); fig.savefig(PREVIEW / f"card_scatter_{tag}.png", bbox_inches="tight")
        plt.close(fig)
    # 3. stream timeline
    allc = INSERT_CASES + DELETE_CASES + ["DeleteNoise"]
    cc = dict(zip(allc, ["#9AA0A6", "#54A24B", "#4C78A8", "#B279A2", "#E45756", "#F58518", "#72B7B2", "#8C564B", "#C8C8C8"]))
    a_up = pd.read_csv(GEO / "atm_stream" / "updates.csv")
    c_up = pd.read_csv(CARD / "card_stream" / "updates.csv")
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(13, 8.5))
    for k, c in enumerate(allc):
        m = a_up.verified_case == c
        a1.scatter(a_up.step[m], [k] * int(m.sum()), s=38, c=cc[c], marker="s" if k < 4 else "o")
    a1.set_yticks(range(len(allc)))
    a1.set_yticklabels([f"{c} ({int((a_up.verified_case == c).sum())})" for c in allc], fontsize=8)
    a1.invert_yaxis(); a1.axhline(3.5, color="#DDD", lw=1)
    a1.set_xlabel("bước (1 bước = 1 ngày)")
    a1.set_title("Stream ATM kịch bản sẵn: case đã kiểm chứng của từng bước (vuông: chèn, tròn: xóa)")
    c_up["bin"] = (c_up.step - 1) // 25
    tab = c_up.groupby(["bin", "verified_case"]).size().unstack(fill_value=0)
    bottom = np.zeros(len(tab))
    for c in allc:
        if c in tab:
            a2.bar(tab.index * 25 + 12.5, tab[c], 22, bottom=bottom, color=cc[c], label=f"{c} ({int(tab[c].sum())})")
            bottom += tab[c].values
    a2.set_xlabel("bước (gom 25 bước; 1 bước = 1 giờ)"); a2.set_ylabel("số update")
    a2.set_title("Stream thẻ: cửa sổ trượt tự nhiên, số case mỗi nhóm bước")
    a2.set_ylim(0, 34); a2.legend(ncol=3, fontsize=8, frameon=False, loc="upper center")
    fig.suptitle("Các stream cập nhật: số lượng từng case đã kiểm chứng (dữ liệu mô phỏng)", fontsize=13, fontweight="bold")
    fig.tight_layout(); fig.savefig(PREVIEW / "stream_timeline.png", bbox_inches="tight"); plt.close(fig)
    print("preview ->", PREVIEW)


def main():
    ap = argparse.ArgumentParser(description="Generate the raw datasets (fixed seed 42).")
    ap.add_argument("--n", type=int, nargs="+",
                    help="also write benchmark variants with this many points, e.g. --n 500 1000 2000 4000 "
                         "(data/raw/bench/, not meant for committing)")
    ap.add_argument("--preview", action="store_true", help="write PNG previews to data/preview/")
    ap.add_argument("--difficulty", nargs="+", choices=["easy", "medium", "hard"],
                    help="variants to write. easy = the base files and hard = the committed hard variant are ALWAYS written "
                         "(byte-identical base columns); add 'medium' to also write the regenerable-only medium files.")
    a = ap.parse_args()
    write_toy(); write_synthetic(); write_nd()
    write_geo()
    card = write_card()
    segs = write_streams(card)
    write_meta()
    import gen_edge
    import gen_enrich
    import gen_hard
    levels = ["hard"] + (["medium"] if "medium" in (a.difficulty or []) else [])   # hard is committed, so always rebuilt
    hard = None
    flags = {}
    if levels:
        tables, flags = gen_hard.write_variants(levels)
        hard = dict(tables=tables, stream_segs=None)
        hard["stream_segs"] = gen_hard.write_hard_stream(flags["hard"]["card"])
    gen_enrich.enrich_all(card, segs, hard)
    gen_edge.write_all(RAW)
    rep, diff_data = gen_hard.run_difficulty(flags)
    import gen_dictionary
    print('DATA_DICTIONARY rows:', gen_dictionary.write(RAW / 'DATA_DICTIONARY.csv'))
    for fam in ("atm", "card"):
        for lvl in ("easy", "medium", "hard"):
            print("difficulty", fam, lvl, {k: v[0] for k, v in rep[fam][lvl]["results"].items()})
    if a.n:
        write_bench(a.n)
    if a.preview:
        write_preview()
        import gen_preview
        gen_preview.write_all(PREVIEW, rep, diff_data)
    import gen_readme
    gen_readme.update()
    print("done", RAW)


if __name__ == "__main__":
    main()
