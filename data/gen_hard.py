"""Part 2: harder variants (medium, hard) of the card and ATM data, plus the difficulty report.

easy = the original files (untouched). medium and hard are parametrised by t in (0, 1]: every
generation parameter moves linearly from the easy value (t = 0) to the hard value (t = 1), medium is
t = 0.65. sklearn is used HERE ONLY (reference clustering for the difficulty report), never in src/.
"""
import json

import numpy as np

SEED = 42
OFF = dict(card_hard=20, card_medium=21, geo_hard=22, geo_medium=23, desc_hard=31, dirt_hard=32, stream_hard=24,
           desc_medium=33, dirt_medium=34)
T = dict(medium=0.65, hard=1.0)


def lerp(a, b, t):
    return a + t * (b - a)


# ====================================================================== card family
EASY_F = {0: [(5.3, .10), (12.5, 1.6), (4.0, 1.5), (9, 1.5), (.12, .035)],
          1: [(5.9, .10), (21.0, 1.2), (1.0, .6), (4, 1.0), (.35, .04)],
          2: [(7.0, .11), (15.0, 2.0), (60, 6), (2, .8), (.45, .05)],
          3: [(4.6, .08), (1.5, .9), (8, 2.5), (14, 2), (.65, .05)]}
HARD_F = {0: [(5.4, .18), (13.5, 2.6), (4.0, 2.0), (8.7, 2.0), (.14, .05)],        # diffuse
          1: [(5.75, .14), (20.0, 2.0), (1.6, .9), (4.7, 1.2), (.28, .045)],          # overlaps retail
          2: [(6.9, .16), (15.0, 2.5), (50, 12), (2.3, .9), (.40, .06)],
          3: [(4.6, .06), (1.5, .7), (8, 1.8), (14, 1.5), (.65, .04)]}              # tight
LARGE_ONLINE = [(6.5, .15), (19.5, 3.0), (22, 9), (3, 1), (.32, .05)]               # online, big basket, farther away
P_EASY, P_HARD = np.array([.38, .28, .16, .18]), np.array([.36, .30, .14, .20])


def card_spec(t):
    f = {s: [(lerp(EASY_F[s][j][0], HARD_F[s][j][0], t), lerp(EASY_F[s][j][1], HARD_F[s][j][1], t)) for j in range(5)]
         for s in range(4)}
    p = lerp(P_EASY, P_HARD, t)
    return dict(f=f, p=p / p.sum(), large_online=.12 * t, n_bridge=int(round(12 * t)), near_anom=.010 * t)


def _sample(rng, f, m):
    from generate_data import card_round
    la = rng.normal(*f[0], m)
    return card_round(np.column_stack([10 ** la] + [rng.normal(*f[i], m) for i in range(1, 5)]))


def make_card_level(level, n=1600):
    """Same return structure as generate_data.make_card (rows, y, uidx, clean, yc, mean, std, params)
    plus 'flags' (per clean row: is_bridge, is_near_anomaly, is_large_online)."""
    import generate_data as gen
    t = T[level]
    sp = card_spec(t)
    rng = np.random.default_rng(SEED + OFF[f"card_{level}"])
    n_dup = int(round(.005 * n))
    n_u = n - n_dup
    n_far = max(int(round(.01 * n)), 6)
    n_near = int(round(sp["near_anom"] * n))
    n_br = sp["n_bridge"]
    n_seg = n_u - n_far - n_near - n_br
    cnt = [int(round(n_seg * sp["p"][s])) for s in range(4)]
    cnt[0] += n_seg - sum(cnt)
    blocks, ys, flag_br, flag_lo = [], [], [], []
    for s in range(4):
        if s == 1:
            m_lo = int(round(cnt[1] * sp["large_online"]))
            blocks += [_sample(rng, sp["f"][1], cnt[1] - m_lo), _sample(rng, LARGE_ONLINE, m_lo)]
            ys += [np.full(cnt[1], 1)]
            flag_lo += [np.zeros(cnt[1] - m_lo, bool), np.ones(m_lo, bool)]
            flag_br += [np.zeros(cnt[1], bool)]
        else:
            blocks.append(_sample(rng, sp["f"][s], cnt[s]))
            ys.append(np.full(cnt[s], s))
            flag_lo.append(np.zeros(cnt[s], bool)); flag_br.append(np.zeros(cnt[s], bool))
    A = np.vstack(blocks)
    y = np.concatenate(ys)
    is_br = np.concatenate(flag_br)
    is_lo = np.concatenate(flag_lo)
    # thin bridge between retail (0) and online (1): points on the line between the two mean vectors
    if n_br:
        m0 = np.array([10 ** sp["f"][0][0][0]] + [sp["f"][0][j][0] for j in range(1, 5)])
        m1 = np.array([10 ** sp["f"][1][0][0]] + [sp["f"][1][j][0] for j in range(1, 5)])
        u = np.linspace(.12, .88, n_br)
        la = lerp(sp["f"][0][0][0], sp["f"][1][0][0], u) + rng.normal(0, .02, n_br)
        B = np.column_stack([10 ** la] + [lerp(m0[j], m1[j], u) + rng.normal(0, s, n_br)
                                          for j, s in zip(range(1, 5), (.3, .25, .25, .008))])
        Bb = gen.card_round(B)
        A = np.vstack([A, Bb])
        y = np.concatenate([y, np.where(u < .5, 0, 1)])
        is_br = np.concatenate([is_br, np.ones(n_br, bool)])
        is_lo = np.concatenate([is_lo, np.zeros(n_br, bool)])
    Zc, mean, std = gen.card_z(A)
    # far anomalies (like the easy file) and near anomalies (a normal-looking txn nudged out of its group)
    far = gen.card_anomalies(rng, n_far, A, min_d=1.6 if t < 1 else 1.4)
    near = []
    while len(near) < n_near:
        i = int(rng.integers(0, len(A)))
        d = rng.normal(size=5); d /= np.linalg.norm(d)
        z = Zc[i] + d * rng.uniform(.5, .8)
        raw = z * std + mean
        raw[0] = 10 ** raw[0]
        a = gen.card_round(raw[None, :])[0]
        zz = gen.card_z(a[None, :], mean, std)[0][0]
        dmin = np.sqrt(((Zc - zz) ** 2).sum(1)).min()
        if .35 <= dmin <= .8:
            near.append(a)
    parts = [A, far] + ([np.array(near)] if n_near else [])
    A = np.vstack(parts)
    y = np.concatenate([y, np.full(n_far + n_near, -1)])
    is_br = np.concatenate([is_br, np.zeros(n_far + n_near, bool)])
    is_lo = np.concatenate([is_lo, np.zeros(n_far + n_near, bool)])
    is_near = np.concatenate([np.zeros(len(A) - n_far - n_near, bool), np.zeros(n_far, bool), np.ones(n_near, bool)])
    perm = rng.permutation(len(A))
    A, y, is_br, is_lo, is_near = A[perm], y[perm], is_br[perm], is_lo[perm], is_near[perm]
    clean, dirty = A.copy(), A.copy()
    cells = rng.choice(dirty.size, int(round(.015 * dirty.size)), replace=False)
    dirty.reshape(-1)[cells] = np.nan
    dup_idx = rng.choice(len(dirty), n_dup, replace=False)
    uidx = rng.permutation(np.concatenate([np.arange(len(dirty)), dup_idx]))
    _, mean, std = gen.card_z(clean)
    params = dict(level=level, t=t, n_rows=int(len(uidx)), n_unique=int(len(A)), n_duplicate_rows=int(n_dup),
                  n_bridge=int(n_br), n_near_anomalies=int(n_near), n_far_anomalies=int(n_far),
                  large_online_share_of_online=round(sp["large_online"], 3), seed=SEED + OFF[f"card_{level}"])
    return dict(rows=dirty[uidx], y=y[uidx], uidx=uidx, clean=clean, yc=y, mean=mean, std=std, params=params,
                flags=dict(is_bridge=is_br, is_near_anomaly=is_near, is_large_online=is_lo))


def card_eval_arrays(rows, y, txn_key):
    """Reference preprocessing used by the difficulty report: drop rows with a missing cell, drop
    duplicate transactions (same key), log10(amount), z-score. Returns Z, y."""
    import generate_data as gen
    keep, seen = [], set()
    for i, k in enumerate(txn_key):
        if k in seen or np.isnan(rows[i]).any():
            continue
        seen.add(k); keep.append(i)
    Z, _, _ = gen.card_z(rows[keep])
    return Z, np.asarray(y)[keep]


# ====================================================================== ATM family
def geo_spec(t):
    return dict(
        dt_c=(lerp(8, 19.2, t), lerp(16, 10.0, t)), dt_sd=lerp(.9, .95, t), campus_sd=lerp(.75, 1.6, t),
        ring_sd=lerp(.17, .2, t), w0=lerp(.1, .25, t), w1=lerp(.3, .5, t),
        hw_a=(lerp(13, 19.8, t), lerp(22, 1.4, t)), hw_b=(lerp(31, 30.1, t), lerp(11, 11.7, t)), hw_sd=lerp(.15, .22, t),
        noise=lerp(.115, .20, t), n_bridge=int(round(lerp(0, 40, t))),
        weights=lerp(np.array([.17, .125, .185, .075, .19, .14]) / .885, np.array([.22, .17, .14, .08, .20, .19]), t))


def make_geo_level(level, n=2000):
    """Returns (P int metres, labels, flags dict). Hotspot ids as in the easy file; -1 = noise."""
    import generate_data as gen
    t = T[level]
    sp = geo_spec(t)
    rng = np.random.default_rng(SEED + OFF[f"geo_{level}"])
    n_noise = int(round(sp["noise"] * n))
    n_br = sp["n_bridge"]
    nh = n - n_noise - n_br
    counts = [int(round(nh * w)) for w in sp["weights"]]
    counts[0] += nh - sum(counts)
    H = {}
    H[0] = np.array(sp["dt_c"]) + gen.trunc_gauss(rng, counts[0], sp["dt_sd"])
    H[1] = np.array([26.0, 18.0]) + gen.trunc_gauss(rng, counts[1], sp["campus_sd"])
    tt = rng.uniform(0, 2 * np.pi, counts[2])
    r = 2.4 + rng.normal(0, sp["ring_sd"], counts[2]).clip(-.5, .5)
    H[2] = np.column_stack([22 + r * np.cos(tt), 7 + r * np.sin(tt)])
    H[3] = np.array([22.0, 7.0]) + gen.trunc_gauss(rng, counts[3], .45, 2.1)
    u = (np.arange(counts[4]) + rng.uniform(-.3, .3, counts[4])) / counts[4]
    th = np.deg2rad(20 + 140 * u)
    w = sp["w0"] + sp["w1"] * np.sin(np.pi * u)
    rr = 7.5 + rng.normal(0, 1, counts[4]).clip(-1.8, 1.8) * w
    H[4] = np.column_stack([11 + rr * np.cos(th), 1 + rr * np.sin(th)])
    u = (np.arange(counts[5]) + rng.uniform(-.3, .3, counts[5])) / counts[5]
    a, b = np.array(sp["hw_a"]), np.array(sp["hw_b"])
    nrm = np.array([-(b - a)[1], (b - a)[0]]) / np.linalg.norm(b - a)
    H[5] = a + u[:, None] * (b - a) + rng.normal(0, sp["hw_sd"], counts[5])[:, None] * nrm
    pts = [np.round(H[k] * 1000).astype(np.int64) for k in range(6)]
    ys = [np.full(len(p), k) for k, p in enumerate(pts)]
    isb = [np.zeros(len(p), bool) for p in pts]
    if n_br:                       # density bridge downtown <-> market plaza
        u = rng.uniform(.0, 1.0, n_br)
        c0, c1 = np.array(sp["dt_c"]), np.array([22.0, 7.0])
        Bp = c0 + u[:, None] * (c1 - c0) + rng.normal(0, .08, (n_br, 2))
        pts.append(np.round(Bp * 1000).astype(np.int64))
        ys.append(np.where(u < .5, 0, 3)); isb.append(np.ones(n_br, bool))
    noise = np.round(rng.uniform([300, 300], [31700, 23700], (n_noise, 2))).astype(np.int64)
    pts.append(noise); ys.append(np.full(n_noise, -1)); isb.append(np.zeros(n_noise, bool))
    P = np.vstack(pts)
    P = np.clip(P, 100, [31900, 23900])
    P = gen.fix_ties(P, 500 ** 2)
    y, isb = np.concatenate(ys), np.concatenate(isb)
    perm = rng.permutation(len(P))
    return P[perm], y[perm], dict(is_bridge=isb[perm])


# ====================================================================== difficulty report
def best_dbscan(Z, y, eps_grid, mp_grid):
    from sklearn.cluster import DBSCAN
    from sklearn.metrics import adjusted_rand_score as ari
    best = (-2.0, None, None)
    for mp in mp_grid:
        for e in eps_grid:
            lab = DBSCAN(eps=float(e), min_samples=int(mp)).fit_predict(Z)
            a = ari(y, lab)
            if a > best[0]:
                best = (a, float(e), int(mp))
    return best


GEO_GRID = (np.round(np.arange(.20, 1.31, .05), 2), [3, 4, 5, 6, 8, 10, 12])
CARD_GRID = (np.round(np.arange(.20, 1.51, .05), 2), [4, 6, 8, 10, 12, 15])


def algorithms(Z, y, k_true, grid, quick=False):
    """Best ARI of every algorithm over a small parameter grid (oracle choice: uses the truth).
    Returns dict name -> (ari, params string)."""
    from sklearn.cluster import HDBSCAN, OPTICS, AgglomerativeClustering, KMeans
    from sklearn.metrics import adjusted_rand_score as ari
    out = {}
    a, e, mp = best_dbscan(Z, y, *grid)
    out["DBSCAN"] = (a, f"eps={e}, MinPts={mp}")
    km = KMeans(n_clusters=k_true, n_init=10, random_state=SEED).fit_predict(Z)
    out["KMeans (true k)"] = (ari(y, km), f"k={k_true}")
    hc = AgglomerativeClustering(n_clusters=k_true, linkage="ward").fit_predict(Z)
    out["Ward HAC (true k)"] = (ari(y, hc), f"k={k_true}")
    if not quick:
        best = (-2.0, "")
        for mpt in (5, 10):
            for xi in (.03, .08):
                lab = OPTICS(min_samples=mpt, xi=xi, min_cluster_size=20).fit_predict(Z)
                v = ari(y, lab)
                if v > best[0]:
                    best = (v, f"min_samples={mpt}, xi={xi}")
        out["OPTICS"] = best
        best = (-2.0, "")
        for sel in ("eom", "leaf"):
            for mcs in (15, 30, 60):
                for ms in (3, 6, 12):
                    lab = HDBSCAN(min_cluster_size=mcs, min_samples=ms, cluster_selection_method=sel).fit_predict(Z)
                    v = ari(y, lab)
                    if v > best[0]:
                        best = (v, f"min_cluster_size={mcs}, min_samples={ms}, selection={sel}")
        out["HDBSCAN"] = best
    return {k: (round(float(v[0]), 4), v[1]) for k, v in out.items()}


def geo_Z(P):
    return P.astype(float)


def difficulty_report(easy_geo, easy_card, hard_geo, hard_card, med_geo, med_card, quick=False):
    """Each argument: (Z, y, k_true). ATM works in metres (Eps in metres), so its grid is scaled by 1000."""
    rep = {}
    for fam, trio, grid in (("atm", (easy_geo, med_geo, hard_geo), (GEO_GRID[0] * 1000, GEO_GRID[1])),
                            ("card", (easy_card, med_card, hard_card), CARD_GRID)):
        rep[fam] = {}
        for lvl, (Z, y, k) in zip(("easy", "medium", "hard"), trio):
            rep[fam][lvl] = dict(n=int(len(Z)), k_true=int(k), noise_share=round(float((np.asarray(y) == -1).mean()), 3),
                                 results=algorithms(Z, y, k, grid, quick))
    return rep


# ====================================================================== hard card stream
def hard_stream_params():
    return dict(W=500, steps=300, eps=0.30, minpts=6, spacing=0.08, hole=4, arrive=[(60, 2), (118, 2), (136, 2)])


def _stream_layout(card_h, prm=None):
    """Sliding-window stream over the hard card family.
    Natural part: arrivals from the hard segments with a drifting mix. Planted part: three thin chains
    ('hybrid customers') between segment pairs (0,1), (1,2), (0,3). Each chain is complete in d0, loses
    interior points (NonSplit), then a run of adjacent points (Split), and the removed run is later
    inserted again (Merge). Cases are verified by reference full re-fits, never trusted from the plan."""
    import generate_data as gen
    from collections import defaultdict
    from datetime import datetime, timedelta
    prm = prm or hard_stream_params()
    W, steps, eps, minpts, s, h = prm["W"], prm["steps"], prm["eps"], prm["minpts"], prm["spacing"], prm["hole"]
    mean, std = card_h["mean"], card_h["std"]
    rng = np.random.default_rng(SEED + OFF["stream_hard"])
    sp = card_spec(1.0)
    n_ins = steps // 2
    # z-space segment centres from a large natural sample
    big = np.vstack([_sample(np.random.default_rng(7 + k), sp["f"][k], 800) for k in range(4)])
    Zb = gen.card_z(big, mean, std)[0]
    cen = np.array([Zb[800 * k:800 * (k + 1)].mean(0) for k in range(4)])

    def raw_from_z(z):
        r = z * std + mean
        r[0] = 10 ** r[0]
        return gen.card_round(r[None, :])[0]
    chains = []
    for (a, b) in prm.get("pairs", [(0, 1), (1, 2), (0, 3)]):
        L = np.linalg.norm(cen[b] - cen[a]); u = (cen[b] - cen[a]) / L
        lam = np.arange(.35, L - .35, s)
        pts = [raw_from_z(cen[a] + l * u) for l in lam]
        chains.append(dict(a=a, b=b, pts=pts, seg=[a if l < L / 2 else b for l in lam]))
    n_chain_d0 = sum(len(c["pts"]) for c in chains)
    # natural draws
    segs_log = {}

    def draw(mix, anom):
        if rng.random() < anom:
            return gen.card_anomalies(rng, 1, big)[0], -1
        k = int(rng.choice(4, p=mix))
        return _sample(rng, sp["f"][k], 1)[0], k
    mix0, mix1 = np.array([.40, .33, .22, .05]), np.array([.30, .22, .08, .40])
    mixes = [(1 - t / (n_ins - 1)) * mix0 + t / (n_ins - 1) * mix1 for t in range(n_ins)]
    # d0 slots: rank = age (0 = oldest = deleted first, at step 2)
    d0_items = [None] * W
    plan = prm.get("slots", [dict(i1=6, i2=14, hole=40), dict(i1=10, i2=22, hole=75), dict(i1=18, i2=30, hole=110)])
    arrive = prm.get("arrive", [(70, 2), (118, 2), (136, 2)])           # (arrival index of first closer, spacing)
    closers = {}                                                          # arrival index -> (raw, seg, chain idx)
    chain_roles = []
    for j, (c, sl) in enumerate(zip(chains, plan)):
        m = len(c["pts"]); mid = m // 2
        hole = list(range(mid - h // 2, mid - h // 2 + h))
        i1, i2 = mid - h - 4, mid + h + 4
        role = {i1: ("int", sl["i1"]), i2: ("int", sl["i2"])}
        for q, idx in enumerate(hole):
            role[idx] = ("hole", sl["hole"] + q)
        for idx in range(m):
            if idx in role:
                pos = role[idx][1]
            else:
                pos = None
            if pos is not None:
                intent = f"hole{j}" if role[idx][0] == "hole" else ""
                d0_items[pos] = (c["pts"][idx], c["seg"][idx], f"chain{j}", intent)
        c["rest"] = [idx for idx in range(m) if idx not in role]
        c["hole"] = hole; c["role"] = role
        k0, gap = arrive[j]
        for q, idx in enumerate(hole):
            raw = raw_from_z(gen.card_z(c["pts"][idx][None, :], mean, std)[0][0] + rng.normal(0, .004, 5))
            closers[k0 + gap * q] = (raw, c["seg"][idx], f"closer{j}", f"closer{j}")
    free = [p for p in range(W) if d0_items[p] is None]
    rest_pts = [(c["pts"][i], c["seg"][i], "chain_rest", "") for c in chains for i in c["rest"]]
    # the rest of every chain lives in the young part of the window (never deleted within 300 steps)
    young = [p for p in free if p >= n_ins + 20]
    take = [young[i] for i in rng.choice(len(young), len(rest_pts), replace=False)]
    for p, it in zip(take, rest_pts):
        d0_items[p] = it
    for p in range(W):
        if d0_items[p] is None:
            x, k = draw(mix0, .01)
            d0_items[p] = (x, k, "natural", "")
    arr = []
    for t in range(n_ins):
        if t in closers:
            arr.append(closers[t])
        else:
            x, k = draw(mixes[t], .02)
            arr.append((x, k, "natural", ""))
    return dict(d0=d0_items, arr=arr, chains=chains, cen=cen, mean=mean, std=std, eps=eps, minpts=minpts, closers=closers)


def build_card_stream_hard(card_h, prm=None):
    """Returns ({filename: text}, info) like generate_data.build_card_stream."""
    import generate_data as gen
    from collections import defaultdict
    from datetime import datetime, timedelta
    prm = prm or hard_stream_params()
    L = _stream_layout(card_h, prm)
    W, steps, eps, minpts = prm["W"], prm["steps"], L["eps"], L["minpts"]
    mean, std = L["mean"], L["std"]
    n_ins = steps // 2
    feats = {i: L["d0"][i][0] for i in range(W)}
    feats.update({W + k: L["arr"][k][0] for k in range(n_ins)})
    segs = {i: L["d0"][i][1] for i in range(W)}
    segs.update({W + k: L["arr"][k][1] for k in range(n_ins)})
    src = {i: L["d0"][i][2] for i in range(W)}
    src.update({W + k: L["arr"][k][2] for k in range(n_ins)})
    intent = {i: L["d0"][i][3] for i in range(W)}
    intent.update({W + k: L["arr"][k][3] for k in range(n_ins)})
    events, nxt, queue = [], W, list(range(W))
    for st in range(1, steps + 1):
        if st % 2 == 1:
            events.append(dict(op="insert", x=gen.card_z(feats[nxt][None, :], mean, std)[0][0], planted="", grp=intent[nxt]))
            queue.append(nxt); nxt += 1
        else:
            k = queue.pop(0)
            events.append(dict(op="delete", key=k, planted="", grp=intent[k]))
    t0 = datetime(2025, 3, 1)

    def to_row(step, op, p, x, planted, case, amb):
        r = feats[p]
        return [step, (t0 + timedelta(hours=step - 1)).strftime("%Y-%m-%d %H:%M"), op, p,
                f"{int(r[0])}", f"{r[1]:.2f}", f"{r[2]:.2f}", f"{int(r[3])}", f"{r[4]:.3f}",
                planted, case, ";".join(map(str, amb))]
    A0 = np.array([feats[i] for i in range(W)])
    Z0 = gen.card_z(A0, mean, std)[0]
    rows, amb0 = gen.run_stream(Z0, list(range(W)), events, eps, minpts, to_row)
    # plant: the first event of every hole group that really splits, of every closer group that really merges
    planted_at = {}
    for g in sorted({e["grp"] for e in events if e["grp"]}):
        want = "PotentialSplit_Split" if g.startswith("hole") else "Merge"
        hit = [i for i, e in enumerate(events) if e["grp"] == g and rows[i][10] == want]
        if not hit:
            if prm.get("debug"):
                print(g, [(i + 1, rows[i][10]) for i, e in enumerate(events) if e["grp"] == g])
                continue
            raise RuntimeError(f"hard stream: group {g} produced no {want}; retune the chain layout")
        planted_at[hit[0]] = want
    for i, e in enumerate(events):
        e["planted"] = planted_at.get(i, "")
    rows, amb0 = gen.run_stream(Z0, list(range(W)), events, eps, minpts, to_row)
    lab0, _ = gen.ref_fit(Z0, eps, minpts)
    d0_rows = [[i, f"{int(a[0])}", f"{a[1]:.2f}", f"{a[2]:.2f}", f"{int(a[3])}", f"{a[4]:.3f}", l]
               for i, (a, l) in enumerate(zip(A0, gen.canon(lab0)))]
    cnt = defaultdict(int)
    for r in rows:
        cnt[r[10]] += 1
    return rows, d0_rows, dict(cnt=dict(cnt), amb0=amb0, segs=segs, src=src, L=L, lab0=lab0)


# ====================================================================== writers
PREFIX = dict(hard=("TXH", "ATH"), medium=("TXI", "ATI"))


def load_card_arrays(path):
    """Base columns of a card CSV -> (rows with NaN, segment ids, txn ids)."""
    import csv
    import generate_data as gen
    with open(path, newline="", encoding="utf-8") as f:
        rd = list(csv.DictReader(f))
    A = np.array([[float(r[c]) if r[c] != "" else np.nan for c in gen.CARD_COLS] for r in rd])
    return A, np.array([int(r["segment_id"]) for r in rd]), [r["txn_id"] for r in rd]


def load_geo_arrays(path):
    import csv
    with open(path, newline="", encoding="utf-8") as f:
        rd = list(csv.DictReader(f))
    P = np.array([[round(float(r["x_km"]) * 1000), round(float(r["y_km"]) * 1000)] for r in rd], float)
    return P, np.array([int(r["hotspot_id"]) for r in rd])


def write_variants(levels):
    """Writes card_txn_5d_<lvl>.csv, atm_txn_2d_<lvl>.csv (+ truth flags). Returns info for the enrichment."""
    import generate_data as gen
    tables, flags = [], {}
    (gen.RAW.parent / "truth").mkdir(parents=True, exist_ok=True)
    for lvl in levels:
        c = make_card_level(lvl)
        pre_c, pre_g = PREFIX[lvl]
        gen.write_card_csv(gen.CARD / f"card_txn_5d_{lvl}.csv", c, prefix=pre_c)
        tables.append((f"card_txn_5d_{lvl}", c, f"card_txn_5d_{lvl}.csv", (f"card_desc_{lvl}", f"dirt_{lvl}"), None))
        n_u = len(c["clean"])
        fl = c["flags"]
        gen.dump(gen.RAW.parent / "truth" / f"card_txn_5d_{lvl}_generation_flags.csv",
                 ["txn_id", "segment_id", "is_bridge", "is_near_anomaly", "is_large_online"],
                 [[f"{pre_c}{u + 1:06d}", int(c["yc"][u]), int(fl["is_bridge"][u]), int(fl["is_near_anomaly"][u]),
                   int(fl["is_large_online"][u])] for u in range(n_u)])
        P, y, gf = make_geo_level(lvl)
        gen.write_geo_csv(gen.GEO / f"atm_txn_2d_{lvl}.csv", P, y, prefix=pre_g)
        gen.dump(gen.RAW.parent / "truth" / f"atm_txn_2d_{lvl}_generation_flags.csv", ["txn_id", "hotspot_id", "is_bridge"],
                 [[f"{pre_g}{i + 1:05d}", int(y[i]), int(gf["is_bridge"][i])] for i in range(len(P))])
        flags[lvl] = dict(card=c, geo=(P, y))
    return tables, flags


def write_hard_stream(card_h):
    """card_stream_hard/{d0,updates}.csv + params.json. Returns the true segment per point id (list)."""
    import generate_data as gen
    rows, d0_rows, info = build_card_stream_hard(card_h)
    L = info["L"]
    base = gen.CARD / "card_stream_hard"
    base.mkdir(parents=True, exist_ok=True)
    (base / "d0.csv").write_text(gen.csv_text(["point_id"] + gen.CARD_COLS + ["ref_label"], d0_rows), encoding="utf-8", newline="\n")
    (base / "updates.csv").write_text(gen.csv_text(["step", "timestamp", "op", "point_id"] + gen.CARD_COLS +
                                                    ["planted_case", "verified_case", "ambiguous_border_ids"], rows),
                                                   encoding="utf-8", newline="\n")
    planted = {}
    for r in rows:
        if r[9]:
            planted[r[9]] = planted.get(r[9], 0) + 1
    params = dict(
        simulated=True, seed=SEED + OFF["stream_hard"], eps=L["eps"], minpts=L["minpts"],
        metric="euclidean (after frozen scaling)", neighbourhood_includes_self=True, n_d0=500, n_steps=len(rows),
        timestamp="1 step = 1 hour from 2025-03-01 00:00; odd steps insert a new txn, even steps delete the oldest",
        frozen_scaling=dict(columns=["log10_amount_vnd"] + gen.CARD_COLS[1:],
                            mean=[float(v) for v in L["mean"]], std=[float(v) for v in L["std"]],
                            note="apply log10 to amount_vnd then (x - mean) / std with these constants (computed on card_txn_5d_hard); do not re-fit on the stream"),
        design=("Hard segments (diffuse retail, tight night cash-out, overlapping online and retail). Three thin chains of "
                "'hybrid customers' join segment pairs (0,1), (1,2), (0,3). A chain is complete in d0, loses points while the window "
                "slides (the oldest points leave first), and the removed run is inserted again later. Natural events from the diffuse "
                "segments add to the planted ones; every verified_case comes from reference full re-fits compared as partitions."),
        d0_reference_clusters=int(max(gen.canon(info["lab0"]))), d0_ambiguous_border_ids=sorted(info["amb0"]),
        verified_case_counts=dict(sorted(info["cnt"].items())), planted_case_counts=dict(sorted(planted.items())),
        case_definitions=gen.CASE_DEFS,
        id_note="point_id never reused. Stream rows carry no missing cells and no duplicates.",
        ambiguous_note="ambiguous_border_ids lists, AFTER the step, border points within eps of cores of >= 2 clusters; exclude them when comparing labels with a full re-fit. planted_case is filled only on the first event of a planted group that really produced the case; every other event is natural.")
    (base / "params.json").write_text(json.dumps(params, indent=2) + "\n", encoding="utf-8", newline="\n")
    print("card_stream_hard steps", len(rows), info["cnt"], "planted", planted)
    return [info["segs"][i] for i in range(500 + 150)]


def run_difficulty(flags):
    """Full report: easy from the committed files, medium and hard in memory (or files). Writes data/difficulty_report.json."""
    import generate_data as gen
    A, y, k = load_card_arrays(gen.CARD / "card_txn_5d.csv")
    easy_card = card_eval_arrays(A, y, k)
    P, yg = load_geo_arrays(gen.GEO / "atm_txn_2d.csv")
    easy_geo = (P, yg)
    trio = {}
    for lvl in ("medium", "hard"):
        c = flags[lvl]["card"] if lvl in flags else make_card_level(lvl)
        Pm, ym = flags[lvl]["geo"] if lvl in flags else make_geo_level(lvl)[:2]
        trio[lvl] = (card_eval_arrays(c["rows"], c["y"], c["uidx"]), (Pm.astype(float), ym))
    args = lambda z, y_: (z, y_, len(set(y_.tolist()) - {-1}))
    rep = difficulty_report(args(*easy_geo), args(*easy_card), args(*trio["hard"][1]), args(*trio["hard"][0]),
                            args(*trio["medium"][1]), args(*trio["medium"][0]))
    rep["note"] = ("Best ARI over a parameter grid chosen WITH the ground truth (oracle), so every number is optimistic. "
                   "ATM: coordinates in metres, DBSCAN eps grid 200..1300 m, MinPts 3..12. Card: log10(amount) + z-score after dropping "
                   "rows with missing cells and duplicates, eps grid 0.2..1.5, MinPts 4..15. KMeans and Ward use the true k "
                   "(ATM 6, card 4); noise points (-1) are their own class in the ARI. OPTICS: min_samples 5/10, xi 0.03/0.08, "
                   "min_cluster_size 20. HDBSCAN (sklearn): min_cluster_size 15/30/60, min_samples 3/6/12.")
    (gen.RAW.parent / "difficulty_report.json").write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8", newline="\n")
    data = dict(atm=dict(easy=easy_geo, medium=trio["medium"][1], hard=trio["hard"][1]),
                card=dict(easy=easy_card, medium=trio["medium"][0], hard=trio["hard"][0]))
    return rep, data
