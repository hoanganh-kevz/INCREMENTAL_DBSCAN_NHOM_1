"""Independent checker for data/raw. Brute force, exact arithmetic (Fractions), no sklearn.
Recomputes cores, labels (before/after) and UpdSeed from points.csv + update.csv and compares
with expected.json. Run: python data/check_data.py   (exit code 1 on any mismatch)
"""
import csv
import io
import json
import sys
from fractions import Fraction
from pathlib import Path

RAW = Path(__file__).resolve().parent / "raw"
fails = []


def rd_raw(p):
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.reader(f))


def rd(p):
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def nbrs(P, q, eps2):
    return {k for k, v in P.items()
            if sum((a - b) ** 2 for a, b in zip(v, P[q])) <= eps2}


def cores(P, eps2, minpts):
    return {k for k in P if len(nbrs(P, k, eps2)) >= minpts}


def cluster(P, eps2, minpts):
    """Plain DBSCAN; returns {id: label} with canonical numbering (by sorted order of first id)."""
    C = cores(P, eps2, minpts)
    lab, seen = {}, set()
    cid = 0
    for k in sorted(P):
        if k in C and k not in seen:
            cid += 1
            stack = [k]; seen.add(k)
            while stack:
                u = stack.pop(); lab[u] = cid
                for v in nbrs(P, u, eps2):
                    lab.setdefault(v, cid) if v not in C else None
                    if v in C and v not in seen:
                        seen.add(v); stack.append(v)
    return {k: lab.get(k, -1) for k in P}


def same_partition(a, b):
    if set(a) != set(b):
        return False
    if {k for k, v in a.items() if v == -1} != {k for k, v in b.items() if v == -1}:
        return False
    m = {}
    for k in a:
        if a[k] == -1:
            continue
        if m.setdefault(a[k], b[k]) != b[k]:
            return False
    return len(set(m.values())) == len(m)


def fr(s):
    return Fraction(s)


def check_toy(d):
    e = json.loads((d / "expected.json").read_text(encoding="utf-8"))
    eps2 = fr(str(e["eps"])) ** 2; mp = e["minpts"]
    P = {r["point_id"]: (fr(r["x"]), fr(r["y"])) for r in rd(d / "points.csv")}
    u = rd(d / "update.csv")[0]
    if u["op"] == "insert":
        Q = {**P, u["point_id"]: (fr(u["x"]), fr(u["y"]))}
    else:
        Q = {k: v for k, v in P.items() if k != u["point_id"]}
    cb, ca = cores(P, eps2, mp), cores(Q, eps2, mp)
    lb, la = cluster(P, eps2, mp), cluster(Q, eps2, mp)
    p = u["point_id"]
    if u["op"] == "insert":      # newly core objects incl. p, UpdSeed = cores of D' in their nbhds
        trig = {q for q in ca if q not in cb}
    else:                        # objects that lost core status incl. deleted p
        trig = {q for q in cb if q not in ca}
    seed = {x for q in trig for x in (nbrs(Q, q, eps2) if q in Q else nbrs(P, q, eps2)) if x in ca}
    got = dict(core_before=sorted(cb), core_after=sorted(ca), updseed=sorted(seed))
    msgs = []
    for key in got:
        if got[key] != sorted(e[key]):
            msgs.append(f"{key}: computed {got[key]} vs doc {sorted(e[key])}")
    if not same_partition(lb, e["labels_before"]):
        msgs.append(f"labels_before: computed {lb} vs doc {e['labels_before']}")
    if not same_partition(la, e["labels_after"]):
        msgs.append(f"labels_after: computed {la} vs doc {e['labels_after']}")
    # exactness margin: no pair distance may sit exactly on Eps
    allp = {**P, **Q}
    tie = [(a, b) for a in allp for b in allp if a < b and
           sum((x - y) ** 2 for x, y in zip(allp[a], allp[b])) == eps2]
    if tie:
        msgs.append(f"pairs at distance exactly Eps (ties): {tie}")
    return msgs


def check_csv(path, dims, nclusters_min):
    rows = rd(path)
    ids = [r["point_id"] for r in rows]
    msgs = []
    if len(set(ids)) != len(ids):
        msgs.append("duplicate ids")
    labs = {int(r["label"]) for r in rows}
    if len(labs - {-1}) < nclusters_min:
        msgs.append(f"labels {labs}")
    for r in rows:
        if any(r[c] == "" for c in r):
            msgs.append("blank cell"); break
    return msgs


for d in sorted((RAW / "toy").iterdir()):
    m = check_toy(d)
    print(("FAIL " if m else "ok   ") + d.name)
    for x in m:
        print("     ", x); fails.append((d.name, x))
for f, k in [("synthetic/moons_n500.csv", 2), ("synthetic/rings_n500.csv", 2),
             ("synthetic/blobs_noise_n500.csv", 3), ("nd/blobs5d_n860.csv", 4)]:
    m = check_csv(RAW / f, None, k)
    print(("FAIL " if m else "ok   ") + f)
    for x in m:
        print("     ", x); fails.append((f, x))


# ===================== step 0: base-column invariants =====================
import hashlib


def _csv_digest(path, ncols):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    body = "\n".join("\x1f".join(r[:ncols]) for r in rows)
    return hashlib.sha256(body.encode()).hexdigest(), len(rows) - 1


def check_invariants():
    """Base columns frozen in base_hashes.json (made before the enrichment) must not change."""
    msgs = []
    inv = json.loads((RAW.parent / "base_hashes.json").read_text(encoding="utf-8"))
    for rel, d in inv["csv_base_columns"].items():
        h, n = _csv_digest(RAW / rel, len(d["columns"]))
        with open(RAW / rel, newline="", encoding="utf-8") as f:
            head = next(csv.reader(f))
        if head[:len(d["columns"])] != d["columns"]:
            msgs.append(f"{rel}: base column names/order changed")
        elif h != d["sha256"] or n != d["n_rows"]:
            msgs.append(f"{rel}: base columns differ from the frozen digest")
    for rel, d in inv["whole_files"].items():
        if hashlib.sha256((RAW / rel).read_bytes()).hexdigest() != d["sha256"]:
            msgs.append(f"{rel}: file bytes changed")
    for rel, keys in inv["json_keys"].items():
        cur = json.loads((RAW / rel).read_text(encoding="utf-8"))
        for k, h in keys.items():
            if k not in cur or hashlib.sha256(json.dumps(cur[k], sort_keys=True).encode()).hexdigest() != h:
                msgs.append(f"{rel}: key {k} changed or removed")
    return msgs


m = check_invariants()
print(("FAIL " if m else "ok   ") + "step 0 invariants (base columns byte-identical to the frozen digests)")
for x in m:
    print("     ", x); fails.append(("invariants", x))


# ===================== toy_edge: independent re-computation (own Fraction code, n-D, ordered updates) =====================
def edge_state(P, e2, mp):
    lab, C = cluster(P, e2, mp), cores(P, e2, mp)
    return lab, C


def edge_ambiguous(P, lab, C, e2):
    out = []
    for k in P:
        if k not in C and lab[k] != -1:
            if len({lab[c] for c in nbrs(P, k, e2) if c in C}) > 1:
                out.append(k)
    return sorted(out)


def edge_case(op, p, Pb, lb, Pa, la, Ca, Cb, e2):
    """Case of one update, written independently from the generator (cluster overlap based)."""
    if op == "insert":
        new_of = {}
        for k, l in la.items():
            if l > 0:
                new_of.setdefault(l, set()).add(k)
        olds = [{lb[k] for k in m if k != p and lb.get(k, -1) > 0} for m in new_of.values()]
        if any(len(o) > 1 for o in olds):
            return "Merge"
        if any(not o for o in olds):
            return "Creation"
        return "Noise" if la[p] == -1 else "Absorption"
    if lb[p] == -1:
        return "DeleteNoise"
    pieces = {}
    for k, l in lb.items():
        if l > 0 and k != p:
            pieces.setdefault(l, set()).add(la[k])
    if any(len({x for x in v if x > 0}) > 1 for v in pieces.values()):
        return "PotentialSplit_Split"
    if any(not {x for x in v if x > 0} for v in pieces.values()) or not pieces:
        return "Removal"
    lost = [q for q in Cb if q not in Ca]
    seeds = sorted({x for q in lost for x in nbrs(Pb, q, e2) if x in Ca})
    for c in {lb[x] for x in seeds}:
        S = [x for x in seeds if lb[x] == c]
        reach, todo = {S[0]}, [S[0]]
        while todo:
            u = todo.pop()
            for v in S:
                if v not in reach and sum((a - b) ** 2 for a, b in zip(Pa[u], Pa[v])) <= e2:
                    reach.add(v); todo.append(v)
        if len(reach) < len(S):
            return "PotentialSplit_NonSplit"
    return "Reduction"


def is_dyadic(s):
    return Fraction(float(s)) == Fraction(s)


def check_edge(d):
    msgs = []
    e = json.loads((d / "expected.json").read_text(encoding="utf-8"))
    eps2 = fr(str(e["eps"])) ** 2
    mp = e["minpts"]
    cols = e["coordinate_columns"]
    pr, ur = rd(d / "points.csv"), rd(d / "update.csv")
    if list(pr[0])[1:] != cols or list(ur[0])[2:] != cols:
        return ["coordinate columns differ from expected.json"]
    P = {r["point_id"]: tuple(fr(r[c]) for c in cols) for r in pr}
    if len(P) != len(pr):
        msgs.append("duplicate point ids in points.csv")
    if len(ur) != e["n_updates"] or len(e["steps"]) != len(ur):
        msgs.append("number of update rows")
    lb, Cb = edge_state(P, eps2, mp)
    if not same_partition(lb, e["labels_before"]) or sorted(Cb) != sorted(e["core_before"]):
        msgs.append("labels_before / core_before differ")
    allp = dict(P)
    seen_ids = set(P)
    for r, st in zip(ur, e["steps"]):
        op, p = r["op"], r["point_id"]
        if (op, p) != (st["op"], st["point_id"]):
            msgs.append(f"step {st['step']}: op/id differ from expected.json"); break
        if op == "insert":
            if p in seen_ids:
                msgs.append(f"step {st['step']}: id {p} reused")
            Pa = {**P, p: tuple(fr(r[c]) for c in cols)}
            allp[p] = Pa[p]
        else:
            if p not in P:
                msgs.append(f"step {st['step']}: delete of unknown id"); break
            Pa = {k: v for k, v in P.items() if k != p}
        seen_ids.add(p)
        la, Ca = edge_state(Pa, eps2, mp)
        amb = edge_ambiguous(Pa, la, Ca, eps2)
        got = {k: v for k, v in la.items() if k not in amb}
        exp_l = {k: v for k, v in st["labels_after"].items() if k not in amb}
        if sorted(Ca) != sorted(st["core_after"]):
            msgs.append(f"step {st['step']}: core_after differ")
        if not same_partition(got, exp_l) or set(la) != set(st["labels_after"]):
            msgs.append(f"step {st['step']}: labels_after differ")
        if amb != sorted(st["ambiguous_border_ids"]):
            msgs.append(f"step {st['step']}: ambiguous border ids {amb} vs {st['ambiguous_border_ids']}")
        trig = [q for q in Ca if q not in Cb] if op == "insert" else [q for q in Cb if q not in Ca]
        src = Pa if op == "insert" else P
        seed = sorted({x for q in trig for x in nbrs(src, q, eps2) if x in Ca})
        if seed != sorted(st["updseed"]):
            msgs.append(f"step {st['step']}: updseed {seed} vs {st['updseed']}")
        case = edge_case(op, p, P, lb, Pa, la, Ca, Cb, eps2)
        if case != st["case"]:
            msgs.append(f"step {st['step']}: case {case} vs {st['case']}")
        P, lb, Cb = Pa, la, Ca
    if not same_partition({k: v for k, v in lb.items() if k not in e["ambiguous_to_ignore"]},
                          {k: v for k, v in e["labels_after"].items() if k not in e["ambiguous_to_ignore"]}):
        msgs.append("final labels_after differ")
    ties = sorted([a, b] for a in allp for b in allp if a < b and sum((x - y) ** 2 for x, y in zip(allp[a], allp[b])) == eps2)
    if ties != e["ties_at_exactly_eps"]:
        msgs.append(f"ties {ties} vs {e['ties_at_exactly_eps']}")
    near = [(a, b) for a in allp for b in allp if a < b and 0 < abs(sum((x - y) ** 2 for x, y in zip(allp[a], allp[b])) - eps2) < Fraction(1, 10 ** 4)]
    if near:
        msgs.append(f"pairs within 1e-4 of Eps^2 but not tied (float risk): {near[:3]}")
    if ties:
        raws = [r[c] for r in pr + ur for c in cols if r[c] != ""]
        if not all(is_dyadic(v) for v in raws):
            msgs.append("tie scenario uses coordinates that are not exact in binary floating point")
    if len(cols) > 2:        # the extra coordinates must matter
        P2 = {k: v[:2] for k, v in {**allp}.items() if k in seen_ids and k in lb}
        l2 = cluster(P2, eps2, mp)
        if same_partition(l2, {k: v for k, v in lb.items() if k in P2}):
            msgs.append("dropping coordinates 3..n gives the same labels: scenario does not prove n-D distances")
    for k in ("why_vi", "tests", "title"):
        if not e.get(k):
            msgs.append(f"missing {k}")
    return msgs


EDGE_DIR = RAW / "toy_edge"
edge_dirs = sorted(p for p in EDGE_DIR.iterdir() if p.is_dir()) if EDGE_DIR.is_dir() else []
edge_fail = 0
for d in edge_dirs:
    m = check_edge(d)
    edge_fail += bool(m)
    if m:
        print("FAIL toy_edge/" + d.name)
        for x in m:
            print("     ", x); fails.append((d.name, x))
need_edge = 16
if len(edge_dirs) < need_edge:
    fails.append(("toy_edge", f"only {len(edge_dirs)} scenarios, need at least {need_edge}"))
cov = set()
for d in edge_dirs:
    cov |= {s["case"] for s in json.loads((d / "expected.json").read_text(encoding="utf-8"))["steps"]}
allcases = {"Noise", "Creation", "Absorption", "Merge", "Removal", "Reduction", "PotentialSplit_NonSplit", "PotentialSplit_Split", "DeleteNoise"}
if cov != allcases:
    fails.append(("toy_edge", f"cases not covered: {sorted(allcases - cov)}"))
print(("FAIL " if edge_fail or len(edge_dirs) < need_edge or cov != allcases else "ok   ") +
      f"toy_edge: {len(edge_dirs)} scenarios, {edge_fail} failing, cases covered {len(cov)}/{len(allcases)}")


# ===================== fictional digital-bank suite (bank_geo, bank_card) =====================
import numpy as np
from sklearn.metrics import adjusted_rand_score          # only for ARI numbers

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_data as gen                              # reuses case definitions + builders

GEO, CARD = RAW / "bank_geo", RAW / "bank_card"
NEEDED = ["bank_geo/hotspots.csv", "bank_card/segments.csv", "bank_geo/atm_txn_2d.csv", "bank_geo/params.json", "bank_card/card_txn_5d.csv",
          "bank_card/params.json"] + [f"{d}/{s}/{f}" for d, s in (("bank_geo", "atm_stream"),
                                                                  ("bank_card", "card_stream"))
                                      for f in ("d0.csv", "updates.csv", "params.json")]


def bf_fit(X, eps, minpts):
    """Independent brute-force DBSCAN (numpy). Returns labels, core mask."""
    X = np.asarray(X, float)
    A = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1) <= eps * eps
    core = A.sum(1) >= minpts
    lab = -np.ones(len(X), int)
    c = 0
    for i in range(len(X)):
        if core[i] and lab[i] < 0:
            lab[i] = c; stack = [i]
            while stack:
                u = stack.pop()
                for v in np.where(A[u])[0]:
                    if lab[v] < 0:
                        lab[v] = c
                        if core[v]:
                            stack.append(v)
            c += 1
    return lab, core


def nclust(lab):
    return len(set(lab.tolist()) - {-1})


def check_geo_static():
    m = []
    rows = rd(GEO / "atm_txn_2d.csv")
    if list(rows[0])[:4] != ["txn_id", "x_km", "y_km", "hotspot_id"]:
        return ["columns"]
    if not 1100 <= len(rows) <= 1300:
        m.append(f"rows {len(rows)}")
    P = np.array([[round(float(r["x_km"]) * 1000), round(float(r["y_km"]) * 1000)] for r in rows], np.int64)
    y = np.array([int(r["hotspot_id"]) for r in rows])
    if set(y) != {-1, 0, 1, 2, 3, 4, 5}:
        m.append(f"hotspot ids {set(y)}")
    if gen.tie_pairs(P, 500 ** 2):
        m.append("tie at distance exactly eps")
    lab, _ = bf_fit(P, 500, gen.GEO_MINPTS)
    ari = adjusted_rand_score(y, lab)
    rep, _, _ = gen.geo_report(P, y, 500, gen.GEO_MINPTS)
    if ari < 0.95:
        m.append(f"DBSCAN ARI {ari:.3f} < 0.95")
    if rep["ari_kmeans6_vs_truth"] > 0.7:
        m.append(f"KMeans ARI {rep['ari_kmeans6_vs_truth']} too good: shapes not hard enough")
    return m


def check_card_static():
    m = []
    rows = rd(CARD / "card_txn_5d.csv")
    cols = ["txn_id"] + gen.CARD_COLS + ["segment_id"]
    if list(rows[0])[:len(cols)] != cols:
        return ["columns"]
    n = len(rows)
    if not 1400 <= n <= 1600:
        m.append(f"rows {n}")
    cells = [r[c] for r in rows for c in gen.CARD_COLS]
    miss = sum(v == "" for v in cells) / len(cells)
    if not 0.01 <= miss <= 0.02:
        m.append(f"missing fraction {miss:.4f} not in 1..2 %")
    keys = [tuple(r.values()) for r in rows]
    dup = (len(keys) - len(set(keys))) / n
    if not 0.003 <= dup <= 0.007:
        m.append(f"duplicate fraction {dup:.4f} not near 0.5 %")
    amt = [float(r["amount_vnd"]) for r in rows if r["amount_vnd"]]
    if max(amt) / min(amt) < 1e3:
        m.append("amount does not span 3 orders of magnitude")
    seen, clean = set(), []
    for r in rows:                              # reference preprocessing (checker only)
        k = tuple(r.values())
        if k in seen or any(r[c] == "" for c in gen.CARD_COLS):
            continue
        seen.add(k); clean.append(r)
    A = np.array([[float(r[c]) for c in gen.CARD_COLS] for r in clean])
    y = np.array([int(r["segment_id"]) for r in clean])
    Z, _, _ = gen.card_z(A)
    p = json.loads((CARD / "params.json").read_text(encoding="utf-8"))
    lab, _ = bf_fit(Z, p["recommended_eps"], p["recommended_minpts"])
    if nclust(lab) != 4 or adjusted_rand_score(y, lab) < 0.9:
        m.append(f"recommended eps gives {nclust(lab)} clusters, ARI {adjusted_rand_score(y, lab):.3f}")
    if (lab[y == -1] != -1).any():
        m.append("a planted anomaly is not noise at the recommended eps")
    return m


INV_EXCEPTIONS = []


def replay(base, kind):
    """Replay a stream with the independent DBSCAN; compare every verified_case / ambiguity list."""
    m = []
    par = json.loads((base / "params.json").read_text(encoding="utf-8"))
    d0, up = rd(base / "d0.csv"), rd(base / "updates.csv")
    if kind == "geo":
        feats, eps = ["x_km", "y_km"], par["eps_km"] * 1000
        conv = lambda r: [round(float(r[c]) * 1000) for c in feats]
    else:
        feats, eps = gen.CARD_COLS, par["eps"]
        mean, std = np.array(par["frozen_scaling"]["mean"]), np.array(par["frozen_scaling"]["std"])

        def conv(r):
            v = np.array([float(r[c]) for c in feats]); v[0] = np.log10(v[0])
            return (v - mean) / std
    mp = par["minpts"]
    ids = [int(r["point_id"]) for r in d0]
    X = np.array([conv(r) for r in d0], float)
    lab, core = bf_fit(X, eps, mp)
    ref = [int(r["ref_label"]) for r in d0]
    if not same_partition(dict(enumerate(lab.tolist())), dict(enumerate(ref))):
        m.append("d0 ref_label differs from the independent fit")
    amb0 = sorted(ids[i] for i in gen.ambiguous_border(X, lab, core, eps))
    if amb0 != sorted(par["d0_ambiguous_border_ids"]):
        m.append("d0 ambiguous border ids differ")
    planted_seen, verified_seen = set(), set()
    for r in up:
        step, op, p = int(r["step"]), r["op"], int(r["point_id"])
        lab_b = dict(zip(ids, lab.tolist()))
        if op == "insert":
            if p in ids:
                m.append(f"step {step}: reused id {p}")
            ids2, X2 = ids + [p], np.vstack([X, conv(r)])
            lab2, core2 = bf_fit(X2, eps, mp)
            case = gen.classify_insert(ids2, lab2.tolist(), p, lab_b, dict(zip(ids, core.tolist())), dict(zip(ids2, core2.tolist())))
        else:
            j = ids.index(p)
            ids2, X2 = ids[:j] + ids[j + 1:], np.delete(X, j, 0)
            lab2, core2 = bf_fit(X2, eps, mp)
            case = gen.classify_delete(ids, X, lab, core, ids2, X2, lab2, core2, p, eps)
        # invariant between verified_case and the number of clusters (every cluster owns >= 1 core)
        Cb, Ca = nclust(lab), nclust(lab2)
        old_cl = {int(lab[i]) for i in range(len(ids)) if core[i]}
        keep = {int(lab[ids.index(k)]) for j, k in enumerate(ids2) if core2[j] and k in ids and core[ids.index(k)]}
        vanished = len(old_cl - keep) if op == "delete" else 0
        vc = r["verified_case"]
        if vc == "PotentialSplit_Split":
            ok = Ca > Cb or vanished > 0      # documented exception: a split and a vanishing cluster in the same step
            if Ca <= Cb:
                INV_EXCEPTIONS.append((str(base.name), step))
        else:
            ok = {"Merge": Ca < Cb, "Creation": Ca > Cb, "Removal": Ca < Cb}.get(vc, Ca == Cb)
        if not ok:
            m.append(f"step {step}: case {vc} contradicts cluster count {Cb} -> {Ca}")
        amb = sorted(ids2[i] for i in gen.ambiguous_border(X2, lab2, core2, eps))
        amb_b = [ids[i] for i in gen.ambiguous_border(X, lab, core, eps)]
        if case != r["verified_case"]:
            m.append(f"step {step}: verified_case {r['verified_case']} but replay gives {case}")
        if r["planted_case"] and r["planted_case"] != r["verified_case"]:
            m.append(f"step {step}: planted {r['planted_case']} not reproduced (verified {r['verified_case']})")
        if r["planted_case"] and (p in amb or p in amb_b):
            m.append(f"step {step}: planted event on ambiguous border point {p}")
        got = sorted(int(v) for v in r["ambiguous_border_ids"].split(";") if v)
        if got != amb:
            m.append(f"step {step}: ambiguous_border_ids {got} vs replay {amb}")
        if r["planted_case"]:
            planted_seen.add(r["planted_case"])
        verified_seen.add(r["verified_case"])
        ids, X, lab, core = ids2, X2, lab2, core2
    if len(up) != par["n_steps"] or [int(r["step"]) for r in up] != list(range(1, len(up) + 1)):
        m.append("step numbering / n_steps")
    return m, planted_seen, verified_seen


def same_bytes(sub, built):
    """Fresh build vs disk. CSV: compare the base columns only (descriptive columns are appended after them)."""
    base = (GEO if sub == "atm_stream" else CARD) / sub
    bad = []
    for fn, txt in built.items():
        if fn.endswith(".json"):
            old, new = json.loads((base / fn).read_text(encoding="utf-8")), json.loads(txt)
            if any(old.get(k) != v for k, v in new.items()):
                bad.append(fn)
        else:
            fresh = list(csv.reader(io.StringIO(txt)))
            disk = [r[:len(fresh[0])] for r in rd_raw(base / fn)]
            if fresh != disk:
                bad.append(fn)
    return bad


bank = []
for f, col, src in (("bank_geo/hotspots.csv", "hotspot_id", "bank_geo/atm_txn_2d.csv"),
                    ("bank_card/segments.csv", "segment_id", "bank_card/card_txn_5d.csv")):
    if (RAW / f).is_file() and (RAW / src).is_file():
        meta = rd(RAW / f)
        if {r[col] for r in meta} != {r[col] for r in rd(RAW / src)} or any(
                not r[k] for r in meta for k in ("name_en", "name_vi", "meaning_vi", "suggested_action_vi")):
            bank.append((f, "ids differ from the data or a field is empty"))
miss = [f for f in NEEDED if not (RAW / f).is_file()]
if miss:
    bank.append(("files", f"missing: {miss}"))
else:
    for name, fn in (("bank_geo/atm_txn_2d.csv", check_geo_static), ("bank_card/card_txn_5d.csv", check_card_static)):
        for x in fn():
            bank.append((name, x))
    mg, planted, ver = replay(GEO / "atm_stream", "geo")
    bank += [("atm_stream", x) for x in mg]
    need = set(gen.INSERT_CASES + gen.DELETE_CASES)
    if planted != need or not need <= ver:
        bank.append(("atm_stream", f"planted {sorted(planted)} / verified {sorted(ver)} do not cover {sorted(need)}"))
    n = len(rd(GEO / "atm_stream" / "updates.csv"))
    if not 100 <= n <= 140:
        bank.append(("atm_stream", f"{n} steps, expected about 120"))
    mc, _, _ = replay(CARD / "card_stream", "card")
    bank += [("card_stream", x) for x in mc]
    n = len(rd(CARD / "card_stream" / "updates.csv"))
    if not 250 <= n <= 350:
        bank.append(("card_stream", f"{n} steps, expected about 300"))
    a1, a2 = gen.build_atm_stream(), gen.build_atm_stream()
    card = gen.make_card(gen.CARD_N)
    c1, c2 = gen.build_card_stream(card), gen.build_card_stream(card)
    if a1 != a2 or c1 != c2:
        bank.append(("determinism", "a stream differs between two runs"))
    for sub, built in (("atm_stream", a1), ("card_stream", c1)):
        d = same_bytes(sub, built)
        if d:
            bank.append((sub, f"files on disk differ from a fresh build: {d}"))
print(("FAIL " if bank else "ok   ") + "bank suite (geo, card, streams, determinism)")
for k, x in bank:
    print("     ", k, x); fails.append((k, x))


# ===================== enrichment, hard variants, difficulty, dictionary, hygiene =====================
import datetime as dtm
from collections import Counter

import gen_dictionary
import gen_edge
import gen_enrich as ge
import gen_hard as gh

TRUTH = RAW.parent / "truth"
CITY = (32.0, 24.0)
CHAN = {"POS", "ONLINE", "ATM", "CONTACTLESS"}
T_FIRST, T_LAST = dtm.datetime(2025, 3, 1), dtm.datetime(2025, 4, 26)


def section(title, msgs):
    print(("FAIL " if msgs else "ok   ") + title)
    for x in msgs:
        print("     ", x); fails.append((title, x))


def nearest_district(districts, x, y):
    c = [(float(d["centroid_x_km"]), float(d["centroid_y_km"]), d["district_id"]) for d in districts]
    return min(c, key=lambda t: (t[0] - x) ** 2 + (t[1] - y) ** 2)[2]


def check_dimensions():
    m = []
    cu, di, ca, at = (rd(CARD / "customers.csv"), rd(GEO / "districts.csv"), rd(CARD / "merchant_categories.csv"), rd(GEO / "atms.csv"))
    if list(cu[0]) != ge.CUSTOMER_COLS:
        m.append("customers columns")
    ids = [r["customer_id"] for r in cu]
    if len(set(ids)) != len(ids) or not 500 <= len(ids) <= 700:
        m.append(f"customers: {len(ids)} rows or duplicate ids")
    dids = {r["district_id"] for r in di}
    if len(di) != 8 or len(dids) != 8:
        m.append("districts: need 8 unique rows")
    if abs(sum(float(r["area_km2"]) for r in di) - CITY[0] * CITY[1]) > 1.0:
        m.append("district areas do not tile the 32 x 24 km city")
    if any(not (0 <= float(r["centroid_x_km"]) <= CITY[0] and 0 <= float(r["centroid_y_km"]) <= CITY[1]) for r in di):
        m.append("district centroid outside the city")
    if {r["area_type"] for r in di} < {"CBD", "university", "market", "riverside", "highway", "residential", "suburb"}:
        m.append("district area types incomplete")
    if len(ca) < 12 or {r["base_risk_level"] for r in ca} != {"low", "medium", "high"} or len({r["category_code"] for r in ca}) != len(ca):
        m.append("merchant_categories")
    if len(at) != 40 or len({r["atm_id"] for r in at}) != 40 or {r["atm_network"] for r in at} != {"own", "partner"}:
        m.append("atms: need 40 unique ATMs in two networks")
    for r in at:
        if r["district_id"] != nearest_district(di, float(r["x_km"]), float(r["y_km"])):
            m.append(f"atm {r['atm_id']} district differs from its coordinates"); break
    occ = Counter(r["occupation_group"] for r in cu)
    if set(occ) != {"office", "student", "trader", "freelancer", "retired", "other"}:
        m.append("occupations")
    for r in cu:
        a, o = int(r["age"]), r["occupation_group"]
        if (o == "student" and not 18 <= a <= 26) or (o == "retired" and a < 56) or not 18 <= a <= 85:
            m.append(f"{r['customer_id']}: age {a} with occupation {o}"); break
        if r["home_district_id"] not in dids or r["digital_active"] not in ("0", "1") or r["has_credit_card"] not in ("0", "1"):
            m.append(f"{r['customer_id']}: bad FK or flag"); break
        if r["digital_active"] == "0" and r["preferred_channel"] in ("mobile_app", "web"):
            m.append(f"{r['customer_id']}: not digital but prefers a digital channel"); break
        if not 1 <= int(r["tenure_months"]) <= 96:
            m.append("tenure_months range"); break
    med = lambda f: float(np.median([int(r["monthly_income_vnd"]) for r in cu if f(r)]))
    tm = [med(lambda r, t=t: r["tier"] == t) for t in ("mass", "affluent", "priority")]
    if not tm[0] < tm[1] < tm[2]:
        m.append(f"income medians do not increase with tier: {tm}")
    if not med(lambda r: r["occupation_group"] == "student") < 5e6 < med(lambda r: r["occupation_group"] == "trader"):
        m.append("student income should be low and trader income high")
    cc = lambda t: np.mean([int(r["has_credit_card"]) for r in cu if r["tier"] == t])
    if not cc("mass") < cc("affluent") < cc("priority"):
        m.append("credit card rate should increase with tier")
    if sum(r["tier"] == "priority" for r in cu if r["occupation_group"] == "student") != 0:
        m.append("students should not be priority tier")
    # determinism
    rows, mix, kinds = ge.make_customers()
    if [[str(v) for v in r] for r in rows] != [[r[c] for c in ge.CUSTOMER_COLS] for r in cu]:
        m.append("customers.csv differs from a fresh build")
    return m


def tod_hours(ts):
    d = dtm.datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
    return d.hour + d.minute / 60 + d.second / 3600


def check_card_table(rel, truth_name, base_n=None):
    """Descriptive columns of a card table against the clean truth files and the defect catalogue."""
    m = []
    path = CARD / rel
    rows = rd(path)
    head = list(rows[0])
    want = ["txn_id"] + gen.CARD_COLS + ["segment_id", "timestamp", "customer_id", "channel", "merchant_category", "txn_district_id", "is_foreign"]
    if head != want:
        return [f"{rel}: columns {head}"]
    cust = {r["customer_id"]: r for r in rd(CARD / "customers.csv")}
    dist = {r["district_id"] for r in rd(GEO / "districts.csv")}
    cats = {r["category_code"] for r in rd(CARD / "merchant_categories.csv")}
    cl = {r["txn_id"]: r for r in rd(TRUTH / f"{truth_name}_clean_descriptive.csv")}
    cb = {r["txn_id"]: r for r in rd(TRUTH / f"{truth_name}_clean_base.csv")}
    by = {}
    for r in rows:
        if r["txn_id"] in by and by[r["txn_id"]] != r:
            m.append(f"{r['txn_id']}: repeated id with different content"); break
        by[r["txn_id"]] = r
    if set(by) != set(cl) or set(by) != set(cb):
        m.append("truth files do not cover exactly the txn_ids of the table")
        return m
    # truth is itself clean and consistent
    hm = []
    for t, r in cl.items():
        if r["customer_id"] not in cust or r["channel"] not in CHAN or r["merchant_category"] not in cats:
            m.append(f"{t}: clean truth has an invalid key"); break
        if (r["txn_district_id"] == "") != (r["is_foreign"] == "1") or (r["txn_district_id"] and r["txn_district_id"] not in dist):
            m.append(f"{t}: clean district / is_foreign inconsistent"); break
        ts = dtm.datetime.strptime(r["timestamp"], "%Y-%m-%d %H:%M:%S")
        if not T_FIRST <= ts < T_LAST:
            m.append(f"{t}: clean timestamp out of range"); break
        if abs(tod_hours(r["timestamp"]) - float(cb[t]["hour_of_day"])) > 0.01:
            hm.append(t)
    if hm:
        m.append(f"{len(hm)} clean timestamps whose time of day differs from hour_of_day by more than 0.01 h")
    # defects measured in the table versus the clean truth
    cnt = Counter()
    for r in rows:
        c = cl[r["txn_id"]]
        if r["channel"] != c["channel"]:
            if r["channel"].strip().upper().replace("-", "") == c["channel"]:
                cnt["inconsistent_spelling_or_case"] += 1
            else:
                m.append(f"{r['txn_id']}: channel differs beyond spelling")
        if r["customer_id"] != c["customer_id"]:
            cnt["unknown_customer_id"] += r["customer_id"] not in cust
        if r["merchant_category"] != c["merchant_category"]:
            cnt["missing_value"] += r["merchant_category"] == ""
        if r["timestamp"] != c["timestamp"]:
            try:
                d = dtm.datetime.strptime(r["timestamp"], "%d/%m/%Y %H:%M:%S")
                cnt["mixed_timestamp_format_dd_mm_yyyy"] += d.strftime("%Y-%m-%d %H:%M:%S") == c["timestamp"]
            except ValueError:
                try:
                    dtm.datetime.strptime(r["timestamp"], "%Y-%m-%d %H:%M:%S")
                    m.append(f"{r['txn_id']}: a valid timestamp differs from the truth")
                except ValueError:
                    cnt["impossible_timestamp"] += 1
        if r["txn_district_id"] != c["txn_district_id"]:
            cnt["invalid_district_id"] += r["txn_district_id"] == "D99"
        if r["is_foreign"] != c["is_foreign"]:
            m.append("is_foreign must not be dirtied")
    cat = {(r["defect_type"], r["column"]): int(r["n_rows"]) for r in rd(TRUTH / "defect_catalogue_card.csv") if r["table"] == truth_name}
    for typ, n in cnt.items():
        if not any(k[0] == typ and v == n for k, v in cat.items()):
            m.append(f"defect {typ}: measured {n}, catalogue {[v for k, v in cat.items() if k[0] == typ]}")
    for typ in ("inconsistent_spelling_or_case", "unknown_customer_id", "missing_value", "mixed_timestamp_format_dd_mm_yyyy", "impossible_timestamp", "invalid_district_id"):
        if cnt[typ] == 0:
            m.append(f"defect type {typ} absent")
    for j, c in enumerate(gen.CARD_COLS):
        n = sum(r[c] == "" for r in rows)
        if cat.get(("missing_value", c)) != n:
            m.append(f"missing cells in {c}: {n} vs catalogue {cat.get(('missing_value', c))}")
    # base columns must hold the clean value wherever present
    for r in rows:
        for c in gen.CARD_COLS:
            if r[c] != "" and abs(float(r[c]) - float(cb[r["txn_id"]][c])) > 1e-9:
                m.append(f"{r['txn_id']}: base value differs from truth"); break
        else:
            continue
        break
    # behavioural consistency (clean truth + segment)
    seg = {r["txn_id"]: int(r["segment_id"]) for r in rows}
    share = lambda s, col, vals: np.mean([cl[t][col] in vals for t in cl if seg[t] == s])
    if min(share(3, "channel", {"ATM"}), share(3, "merchant_category", {"ATM"})) < .8:
        m.append("night_cash_out should be ATM / cash_atm")
    if share(1, "channel", {"ONLINE"}) < .85 or share(1, "merchant_category", {"ECO"}) < .4:
        m.append("online_shopping should be online / ecommerce")
    if share(2, "merchant_category", {"TRV"}) < .55:
        m.append("travel_big_ticket should be travel category")
    if share(-1, "is_foreign", {"1"}) < .3:
        m.append("anomalies should often be foreign")
    cu_home = {k: v["home_district_id"] for k, v in cust.items()}
    for t, r in cl.items():
        if r["is_foreign"] == "1":
            continue
        home = cu_home[r["customer_id"]]
        if seg[t] in (0, 1) and r["txn_district_id"] != home:
            m.append(f"{t}: local segment away from home district"); break
        if seg[t] in (2, -1) and r["txn_district_id"] == home:
            m.append(f"{t}: far-from-home segment in the home district"); break
    # calendar effects
    days = Counter(dtm.datetime.strptime(r["timestamp"], "%Y-%m-%d %H:%M:%S").date() for r in cl.values())
    mean_d = np.mean(list(days.values()))
    d5 = np.mean([days[dtm.date(2025, 3, 5)], days[dtm.date(2025, 4, 5)]])
    dend = np.mean([days[dtm.date(2025, 3, d)] for d in (29, 30, 31)])
    if d5 < 1.3 * mean_d or dend < 1.1 * mean_d:
        m.append(f"payday spikes too weak: day 5 {d5:.1f}, month end {dend:.1f}, mean {mean_d:.1f}")
    wk = np.mean([dtm.datetime.strptime(r["timestamp"], "%Y-%m-%d %H:%M:%S").weekday() >= 5 for t, r in cl.items() if seg[t] in (0, 1)])
    if wk < 2 / 7 * 1.1:
        m.append(f"weekend share of retail and online too low: {wk:.2f}")
    # customers: most dominated by one segment
    mixr = rd(TRUTH / "customer_segment_mix.csv")
    dom = np.mean([r["kind"] == "dominated" for r in mixr])
    if not .75 <= dom <= .95 or any(max(float(r[f"p_seg{k}"]) for k in range(4)) < .8 for r in mixr if r["kind"] == "dominated"):
        m.append("customer dominance shares")
    return m


def check_atm_table():
    m = []
    rows = rd(GEO / "atm_txn_2d.csv")
    want = ["txn_id", "x_km", "y_km", "hotspot_id", "timestamp", "atm_id", "customer_id", "amount_vnd", "atm_network", "district_id", "atm_distance_km"]
    if list(rows[0]) != want:
        return ["columns"]
    atms = {r["atm_id"]: r for r in rd(GEO / "atms.csv")}
    cust = {r["customer_id"]: r for r in rd(CARD / "customers.csv")}
    di = rd(GEO / "districts.csv")
    by_h = {}
    for a in atms.values():
        by_h.setdefault(a["hotspot_id"], []).append(a)
    ids = [r["txn_id"] for r in rows]
    if len(set(ids)) != len(ids):
        m.append("duplicate txn_id")
    occ = {h: [] for h in ("-1", "0", "1", "2", "3", "4", "5")}
    hours = {h: [] for h in occ}
    hw_days = Counter()
    for r in rows:
        x, y, h = float(r["x_km"]), float(r["y_km"]), r["hotspot_id"]
        a = atms.get(r["atm_id"])
        if a is None or r["customer_id"] not in cust:
            m.append(f"{r['txn_id']}: FK violated"); break
        if r["atm_network"] != a["atm_network"]:
            m.append("atm_network differs from atms.csv"); break
        pool = by_h[h] if h != "-1" else list(atms.values())
        dd = {p["atm_id"]: ((float(p["x_km"]) - x) ** 2 + (float(p["y_km"]) - y) ** 2) ** .5 for p in pool}
        if min(dd, key=dd.get) != r["atm_id"] or abs(dd[r["atm_id"]] - float(r["atm_distance_km"])) > .0015:
            m.append(f"{r['txn_id']}: atm_id is not the nearest ATM of its hotspot or distance wrong"); break
        if r["district_id"] != nearest_district(di, x, y):
            m.append(f"{r['txn_id']}: district_id differs from the coordinates"); break
        amt = int(r["amount_vnd"])
        if amt % 50000 or not 50000 <= amt <= 5000000:
            m.append(f"{r['txn_id']}: amount {amt}"); break
        ts = dtm.datetime.strptime(r["timestamp"], "%Y-%m-%d %H:%M:%S")
        if not T_FIRST <= ts < T_LAST:
            m.append("timestamp out of range"); break
        occ[h].append(cust[r["customer_id"]]["occupation_group"])
        hours[h].append(ts.hour + ts.minute / 60)
        if h == "5":
            hw_days[ts.date()] += 1
    if len(by_h.get("-1", [])) != ge.N_STANDALONE:
        m.append("stand-alone ATM count")
    fr = lambda h, o: np.mean([x in o for x in occ[h]])
    if fr("1", {"student"}) < .6 or fr("0", {"office"}) < .45 or fr("3", {"trader"}) < .5:
        m.append("customer occupations do not match the hotspots")
    if np.median(hours["1"]) < 18.5 or np.median(hours["2"]) > 10 or np.median(hours["3"]) > 9:
        m.append("hour curves: campus should peak in the evening, the market early in the morning")
    dt_h = np.array(hours["0"])
    if np.mean(((dt_h >= 11) & (dt_h < 14)) | ((dt_h >= 17) & (dt_h < 20))) < .55:
        m.append("downtown should peak at lunch and after work")
    hd = np.mean([hw_days[dtm.date(2025, 4, d)] for d in (5, 6, 7)])
    if hd < 1.5 * np.mean([v for k, v in hw_days.items() if k.day not in (5, 6, 7) or k.month != 4]):
        m.append("highway should be busier on the Hung Kings long weekend")
    return m


def check_streams_enriched():
    m = []
    di = rd(GEO / "districts.csv")
    for fn in ("d0.csv", "updates.csv"):
        for r in rd(GEO / "atm_stream" / fn):
            if r["district_id"] != nearest_district(di, float(r["x_km"]), float(r["y_km"])):
                m.append(f"atm_stream/{fn}: district_id differs from coordinates"); break
    cust = {r["customer_id"] for r in rd(CARD / "customers.csv")}
    cats = {r["category_code"] for r in rd(CARD / "merchant_categories.csv")}
    dids = {r["district_id"] for r in di}
    for sub in ("card_stream", "card_stream_hard"):
        d0, up = rd(CARD / sub / "d0.csv"), rd(CARD / sub / "updates.csv")
        per = {}
        for fn, rows in (("d0", d0), ("updates", up)):
            for r in rows:
                p = int(r["point_id"])
                tup = tuple(r[c] for c in ("point_arrival_timestamp", "event_timestamp", "segment_id", "customer_id", "channel", "merchant_category", "txn_district_id", "is_foreign"))
                if per.setdefault(p, tup) != tup:
                    m.append(f"{sub}: point {p} has inconsistent descriptive values"); break
                if r["customer_id"] not in cust or r["channel"] not in CHAN or r["merchant_category"] not in cats or (r["txn_district_id"] and r["txn_district_id"] not in dids):
                    m.append(f"{sub}/{fn}: FK or domain violated"); break
                if (r["txn_district_id"] == "") != (r["is_foreign"] == "1"):
                    m.append(f"{sub}: is_foreign vs blank district"); break
                ev = dtm.datetime.strptime(r["event_timestamp"], "%Y-%m-%d %H:%M:%S")
                ar = dtm.datetime.strptime(r["point_arrival_timestamp"], "%Y-%m-%d %H:%M")
                if abs(ev.hour + ev.minute / 60 + ev.second / 3600 - float(r["hour_of_day"])) > .01:
                    m.append(f"{sub}: event_timestamp time of day differs from hour_of_day"); break
                if not 0 <= (ar - ev).total_seconds() < 24 * 3600:
                    m.append(f"{sub}: event_timestamp must precede the arrival by less than 24 h"); break
                if fn == "updates" and r["op"] == "insert" and r["point_arrival_timestamp"] != r["timestamp"]:
                    m.append(f"{sub}: arrival of an inserted point differs from the step timestamp"); break
            else:
                continue
            break
        segs = Counter(int(r["segment_id"]) for r in d0)
        if not {0, 1, 2} <= set(segs):
            m.append(f"{sub}: d0 segment ids {dict(segs)}")
    return m


def check_hard_files():
    m = []
    for f in ("bank_card/card_txn_5d_hard.csv", "bank_geo/atm_txn_2d_hard.csv", "bank_card/card_stream_hard/d0.csv",
              "bank_card/card_stream_hard/updates.csv", "bank_card/card_stream_hard/params.json"):
        if not (RAW / f).is_file():
            return [f"missing {f}"]
    easy_head = list(rd(CARD / "card_txn_5d.csv")[0])
    hard = rd(CARD / "card_txn_5d_hard.csv")
    if list(hard[0]) != easy_head:
        m.append("card hard must have the same columns as the enriched card file")
    ghard = rd(GEO / "atm_txn_2d_hard.csv")
    if list(ghard[0]) != ["txn_id", "x_km", "y_km", "hotspot_id"]:
        m.append("atm hard columns")
    if not 500 <= len(hard) <= 4000 or not 500 <= len(ghard) <= 4000:
        m.append("hard files outside 500..4000 rows")
    # determinism: regenerate and compare the base columns
    for lvl, path_c, path_g in (("hard", CARD / "card_txn_5d_hard.csv", GEO / "atm_txn_2d_hard.csv"),):
        c = gh.make_card_level(lvl)
        buf = io.StringIO()
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            gen.write_card_csv(Path(td) / "c.csv", c, prefix=gh.PREFIX[lvl][0])
            fresh = rd_raw(Path(td) / "c.csv")
            P, y, _ = gh.make_geo_level(lvl)
            gen.write_geo_csv(Path(td) / "g.csv", P, y, prefix=gh.PREFIX[lvl][1])
            freshg = rd_raw(Path(td) / "g.csv")
        if [r[:7] for r in rd_raw(path_c)] != [r[:7] for r in fresh] or rd_raw(path_g) != freshg:
            m.append(f"{lvl} files differ from a fresh build")
    # hardness features (measured, not trusted)
    from sklearn.neighbors import NearestNeighbors
    A, y, k = gh.load_card_arrays(CARD / "card_txn_5d_hard.csv")
    Z, yy = gh.card_eval_arrays(A, y, k)
    d, i = NearestNeighbors(n_neighbors=6).fit(Z).kneighbors(Z)
    dens = {s: d[yy == s, 5].mean() for s in (0, 1, 2, 3)}
    if max(dens.values()) / min(dens.values()) < 2.0:
        m.append(f"unequal densities: 5-NN distance ratio {max(dens.values()) / min(dens.values()):.2f} < 2")
    nb = yy[i[:, 1:]]
    mixed = np.mean([(nb[yy == s] != s).any(1).mean() for s in (0, 1)])
    if mixed < .02:
        m.append(f"retail and online barely overlap (mixed-neighbour share {mixed:.3f})")
    fl = rd(TRUTH / "card_txn_5d_hard_generation_flags.csv")
    if sum(r["is_bridge"] == "1" for r in fl) < 8 or sum(r["is_near_anomaly"] == "1" for r in fl) < 10 or sum(r["is_large_online"] == "1" for r in fl) < 20:
        m.append("hard card generation flags: bridge / near anomalies / large online too few")
    P, yg = gh.load_geo_arrays(GEO / "atm_txn_2d_hard.csv")
    ns = float(np.mean(yg == -1))
    if not .17 <= ns <= .23:
        m.append(f"atm hard noise share {ns:.3f}, expected about 0.20")
    if gen.tie_pairs(P.astype(np.int64), 500 ** 2):
        m.append("atm hard has a pair at distance exactly 0.5 km")
    gfl = rd(TRUTH / "atm_txn_2d_hard_generation_flags.csv")
    if sum(r["is_bridge"] == "1" for r in gfl) < 20:
        m.append("atm hard density bridge too small")
    # stream
    mc, planted, ver = replay(CARD / "card_stream_hard", "card")
    m += [f"card_stream_hard: {x}" for x in mc]
    par = json.loads((CARD / "card_stream_hard" / "params.json").read_text(encoding="utf-8"))
    want = {"Merge", "PotentialSplit_Split", "PotentialSplit_NonSplit"}
    cnt = par["verified_case_counts"]
    for c in want:
        if cnt.get(c, 0) < 3:
            m.append(f"card_stream_hard: only {cnt.get(c, 0)} {c} events, need 3")
    for c in ("Merge", "PotentialSplit_Split"):
        if par["planted_case_counts"].get(c, 0) < 3:
            m.append(f"card_stream_hard: fewer than 3 planted {c}")
    rows, d0r, info = gh.build_card_stream_hard(gh.make_card_level("hard"))
    fresh = [[str(v) for v in r] for r in rows]
    disk = [r[:len(fresh[0])] for r in rd_raw(CARD / "card_stream_hard" / "updates.csv")[1:]]
    if fresh != disk:
        m.append("card_stream_hard/updates.csv differs from a fresh build")
    return m


def check_difficulty():
    m = []
    rep = json.loads((RAW.parent / "difficulty_report.json").read_text(encoding="utf-8"))
    A, y, k = gh.load_card_arrays(CARD / "card_txn_5d.csv")
    data = {"card": {"easy": gh.card_eval_arrays(A, y, k)}, "atm": {"easy": gh.load_geo_arrays(GEO / "atm_txn_2d.csv")}}
    for lvl in ("hard",):
        A, y, k = gh.load_card_arrays(CARD / f"card_txn_5d_{lvl}.csv")
        data["card"][lvl] = gh.card_eval_arrays(A, y, k)
        data["atm"][lvl] = gh.load_geo_arrays(GEO / f"atm_txn_2d_{lvl}.csv")
    c = gh.make_card_level("medium")
    data["card"]["medium"] = gh.card_eval_arrays(c["rows"], c["y"], c["uidx"])
    P, yg, _ = gh.make_geo_level("medium")
    data["atm"]["medium"] = (P.astype(float), yg)
    best = {}
    for fam, grid in (("atm", (gh.GEO_GRID[0] * 1000, gh.GEO_GRID[1])), ("card", gh.CARD_GRID)):
        for lvl in ("easy", "medium", "hard"):
            Z, yy = data[fam][lvl]
            a, e, mp = gh.best_dbscan(Z, yy, *grid)
            best[(fam, lvl)] = a
            r = rep[fam][lvl]["results"]["DBSCAN"][0]
            if abs(a - r) > 2e-3:
                m.append(f"{fam} {lvl}: recomputed best DBSCAN ARI {a:.4f} vs report {r:.4f}")
            if rep[fam][lvl]["n"] != len(Z):
                m.append(f"{fam} {lvl}: n {len(Z)} vs report {rep[fam][lvl]['n']}")
    MARGIN = 0.08
    for fam in ("atm", "card"):
        e_, md, h = best[(fam, "easy")], best[(fam, "medium")], best[(fam, "hard")]
        if not (e_ - md >= MARGIN and md - h >= MARGIN):
            m.append(f"{fam}: difficulty ordering easy {e_:.3f} > medium {md:.3f} > hard {h:.3f} needs margin {MARGIN}")
        for lvl in ("easy", "medium", "hard"):
            res = rep[fam][lvl]["results"]
            if set(res) != {"DBSCAN", "KMeans (true k)", "Ward HAC (true k)", "OPTICS", "HDBSCAN"}:
                m.append(f"{fam} {lvl}: report misses an algorithm")
    return m


def check_dictionary():
    m = []
    path = RAW / "DATA_DICTIONARY.csv"
    rows = rd(path)
    if list(rows[0]) != gen_dictionary.COLS:
        return ["dictionary columns"]
    if path.read_bytes().decode().replace("\r\n", "\n") != (lambda: (lambda b: b.getvalue())(
            (lambda b: (csv.writer(b, lineterminator="\n").writerows([gen_dictionary.COLS] + gen_dictionary.build()), b)[1])(io.StringIO())))():
        m.append("DATA_DICTIONARY.csv differs from a fresh build")
    for r in rows:
        if r["role"] not in ("key", "feature", "descriptive", "label", "truth") or not r["description_en"] or not r["description_vi"]:
            m.append(f"{r['table']}.{r['column']}: role or description"); break
    keyed = {}
    for r in rows:
        keyed.setdefault(r["table"], set()).add(r["column"])
    seen_tables = set()
    for base in (RAW, TRUTH):
        for p in sorted(base.rglob("*.csv")):
            rel = p.relative_to(RAW.parent).as_posix()
            if "raw/bench/" in rel or p.name == "DATA_DICTIONARY.csv":
                continue
            parts = rel.split("/")
            if parts[:2] in (["raw", "toy"], ["raw", "toy_edge"]) and len(parts) == 4:
                key = "/".join(parts[:2] + ["<scenario>", parts[3]])
            else:
                key = rel
            seen_tables.add(key)
            if key not in keyed:
                m.append(f"table {rel} missing in the dictionary"); continue
            cols = set(rd_raw(p)[0])
            if not cols <= keyed[key]:
                m.append(f"{rel}: columns not in the dictionary: {sorted(cols - keyed[key])}")
            if "<scenario>" not in key and not keyed[key] <= cols:
                m.append(f"{rel}: dictionary lists columns the file does not have: {sorted(keyed[key] - cols)}")
    for key in keyed:
        if key not in seen_tables and "medium" not in key and "<scenario>" not in key:
            m.append(f"dictionary table {key} has no file")
    return m


def check_hygiene():
    m = []
    root = RAW.parent
    bad = ("\u2014", "\u2013")
    total = 0
    own_tools = [root.parent / "tools" / n for n in ("make_base_hashes.py", "make_d07_figures.py")]
    for base in (root, *[t for t in own_tools if t.exists()]):
        for p in ([base] if base.is_file() else base.rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts:
                regenerable = {"bench", "baseline_bench", "baseline", "preview"} & set(p.parts)      # sinh lại được, nằm trong data/.gitignore
                total += p.stat().st_size if base == root and not regenerable else 0
                if p.suffix in (".py", ".md", ".csv", ".json", ".txt"):
                    t = p.read_text(encoding="utf-8", errors="ignore")
                    if any(b in t for b in bad):
                        m.append(f"em or en dash in {p.relative_to(root.parent)}")
    if total > 8 * 1024 * 1024:
        m.append(f"data/ is {total / 2**20:.1f} MB, limit 8 MB")
    return m


def check_readme():
    m = []
    t = (RAW.parent / "README_data.md").read_text(encoding="utf-8")
    for w in ["toy_edge", "card_stream_hard", "DATA_DICTIONARY.csv", "customers.csv", "districts.csv", "merchant_categories.csv", "atms.csv",
              "data/truth", "difficulty_report.json", "base_hashes.json", "atm_txn_2d_hard.csv", "card_txn_5d_hard.csv", "--difficulty",
              "python data/generate_data.py", "python data/check_data.py", "Eps", "mô phỏng", "ethical"] + [d.name for d in sorted((RAW / "toy_edge").iterdir())]:
        if w.lower() not in t.lower():
            m.append(f"README_data.md does not mention {w}")
    return m


def check_fresh_builds():
    """Everything the generator can rebuild in memory must equal the files on disk."""
    m = []
    files, _ = gen_edge.build_all()
    for rel, txt in files.items():
        if (RAW / rel).read_bytes().decode() != txt:
            m.append(f"{rel} differs from a fresh build")
    di = ge.district_rows()
    if [r[:] for r in di] != [r[:] for r in [list(x.values()) for x in rd(GEO / "districts.csv")]]:
        m.append("districts.csv differs from a fresh build")
    return m


section("enrichment: dimension tables (customers, districts, categories, ATMs)", check_dimensions())
section("enrichment: card_txn_5d.csv vs clean truth and defect catalogue", check_card_table("card_txn_5d.csv", "card_txn_5d"))
section("enrichment: atm_txn_2d.csv (FK, nearest ATM, hour curves)", check_atm_table())
section("enrichment: streams (district_id, two clocks, consistency)", check_streams_enriched())
section("part 2: hard variants (files, hardness features, hard stream Merge/Split/NonSplit)", check_hard_files())
if (CARD / "card_txn_5d_medium.csv").is_file():
    section("part 2: medium card variant (optional)", check_card_table("card_txn_5d_medium.csv", "card_txn_5d_medium"))
print(f"     invariant verified_case vs cluster count: {len(INV_EXCEPTIONS)} documented Split exceptions {INV_EXCEPTIONS}")
section("part 2: difficulty report recomputed (ordering easy > medium > hard)", check_difficulty())
section("toy_edge and districts rebuilt from code", check_fresh_builds())
section("data dictionary covers every table and column", check_dictionary())
section("text hygiene (no em/en dashes) and size under 8 MB", check_hygiene())
section("README_data.md mentions every deliverable", check_readme())
sys.exit(1 if fails else 0)
