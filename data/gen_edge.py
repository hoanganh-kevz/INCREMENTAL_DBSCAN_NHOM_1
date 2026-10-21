"""Part 3: boundary scenarios for the incremental algorithm, data/raw/toy_edge/<id>/.

Expected results are computed here with an exact, pure-Python Fraction DBSCAN (no sklearn, no numpy)
and re-computed by a second implementation inside check_data.py. Coordinates are decimals that are
exact in binary floating point (multiples of 1/16) wherever a distance is meant to equal Eps, so
float implementations compare exactly too.
"""
import csv
import json
from collections import deque
from fractions import Fraction
from pathlib import Path


def F(s):
    return Fraction(str(s))


# ---------------------------------------------------------------- exact reference
def d2(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b))


def nbrs(P, q, e2):
    return {k for k, v in P.items() if d2(v, P[q]) <= e2}


def core_set(P, e2, mp):
    return {k for k in P if len(nbrs(P, k, e2)) >= mp}


def cluster_labels(P, e2, mp):
    """Labels 1.. (numbered by smallest core id in sorted order), -1 noise. A border point shared
    by several clusters gets the first one that reaches it; those points are reported as ambiguous."""
    C = core_set(P, e2, mp)
    lab, cid = {}, 0
    for k in sorted(P):
        if k in C and k not in lab:
            cid += 1
            lab[k] = cid
            stack = [k]
            while stack:
                u = stack.pop()
                for v in sorted(nbrs(P, u, e2)):
                    if v in lab:
                        continue
                    lab[v] = cid
                    if v in C:
                        stack.append(v)
    return {k: lab.get(k, -1) for k in P}, C


def ambiguous_points(P, lab, C, e2):
    out = []
    for k in P:
        if k in C or lab[k] == -1:
            continue
        if len({lab[c] for c in nbrs(P, k, e2) if c in C}) > 1:
            out.append(k)
    return sorted(out)


def core_hops(P, C, e2, a, b):
    """Shortest number of eps-hops between cores a and b through cores only (None if disconnected)."""
    dist, q = {a: 0}, deque([a])
    while q:
        u = q.popleft()
        if u == b:
            return dist[u]
        for v in nbrs(P, u, e2):
            if v in C and v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)
    return None


def classify(op, p, Pb, lb, Cb, Pa, la, Ca, e2):
    """Case name of one update (same vocabulary as the streams)."""
    if op == "insert":
        members = {}
        for k, l in la.items():
            if l >= 0:
                members.setdefault(l, set()).add(k)
        olds = {c: {lb[k] for k in m if k != p and lb[k] >= 0} for c, m in members.items()}
        if any(len(o) >= 2 for o in olds.values()):
            return "Merge"
        if any(len(o) == 0 for o in olds.values()):
            return "Creation"
        return "Noise" if la[p] == -1 else "Absorption"
    if lb[p] == -1:
        return "DeleteNoise"
    split = vanished = False
    for c in {v for v in lb.values() if v >= 0}:
        new = {la[k] for k in la if lb[k] == c and la[k] >= 0}
        split |= len(new) >= 2
        vanished |= len(new) == 0
    if split:
        return "PotentialSplit_Split"
    if vanished:
        return "Removal"
    seeds = updseed("delete", p, Pb, Cb, Pa, Ca, e2)
    for c in {lb[k] for k in seeds}:
        S = sorted(k for k in seeds if lb[k] == c)
        comp, stack = {S[0]}, [S[0]]
        while stack:
            u = stack.pop()
            for v in S:
                if v not in comp and d2(Pa[u], Pa[v]) <= e2:
                    comp.add(v); stack.append(v)
        if len(comp) < len(S):
            return "PotentialSplit_NonSplit"
    return "Reduction"


def updseed(op, p, Pb, Cb, Pa, Ca, e2):
    if op == "insert":
        trig = {q for q in Ca if q not in Cb}
        return sorted({x for q in trig for x in nbrs(Pa, q, e2) if x in Ca})
    trig = {q for q in Cb if q not in Ca}
    return sorted({x for q in trig for x in nbrs(Pb, q, e2) if x in Ca})


def run_scenario(s):
    """s: dict(eps, minpts, pts {id: tuple str}, ups [(op, id, tuple|None)]). Returns expected dict."""
    e2 = F(s["eps"]) ** 2
    mp = s["minpts"]
    P = {k: tuple(F(c) for c in v) for k, v in s["pts"].items()}
    lab0, C0 = cluster_labels(P, e2, mp)
    steps, Pb, lb, Cb = [], P, lab0, C0
    allpts = dict(P)
    for i, (op, pid, xy) in enumerate(s["ups"], 1):
        if op == "insert":
            Pa = {**Pb, pid: tuple(F(c) for c in xy)}
            allpts[pid] = Pa[pid]
        else:
            Pa = {k: v for k, v in Pb.items() if k != pid}
        la, Ca = cluster_labels(Pa, e2, mp)
        seed = updseed(op, pid, Pb, Cb, Pa, Ca, e2)
        case = classify(op, pid, Pb, lb, Cb, Pa, la, Ca, e2)
        st = dict(step=i, op=op, point_id=pid, case=case, labels_after=la, core_after=sorted(Ca), updseed=seed,
                  ambiguous_border_ids=ambiguous_points(Pa, la, Ca, e2))
        if op == "delete" and len(seed) >= 2:
            hops = [core_hops(Pa, Ca, e2, a, b) for a in seed for b in seed if a < b and lb[a] == lb[b]]
            st["max_core_hops_between_seeds"] = None if any(h is None for h in hops) else (max(hops) if hops else None)
        steps.append(st)
        Pb, lb, Cb = Pa, la, Ca
    ties = sorted([a, b] for a in allpts for b in allpts if a < b and d2(allpts[a], allpts[b]) == e2)
    return dict(P0=P, labels_before=lab0, core_before=sorted(C0), steps=steps, final_labels=lb, final_core=sorted(Cb),
                ties=ties, allpts=allpts)


# ---------------------------------------------------------------- scenarios
SQ = lambda x, y, tag, step=0.5: {f"{tag}{i}": (x + dx, y + dy) for i, (dx, dy) in enumerate(
    [(0, 0), (step, 0), (0, step), (step, step)], 1)}
S = []


def add(sid, title, eps, mp, pts, ups, why, supports, check, dims=2):
    S.append(dict(id=sid, title=title, eps=eps, minpts=mp, pts=pts, ups=ups, why_vi=why, supports=supports,
                  check=check, dims=dims))


def sqpts(**kw):
    out = {}
    for tag, (x, y) in kw.items():
        out.update({k: (str(v[0]), str(v[1])) for k, v in SQ(x, y, tag).items()})
    return out


K = sqpts(K=(0, 0))                                                       # one 4-point core square
add("e01_insert_at_eps_inclusive", "Insert at exactly Eps from a core (inclusive boundary)", 1, 4,
    {**K, }, [("insert", "p", ("1.5", "0"))],
    "p cách điểm lõi K2 đúng bằng Eps (=1). Biên Eps là bao hàm (<=), nên p nằm trong lân cận của K2: p là điểm biên của cụm, không phải nhiễu. Cài đặt dùng < sẽ sai.",
    ("insert", "equivalence"),
    lambda r: r["steps"][0]["labels_after"]["p"] == 1 and "p" not in r["final_core"] and r["ties"] == [["K2", "p"]])
add("e02_insert_just_outside_eps", "Insert just outside Eps (1/16 beyond the boundary)", 1, 4,
    {**K}, [("insert", "p", ("1.5625", "0"))],
    "p cách K2 là 1.0625 > Eps nên không thuộc lân cận của điểm lõi nào: p là nhiễu, các nhãn cũ không đổi. Cặp với e01 để kiểm tra đúng ranh giới.",
    ("insert", "equivalence"),
    lambda r: r["steps"][0]["labels_after"]["p"] == -1 and r["ties"] == [] and r["steps"][0]["case"] == "Noise")
add("e03_insert_tie_makes_core", "Insert at exactly Eps turns a noise point into a core", 1, 4,
    {"A": ("0", "0"), "B": ("0.5", "0"), "C": ("0", "0.5")}, [("insert", "p", ("-1", "0"))],
    "Trước khi chèn cả A, B, C đều là nhiễu (3 < MinPts). p cách A đúng Eps nên N_Eps(A) = {A,B,C,p} có 4 phần tử, A thành lõi và tạo cụm mới. Nếu tính biên là < thì không có cụm nào.",
    ("insert", "equivalence"),
    lambda r: r["core_before"] == [] and r["final_core"] == ["A"] and r["steps"][0]["case"] == "Creation")
add("e04_insert_duplicate_coordinate", "Insert a duplicate of an existing coordinate", 1, 4,
    {"A": ("0", "0"), "B": ("0.5", "0"), "C": ("0", "0.5")}, [("insert", "p", ("0", "0"))],
    "p có đúng toạ độ của A (khoảng cách 0, hai điểm khác id). Mỗi điểm đều phải đếm p vào lân cận; kết quả: cụm mới gồm A, B, C, p. Cài đặt dùng toạ độ làm khoá sẽ gộp nhầm.",
    ("insert", "equivalence"),
    lambda r: r["steps"][0]["case"] == "Creation" and len(r["final_core"]) == 4)
add("e05_insert_border_to_core_cascade", "Insert promotes a border point to core and a noise trio to core (cascade)", 1, 4,
    {**K, "b1": ("1.25", "0"), "n1": ("1.25", "1.4"), "n2": ("1.6", "1.3"), "n3": ("1.9", "1.0")},
    [("insert", "p", ("1.25", "0.5"))],
    "Trước: b1 là điểm biên (3 láng giềng) và n1, n2, n3 là ba điểm nhiễu đôi một gần nhau (3 < MinPts). p chỉ nằm trong Eps của b1 và của cả ba điểm n. Sau khi chèn: p lõi, b1 lên lõi, và n1, n2, n3 cùng lên lõi (chuỗi lõi mới) rồi được hút vào cụm K. UpdSeed gồm các lõi nằm trong lân cận của mọi điểm mới thành lõi.",
    ("insert", "equivalence"),
    lambda r: set(r["core_before"]) == {"K1", "K2", "K3", "K4"} and {"b1", "n1", "n2", "n3", "p"} <= set(r["final_core"])
    and r["steps"][0]["case"] == "Absorption" and r["labels_before"]["n1"] == -1 and r["final_labels"]["n3"] > 0)

_c3 = {}
_c3.update({f"R{i}": (str(.95 + dx), str(dy)) for i, (dx, dy) in enumerate([(0, -.1), (0, .1), (.2, -.1), (.2, .1)], 1)})
_c3.update({f"L{i}": (str(-.95 - dx), str(dy)) for i, (dx, dy) in enumerate([(0, -.1), (0, .1), (.2, -.1), (.2, .1)], 1)})
_c3.update({f"T{i}": (str(dx), str(.95 + dy)) for i, (dx, dy) in enumerate([(-.1, 0), (.1, 0), (-.1, .2), (.1, .2)], 1)})
add("e06_insert_bridges_three_clusters", "One insert merges three clusters", 1, 4, _c3, [("insert", "p", ("0", "0"))],
    "Ba cụm 4 điểm cách nhau hơn Eps. p ở giữa, cách 2 điểm gần nhất của mỗi cụm dưới Eps nên p là lõi và nối cả ba cụm thành một. Kiểm tra gộp nhiều cụm cùng lúc, không chỉ cặp.",
    ("insert", "equivalence"),
    lambda r: len({v for v in r["labels_before"].values() if v > 0}) == 3 and len({v for v in r["final_labels"].values() if v > 0}) == 1
    and r["steps"][0]["case"] == "Merge")
add("e07_insert_merge_absorbs_noise", "Merge that also absorbs a noise point (and leaves another noise)", 1, 4,
    {**sqpts(A=(0, 0), B=(2, 0)), "n1": ("1.25", "1.2"), "n2": ("1.25", "-0.9")}, [("insert", "p", ("1.25", "0.25"))],
    "p nối cụm A và B (Merge). Điểm n1 cách p 0.95 nên từ nhiễu thành điểm biên của cụm gộp; n2 cách p 1.15 nên vẫn là nhiễu. Một update vừa Merge vừa đổi nhãn nhiễu.",
    ("insert", "equivalence"),
    lambda r: r["labels_before"]["n1"] == -1 and r["final_labels"]["n1"] > 0 and r["final_labels"]["n2"] == -1
    and r["steps"][0]["case"] == "Merge")
add("e08_insert_creation_exactly_minpts", "Cluster is created exactly when MinPts points have accumulated", 1, 4,
    {"n0": ("0", "0"), "z": ("5", "5")},
    [("insert", "q1", ("0.5", "0")), ("insert", "q2", ("0", "0.5")), ("insert", "q3", ("0.5", "0.5"))],
    "Ba lần chèn đầu cho 2, 3 điểm gần nhau vẫn là nhiễu; lần thứ ba đủ MinPts = 4 điểm đôi một trong Eps nên cả bốn thành lõi cùng lúc (Creation). z ở xa là nhiễu suốt kịch bản.",
    ("insert", "equivalence"),
    lambda r: [s["case"] for s in r["steps"]] == ["Noise", "Noise", "Creation"] and r["final_labels"]["z"] == -1)
add("e09_insert_noise_near_border", "Noise inserted next to a border point but outside every core Eps", 1, 4,
    {**K, "b": ("1.4", "0")}, [("insert", "p", ("2.2", "0"))],
    "b là điểm biên của cụm K (cách 2 lõi dưới Eps). p cách b 0.85 nhưng cách mọi lõi hơn Eps; điểm biên không mở rộng cụm, nên p vẫn là nhiễu và không thay đổi gì.",
    ("insert", "equivalence"),
    lambda r: r["labels_before"]["b"] == 1 and "b" not in r["core_before"] and r["final_labels"]["p"] == -1
    and r["steps"][0]["updseed"] == [] and r["steps"][0]["case"] == "Noise")
add("e10_insert_border_of_two_clusters", "Inserted point is a border of two clusters (ambiguous)", 1, 4,
    {**sqpts(A=(0, 0)), "B1": ("2.4", "0.5"), "B2": ("2.9", "0.5"), "B3": ("2.4", "1.0"), "B4": ("2.9", "1.0")},
    [("insert", "p", ("1.45", "0.45"))],
    "p cách đúng một lõi của A (0.95) và một lõi của B (0.95), tổng lân cận 3 < MinPts nên p không phải lõi, hai cụm không gộp. p là điểm biên của cả hai cụm: nhãn hợp lệ là A hoặc B. Hãy bỏ p khi so với re-fit.",
    ("insert", "equivalence"),
    lambda r: r["steps"][0]["ambiguous_border_ids"] == ["p"] and r["steps"][0]["case"] in ("Absorption",)
    and len({v for v in r["final_labels"].values() if v > 0}) == 2)
add("e11_delete_border_point", "Delete a border point (no core changes status)", 1, 4,
    {**K, "b": ("1.25", "0.25")}, [("delete", "b", None)],
    "b là điểm biên. Hai lõi K2, K4 có 5 láng giềng kể cả b nên sau khi xóa còn 4, vẫn là lõi. Chỉ b biến mất; không điểm nào đổi nhãn, UpdSeed rỗng.",
    ("delete", "equivalence"),
    lambda r: r["steps"][0]["updseed"] == [] and r["steps"][0]["case"] == "Reduction" and set(r["final_core"]) == set(r["core_before"]))
add("e12_delete_noise_point", "Delete a noise point", 1, 4,
    {**K, "b": ("1.4", "0"), "n": ("2.2", "0")}, [("delete", "n", None)],
    "n là nhiễu nằm cạnh điểm biên b. Xóa n không làm đổi lân cận của lõi nào, mọi nhãn còn lại giữ nguyên (DeleteNoise).",
    ("delete", "equivalence"),
    lambda r: r["labels_before"]["n"] == -1 and r["steps"][0]["case"] == "DeleteNoise" and r["steps"][0]["updseed"] == [])
add("e13_delete_splits_into_three", "Delete the hub core: one cluster splits into three", 1, 4,
    {**_c3, "p": ("0", "0")}, [("delete", "p", None)],
    "p là lõi nối ba cụm 4 điểm. Xóa p thì ba nhóm lõi không còn nối trực tiếp và không có đường thay thế: cụm tách thành ba (Split). Kiểm tra tách nhiều hơn hai phần.",
    ("delete", "equivalence"),
    lambda r: len({v for v in r["labels_before"].values() if v > 0}) == 1 and len({v for v in r["final_labels"].values() if v > 0}) == 3
    and r["steps"][0]["case"] == "PotentialSplit_Split")
_d = {"L1": ("0", "0"), "L2": ("0.5", "0"), "L3": ("0", "0.5"), "R1": ("2", "0"), "R2": ("2.5", "0"), "R3": ("2", "0.5"),
      "p": ("1.25", "0")}
for i, (x, y) in enumerate([(0, -.75), (0, -1.5), (0, -2.25), (.75, -2.25), (1.5, -2.25), (2, -1.5), (2, -.75)], 1):
    _d[f"d{i}"] = (str(x), str(y))
add("e14_delete_nonsplit_long_detour", "Delete a bridge core; the alternative path is long (BFS depth above 2)", 1, 3, _d,
    [("delete", "p", None)],
    "MinPts = 3. p nối trực tiếp L và R. Hai lõi hạt giống L2 và R1 không cách nhau trong Eps nên phải tìm đường; đường thay thế đi vòng qua 7 điểm d1..d7 nên độ sâu BFS lớn hơn 2 (file ghi max_core_hops_between_seeds). Không tách (NonSplit). Thuật toán chỉ BFS sâu 2 sẽ báo tách sai.",
    ("delete", "equivalence"),
    lambda r: r["steps"][0]["case"] == "PotentialSplit_NonSplit" and (r["steps"][0].get("max_core_hops_between_seeds") or 0) > 2)
add("e15_delete_shared_border_to_noise", "Delete a core; a border shared by two cores becomes noise", 1, 4,
    {"c1": ("0", "0"), "c2": ("0.5", "0"), "c3": ("0.25", "-0.9"), "b": ("0.25", "0.9")}, [("delete", "c1", None)],
    "c1 và c2 là lõi duy nhất; c3 và b là điểm biên chung của cả hai. Xóa c1 làm N_Eps(c2) còn 3 nên c2 mất tính lõi, cả cụm biến mất và b (điểm biên chung) thành nhiễu (Removal).",
    ("delete", "equivalence"),
    lambda r: r["core_before"] == ["c1", "c2"] and r["final_core"] == [] and r["final_labels"]["b"] == -1
    and r["steps"][0]["case"] == "Removal")
_a = {**{f"A{i}": (str(x), str(y)) for i, (x, y) in enumerate([(0, 0), (.5, 0), (0, .5), (.5, .5), (.5, -.4)], 1)},
      **{f"B{i}": (str(x), str(y)) for i, (x, y) in enumerate([(2, 0), (2.5, 0), (2, .5), (2.5, .5), (2, -.4)], 1)}}
add("e16_delete_bridge_leaves_ambiguous_border", "Delete a bridging core; a border point is left shared by two clusters", 1, 4,
    {**_a, "p": ("1.25", "-0.35"), "b": ("1.25", "0.75")}, [("delete", "p", None)],
    "p là lõi nối A và B. b là điểm biên cách một lõi của A (A4) và một lõi của B (B3). Trước khi xóa b thuộc cụm gộp; sau khi xóa p, hai cụm tách và b nằm trong Eps của lõi cả hai cụm: gán nhãn mơ hồ, hãy bỏ b khi so với re-fit.",
    ("delete", "equivalence"),
    lambda r: r["steps"][0]["case"] == "PotentialSplit_Split" and r["steps"][0]["ambiguous_border_ids"] == ["b"])
add("e17_delete_then_reinsert_round_trip", "Delete a bridge then re-insert the same coordinates (new id)", 1, 4,
    {**_a, "p": ("1.25", "-0.35")}, [("delete", "p", None), ("insert", "q", ("1.25", "-0.35"))],
    "Bước 1: xóa p, cụm tách đôi. Bước 2: chèn lại cùng toạ độ nhưng id mới q, cụm gộp lại. Phân hoạch cuối giống phân hoạch đầu (q thay p). Kiểm tra id không bị dùng lại và trạng thái nội bộ sạch sau Split.",
    ("insert", "delete", "equivalence"),
    lambda r: [s["case"] for s in r["steps"]] == ["PotentialSplit_Split", "Merge"])
_g = {f"a{i}": c for i, c in enumerate([("0", "0", "0", "0"), ("0.3", "0", "0", "0"), ("0", "0.3", "0", "0"), ("0", "0", "0.3", "0"), ("0", "0", "0", "0.3")], 1)}
_g.update({f"b{i}": c for i, c in enumerate([("1.7", "0", "0", "0"), ("1.4", "0", "0", "0"), ("1.7", "0.3", "0", "0"), ("1.7", "0", "0.3", "0"), ("1.7", "0", "0", "0.3")], 1)})
_g.update({"e1": ("0.5", "0.5", "3", "3"), "e2": ("0.55", "0.5", "3", "3"), "e3": ("0.5", "0.55", "3", "3")})
add("e18_insert_4d_merge", "Insert merges two clusters in 4-D (distances use all 4 coordinates)", 1, 4, _g,
    [("insert", "p", ("0.85", "0", "0", "0"))],
    "Dữ liệu 4 chiều (x1..x4). p nối hai cụm a* và b*. Ba điểm e1..e3 có (x1,x2) sát cụm nhưng x3 = x4 = 3 nên là nhiễu thật; một cài đặt chỉ dùng 2 toạ độ đầu sẽ coi chúng là điểm biên. File kiểm tra: kết quả 4D khác kết quả khi cắt còn 2 chiều.",
    ("insert", "equivalence"),
    lambda r: r["steps"][0]["case"] == "Merge" and all(r["final_labels"][k] == -1 for k in ("e1", "e2", "e3")), dims=4)
add("e19_minpts2_chain", "MinPts = 2: every non-noise point is core, chains link by single steps", 1, 2,
    {"a1": ("0", "0"), "a2": ("0.9", "0"), "a3": ("1.8", "0"), "c1": ("3.6", "0"), "c2": ("4.5", "0"), "c3": ("5.4", "0"),
     "z": ("0", "3")},
    [("insert", "p", ("2.7", "0")), ("insert", "q", ("6.3", "0"))],
    "MinPts = 2 nên mọi điểm có ít nhất một láng giềng đều là lõi (không có điểm biên) và Eps-reachability thành liên kết đơn. p nối hai chuỗi (Merge), q nối dài thêm đầu chuỗi (Absorption); z cô lập vẫn là nhiễu.",
    ("insert", "equivalence"),
    lambda r: [s["case"] for s in r["steps"]] == ["Merge", "Absorption"] and r["final_labels"]["z"] == -1
    and set(r["final_core"]) == set(r["allpts"]) - {"z"})


# ---------------------------------------------------------------- build / write
def cols_for(dims):
    return ["x", "y"] if dims == 2 else [f"x{i}" for i in range(1, dims + 1)]


def csv_text(cols, rows):
    import io
    b = io.StringIO()
    w = csv.writer(b, lineterminator="\r\n")
    w.writerow(cols); w.writerows(rows)
    return b.getvalue()


def project(pts, k):
    return {a: v[:k] for a, v in pts.items()}


def build_all():
    """{relative path: text} for every scenario. Raises if a scenario's own assertion fails."""
    out, summary = {}, []
    for s in S:
        r = run_scenario(s)
        if not s["check"](dict(r, **{k: r[k] for k in ("labels_before", "core_before", "steps", "ties")},
                               final_labels=r["final_labels"], final_core=r["final_core"])):
            raise RuntimeError(f"scenario {s['id']} does not behave as designed")
        if s["dims"] > 2:                     # prove that dropping the extra coordinates changes the answer
            r2 = run_scenario(dict(s, pts=project(s["pts"], 2), ups=[(o, i, xy[:2] if xy else None) for o, i, xy in s["ups"]]))
            if r2["final_labels"] == r["final_labels"]:
                raise RuntimeError(f"{s['id']}: 2-D projection gives the same labels, scenario proves nothing")
        cols = cols_for(s["dims"])
        d = f"toy_edge/{s['id']}/"
        out[d + "points.csv"] = csv_text(["point_id"] + cols, [[k, *v] for k, v in s["pts"].items()])
        out[d + "update.csv"] = csv_text(["op", "point_id"] + cols,
                                         [[o, i, *(xy if xy else [""] * s["dims"])] for o, i, xy in s["ups"]])
        steps = []
        for st in r["steps"]:
            st = dict(st)
            st["labels_after"] = {k: st["labels_after"][k] for k in sorted(st["labels_after"])}
            steps.append(st)
        exp = dict(title=s["title"], eps=float(F(s["eps"])), minpts=s["minpts"], dims=s["dims"], coordinate_columns=cols,
                   neighbourhood_includes_self=True, metric="euclidean",
                   label_convention="int cluster id (1,2,..), -1 = noise; ids only identify the partition; points listed in ambiguous_border_ids may carry either adjacent cluster id",
                   n_updates=len(s["ups"]),
                   labels_before={k: r["labels_before"][k] for k in sorted(r["labels_before"])},
                   labels_after={k: r["final_labels"][k] for k in sorted(r["final_labels"])},
                   core_before=r["core_before"], core_after=r["final_core"],
                   steps=steps,
                   ambiguous_to_ignore=sorted({a for st in steps for a in st["ambiguous_border_ids"]}),
                   ties_at_exactly_eps=r["ties"],
                   tests={"insert": "insert" in s["supports"], "delete": "delete" in s["supports"], "equivalence": True},
                   why_vi=s["why_vi"])
        if len(s["ups"]) == 1:
            exp["updseed"] = steps[0]["updseed"]
        else:
            exp["updseed_by_step"] = [st["updseed"] for st in steps]
        out[d + "expected.json"] = json.dumps(exp, indent=2, ensure_ascii=False) + "\n"
        summary.append((s["id"], [st["case"] for st in steps], s["supports"]))
    return out, summary


def write_all(raw):
    files, summary = build_all()
    import shutil
    base = Path(raw) / "toy_edge"
    if base.exists():
        shutil.rmtree(base)
    for rel, txt in files.items():
        p = Path(raw) / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", newline="", encoding="utf-8") as f:
            f.write(txt)
    print("toy_edge:", len(S), "scenarios")
    return summary
