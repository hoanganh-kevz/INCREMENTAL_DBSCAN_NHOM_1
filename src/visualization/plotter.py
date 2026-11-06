"""Trực quan hóa cụm (D07): ClusterVisualizer vẽ MỘT ảnh chụp (snapshot) của mô hình phân cụm.

Snapshot là dict gồm bốn khóa, cùng độ dài n:
    ids          id của điểm (số nguyên hoặc chuỗi; chỉ cần phân biệt được các điểm)
    coords       ma trận (n, 2) tọa độ để vẽ trực tiếp
    labels       nhãn cụm, -1 là nhiễu
    point_types  loại điểm: 3 = lõi (core), 2 = biên (border), 1 = nhiễu (noise), 0 = chưa phân loại (có thể bỏ, để None)
Đây đúng là dạng `RefitDBSCAN.snapshot()` của baseline/, nên lớp này chạy được ngay hôm nay và chạy được với IncrementalDBSCAN
thật của nhóm mà không phải sửa gì, miễn là đổi MỘT chỗ: hàm `snapshot_from_model` ở dưới (xem docstring của nó).

Quy ước vẽ: lõi = hình tròn đặc, biên = tam giác, nhiễu = dấu x xám; mỗi cụm một màu (bảng Okabe-Ito, an toàn cho người mù màu),
vùng tô nhạt của cụm = hợp các hình tròn bán kính Eps quanh các điểm lõi (khi biết Eps) hoặc bao lồi (khi không biết).
Điểm được chèn = ngôi sao, điểm bị xóa = dấu X (vẽ tại vị trí cũ), UpdSeed = vòng tròn viền đen, Eps = đường tròn nét đứt.

Chỉ dùng numpy và matplotlib; không dùng sklearn hay scipy (quy ước của src/). Không đặt backend: tự chọn Agg khi chạy không màn hình.
"""
from contextlib import contextmanager

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Polygon

# Mã loại điểm, trùng với baseline.sklearn_dbscan.PointType (và thứ tự trong docs/oop_design.md).
UNCLASSIFIED, NOISE, BORDER, CORE = 0, 1, 2, 3
_TYPE_BY_NAME = {"UNCLASSIFIED": UNCLASSIFIED, "NOISE": NOISE, "BORDER": BORDER, "CORE": CORE}

# Okabe-Ito (an toàn cho người mù màu) thêm hai màu của Paul Tol; quá 9 cụm thì màu lặp lại (chấp nhận được khi các cụm cách xa nhau).
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#56B4E9", "#CC79A7", "#882255", "#332288", "#999933"]
NOISE_COLOR = "#8C8C8C"
_HALO = [patheffects.withStroke(linewidth=2.2, foreground="white")]        # viền trắng quanh chữ để đọc được khi nằm trên đường kẻ
INK = "#1A1A1A"

# Giao diện cho báo cáo A4 (Word hoặc PDF): phông DejaVu Sans có sẵn trong matplotlib và đủ dấu tiếng Việt.
REPORT_RC = {
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "figure.dpi": 100, "savefig.dpi": 200,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False, "pdf.fonttype": 42, "svg.fonttype": "none",
}


@contextmanager
def report_style():
    """Áp REPORT_RC trong một khối with rồi trả lại đúng các khóa đó. KHÔNG dùng plt.rc_context: nó khôi phục cả khóa `interactive`, và trong
    Jupyter (backend inline) việc tạo hình đầu tiên bên trong rc_context làm mọi hình sau đó không còn được hiển thị."""
    old = {k: matplotlib.rcParams[k] for k in REPORT_RC}
    matplotlib.rcParams.update(REPORT_RC)
    try:
        yield
    finally:
        matplotlib.rcParams.update(old)


# ------------------------------------------------------------------ snapshot
def _type_code(value):
    """Mã loại điểm (0 đến 3) từ số nguyên, IntEnum hoặc Enum (đọc theo TÊN CORE, BORDER, NOISE, UNCLASSIFIED nếu có)."""
    name = getattr(value, "name", None)
    if isinstance(name, str) and name.upper() in _TYPE_BY_NAME:
        return _TYPE_BY_NAME[name.upper()]
    if isinstance(value, str):
        return _TYPE_BY_NAME.get(value.upper(), UNCLASSIFIED)
    return int(value)


def make_snapshot(ids, coords, labels, point_types=None):
    """Chuẩn hóa bốn dãy thành snapshot (mảng numpy, bản sao). point_types None được điền 0 (chưa phân loại)."""
    coords = np.asarray(coords, dtype=float)
    if coords.ndim == 1:
        coords = coords.reshape(-1, 1) if coords.size else coords.reshape(0, 2)
    if coords.ndim != 2:
        raise ValueError("coords cần ma trận 2-D (n_điểm, n_chiều)")
    n = len(coords)
    ids = np.array(list(ids))
    labels = np.array(labels, dtype=int).ravel()
    types = np.zeros(n, dtype=int) if point_types is None else np.array([_type_code(t) for t in point_types], dtype=int)
    if not (len(ids) == len(labels) == len(types) == n):
        raise ValueError(f"độ dài không khớp: ids {len(ids)}, coords {n}, labels {len(labels)}, point_types {len(types)}")
    return {"ids": ids, "coords": coords.copy(), "labels": labels, "point_types": types}


def snapshot_from_model(model):
    """ADAPTER DUY NHẤT từ một mô hình phân cụm sang snapshot. Nhóm chưa chốt tên phương thức, nên mọi thay đổi chỉ sửa ở đây.

    Thử lần lượt:
      1. model.snapshot() trả dict có ids, coords, labels, point_types (RefitDBSCAN hiện nay; nên cài cùng giao diện này cho lớp thật);
      2. model.points hoặc model._points: dãy (hoặc dict id -> điểm) các đối tượng giống DataPoint trong docs/oop_design.md, mỗi
         điểm có .id, .coords, .cluster_id (None nghĩa là nhiễu) và .point_type (số, IntEnum hoặc Enum có tên CORE, BORDER, NOISE).
    Nếu cả hai không có, báo lỗi nêu rõ cần cài gì. Một dict đã đúng dạng snapshot cũng được chấp nhận.
    """
    if isinstance(model, dict):
        return make_snapshot(model["ids"], model["coords"], model["labels"], model.get("point_types"))
    if callable(getattr(model, "snapshot", None)):
        snap = model.snapshot()
        return make_snapshot(snap["ids"], snap["coords"], snap["labels"], snap.get("point_types"))
    pts = getattr(model, "points", None)
    pts = getattr(model, "_points", None) if pts is None else pts
    if pts is not None:
        pts = list(pts.values()) if isinstance(pts, dict) else list(pts)
        if not pts:
            return make_snapshot([], np.empty((0, 2)), [], [])
        labels = [-1 if p.cluster_id is None else int(p.cluster_id) for p in pts]
        return make_snapshot([p.id for p in pts], np.vstack([np.asarray(p.coords, float).ravel() for p in pts]), labels,
                             [_type_code(p.point_type) for p in pts])
    raise TypeError("không lấy được snapshot: mô hình cần phương thức snapshot() trả dict (ids, coords, labels, point_types) "
                    "hoặc thuộc tính points gồm các đối tượng có id, coords, cluster_id, point_type")


def type_counts(snap):
    """Đếm core, border, noise, other (điểm thuộc cụm nhưng chưa phân loại) và số cụm của một snapshot."""
    lab, typ = snap["labels"], snap["point_types"]
    noise = lab == -1
    core, border = (typ == CORE) & ~noise, (typ == BORDER) & ~noise
    return {"core": int(core.sum()), "border": int(border.sum()), "noise": int(noise.sum()),
            "other": int((~noise & ~core & ~border).sum()), "clusters": int(len(set(lab.tolist()) - {-1})), "n": int(len(lab))}


def align_labels(reference, snapshot, min_new_label=0):
    """Đổi nhãn của `snapshot` để cụm giữ MÀU của cụm trùng nhiều điểm nhất trong `reference` (khớp một-một, tham lam theo số điểm chung).

    reference, snapshot: snapshot hoặc dict có ids và labels. Cụm không khớp được nhận nhãn mới bắt đầu từ
    max(nhãn của reference, min_new_label - 1) + 1. Nhiễu (-1) giữ nguyên. Trả mảng nhãn mới (bản sao).
    """
    ref = dict(zip(np.asarray(reference["ids"]).tolist(), np.asarray(reference["labels"]).tolist()))
    ids, labels = np.asarray(snapshot["ids"]).tolist(), np.asarray(snapshot["labels"], dtype=int)
    counts = {}
    for i, l in zip(ids, labels.tolist()):
        r = ref.get(i)
        if l != -1 and r is not None and r != -1:
            counts[(l, r)] = counts.get((l, r), 0) + 1
    mapping, taken = {}, set()
    for (l, r), _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0][1], kv[0][0])):
        if l not in mapping and r not in taken:
            mapping[l] = r
            taken.add(r)
    nxt = max([max(ref.values(), default=-1), min_new_label - 1]) + 1
    out = labels.copy()
    for l in sorted(set(labels.tolist()) - {-1}):
        if l not in mapping:
            mapping[l] = nxt
            nxt += 1
    for l, m in mapping.items():
        out[labels == l] = m
    return out


def _convex_hull(pts):
    """Bao lồi (monotone chain, numpy thuần) của các điểm 2-D; trả mảng các đỉnh ngược chiều kim đồng hồ (có thể < 3 đỉnh)."""
    p = np.unique(np.asarray(pts, dtype=float), axis=0)
    if len(p) <= 2:
        return p

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower, upper = [], []
    for q in p.tolist():
        while len(lower) >= 2 and cross(lower[-2], lower[-1], q) <= 0:
            lower.pop()
        lower.append(q)
    for q in p.tolist()[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], q) <= 0:
            upper.pop()
        upper.append(q)
    return np.array(lower[:-1] + upper[:-1])


def label_colors(labels):
    """Mảng màu (n, 3) theo nhãn: nhiễu xám, cụm lấy PALETTE[nhãn % độ dài]."""
    from matplotlib.colors import to_rgb
    return np.array([to_rgb(NOISE_COLOR if l < 0 else PALETTE[int(l) % len(PALETTE)]) for l in labels]).reshape(-1, 3)


def _normalize_updates(update):
    if update is None:
        return []
    items = [update] if isinstance(update, dict) else list(update)
    for u in items:
        if u.get("op") not in ("insert", "delete"):
            raise ValueError(f"update cần khóa op là 'insert' hoặc 'delete', nhận được {u.get('op')!r}")
    return items


# ------------------------------------------------------------------ lớp vẽ
class ClusterVisualizer:
    """Vẽ snapshot kết quả phân cụm. Mọi phương thức nhận snapshot (dict) hoặc một mô hình có snapshot_from_model chấp nhận."""

    def __init__(self, point_size=38):
        self.point_size = float(point_size)

    # -- chuẩn bị
    @staticmethod
    def as_snapshot(obj):
        return snapshot_from_model(obj)

    @staticmethod
    def _limits(arrays, extra=None, pad=0.12, min_aspect=0.6):
        pts = np.vstack([a for a in arrays if len(a)] or [np.zeros((1, 2))])
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        if extra is not None and len(extra):
            lo, hi = np.minimum(lo, extra[:, :2].min(axis=0)), np.maximum(hi, extra[:, :2].max(axis=0))
        span = np.maximum(hi - lo, 1e-9)
        span = np.where(span < 1e-6, 1.0, span)
        m = span * pad
        xl, yl = [lo[0] - m[0], hi[0] + m[0]], [lo[1] - m[1], hi[1] + m[1]]
        w, h = xl[1] - xl[0], yl[1] - yl[0]
        if h < min_aspect * w:                      # hình quá dẹt: nới trục y đối xứng để panel không còn một sợi
            c = (yl[0] + yl[1]) / 2
            yl = [c - min_aspect * w / 2, c + min_aspect * w / 2]
        elif w < min_aspect * h:
            c = (xl[0] + xl[1]) / 2
            xl = [c - min_aspect * h / 2, c + min_aspect * h / 2]
        return tuple(xl), tuple(yl)

    def _to_2d(self, snap, basis=None):
        """Báo lỗi nếu dữ liệu nhiều chiều."""
        if snap["coords"].shape[1] != 2:
            raise ValueError("Trực quan hóa chỉ hỗ trợ dữ liệu 2D.")
        return snap, None

    # -- vẽ lõi
    def _draw(self, ax, snap, eps=None, highlight_ids=None, seed_ids=None, marks=None, fill="auto", annotate_ids=None,
              legend=True, axis_labels=None, compact=False, limits=None, title=None, proj=None, colors=None):
        s = self.point_size * (0.42 if compact else 1.0)
        xy, lab, typ, ids = snap["coords"], snap["labels"], snap["point_types"], snap["ids"]
        idx = {i: k for k, i in enumerate(ids.tolist())}
        eps_ok = eps is not None
        mode = ("eps" if eps_ok else "hull") if fill == "auto" else fill
        if mode == "eps" and not eps_ok:
            mode = "hull"
        noise = lab == -1
        core, border = (typ == CORE) & ~noise, (typ == BORDER) & ~noise
        other = ~noise & ~core & ~border
        col = label_colors(lab) if colors is None else colors

        if mode == "eps":
            patches, fcs = [], []
            for k in np.flatnonzero(core | (other & ~noise)):
                patches.append(Circle(xy[k], eps))
                fcs.append(col[k])
            if patches:
                ax.add_collection(PatchCollection(patches, facecolors=fcs, alpha=0.16, linewidths=0, zorder=1))
        elif mode == "hull":
            for l in sorted(set(lab.tolist()) - {-1}):
                h = _convex_hull(xy[lab == l])
                if len(h) >= 3:
                    c = PALETTE[l % len(PALETTE)]
                    ax.add_patch(Polygon(h, closed=True, facecolor=c, alpha=0.14, edgecolor=c, linewidth=0.8, zorder=1))

        def scat(mask, marker, gid, **kw):
            if mask.any():
                sc = ax.scatter(xy[mask, 0], xy[mask, 1], marker=marker, s=s, zorder=3, **kw)
                sc.set_gid(gid)
        scat(core, "o", "core", c=col[core], edgecolors=INK, linewidths=0.45)
        scat(border, "^", "border", c=col[border], edgecolors=INK, linewidths=0.45)
        scat(other, "o", "point", c=col[other], edgecolors=INK, linewidths=0.45)
        scat(noise, "x", "noise", c=NOISE_COLOR, linewidths=0.9 if not compact else 0.7)

        shown = {"seed": False, "insert": False, "delete": False, "highlight": False}
        for name, group in (("seed_ids", seed_ids), ("highlight_ids", highlight_ids), ("annotate_ids", annotate_ids)):
            missing = [i for i in (group or []) if i not in idx]
            if missing:
                raise KeyError(f"{name} có id không nằm trong snapshot: {missing} (id của snapshot: {list(idx)[:6]}...)")
        if seed_ids is not None:
            seeds = [idx[i] for i in seed_ids]
            if seeds:
                ax.scatter(xy[seeds, 0], xy[seeds, 1], s=s * 3.2, facecolors="none", edgecolors=INK, linewidths=1.0 if compact else 1.3,
                           zorder=4).set_gid("seed")
                shown["seed"] = True
        circle_centres = []
        if highlight_ids is not None:
            hl = [idx[i] for i in highlight_ids]
            if hl:
                ax.scatter(xy[hl, 0], xy[hl, 1], s=s * 3.8, facecolors="none", edgecolors=INK, linewidths=1.4, zorder=5).set_gid("highlight")
                circle_centres += [xy[k] for k in hl]
                shown["highlight"] = True
        pending = []                                    # nhãn điểm cập nhật, ghi sau khi biết khung trục
        for m in marks or []:
            ghost = m.get("ghost", False)
            star = m["kind"] == "insert"
            ax.scatter([m["xy"][0]], [m["xy"][1]], marker="*" if star else "X", s=s * (6.5 if star else 4.0),
                       facecolors="none" if ghost else INK, edgecolors=INK if ghost else "white",
                       linewidths=1.0 if ghost else 0.7, zorder=6).set_gid(f"{m['kind']}{'_ghost' if ghost else ''}")
            if not ghost:
                circle_centres.append(np.asarray(m["xy"], float))
                shown[m["kind"]] = True
            pending.append((None if m.get("label") is None else str(m["label"]), np.asarray(m["xy"], float), not ghost))
        rings = [c for c in circle_centres if not any(np.array_equal(c, q[1]) for q in pending)]      # vòng của điểm nhấn mạnh
        for i in annotate_ids or []:
            ax.annotate(str(i), xy[idx[i]], xytext=(3, 3), textcoords="offset points", fontsize=6.5, color=INK, zorder=7, path_effects=_HALO)

        extra = None
        if eps_ok and circle_centres:
            c = np.vstack(circle_centres)
            extra = np.vstack([c - eps, c + eps])
        xl, yl = limits if limits is not None else self._limits([xy], extra)
        ax.set_xlim(*xl)
        ax.set_ylim(*yl)
        # các điểm cập nhật nằm sát nhau (nối chuỗi, mỗi bước dưới 9% khung trục) thành một nhóm: một nhãn chung đặt ở điểm trên cùng
        # bên phải, và chỉ điểm cập nhật sau cùng của nhóm có vòng Eps, để nhiều bước liền nhau không chồng chữ và chồng vòng
        span = max(xl[1] - xl[0], yl[1] - yl[0])
        groups = []
        for item in pending:
            near = [g for g in groups if any(np.linalg.norm(o[1] - item[1]) < 0.09 * span for o in g)]
            merged = [item] + [o for g in near for o in g]
            groups = [g for g in groups if all(g is not h for h in near)] + [merged]
        order = {id(it): k for k, it in enumerate(pending)}
        offsets = [(3.5, 3.5), (3.5, -9.0), (-11.0, 3.5), (-11.0, -9.0)]
        for k, g in enumerate(groups):
            g = sorted(g, key=lambda it: order[id(it)])
            texts = [it[0] for it in g if it[0] is not None]
            if texts:
                top = max(g, key=lambda it: (it[1][1], it[1][0]))[1]
                ax.annotate(", ".join(dict.fromkeys(texts)), top, xytext=offsets[k % len(offsets)], textcoords="offset points",
                            fontsize=6.5 if compact else 8, fontweight="bold", color=INK, zorder=7, path_effects=_HALO)
            solid = [it for it in g if it[2]]
            if solid:
                rings.append(solid[-1][1])
        if eps_ok:
            for c in rings:
                ax.add_patch(Circle(c, eps, fill=False, linestyle=(0, (3, 2)), edgecolor=INK, linewidth=0.8, zorder=5))
        ax.set_aspect("equal", adjustable="box")
        if compact:
            ax.set_anchor("N")
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(True)
                sp.set_linewidth(0.6)
                sp.set_color("#999999")
        else:
            labs = axis_labels or ("x", "y")
            ax.set_xlabel(labs[0])
            ax.set_ylabel(labs[1])
        if title:
            ax.set_title(title, pad=3 if compact else 6, fontsize=6.8 if compact else None)
        if legend:
            self._legend(ax, snap, shown, eps_ok and bool(circle_centres))
        return ax

    def _legend(self, ax, snap, shown, with_eps, fig=None, counts=True, loc="upper center", anchor=(0.5, -0.14), ncol=3, present=None):
        """present (khi counts=False): dict core, border, noise cho biết loại điểm nào có mặt trong hình; thiếu thì liệt kê đủ ba loại."""
        c = type_counts(snap) if counts else None
        has = present or {"core": True, "border": True, "noise": True}
        ink = {"markerfacecolor": "#BBBBBB", "markeredgecolor": INK, "markeredgewidth": 0.5, "linestyle": ""}

        def lab(text, key):
            return f"{text}: {c[key]}" if c is not None else text
        handles = []
        if (c is None and has["core"]) or (c is not None and c["core"]):
            handles.append(Line2D([], [], marker="o", markersize=6, label=lab("Điểm lõi (core)", "core"), **ink))
        if (c is None and has["border"]) or (c is not None and c["border"]):
            handles.append(Line2D([], [], marker="^", markersize=6, label=lab("Điểm biên (border)", "border"), **ink))
        if c is not None and c["other"]:
            handles.append(Line2D([], [], marker="o", markersize=6, label=lab("Điểm thuộc cụm", "other"), **ink))
        if (c is None and has["noise"]) or (c is not None and c["noise"]):
            handles.append(Line2D([], [], marker="x", color=NOISE_COLOR, markersize=6, linestyle="", markeredgewidth=1.0,
                                  label=lab("Nhiễu (noise)", "noise")))
        if shown.get("seed"):
            handles.append(Line2D([], [], marker="o", markersize=8, markerfacecolor="none", markeredgecolor=INK, linestyle="", label="UpdSeed"))
        if shown.get("insert"):
            handles.append(Line2D([], [], marker="*", markersize=9, markerfacecolor=INK, markeredgecolor="white", linestyle="", label="Điểm chèn (insert)"))
        if shown.get("delete"):
            handles.append(Line2D([], [], marker="X", markersize=7, markerfacecolor=INK, markeredgecolor="white", linestyle="", label="Điểm xóa (delete)"))
        if shown.get("highlight"):
            handles.append(Line2D([], [], marker="o", markersize=8, markerfacecolor="none", markeredgecolor=INK, linestyle="", label="Điểm được nhấn mạnh"))
        if with_eps:
            handles.append(Line2D([], [], color=INK, linestyle=(0, (3, 2)), linewidth=0.9, label="Vòng Eps"))
        if c is not None:
            handles.append(Line2D([], [], linestyle="", label=f"Số cụm: {c['clusters']}"))
        target = fig if fig is not None else ax
        target.legend(handles=handles, loc=loc, bbox_to_anchor=anchor, ncol=ncol, frameon=False, handletextpad=0.4, columnspacing=1.2)

    # -- API công khai
    def plot_2d(self, snapshot, ax=None, eps=None, highlight_ids=None, title=None, *, fill="auto", seed_ids=None,
                annotate_ids=None, legend=True, axis_labels=None, compact=False, limits=None):
        """Vẽ một snapshot lên `ax` (tạo mới nếu None) và trả về Axes.

        snapshot: dict (ids, coords, labels, point_types) hoặc mô hình (xem snapshot_from_model); chỉ vẽ dữ liệu 2D.
        eps: bán kính Eps (đơn vị tọa độ); có Eps thì vùng tô cụm là hợp các hình tròn Eps quanh điểm lõi và điểm nhấn mạnh có vòng Eps
        highlight_ids: id cần nhấn mạnh (vòng tròn đậm). seed_ids: id vẽ vòng UpdSeed.
        fill: 'auto' | 'eps' | 'hull' | None. legend=True hiện chú thích có số lượng core, border, noise và số cụm. Mỗi loại điểm là một
        PathCollection có gid 'core', 'border', 'noise', 'point' (số điểm = len(get_offsets()))."""
        snap = snapshot_from_model(snapshot)
        with report_style():
            if ax is None:
                _, ax = plt.subplots(figsize=(4.6, 4.6) if legend else (4.0, 3.6))
            s2, proj = self._to_2d(snap)
            return self._draw(ax, s2, eps, highlight_ids, seed_ids, None, fill, annotate_ids, legend, axis_labels, compact, limits, title, proj)

    def _marks(self, update, before, after):
        """Danh sách dấu cập nhật cho (before, after): mỗi dấu {kind, xy, label, ghost} theo panel, xy theo tọa độ gốc d chiều."""
        mb, ma = [], []
        for u in _normalize_updates(update):
            pid = u.get("point_id")
            xy = u.get("coords")
            if u["op"] == "insert":
                if xy is None:
                    k = [j for j, i in enumerate(after["ids"].tolist()) if i == pid]
                    if not k:
                        raise ValueError(f"không thấy điểm chèn {pid!r} trong snapshot sau")
                    xy = after["coords"][k[0]]
                mb.append({"kind": "insert", "xy": np.asarray(xy, float), "label": pid, "ghost": True})
                ma.append({"kind": "insert", "xy": np.asarray(xy, float), "label": pid, "ghost": False})
            else:
                if xy is None:
                    k = [j for j, i in enumerate(before["ids"].tolist()) if i == pid]
                    if not k:
                        raise ValueError(f"không thấy điểm xóa {pid!r} trong snapshot trước")
                    xy = before["coords"][k[0]]
                mb.append({"kind": "delete", "xy": np.asarray(xy, float), "label": pid, "ghost": False})
                ma.append({"kind": "delete", "xy": np.asarray(xy, float), "label": None, "ghost": True})
        return mb, ma

    def plot_before_after(self, before, after, update=None, eps=None, seed_ids=None, *, axes=None, titles=("Trước", "Sau"),
                          suptitle=None, legend=True, compact=False, axis_labels=None, fill="auto", match_colors=True):
        """Hai panel (trước, sau) trên cùng tọa độ và cùng màu cụm (nhãn của `after` được khớp về `before`). Trả Figure (hoặc Figure của axes).

        update: dict {op, point_id, coords (tùy chọn)} hoặc danh sách dict theo thứ tự (kịch bản nhiều dòng cho thấy trạng thái cuối).
        op='insert': ngôi sao tại vị trí điểm, đặc ở panel sau, rỗng ở panel trước. op='delete': dấu X đặc đè lên điểm ở panel trước,
        rỗng ở panel sau (vị trí cũ). Vòng Eps nét đứt quanh dấu đặc. seed_ids: id UpdSeed, vẽ vòng đen ở panel sau.
        Cả hai trạng thái phải có tọa độ 2D. axes: cặp Axes có sẵn (dùng cho bộ ảnh)."""
        b, a = snapshot_from_model(before), snapshot_from_model(after)
        mb, ma = self._marks(update, b, a)
        with report_style():
            if axes is None:
                fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.9 if legend else 3.4))
            else:
                fig = axes[0].figure
            if b["coords"].shape[1] != 2 or a["coords"].shape[1] != 2:
                raise ValueError("Hai trạng thái phải có tọa độ 2D.")

            basis = None
            b2, pb = self._to_2d(b)
            a2, pa = self._to_2d(a)
            if match_colors:
                a2 = dict(a2)
                a2["labels"] = align_labels(b, a)
            extra = np.array([mk["xy"] for mk in mb + ma]) if (mb or ma) else None
            lim_extra = None
            if extra is not None and eps is not None and basis is None:
                lim_extra = np.vstack([extra - eps, extra + eps])
            elif extra is not None:
                lim_extra = extra
            limits = self._limits([b2["coords"], a2["coords"]], lim_extra)
            for ax, snap, marks, prx, seeds, ttl in ((axes[0], b2, mb, pb, None, titles[0]), (axes[1], a2, ma, pa, seed_ids, titles[1])):
                c = type_counts(snap)
                sub = f"{c['clusters']} cụm, {c['noise']} nhiễu" if compact else f"{c['n']} điểm, {c['clusters']} cụm, {c['noise']} nhiễu"
                self._draw(ax, snap, eps, None, seeds, marks, fill, None, False, axis_labels, compact, limits, f"{ttl} · {sub}", prx,
                           colors=label_colors(snap["labels"]))
            if legend:
                shown = {"seed": bool(seed_ids), "insert": any(m["kind"] == "insert" for m in ma), "delete": any(m["kind"] == "delete" for m in mb)}
                both = [type_counts(b), type_counts(a)]
                present = {k: any(t[k] for t in both) for k in ("core", "border", "noise")}
                self._legend(None, a2, shown, eps is not None and basis is None and bool(mb or ma), fig=fig, counts=False,
                             anchor=(0.5, 0.0), ncol=4, present=present)
            if suptitle:
                fig.suptitle(suptitle, fontsize=10, fontweight="bold")
            if legend and axes is not None:
                fig.subplots_adjust(bottom=0.2)
            return fig

    def plot_case_gallery(self, cases, n_pair_cols=2, width=7.5, panel_gap=0.04, pair_gap=0.18, margin=0.18, header_h=0.46,
                          legend_h=0.55, fill="hull"):
        """Bộ ảnh nhiều kịch bản trước và sau (khổ rộng `width` inch, vừa chiều ngang A4). Trả Figure.

        cases: danh sách dict {before, after, update, seed_ids, eps, header, note}; header in đậm trên mỗi cặp panel, note là dòng chú thích nhỏ.
        Mỗi hàng chứa n_pair_cols cặp (trước, sau); chú thích chung ở cuối. fill mặc định 'hull' (bao lồi) vì vùng Eps của toy lớn hơn khoảng cách
        giữa các điểm và che hết hình khi panel nhỏ; truyền fill='eps' để tô hợp hình tròn Eps."""
        n = len(cases)
        rows = -(-n // n_pair_cols)
        pw = (width - 2 * margin - (n_pair_cols - 1) * pair_gap - n_pair_cols * panel_gap) / (2 * n_pair_cols)

        def build(panel_hs):
            """Dựng Figure với chiều cao panel riêng từng hàng; trả (fig, chiều cao thật của các panel theo hàng)."""
            tops = [margin + sum(h + header_h + 0.12 for h in panel_hs[:r]) for r in range(rows)]
            height = margin + sum(h + header_h + 0.12 for h in panel_hs) + legend_h
            fig = plt.figure(figsize=(width, height))
            used = [0.0] * rows
            for k, case in enumerate(cases):
                r, c = divmod(k, n_pair_cols)
                x0 = margin + c * (2 * pw + panel_gap + pair_gap)
                ytop = height - tops[r]
                axs = [fig.add_axes([(x0 + j * (pw + panel_gap)) / width, (ytop - header_h - panel_hs[r]) / height, pw / width,
                                     panel_hs[r] / height]) for j in range(2)]
                self.plot_before_after(case["before"], case["after"], case.get("update"), case.get("eps"), case.get("seed_ids"),
                                       axes=axs, legend=False, compact=True, fill=fill)
                for ax in axs:
                    ax.apply_aspect()
                    used[r] = max(used[r], ax.get_position().height * height)
                fig.text((x0 + pw + panel_gap / 2) / width, (ytop - 0.02) / height, case.get("header", ""), ha="center", va="top",
                         fontsize=8, fontweight="bold")
                if case.get("note"):
                    fig.text((x0 + pw + panel_gap / 2) / width, (ytop - 0.2) / height, case["note"], ha="center", va="top",
                             fontsize=6.4, color="#444444")
            ink = {"markerfacecolor": "#BBBBBB", "markeredgecolor": INK, "markeredgewidth": 0.5, "linestyle": ""}
            entries = [(has["core"], Line2D([], [], marker="o", markersize=5, label="Điểm lõi (core)", **ink)),
                       (has["border"], Line2D([], [], marker="^", markersize=5, label="Điểm biên (border)", **ink)),
                       (has["noise"], Line2D([], [], marker="x", color=NOISE_COLOR, markersize=5, linestyle="", label="Nhiễu (noise)")),
                       (has_ins, Line2D([], [], marker="*", markersize=10, markerfacecolor=INK, markeredgecolor="white", linestyle="",
                                        label="Điểm vừa chèn (sao đặc)")),
                       (has_ins, Line2D([], [], marker="*", markersize=8, markerfacecolor="none", markeredgecolor=INK, linestyle="",
                                        label="Vị trí sắp chèn (sao rỗng)")),
                       (has_del, Line2D([], [], marker="X", markersize=6, markerfacecolor=INK, markeredgecolor="white", linestyle="",
                                        label="Điểm bị xóa (X đặc)")),
                       (has_del, Line2D([], [], marker="X", markersize=6, markerfacecolor="none", markeredgecolor=INK, linestyle="",
                                        label="Vị trí đã xóa (X rỗng)")),
                       (has_seed, Line2D([], [], marker="o", markersize=7, markerfacecolor="none", markeredgecolor=INK, linestyle="", label="UpdSeed")),
                       (has_eps, Line2D([], [], color=INK, linestyle=(0, (3, 2)), linewidth=0.9, label="Vòng Eps quanh điểm cập nhật"))]
            handles = [h for ok, h in entries if ok]
            ncol = 4 if len(handles) <= 8 else 5
            fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=ncol, frameon=False, fontsize=6.4,
                       handletextpad=0.3, columnspacing=1.0)
            fig.text(0.5, (legend_h - 0.1) / height, "Màu = cụm (giữ nguyên giữa trước và sau)", ha="center", va="bottom",
                     fontsize=6.4, color="#444444")
            return fig, used

        counts = [type_counts(snapshot_from_model(c[k])) for c in cases for k in ("before", "after")]
        has = {k: any(t[k] for t in counts) for k in ("core", "border", "noise")}
        ups = [u for c in cases for u in _normalize_updates(c.get("update"))]
        has_ins, has_del = any(u["op"] == "insert" for u in ups), any(u["op"] == "delete" for u in ups)
        has_seed = any(c.get("seed_ids") for c in cases)
        has_eps = bool(ups) and any(c.get("eps") is not None for c in cases)
        with report_style():
            fig, used = build([pw] * rows)          # lượt 1: đo chiều cao thật của panel (tỉ lệ khung hình đều nhau), lượt 2: co khoảng trống
            plt.close(fig)
            fig, _ = build([max(u, 0.3) + 0.02 for u in used])
        return fig

    @staticmethod
    def save(fig, path, dpi=200):
        """Lưu Figure ra file (tạo thư mục nếu thiếu), nền trắng, cắt lề thừa. Trả đường dẫn."""
        from pathlib import Path
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with report_style():
            fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        return path
