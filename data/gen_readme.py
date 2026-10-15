"""Fills the generated blocks of README_data.md (<!-- NAME:START --> ... <!-- NAME:END -->) from the files on disk."""
import csv
import json
import re
from pathlib import Path

DATA = Path(__file__).resolve().parent
RAW = DATA / "raw"

SUPPORT_VI = {"insert": "TV2 (chèn)", "delete": "TV3 (xóa)", "equivalence": "TV5 (tương đương với re-fit)"}
EDGE_TASK = {"e18_insert_4d_merge": "C12; chứng minh khoảng cách không viết cứng 2D", "e19_minpts2_chain": "C12; MinPts nhỏ nhất, không có điểm biên",
             "e17_delete_then_reinsert_round_trip": "C12; kiểm tra id mới và trạng thái nội bộ sau Split"}


def block(text, name, body):
    pat = re.compile(rf"(<!-- {name}:START -->\n).*?(<!-- {name}:END -->)", re.S)
    if not pat.search(text):
        raise RuntimeError(f"README_data.md has no {name} block")
    return pat.sub(lambda m: m.group(1) + body.rstrip() + "\n" + m.group(2), text)


def rows_of(p):
    with open(p, newline="", encoding="utf-8") as f:
        return sum(1 for _ in f) - 1


def filemap():
    out = ["| File | Dòng | Dung lượng |", "|---|---|---|"]
    for base in (RAW, DATA / "truth", DATA / "processed"):
        for p in sorted(base.rglob("*")):
            if not p.is_file() or p.name == ".gitkeep" or {"bench", "baseline_bench"} & set(p.parts) or "toy" in p.parts[-3:-1] or "toy_edge" in p.parts[-3:-1]:
                continue
            rel = p.relative_to(DATA).as_posix()
            n = rows_of(p) if p.suffix == ".csv" else ""
            out.append(f"| `{rel}` | {n} | {p.stat().st_size / 1024:.1f} KB |")
    for name in ("toy", "toy_edge"):
        d = RAW / name
        k = len([x for x in d.iterdir() if x.is_dir()])
        sz = sum(p.stat().st_size for p in d.rglob("*") if p.is_file())
        out.append(f"| `raw/{name}/<scenario>/` ({k} thư mục x 3 file) | | {sz / 1024:.1f} KB |")
    total = sum(p.stat().st_size for p in DATA.rglob("*")
                if p.is_file() and "__pycache__" not in p.parts and not {"bench", "baseline_bench"} & set(p.parts))
    out.append(f"| **Toàn bộ `data/`** | | **{total / 2 ** 20:.2f} MB** |")
    return "\n".join(out)


def defects():
    rows = list(csv.DictReader(open(DATA / "truth" / "defect_catalogue_card.csv", newline="", encoding="utf-8")))
    out = ["| Phạm vi | Bảng | Loại lỗi | Cột | Số dòng bị ảnh hưởng |", "|---|---|---|---|---|"]
    for r in rows:
        out.append(f"| {r['scope']} | {r['table']} | {r['defect_type']} | {r['column']} | {r['n_rows']} |")
    return "\n".join(out)


def difficulty():
    rep = json.loads((DATA / "difficulty_report.json").read_text(encoding="utf-8"))
    out = []
    for fam, ttl in (("atm", "ATM 2D (gom theo mét, nhãn thật = hotspot_id)"), ("card", "Thẻ 5D (log10 amount + z-score, nhãn thật = segment_id)")):
        out += [f"**{ttl}**", "", "| Thuật toán | Dễ | Vừa | Khó | Tham số tốt nhất (mức khó) |", "|---|---|---|---|---|"]
        for alg in ("DBSCAN", "HDBSCAN", "OPTICS", "KMeans (true k)", "Ward HAC (true k)"):
            v = [rep[fam][l]["results"][alg][0] for l in ("easy", "medium", "hard")]
            out.append(f"| {alg} | {v[0]:.3f} | {v[1]:.3f} | {v[2]:.3f} | " + "; ".join(
                f"{l}: {rep[fam][l]['results'][alg][1]}" for l in ("easy", "medium", "hard")) + " |")
        out.append(f"| (số điểm / nhiễu thật) | " + " | ".join(f"{rep[fam][l]['n']} / {rep[fam][l]['noise_share'] * 100:.0f} %" for l in ("easy", "medium", "hard")) + " | |")
        out.append("")
        for l in ("easy", "medium", "hard"):
            r = rep[fam][l]["results"]
            beat = [f"{a} ({r[a][0]:.2f} so với DBSCAN {r['DBSCAN'][0]:.2f})" for a in r if a != "DBSCAN" and r[a][0] > r["DBSCAN"][0] + 1e-9]
            out.append(f"- Mức {l}: " + ("thuật toán đánh bại DBSCAN: " + ", ".join(beat) if beat else "DBSCAN tốt nhất hoặc hòa nhất.")) 
        out.append("")
    return "\n".join(out)


def edge():
    base = RAW / "toy_edge"
    out = ["| Kịch bản | Case (theo bước) | Eps / MinPts | Kiểm tra mã của | Hỗ trợ task | Ghi chú |", "|---|---|---|---|---|---|"]
    for d in sorted(p for p in base.iterdir() if p.is_dir()):
        e = json.loads((d / "expected.json").read_text(encoding="utf-8"))
        who = ", ".join(SUPPORT_VI[k] for k in ("insert", "delete", "equivalence") if e["tests"][k])
        extra = []
        if e["ambiguous_to_ignore"]:
            extra.append("có điểm biên mơ hồ: " + ",".join(e["ambiguous_to_ignore"]))
        if e["ties_at_exactly_eps"]:
            extra.append("khoảng cách đúng bằng Eps")
        if e["dims"] > 2:
            extra.append(f"{e['dims']} chiều")
        hops = [s.get("max_core_hops_between_seeds") for s in e["steps"] if s.get("max_core_hops_between_seeds")]
        if hops:
            extra.append(f"độ sâu BFS giữa các hạt giống = {max(hops)}")
        out.append(f"| `{d.name}` | {' → '.join(s['case'] for s in e['steps'])} | {e['eps']:g} / {e['minpts']} | {who} | "
                   f"{EDGE_TASK.get(d.name, 'C12')} | {e['title']}" + (f" ({'; '.join(extra)})" if extra else "") + " |")
    return "\n".join(out)


def streamhard():
    p = json.loads((RAW / "bank_card" / "card_stream_hard" / "params.json").read_text(encoding="utf-8"))
    out = ["| verified_case | Số lần | Trong đó cài sẵn (planted_case) |", "|---|---|---|"]
    for k, v in p["verified_case_counts"].items():
        out.append(f"| {k} | {v} | {p['planted_case_counts'].get(k, 0)} |")
    out.append(f"| **Tổng** | **{p['n_steps']}** | **{sum(p['planted_case_counts'].values())}** |")
    return "\n".join(out)


def update():
    p = DATA / "README_data.md"
    if not p.exists():
        return
    t = p.read_text(encoding="utf-8")
    for name, fn in (("FILEMAP", filemap), ("DEFECTS", defects), ("DIFFICULTY", difficulty), ("EDGE", edge), ("STREAMHARD", streamhard)):
        if f"<!-- {name}:START -->" in t:
            t = block(t, name, fn())
    p.write_text(t, encoding="utf-8", newline="\n")
