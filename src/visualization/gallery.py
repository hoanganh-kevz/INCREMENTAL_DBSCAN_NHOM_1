"""Bộ ảnh các trường hợp chèn và xóa dựng từ kịch bản THẬT (toy và toy_edge), không vẽ tay: mỗi cặp panel là kết quả của engine được truyền vào.

    cases = build_gallery_cases("toy", RefitDBSCAN)          # engine_factory(eps, min_pts)
    fig = ClusterVisualizer().plot_case_gallery(cases, n_pair_cols=2)
"""
from .scenarios import list_scenarios, load_scenario, run_scenario

# (tiêu đề đậm, chú thích một dòng) cho 8 kịch bản toy, theo 7 trường hợp của báo cáo (Potential Split có hai ví dụ A và B)
TOY_CAPTIONS = {
    "r02_1_noise": ("1. Noise", "chèn p, p là nhiễu"),
    "r02_2_creation": ("2. Creation", "chèn p, tạo cụm mới"),
    "r02_3_absorption": ("3. Absorption", "chèn p, p vào cụm cũ"),
    "r02_4_merge": ("4. Merge", "chèn p, hai cụm hợp nhất"),
    "r03_1_removal": ("5. Removal", "xóa p, cụm biến mất"),
    "r03_2_reduction": ("6. Reduction", "xóa p, cụm thu nhỏ"),
    "r03_3a_split": ("7a. Split", "xóa p, cụm tách đôi"),
    "r03_3b_nonsplit": ("7b. Non-split", "xóa p, cụm vẫn liền"),
}

EDGE_NOTES = {
    "e01_insert_at_eps_inclusive": "chèn đúng cách Eps (bao hàm)",
    "e02_insert_just_outside_eps": "chèn ngay ngoài Eps",
    "e03_insert_tie_makes_core": "đúng Eps: nhiễu thành lõi",
    "e04_insert_duplicate_coordinate": "chèn trùng tọa độ",
    "e05_insert_border_to_core_cascade": "biên thành lõi, lan chuyền",
    "e06_insert_bridges_three_clusters": "một điểm nối 3 cụm",
    "e07_insert_merge_absorbs_noise": "hợp nhất và hút nhiễu",
    "e08_insert_creation_exactly_minpts": "3 lần chèn, đủ MinPts mới tạo cụm",
    "e09_insert_noise_near_border": "nhiễu cạnh điểm biên",
    "e10_insert_border_of_two_clusters": "biên của hai cụm (mơ hồ)",
    "e11_delete_border_point": "xóa điểm biên",
    "e12_delete_noise_point": "xóa điểm nhiễu",
    "e13_delete_splits_into_three": "xóa lõi trung tâm: tách 3",
    "e14_delete_nonsplit_long_detour": "xóa cầu, đường vòng dài",
    "e15_delete_shared_border_to_noise": "biên chung thành nhiễu",
    "e16_delete_bridge_leaves_ambiguous_border": "xóa cầu, còn biên mơ hồ",
    "e17_delete_then_reinsert_round_trip": "xóa rồi chèn lại (2 bước)",
    "e18_insert_4d_merge": "4 chiều (PCA): chèn gộp 2 cụm",
    "e19_minpts2_chain": "MinPts = 2, chuỗi nối (2 bước)",
}


def _short_case(case):
    return case.replace("PotentialSplit_", "")


def build_gallery_cases(kind, engine_factory, names=None):
    """Chạy các kịch bản `kind` ('toy' hoặc 'toy_edge') trên engine và trả danh sách case cho ClusterVisualizer.plot_case_gallery.

    Mỗi case: before, after, update, seed_ids, eps, header, note, name. names chọn bớt kịch bản (mặc định tất cả)."""
    out = []
    for name in names or list_scenarios(kind):
        sc = load_scenario(kind, name)
        run = run_scenario(engine_factory, sc)
        if kind == "toy":
            header, note = TOY_CAPTIONS[name]
        else:
            cases = [_short_case(s["case"]) for s in sc["expected"]["steps"]]
            tag = cases[-1] if len(set(cases)) == 1 or len(cases) > 2 else " → ".join(cases)
            header, note = f"{name[:3]} {tag}", EDGE_NOTES[name]
        out.append({"name": name, "before": run["before"], "after": run["after"], "update": run["update"], "seed_ids": run["seed_ids"],
                    "eps": sc["eps"] if run["before"]["coords"].shape[1] == 2 else None, "header": header, "note": note,
                    "expected": sc["expected"]})
    return out
