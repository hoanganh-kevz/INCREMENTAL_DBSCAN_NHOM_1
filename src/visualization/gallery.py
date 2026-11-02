"""Tạo dữ liệu cho gallery các ví dụ insert và delete.

    cases = build_gallery_cases("incremental_examples", RefitDBSCAN)          # engine_factory(eps, min_pts)
    fig = ClusterVisualizer().plot_case_gallery(cases, n_pair_cols=2)
"""
from .scenarios import list_scenarios, load_scenario, run_scenario

# (tiêu đề đậm, chú thích một dòng) cho 8 ví dụ, theo 7 trường hợp của báo cáo (Potential Split có hai ví dụ A và B)
EXAMPLE_CAPTIONS = {
    "r02_1_noise": ("1. Noise", "chèn p, p là nhiễu"),
    "r02_2_creation": ("2. Creation", "chèn p, tạo cụm mới"),
    "r02_3_absorption": ("3. Absorption", "chèn p, p vào cụm cũ"),
    "r02_4_merge": ("4. Merge", "chèn p, hai cụm hợp nhất"),
    "r03_1_removal": ("5. Removal", "xóa p, cụm biến mất"),
    "r03_2_reduction": ("6. Reduction", "xóa p, cụm thu nhỏ"),
    "r03_3a_split": ("7a. Split", "xóa p, cụm tách đôi"),
    "r03_3b_nonsplit": ("7b. Non-split", "xóa p, cụm vẫn liền"),
}

def build_gallery_cases(kind, engine_factory, names=None):
    """Chạy các ví dụ cập nhật trên engine và trả danh sách case cho ClusterVisualizer.plot_case_gallery.

    Mỗi case: before, after, update, seed_ids, eps, header, note, name. names chọn bớt kịch bản (mặc định tất cả)."""
    if kind != "incremental_examples":
        raise ValueError("Gallery chỉ hỗ trợ các ví dụ trong data/raw/incremental_examples")
    out = []
    for name in names or list_scenarios(kind):
        sc = load_scenario(kind, name)
        run = run_scenario(engine_factory, sc)
        header, note = EXAMPLE_CAPTIONS[name]
        out.append({"name": name, "before": run["before"], "after": run["after"], "update": run["update"], "seed_ids": run["seed_ids"],
                    "eps": sc["eps"] if run["before"]["coords"].shape[1] == 2 else None, "header": header, "note": note,
                    "expected": sc["expected"]})
    return out
