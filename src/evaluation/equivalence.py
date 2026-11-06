"""Đối chiếu Incremental DBSCAN với DBSCAN chạy lại toàn bộ bằng scikit-learn .

Việc sử dụng thư viện chỉ giới hạn trong phần đánh giá và so sánh.
Thuật toán Incremental DBSCAN vẫn do nhóm tự cài đặt.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from baseline.sklearn_dbscan import (
    RefitDBSCAN,
    ambiguous_by_step,
    compare_partitions,
    read_stream,
)
from src.clustering.incremental_dbscan import IncrementalDBSCAN
from src.visualization.scenarios import (
    list_scenarios,
    load_scenario,
    run_scenario,
    stream_snapshots,
)

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
STREAMS = (
    "card_stream",
)


def _compare_snapshots(
    case: str,
    step: int,
    operation: str,
    custom: dict[str, Any],
    reference: dict[str, Any],
    *,
    ignore_ids: set[Any] | None = None,
    eps: float,
    min_pts: int,
) -> dict[str, Any]:
    """So sánh hai trạng thái phân cụm theo ID điểm"""
    custom_ids = [value.item() if hasattr(value, "item") else value for value in custom["ids"]]
    reference_ids = [value.item() if hasattr(value, "item") else value for value in reference["ids"]]
    custom_labels = dict(zip(custom_ids, map(int, custom["labels"])))
    reference_labels = dict(zip(reference_ids, map(int, reference["labels"])))
    if set(custom_labels) != set(reference_labels):
        raise ValueError(f"{case} step {step}: custom và sklearn có tập điểm khác nhau")

    ids = sorted(custom_labels, key=str)
    custom_types = dict(zip(custom_ids, map(int, custom["point_types"])))
    reference_types = dict(zip(reference_ids, map(int, reference["point_types"])))
    comparison = compare_partitions(
        custom_labels,
        reference_labels,
        ids=ids,
        ignore_ids=ignore_ids,
        types_a=[custom_types[point_id] for point_id in ids],
        types_b=[reference_types[point_id] for point_id in ids],
    )
    custom_values = list(custom_labels.values())
    reference_values = list(reference_labels.values())
    core_equal = comparison["core_equal"]
    equivalent = comparison["partition_equal"] and core_equal is True
    return {
        "case": case,
        "step": step,
        "operation": operation,
        "n_points": len(ids),
        "n_compared": comparison["n_compared"],
        "n_ignored_ambiguous_border": comparison["n_ignored"],
        "eps": eps,
        "min_pts": min_pts,
        "n_clusters_incremental": len(set(custom_values) - {-1}),
        "n_clusters_sklearn": len(set(reference_values) - {-1}),
        "n_noise_incremental": sum(value == -1 for value in custom_values),
        "n_noise_sklearn": sum(value == -1 for value in reference_values),
        "ari": comparison["ari"],
        "noise_equal": comparison["noise_equal"],
        "core_equal": core_equal,
        "equivalent": equivalent,
    }


def _scenario_rows(kind: str, name: str, raw_dir: Path = RAW) -> list[dict[str, Any]]:
    scenario = load_scenario(kind, name, raw_dir)
    custom = run_scenario(IncrementalDBSCAN, scenario)
    reference = run_scenario(RefitDBSCAN, scenario)
    expected = scenario["expected"]
    global_ambiguous = set(expected.get("ambiguous_to_ignore", []))
    rows = [
        _compare_snapshots(
            f"{kind}/{name}",
            0,
            "fit",
            custom["before"],
            reference["before"],
            ignore_ids=global_ambiguous,
            eps=scenario["eps"],
            min_pts=scenario["min_pts"],
        )
    ]
    expected_steps = expected.get("steps") or []
    for index, (custom_step, reference_step, update) in enumerate(
        zip(custom["steps"], reference["steps"], scenario["updates"]), start=1
    ):
        step_info = expected_steps[index - 1] if index <= len(expected_steps) else {}
        ambiguous = global_ambiguous | set(step_info.get("ambiguous_border_ids", []))
        row = _compare_snapshots(
            f"{kind}/{name}",
            index,
            update["op"],
            custom_step,
            reference_step,
            ignore_ids=ambiguous,
            eps=scenario["eps"],
            min_pts=scenario["min_pts"],
        )
        row["point_id"] = update["point_id"]
        row["verified_case"] = step_info.get("case", "")
        rows.append(row)
    return rows


def _stream_rows(
    stream: str,
    raw_dir: Path = RAW,
    max_steps: int | None = None,
) -> list[dict[str, Any]]:
    stream_dir = raw_dir / stream
    d0, updates, params, _ = read_stream(stream_dir)
    ambiguous = ambiguous_by_step(params, updates)
    custom_frames = stream_snapshots(IncrementalDBSCAN, stream_dir, max_steps=max_steps)
    reference_frames = stream_snapshots(RefitDBSCAN, stream_dir, max_steps=max_steps)
    if len(custom_frames) != len(reference_frames):
        raise ValueError(f"{stream}: custom và sklearn có số step khác nhau")

    eps = float(params.get("eps", params.get("eps_km")))
    rows = []
    for custom_frame, reference_frame in zip(custom_frames, reference_frames):
        step = int(custom_frame["step"])
        row = _compare_snapshots(
            stream,
            step,
            custom_frame["op"],
            custom_frame["snapshot"],
            reference_frame["snapshot"],
            ignore_ids=set(ambiguous.get(step, [])),
            eps=eps,
            min_pts=int(params["minpts"]),
        )
        row["point_id"] = custom_frame["point_id"]
        row["verified_case"] = custom_frame.get("case") or ""
        rows.append(row)
    return rows


def run_equivalence_suite(
    raw_dir: Path = RAW,
    *,
    include_scenarios: bool = True,
    include_streams: bool = True,
    max_stream_steps: int | None = None,
) -> pd.DataFrame:
    """Đối chiếu kết quả phân cụm ban đầu và từng bước cập nhật được chọn."""
    rows: list[dict[str, Any]] = []
    if include_scenarios:
        for kind in ("incremental_examples",):
            for name in list_scenarios(kind, raw_dir):
                rows.extend(_scenario_rows(kind, name, raw_dir))
    if include_streams:
        for stream in STREAMS:
            rows.extend(_stream_rows(stream, raw_dir, max_stream_steps))
    return pd.DataFrame(rows)


def write_equivalence_results(
    output_path: Path = ROOT / "results" / "equivalence_results.csv",
    *,
    max_stream_steps: int | None = None,
) -> pd.DataFrame:
    """Thực hiện toàn bộ quá trình đối chiếu và lưu kết quả từng lần phân cụm, cập nhật vào file CSV."""
    results = run_equivalence_suite(max_stream_steps=max_stream_steps)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False, encoding="utf-8", lineterminator="\n")
    return results


def main() -> int:
    results = write_equivalence_results()
    failed = results.loc[~results["equivalent"]]
    print(f"Đã ghi {len(results)} kết quả so sánh vào results/equivalence_results.csv")
    if failed.empty:
        print("Tất cả trạng thái của Incremental DBSCAN đều khớp với kết quả chạy lại DBSCAN bằng scikit-learn trên toàn bộ dữ liệu.")
        return 0
    print(f"Phát hiện {len(failed)}  kết quả so sánh không trùng khớp, kiểm tra file CSV để xem chi tiết từng trường hợp và bước cập nhật.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
