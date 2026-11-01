"""So sánh Incremental DBSCAN OOP với ba thuật toán density-based của sklearn.

Chạy từ gốc repo:
    python -m baseline.benchmark_density_algorithms --repeats 5 --max-steps 20

Mỗi ca 2D/3D/4D lấy số chiều đầu từ cùng stream thẻ; mọi phương pháp trong một ca
nhận cùng D0, thứ tự cập nhật, tọa độ, Euclidean và MinPts.
Incremental DBSCAN cập nhật cục bộ; sklearn DBSCAN, HDBSCAN và OPTICS được fit
lại toàn bộ sau mỗi cập nhật vì các estimator này không có giao diện insert/delete.
"""

from __future__ import annotations

import argparse
import inspect
import json
import platform
import random
import time
from pathlib import Path
from typing import Callable

import matplotlib
import numpy as np
import pandas as pd
import sklearn
from sklearn.cluster import DBSCAN as SklearnDBSCAN
from sklearn.cluster import HDBSCAN, OPTICS
from sklearn.metrics import adjusted_rand_score

matplotlib.use("Agg")

from src.experiments.engine import IncrementalEngine
from src.preprocessing.stream_scaling import stream_coordinates
from src.visualization.plotter import report_style
from .sklearn_dbscan import IdMap, read_stream

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
RESULTS = ROOT / "results"
FIGURES = ROOT / "report" / "figures"
RUNS_CSV = RESULTS / "density_algorithm_benchmark_runs.csv"
REPEAT_RESULTS_CSV = RESULTS / "density_algorithm_benchmark_repeats.csv"
SUMMARY_CSV = RESULTS / "density_algorithm_benchmark_summary.csv"
README_TXT = RESULTS / "density_algorithm_benchmark_README.txt"
FIGURE_PATH = FIGURES / "density_algorithm_benchmark.png"

CARD_STREAM = RAW / "card_stream"
STREAMS = {
    "Thẻ 2D": (CARD_STREAM, 2),
    "Thẻ 3D": (CARD_STREAM, 3),
    "Thẻ 4D": (CARD_STREAM, 4),
}
METHODS = (
    "Incremental DBSCAN (OOP)",
    "sklearn DBSCAN (re-fit)",
    "sklearn HDBSCAN (re-fit)",
    "sklearn OPTICS (re-fit)",
)
METHOD_COLORS = ("#1f77b4", "#7f7f7f", "#9467bd", "#2ca02c")


def _make_sklearn_estimator(method: str, eps: float, min_pts: int):
    """Tạo estimator tĩnh với cấu hình cố định cho benchmark."""
    if method == "sklearn DBSCAN (re-fit)":
        return SklearnDBSCAN(eps=eps, min_samples=min_pts, metric="euclidean", n_jobs=1)
    if method == "sklearn HDBSCAN (re-fit)":
        options = {
            "min_cluster_size": max(2, min_pts),
            "min_samples": min_pts,
            "metric": "euclidean",
            "cluster_selection_method": "eom",
            "n_jobs": 1,
        }
        # sklearn 1.9 added copy; set it explicitly there, while staying compatible with 1.4+.
        if "copy" in inspect.signature(HDBSCAN).parameters:
            options["copy"] = True
        return HDBSCAN(**options)
    if method == "sklearn OPTICS (re-fit)":
        return OPTICS(
            min_samples=min_pts,
            max_eps=eps + 1e-6,
            metric="euclidean",
            cluster_method="dbscan",
            eps=eps,
            n_jobs=1,
        )
    raise ValueError(f"Không phải estimator sklearn: {method}")


def _parameters(method: str, eps: float, min_pts: int, features: list[str]) -> str:
    """Lưu đầy đủ tham số để đọc kết quả mà không phải suy ra từ mã nguồn."""
    common = {"dimensions": len(features), "features": features, "metric": "euclidean"}
    if method == "Incremental DBSCAN (OOP)":
        config = {**common, "eps": eps, "min_pts": min_pts, "update": "local insert/delete"}
    elif method == "sklearn DBSCAN (re-fit)":
        config = {**common, "eps": eps, "min_samples": min_pts, "update": "full refit"}
    elif method == "sklearn HDBSCAN (re-fit)":
        config = {
            **common,
            "min_cluster_size": max(2, min_pts),
            "min_samples": min_pts,
            "cluster_selection_method": "eom",
            "n_jobs": 1,
            "update": "full refit",
        }
        if "copy" in inspect.signature(HDBSCAN).parameters:
            config["copy"] = True
    else:
        config = {
            **common,
            "min_samples": min_pts,
            "eps": eps,
            "max_eps": eps + 1e-6,
            "cluster_method": "dbscan",
            "n_jobs": 1,
            "update": "full refit",
        }
    return json.dumps(config, ensure_ascii=False, sort_keys=True)


class _FullRefitDensityModel:
    """Adapter insert/delete dùng để đo estimator sklearn chạy lại toàn bộ dữ liệu."""

    def __init__(self, method: str, eps: float, min_pts: int) -> None:
        self.method = method
        self.eps = eps
        self.min_pts = min_pts
        self.points: dict[int, np.ndarray] = {}
        self.labels = np.empty(0, dtype=int)
        self.next_id = 0

    def fit(self, data: np.ndarray):
        self.points = {i: np.asarray(row, dtype=float).copy() for i, row in enumerate(data)}
        self.next_id = len(self.points)
        self._refit()
        return self

    def insert(self, coords) -> int:
        point_id = self.next_id
        self.next_id += 1
        self.points[point_id] = np.asarray(coords, dtype=float).copy()
        self._refit()
        return point_id

    def delete(self, point_id: int) -> None:
        if point_id not in self.points:
            raise ValueError(f"Không tìm thấy point_id={point_id}")
        del self.points[point_id]
        self._refit()

    def _refit(self) -> None:
        if not self.points:
            self.labels = np.empty(0, dtype=int)
            return
        data = np.vstack(list(self.points.values()))
        estimator = _make_sklearn_estimator(self.method, self.eps, self.min_pts)
        estimator.fit(data)
        self.labels = np.asarray(estimator.labels_, dtype=int)

    def snapshot(self) -> dict:
        ids = list(self.points)
        dimension = len(next(iter(self.points.values()))) if self.points else 0
        coords = np.vstack([self.points[point_id] for point_id in ids]) if ids else np.empty((0, dimension))
        return {"ids": np.asarray(ids, dtype=int), "coords": coords, "labels": self.labels.copy()}


def _new_model(method: str, eps: float, min_pts: int):
    if method == "Incremental DBSCAN (OOP)":
        return IncrementalEngine(eps, min_pts)
    return _FullRefitDensityModel(method, eps, min_pts)


def _snapshot_state(model, ids: IdMap) -> tuple[dict[int, int], int, int, float]:
    """Trả labels theo point_id của stream cùng số cụm và tỷ lệ noise."""
    snap = model.snapshot()
    state = {
        ids.stream_id(engine_id): int(label)
        for engine_id, label in zip(snap["ids"].tolist(), snap["labels"].tolist())
    }
    n_noise = sum(label == -1 for label in state.values())
    n_clusters = len(set(state.values()) - {-1})
    noise_fraction = n_noise / len(state) if state else 0.0
    return state, n_clusters, n_noise, noise_fraction


def _run_one_stream(
    method: str,
    stream_name: str,
    d0: pd.DataFrame,
    updates: pd.DataFrame,
    feature_cols: list[str],
    eps: float,
    min_pts: int,
    initial_coords: np.ndarray,
    update_coords: np.ndarray,
) -> tuple[list[dict], dict[int, dict[int, int]]]:
    model = _new_model(method, eps, min_pts)
    ids = IdMap()
    records: list[dict] = []
    states: dict[int, dict[int, int]] = {}

    started = time.perf_counter()
    model.fit(initial_coords)
    initial_ms = (time.perf_counter() - started) * 1000
    for stream_id, engine_id in zip(d0["point_id"].tolist(), model.snapshot()["ids"].tolist()):
        ids.bind(stream_id, engine_id)

    def record(step: int, op: str, point_id, case: str, elapsed_ms: float, n_before: int) -> None:
        state, n_clusters, n_noise, noise_fraction = _snapshot_state(model, ids)
        states[step] = state
        records.append({
            "stream": stream_name,
            "method": method,
            "step": step,
            "op": op,
            "point_id": point_id,
            "case": case,
            "n_before": n_before,
            "n_after": len(state),
            "dimensions": int(initial_coords.shape[1]),
            "features": ",".join(feature_cols),
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "noise_fraction": noise_fraction,
            "elapsed_ms": elapsed_ms,
            "eps": eps,
            "min_pts_reference": min_pts,
            "parameters": _parameters(method, eps, min_pts, feature_cols),
        })

    record(0, "fit", None, "", initial_ms, 0)
    for index, (_, row) in enumerate(updates.iterrows()):
        step = int(row["step"])
        stream_id = int(row["point_id"])
        op = str(row["op"])
        case_value = row.get("verified_case", "")
        case = "" if pd.isna(case_value) else str(case_value)
        n_before = len(model.snapshot()["ids"])
        if op == "insert":
            started = time.perf_counter()
            engine_id = model.insert(update_coords[index])
            elapsed_ms = (time.perf_counter() - started) * 1000
            ids.bind(stream_id, engine_id)
        elif op == "delete":
            engine_id = ids.release(stream_id)
            started = time.perf_counter()
            model.delete(engine_id)
            elapsed_ms = (time.perf_counter() - started) * 1000
        else:
            raise ValueError(f"Toán tử cập nhật không hợp lệ ở step {step}: {op}")
        record(step, op, stream_id, case, elapsed_ms, n_before)
    return records, states


def _add_agreement_scores(records: list[dict], states_by_method: dict[str, dict[int, dict[int, int]]]) -> None:
    reference = states_by_method["Incremental DBSCAN (OOP)"]
    methods = list(states_by_method)
    score_by_method_step = {}
    for method in methods:
        for step, state in states_by_method[method].items():
            ref = reference[step]
            if set(ref) != set(state):
                raise RuntimeError(f"Tập điểm khác nhau ở {method}, step={step}")
            ordered_ids = sorted(ref)
            ref_labels = [ref[point_id] for point_id in ordered_ids]
            labels = [state[point_id] for point_id in ordered_ids]
            ari = float(adjusted_rand_score(ref_labels, labels))
            noise_equal = {i for i, label in ref.items() if label == -1} == {i for i, label in state.items() if label == -1}
            score_by_method_step[(method, step)] = (ari, noise_equal)
    for row in records:
        row["ari_vs_incremental"] = score_by_method_step[(row["method"], row["step"])][0]
        row["noise_matches_incremental"] = score_by_method_step[(row["method"], row["step"])][1]


def _load_prefix(stream_dir: Path, dimensions: int, max_steps: int):
    d0, updates, params, feature_cols = read_stream(stream_dir)
    if not 1 <= dimensions <= len(feature_cols):
        raise ValueError(f"Số chiều {dimensions} không hợp lệ cho {len(feature_cols)} đặc trưng.")
    updates = updates.sort_values("step").head(max_steps).reset_index(drop=True)
    initial_coords = stream_coordinates(params, d0, feature_cols)[:, :dimensions]
    update_coords = stream_coordinates(params, updates, feature_cols)[:, :dimensions]
    frozen_scaling = params.get("frozen_scaling")
    scaled_features = frozen_scaling["columns"] if frozen_scaling else feature_cols
    return d0, updates, params, scaled_features[:dimensions], initial_coords, update_coords


def run_benchmark(
    *,
    stream_names: list[str] | None = None,
    repeats: int = 5,
    warmups: int = 1,
    max_steps: int = 20,
    seed: int = 42,
    log: Callable[[str], None] = print,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Benchmark các prefix của cùng stream thẻ ở 2D/3D/4D; sklearn refit toàn bộ sau từng update."""
    if repeats < 2:
        raise ValueError("repeats phải từ 2 trở lên")
    if warmups < 0:
        raise ValueError("warmups không được âm")
    if max_steps < 1:
        raise ValueError("max_steps phải lớn hơn 0")
    chosen = [name for name in STREAMS if stream_names is None or name in stream_names]
    unknown = set(stream_names or ()) - set(STREAMS)
    if unknown:
        raise ValueError(f"Không tìm thấy stream: {', '.join(sorted(unknown))}")

    all_records = []
    rng = random.Random(seed)
    for stream_name in chosen:
        stream_dir, dimensions = STREAMS[stream_name]
        d0, updates, params, feature_cols, initial_coords, update_coords = _load_prefix(
            stream_dir, dimensions, max_steps
        )
        eps = float(params.get("eps", params.get("eps_km")))
        min_pts = int(params["minpts"])

        # Warm-up chỉ fit D0; các phép đo bên dưới vẫn phát lại từng chuỗi từ đầu.
        for method in METHODS:
            for _ in range(warmups):
                _new_model(method, eps, min_pts).fit(initial_coords)

        stream_records = []
        for repeat in range(1, repeats + 1):
            order = list(METHODS)
            rng.shuffle(order)
            states_by_method = {}
            repeat_records = []
            for method in order:
                records, states = _run_one_stream(
                    method, stream_name, d0, updates, feature_cols, eps, min_pts,
                    initial_coords, update_coords,
                )
                for row in records:
                    row["repeat"] = repeat
                repeat_records.extend(records)
                states_by_method[method] = states
            _add_agreement_scores(repeat_records, states_by_method)
            stream_records.extend(repeat_records)
            stream_records.sort(key=lambda row: (METHODS.index(row["method"]), row["repeat"], row["step"]))
            updates_count = len(updates)
            log(
                f"{stream_name}: hoàn tất lượt {repeat}/{repeats}, "
                f"{updates_count} update, Eps={eps:g}, MinPts={min_pts}"
            )
        all_records.extend(stream_records)

    runs = pd.DataFrame(all_records)
    repeat_results = summarize_repeats(runs)
    summary = summarize_runs(repeat_results, max_steps=max_steps)
    return runs, repeat_results, summary


def summarize_repeats(runs: pd.DataFrame) -> pd.DataFrame:
    """Tạo một dòng cho từng stream/phương pháp/lượt để so sánh trực tiếp."""
    rows = []
    for (stream_name, method, repeat), part in runs.groupby(["stream", "method", "repeat"], sort=False):
        initial = part[part["op"] == "fit"].iloc[0]
        updates = part[part["op"].isin(("insert", "delete"))]
        inserts = updates[updates["op"] == "insert"]["elapsed_ms"]
        deletes = updates[updates["op"] == "delete"]["elapsed_ms"]
        final = part.sort_values("step").iloc[-1]
        rows.append({
            "stream": stream_name,
            "method": method,
            "repeat": int(repeat),
            "n_d0": int(initial["n_after"]),
            "dimensions": int(initial["dimensions"]),
            "features": initial["features"],
            "n_updates": int(len(updates)),
            "n_insert": int((updates["op"] == "insert").sum()),
            "n_delete": int((updates["op"] == "delete").sum()),
            "eps": float(initial["eps"]),
            "min_pts_reference": int(initial["min_pts_reference"]),
            "parameters": initial["parameters"],
            "initial_fit_ms": float(initial["elapsed_ms"]),
            "insert_total_ms": float(inserts.sum()),
            "insert_median_ms_per_update": float(inserts.median()) if len(inserts) else 0.0,
            "delete_total_ms": float(deletes.sum()),
            "delete_median_ms_per_update": float(deletes.median()) if len(deletes) else 0.0,
            "update_total_ms": float(updates["elapsed_ms"].sum()),
            "ari_vs_incremental_mean": float(updates["ari_vs_incremental"].mean()) if len(updates) else 1.0,
            "ari_vs_incremental_final": float(final["ari_vs_incremental"]),
            "noise_matches_incremental_fraction": float(updates["noise_matches_incremental"].mean()) if len(updates) else 1.0,
            "final_clusters": int(final["n_clusters"]),
            "final_noise_fraction": float(final["noise_fraction"]),
        })
    return pd.DataFrame(rows)


def summarize_runs(repeats: pd.DataFrame, *, max_steps: int) -> pd.DataFrame:
    """Tổng hợp các lượt lặp, giữ median/mean/std và min/max thời gian cập nhật."""
    rows = []
    for (stream_name, method), part in repeats.groupby(["stream", "method"], sort=False):
        totals = part["update_total_ms"]
        rows.append({
            "stream": stream_name,
            "method": method,
            "repeats": int(part["repeat"].nunique()),
            "n_d0": int(part["n_d0"].iloc[0]),
            "dimensions": int(part["dimensions"].iloc[0]),
            "features": part["features"].iloc[0],
            "max_steps_requested": max_steps,
            "n_updates_per_repeat": int(part["n_updates"].median()),
            "n_insert_per_repeat": int(part["n_insert"].median()),
            "n_delete_per_repeat": int(part["n_delete"].median()),
            "eps": float(part["eps"].iloc[0]),
            "min_pts_reference": int(part["min_pts_reference"].iloc[0]),
            "parameters": part["parameters"].iloc[0],
            "initial_fit_median_ms": float(part["initial_fit_ms"].median()),
            "insert_median_ms_per_update": float(part["insert_median_ms_per_update"].median()),
            "delete_median_ms_per_update": float(part["delete_median_ms_per_update"].median()),
            "update_total_median_ms": float(totals.median()),
            "update_total_mean_ms": float(totals.mean()),
            "update_total_std_ms": float(totals.std(ddof=1)) if len(totals) > 1 else 0.0,
            "update_total_min_ms": float(totals.min()),
            "update_total_max_ms": float(totals.max()),
            "ari_vs_incremental_mean": float(part["ari_vs_incremental_mean"].mean()),
            "ari_vs_incremental_final_mean": float(part["ari_vs_incremental_final"].mean()),
            "noise_matches_incremental_fraction": float(part["noise_matches_incremental_fraction"].mean()),
            "final_clusters_median": float(part["final_clusters"].median()),
            "final_noise_fraction_median": float(part["final_noise_fraction"].median()),
        })
    return pd.DataFrame(rows)


def plot_summary(summary: pd.DataFrame):
    """Vẽ median tổng thời gian cập nhật; whisker min–max là độ biến thiên giữa các lượt."""
    import matplotlib.pyplot as plt

    streams = list(dict.fromkeys(summary["stream"].tolist()))
    x = np.arange(len(streams), dtype=float)
    width = 0.19
    with report_style():
        fig, ax = plt.subplots(figsize=(10.2, 4.5))
        for method_index, (method, color) in enumerate(zip(METHODS, METHOD_COLORS)):
            rows = summary[summary["method"] == method].set_index("stream").reindex(streams)
            median = rows["update_total_median_ms"].to_numpy(dtype=float)
            low = median - rows["update_total_min_ms"].to_numpy(dtype=float)
            high = rows["update_total_max_ms"].to_numpy(dtype=float) - median
            positions = x + (method_index - (len(METHODS) - 1) / 2) * width
            ax.bar(positions, median, width, color=color, label=method)
            ax.errorbar(positions, median, yerr=np.vstack((low, high)), fmt="none", ecolor="#333333", capsize=2, linewidth=0.8)
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(streams, fontsize=8)
        ax.set_ylabel("Tổng thời gian insert/delete (ms, log)")
        ax.set_title("Incremental DBSCAN OOP so với sklearn fit lại toàn bộ\nTrung vị chuỗi và khoảng min–max qua các lượt", loc="left", fontsize=10)
        ax.legend(frameon=False, fontsize=7.5, loc="center left", bbox_to_anchor=(1.01, 0.5))
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout(rect=(0, 0, 0.78, 1))
    return fig


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--streams", nargs="+", choices=list(STREAMS), default=None)
    parser.add_argument("--repeats", type=int, default=5, help="Số lần phát lại cùng chuỗi (tối thiểu 2).")
    parser.add_argument("--warmups", type=int, default=1, help="Số lần warm-up fit D0, không tính vào kết quả.")
    parser.add_argument("--max-steps", type=int, default=20, help="Độ dài prefix stream dùng trong mỗi lượt.")
    parser.add_argument("--seed", type=int, default=42, help="Seed cố định để xáo thứ tự các thuật toán.")
    args = parser.parse_args(argv)
    if args.repeats < 2:
        parser.error("--repeats phải từ 2 trở lên")
    if args.warmups < 0:
        parser.error("--warmups không được âm")
    if args.max_steps < 1:
        parser.error("--max-steps phải lớn hơn 0")
    if args.seed < 0:
        parser.error("--seed phải là số nguyên không âm")

    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    runs, repeat_results, summary = run_benchmark(
        stream_names=args.streams,
        repeats=args.repeats,
        warmups=args.warmups,
        max_steps=args.max_steps,
        seed=args.seed,
        log=lambda message: print(message, flush=True),
    )
    runs.to_csv(RUNS_CSV, index=False, encoding="utf-8", lineterminator="\n")
    repeat_results.to_csv(REPEAT_RESULTS_CSV, index=False, encoding="utf-8", lineterminator="\n")
    summary.to_csv(SUMMARY_CSV, index=False, encoding="utf-8", lineterminator="\n")
    from src.visualization.plotter import ClusterVisualizer
    ClusterVisualizer.save(plot_summary(summary), FIGURE_PATH)
    (README_TXT).write_text(
        "Benchmark chuỗi cập nhật: Incremental DBSCAN OOP so với DBSCAN, HDBSCAN và OPTICS của sklearn chạy lại toàn bộ.\n"
        f"Máy: {platform.platform()}, {platform.processor() or platform.machine()}, Python {platform.python_version()}, "
        f"scikit-learn {sklearn.__version__}, numpy {np.__version__}, pandas {pd.__version__}.\n"
        f"Mỗi trường hợp chiều: {args.warmups} warm-up fit D0 và {args.repeats} lượt phát lại prefix tối đa {args.max_steps} update; "
        f"seed xáo thứ tự thuật toán: {args.seed}.\n"
        "Ba trường hợp 2D/3D/4D lấy lần lượt 2, 3, 4 đặc trưng đầu của cùng stream thẻ đã chuẩn hóa đóng băng; D0 và thứ tự update giống nhau.\n"
        "Danh sách đặc trưng: log10_amount_vnd, hour_of_day; thêm distance_from_home_km cho 3D; thêm txns_last_7d cho 4D.\n"
        "Cả bốn phương pháp trong từng trường hợp nhận cùng D0, tọa độ, metric Euclidean và MinPts tham chiếu. Incremental DBSCAN chỉ cập nhật vùng ảnh hưởng;\n"
        "ba estimator sklearn không có insert/delete nên adapter cập nhật tập điểm rồi fit lại toàn bộ ở từng step.\n"
        "DBSCAN dùng Eps/MinPts trong params.json; HDBSCAN dùng min_cluster_size=min_samples=MinPts và EOM;\n"
        "OPTICS dùng min_samples=MinPts, eps=Eps và cluster_method='dbscan'; max_eps=Eps+1e-6 để trích cụm tại cùng ngưỡng.\n"
        "HDBSCAN có tham số phân cấp riêng; OPTICS vẫn tính thứ tự reachability trước khi trích cụm kiểu DBSCAN.\n"
        "CSV repeats có một dòng cho mỗi stream/phương pháp/lượt chạy; CSV runs lưu từng step/lượt; CSV summary tổng hợp median/mean/std/min/max thời gian qua các lượt.\n"
        "Các CSV có cả cấu hình tham số, số cụm/noise, ARI trung bình/cuối và tỷ lệ noise khớp Incremental DBSCAN OOP.\n"
        "Thời gian phụ thuộc máy; kết quả trên prefix stream không đại diện toàn bộ 300 bước.\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"\nĐã ghi: {RUNS_CSV.relative_to(ROOT)}")
    print(f"Đã ghi: {REPEAT_RESULTS_CSV.relative_to(ROOT)}")
    print(f"Đã ghi: {SUMMARY_CSV.relative_to(ROOT)}")
    print(f"Đã ghi: {FIGURE_PATH.relative_to(ROOT)}")
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
