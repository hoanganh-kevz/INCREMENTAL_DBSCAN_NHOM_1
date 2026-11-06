from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Iterable


def _prepare_inputs(
    data: Iterable[Iterable[float]], labels: Iterable[Any]
) -> tuple[list[tuple[float, ...]], list[Any]]:
    """Kiểm tra tính hợp lệ và chuẩn hóa dữ liệu đặc trưng cùng nhãn cụm."""
    try:
        rows = [tuple(float(value) for value in row) for row in data]
    except (TypeError, ValueError) as exc:
        raise ValueError("data phải là ma trận số hai chiều") from exc

    try:
        label_list = list(labels)
    except TypeError as exc:
        raise ValueError("labels phải là một dãy nhãn") from exc

    if not rows:
        raise ValueError("data không được rỗng")
    if len(rows) != len(label_list):
        raise ValueError("data và labels phải có cùng số phần tử")
    if not rows[0]:
        raise ValueError("mỗi điểm phải có ít nhất một đặc trưng")
    if any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("các điểm phải có cùng số chiều")
    if any(not math.isfinite(value) for row in rows for value in row):
        raise ValueError("data chỉ được chứa giá trị hữu hạn")
    try:
        for label in label_list:
            hash(label)
    except TypeError as exc:
        raise ValueError("mỗi nhãn phải là giá trị có thể băm") from exc
    return rows, label_list


def silhouette_samples(
    data: Iterable[Iterable[float]],
    labels: Iterable[Any],
    *,
    noise_label: Any = -1,
) -> list[float | None]:
    """Tính hệ số Silhouette theo khoảng cách Euclid cho từng điểm dữ liệu.

    Các điểm nhiễu không tham gia tính toán và được biểu diễn bằng 'None'
    trong danh sách kết quả, đồng thời giữ nguyên thứ tự đầu vào.
    """
    rows, label_list = _prepare_inputs(data, labels)
    included = [index for index, label in enumerate(label_list) if label != noise_label]
    cluster_sizes = Counter(label_list[index] for index in included)
    cluster_labels = list(cluster_sizes)

    if len(cluster_labels) < 2:
        raise ValueError("Silhouette cần ít nhất hai cụm sau khi bỏ noise")
    if len(included) <= len(cluster_labels):
        raise ValueError("Silhouette cần nhiều điểm hơn số cụm")

    result: list[float | None] = [None] * len(rows)
    for index in included:
        own_label = label_list[index]
        own_size = cluster_sizes[own_label]
        if own_size == 1:
            result[index] = 0.0
            continue

        distance_sums: dict[Any, float] = defaultdict(float)
        for other_index in included:
            if other_index == index:
                continue
            other_label = label_list[other_index]
            distance_sums[other_label] += math.dist(rows[index], rows[other_index])

        within_cluster_distance = distance_sums[own_label] / (own_size - 1)
        nearest_other_cluster_distance = min(
            distance_sums[other_label] / size
            for other_label, size in cluster_sizes.items()
            if other_label != own_label
        )
        denominator = max(within_cluster_distance, nearest_other_cluster_distance)
        result[index] = (
            0.0
            if denominator == 0.0
            else (nearest_other_cluster_distance - within_cluster_distance) / denominator
        )
    return result


def silhouette_score(
    data: Iterable[Iterable[float]],
    labels: Iterable[Any],
    *,
    noise_label: Any = -1,
) -> float:
    """Return the mean silhouette after excluding noise points."""
    samples = silhouette_samples(data, labels, noise_label=noise_label)
    included_samples = [value for value in samples if value is not None]
    return sum(included_samples) / len(included_samples)
