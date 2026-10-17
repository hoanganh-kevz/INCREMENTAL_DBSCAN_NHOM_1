# Thiết kế lớp theo OOP (C02)

## Sơ đồ lớp

```mermaid
classDiagram
    direction TB

    class DataPoint {
        +id: int
        +coords: ndarray
        +cluster_id: Optional[int]
        +point_type: PointType
        +neighbor_ids: set~int~
        +neighbor_count: int
    }

    class PointType {
        <<enumeration>>
        UNCLASSIFIED
        NOISE
        BORDER
        CORE
    }

    class DistanceMetric {
        +euclidean(first_coords: ndarray, second_coords: ndarray): float
    }

    class ManualScaler {
        +fit(data: ndarray): ManualScaler
        +transform(data: ndarray): ndarray
        +fit_transform(data: ndarray): ndarray
        +inverse_transform(data: ndarray): ndarray
    }

    class DBSCAN {
        -eps: float
        -min_pts: int
        -points: dict~int,DataPoint~
        +fit(data: ndarray): DBSCAN
        +get_labels(): list~int~
        #_region_query(point_id: int): set~int~
        #_expand_cluster(seed_id: int, neighbor_ids: set~int~, cluster_id: int): None
    }

    class IncrementalDBSCAN {
        +insert(point: DataPoint): int
        +delete(point_id: int): None
    }

    class ClusterEvaluator {
        +evaluate(data: ndarray, labels: list~int~): dict
    }

    class ClusterVisualizer {
        +plot_2d(data: ndarray, labels: list~int~)
        +plot_before_after(before, after)
    }

    class ExperimentRunner {
        +run(initial_data: ndarray, updates: list): list
    }

    IncrementalDBSCAN --|> DBSCAN
    DBSCAN "1" o-- "0..*" DataPoint : quản lý theo ID ổn định
    DataPoint --> PointType : point_type
    DBSCAN ..> DistanceMetric : tính khoảng cách
    ClusterEvaluator ..> DistanceMetric : tính khoảng cách đánh giá
    ExperimentRunner ..> ManualScaler : tiền xử lý dữ liệu
    ExperimentRunner ..> IncrementalDBSCAN : chạy cập nhật
    ExperimentRunner ..> ClusterEvaluator : đánh giá kết quả
    ExperimentRunner ..> ClusterVisualizer : trực quan hóa kết quả
    ClusterVisualizer ..> DataPoint : hiển thị điểm và nhãn
```

## Trách nhiệm và giao diện

| Lớp | Trách nhiệm | Giao diện chính |
|---|---|---|
| `DataPoint` | Lưu vector n-D, ID ổn định, nhãn cụm, loại điểm và cache ID láng giềng. | `id`, `coords`, `cluster_id`, `point_type`, `neighbor_ids`, `neighbor_count` |
| `PointType` | Liệt kê trạng thái chưa phân loại, nhiễu, biên và lõi của điểm. | `UNCLASSIFIED`, `NOISE`, `BORDER`, `CORE` |
| `DistanceMetric` | Tính khoảng cách giữa hai vector, độc lập với `DataPoint` và thuật toán gom cụm. | `euclidean(first_coords, second_coords)` |
| `ManualScaler` | Ước lượng tham số co giãn và biến đổi đặc trưng trước khi gom cụm. | `fit`, `transform`, `fit_transform`, `inverse_transform` |
| `DBSCAN` | Giữ tham số và thực hiện DBSCAN theo lô: tìm vùng lân cận, mở rộng cụm, gán nhãn. | `fit`, `get_labels` |
| `IncrementalDBSCAN` | Mở rộng DBSCAN với thao tác cập nhật điểm. | Kế thừa `fit`/`get_labels`; thêm `insert(point)`/`delete(point_id)` |
| `ClusterEvaluator` | Tính các chỉ số đánh giá từ dữ liệu và nhãn cụm. | `evaluate(data, labels)` |
| `ClusterVisualizer` | Vẽ dữ liệu 2D, nhãn cụm và trạng thái điểm. | `plot_2d`, `plot_before_after` |
| `ExperimentRunner` | Điều phối kịch bản chạy và thu thập kết quả. | `run(initial_data, updates)` |

`DBSCAN` và `IncrementalDBSCAN` dùng chung giao diện gồm `fit()` và
`get_labels()`; lớp incremental chỉ bổ sung các phương thức cập nhật. Mỗi lớp
có một trách nhiệm chính và chỉ phụ thuộc vào những giao diện cần thiết. Đây
là thiết kế cho C02; các nhiệm vụ sau chịu trách nhiệm cài đặt từng lớp và
thuật toán.

`DBSCAN` lưu điểm theo ID ổn định. Mỗi `DataPoint` giữ tập ID láng giềng;
`neighbor_count` được suy ra từ tập này để không bị lệch khi cache thay đổi.
Các phương thức cập nhật sau cần duy trì cache bằng ID, không bằng vị trí trong
danh sách.

## Phạm vi C02

Tài liệu này chỉ chốt sơ đồ lớp, trách nhiệm và giao diện dự kiến. Việc cài đặt
các lớp và thuật toán được thực hiện ở những task code riêng sau C02.
