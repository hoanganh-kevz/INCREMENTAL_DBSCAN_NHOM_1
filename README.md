# Incremental DBSCAN

**Công cụ hỗ trợ phân cụm dữ liệu theo mật độ, cập nhật cụm khi thêm hoặc xóa điểm, và đối chiếu kết quả với DBSCAN chuẩn.**

Tự cài đặt **DBSCAN** và **Incremental DBSCAN** bằng Python theo hướng đối tượng, không dùng thư viện cho phần thuật toán. Khi có điểm mới được thêm vào hoặc xóa đi, mô hình chỉ cập nhật vùng bị ảnh hưởng thay vì chạy lại toàn bộ.

## Ý tưởng chính

DBSCAN gom cụm theo mật độ với hai tham số `eps` và `MinPts`. Mỗi điểm là core, border hoặc noise. Khi dữ liệu thay đổi, chạy lại từ đầu rất tốn kém. Incremental DBSCAN chỉ xét các điểm nằm trong vùng lân cận của điểm vừa thêm hoặc xóa (tập *affected* và *UpdSeed*), rồi quyết định cụm đó được tạo mới, hấp thụ, gộp hoặc tách.

Dự án cài đặt 4 trường hợp **insert** (noise, creation, absorption, merge) và 3 trường hợp **delete** (removal, reduction, potential split). Sau mỗi lần cập nhật, kết quả được đối chiếu với DBSCAN chạy lại toàn bộ để kiểm tra tính đúng, ưu tiên ARI = 1 trên các kịch bản chuẩn.

## Cấu trúc thư mục

```
INCREMENTAL_DBSCAN_NHOM_1/
├─ README.md
├─ requirements.txt
├─ run_demo.py
├─ docs/results_discussion.md  # Bản thảo V07 để tích hợp vào báo cáo PDF
├─ report/figures/             # Hình kết quả dùng trong báo cáo
├─ src/
│  ├─ core/            # DataPoint, DistanceMetric
│  ├─ preprocessing/   # ManualScaler
│  ├─ clustering/      # DBSCAN, IncrementalDBSCAN
│  ├─ evaluation/      # Hàm đánh giá Silhouette tự cài và đối chiếu thư viện
│  ├─ visualization/   # ClusterVisualizer (minh họa 2D trước/sau update)
│  └─ experiments/     # IncrementalEngine, phân tích update và adapter thực nghiệm
├─ baseline/           # DBSCAN của thư viện, chỉ dùng để đối chứng
├─ data/raw/ + data/processed/
├─ notebooks/Incremental_DBSCAN_Demo.ipynb
└─ tests/
```

Phần thuật toán trong `src/clustering` không gọi `sklearn` hay `scipy`. Thư viện chỉ xuất hiện ở `baseline/`, phần đánh giá đối chứng và vẽ hình. Mỗi điểm dữ liệu là vector n chiều, không hard-code x/y; hình minh họa 2D chỉ là một trường hợp riêng.

## Cài đặt và chạy

Yêu cầu Python 3.10 trở lên. Các phụ thuộc chạy và vẽ hình được khai báo trong `requirements.txt`; DBSCAN tự cài trong `src/clustering` không dùng scikit-learn. Trên Windows PowerShell, chạy 5 lệnh sau từ thư mục gốc repo:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python run_demo.py --seed 42
jupyter notebook notebooks/Incremental_DBSCAN_Demo.ipynb
```

Trên macOS/Linux, thay lệnh kích hoạt môi trường ở dòng 2 bằng `source .venv/bin/activate`. Demo 2D bắt đầu với hai cụm, insert điểm cầu nối để gộp cụm, rồi delete điểm đó để tách cụm lại. Seed mặc định là 42; có thể truyền seed khác bằng `--seed`.

Chạy toàn bộ test (4 case insert, 3 case delete, dữ liệu n chiều, tương đương với full DBSCAN):

```bash
pytest tests/
```

Notebook đi theo thứ tự: tiền xử lý, mô hình, kết quả, trực quan hóa, đánh giá, so sánh và chọn lựa. Notebook chỉ import từ `src/`, không chứa logic thuật toán riêng.

Nếu chạy trên Google Colab, thêm cell đầu:

```python
!git clone https://github.com/hoanganh-kevz/INCREMENTAL_DBSCAN_NHOM_1
%cd INCREMENTAL_DBSCAN_NHOM_1
!pip install -r requirements.txt
```

## Dữ liệu

Dữ liệu gốc nằm ở `data/raw/`, dữ liệu đã xử lý ở `data/processed/`, kèm file mô tả và đường dẫn tương đối. Seed của từng bộ sinh dữ liệu được ghi tại `data/README_data.md`; demo cho phép đặt seed bằng `--seed`, mặc định là 42. Luồng cập nhật (update stream) dùng cho thí nghiệm insert/delete cũng nằm trong thư mục này.

## Đánh giá

Nhóm so sánh Incremental DBSCAN với việc chạy lại DBSCAN đầy đủ trên cùng dữ liệu và tham số, về độ tương đương kết quả và thời gian chạy. Chất lượng cụm được đo bằng Silhouette tự cài cùng thống kê số cụm và số noise. Vì Silhouette không phù hợp hoàn toàn với cụm theo mật độ, báo cáo có phần nêu rõ giới hạn này và cách xử lý noise. Phần so sánh rộng hơn gồm OPTICS, HDBSCAN, KMeans và HAC.

Chạy các phần đánh giá V04 và V05:

```bash
python -m src.evaluation.equivalence
python -m src.evaluation.parameter_selection
```

V04 ghi kết quả đối chiếu từng bước vào `results/equivalence_results.csv`. V05 ghi bảng chọn tham số và profile k-distance vào `results/parameter_selection.csv`, `results/k_distance_moons.csv` và `results/k_distance_moons.png`.

Chạy benchmark V06 để so sánh Incremental DBSCAN OOP với DBSCAN, HDBSCAN và OPTICS của scikit-learn trên cùng các stream insert/delete:

```bash
python -m baseline.benchmark_density_algorithms --repeats 5 --warmups 1 --max-steps 20 --seed 42
```

Mặc định chạy 5 lượt trên tối đa 20 cập nhật của cùng stream thẻ, lần lượt dùng 2, 3 và 4 đặc trưng đầu sau chuẩn hóa đóng băng. Cả ba trường hợp dùng chung D0, thứ tự insert/delete, Eps = 0,4 và MinPts = 6; chỉ số chiều thay đổi. Tăng `--max-steps` để phát lại nhiều bước hơn. Ba estimator scikit-learn được fit lại toàn bộ sau mỗi insert/delete; Incremental DBSCAN OOP cập nhật cục bộ. Kết quả từng bước, từng lượt, bảng tổng hợp và biểu đồ lần lượt nằm ở `results/density_algorithm_benchmark_runs.csv`, `results/density_algorithm_benchmark_repeats.csv`, `results/density_algorithm_benchmark_summary.csv` và `report/figures/density_algorithm_benchmark.png`. Tham số từng phương pháp và giới hạn so sánh được ghi trong `docs/algorithm_comparison.md` và file README kết quả.

Phần phân tích kết quả và giới hạn cho báo cáo (V07) được soạn tại `docs/results_discussion.md`; README chỉ giữ hướng dẫn cài đặt và chạy.

## Thành viên

| TV | Họ và tên | Vai trò |
|---|---|---|
| TV1 | Nguyễn Hoàng Anh | _Leader / DBSCAN / tích hợp / Evaluation + comparison + QA_ |
| TV2 | Lý Huỳnh Phương Anh | _Incremental theory + INSERT_ |
| TV3 | Từ Thiên Gia Bảo | _OOP + DBSCAN + DELETE_ |
| TV4 | Nguyễn Phạm Gia Bảo | _Data + experiment + visualization_ |

Môn học: _Máy học_ · Giảng viên: _Nguyễn An Tế_ · Trường: _Đại học Kinh tế TPHCM_
