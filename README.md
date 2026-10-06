# Incremental DBSCAN

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
├─ report/Report_Incremental_DBSCAN.pdf
├─ src/
│  ├─ core/            # DataPoint, DistanceMetric
│  ├─ preprocessing/   # ManualScaler
│  ├─ clustering/      # DBSCAN, IncrementalDBSCAN
│  ├─ evaluation/      # ClusterEvaluator (Silhouette, thống kê cụm/noise)
│  ├─ visualization/   # ClusterVisualizer (minh họa 2D trước/sau update)
│  └─ experiments/     # ExperimentRunner (chạy kịch bản, benchmark)
├─ baseline/           # DBSCAN của thư viện, chỉ dùng để đối chứng
├─ data/raw/ + data/processed/
├─ notebooks/Incremental_DBSCAN_Demo.ipynb
└─ tests/
```

Phần thuật toán trong `src/clustering` không gọi `sklearn` hay `scipy`. Thư viện chỉ xuất hiện ở `baseline/`, phần đánh giá đối chứng và vẽ hình. Mỗi điểm dữ liệu là vector n chiều, không hard-code x/y; hình minh họa 2D chỉ là một trường hợp riêng.

## Cài đặt

Cần Python 3.10 trở lên.

```bash
git clone https://github.com/hoanganh-kevz/INCREMENTAL_DBSCAN_NHOM_1
cd INCREMENTAL_DBSCAN_NHOM_1
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Chạy thử

Chạy nhanh một kịch bản insert/delete và in kết quả:

```bash
python run_demo.py
```

Chạy toàn bộ test (4 case insert, 3 case delete, dữ liệu n chiều, tương đương với full DBSCAN):

```bash
pytest tests/
```

Để xem bản trình bày đầy đủ, mở `notebooks/Incremental_DBSCAN_Demo.ipynb` rồi chọn **Run All**. Notebook đi theo thứ tự: tiền xử lý, mô hình, kết quả, trực quan hóa, đánh giá, so sánh và chọn lựa. Notebook chỉ import từ `src/`, không chứa logic thuật toán riêng.

Nếu chạy trên Google Colab, thêm cell đầu:

```python
!git clone https://github.com/hoanganh-kevz/INCREMENTAL_DBSCAN_NHOM_1
%cd INCREMENTAL_DBSCAN_NHOM_1
!pip install -r requirements.txt
```

## Dữ liệu

Dữ liệu gốc nằm ở `data/raw/`, dữ liệu đã xử lý ở `data/processed/`, kèm file mô tả, seed cố định và đường dẫn tương đối để chạy lại trên máy khác. Luồng cập nhật (update stream) dùng cho thí nghiệm insert/delete cũng nằm trong thư mục này.

## Đánh giá

Nhóm so sánh Incremental DBSCAN với việc chạy lại DBSCAN đầy đủ trên cùng dữ liệu và tham số, về độ tương đương kết quả và thời gian chạy. Chất lượng cụm được đo bằng Silhouette tự cài cùng thống kê số cụm và số noise. Vì Silhouette không phù hợp hoàn toàn với cụm theo mật độ, báo cáo có phần nêu rõ giới hạn này và cách xử lý noise. Phần so sánh rộng hơn gồm OPTICS, HDBSCAN, KMeans và HAC.

## Thành viên

| Họ và tên | Vai trò |
|---|---|
| Lý Huỳnh Phương Anh | _cập nhật sau khi vote_ |
| Nguyễn Hoàng Anh | _cập nhật sau khi vote_ |
| Hồ Ngọc Bảo | _cập nhật sau khi vote_ |
| Nguyễn Phạm Gia Bảo | _cập nhật sau khi vote_ |
| Từ Thiên Gia Bảo | _cập nhật sau khi vote_ |

Môn học: _[điền tên môn]_ · Giảng viên: _[điền tên giảng viên]_ · Trường: _[điền tên trường]_
