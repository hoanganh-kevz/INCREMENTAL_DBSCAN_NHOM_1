# So sánh thuật toán phân cụm (V06)

Bảng này phục vụ lựa chọn phương pháp và diễn giải kết quả. Các đặc điểm hiệu năng phụ thuộc metric, chỉ mục không gian, số chiều và cách cài đặt; độ phức tạp bên dưới là hướng dẫn phổ biến, không phải cam kết cho mọi thư viện.

| Thuật toán | Họ phương pháp | Hình dạng cụm | Xử lý noise | Tham số tiêu biểu | Cập nhật tăng dần | Quy mô / độ phức tạp thường gặp |
|---|---|---|---|---|---|---|
| DBSCAN | Mật độ | Tìm được cụm không lồi, nối bởi các vùng đủ mật độ | Có, gán điểm không thuộc cụm là noise | `eps`, `MinPts` | Thường phải phân cụm lại toàn bộ khi dữ liệu đổi | Có chỉ mục không gian thường nhanh hơn; trường hợp xấu nhất O(n²). |
| Incremental DBSCAN | Mật độ, cập nhật cục bộ | Như DBSCAN; cụm có thể tách hoặc nhập sau cập nhật | Có; noise có thể được hấp thụ hoặc tạo ra | `eps`, `MinPts`, cùng trạng thái lân cận/cụm | Có; xử lý vùng bị ảnh hưởng khi insert/delete | Chi phí phụ thuộc kích thước vùng ảnh hưởng; trường hợp xấu nhất vẫn có thể là O(n²). |
| OPTICS | Mật độ, thứ tự hóa | Hỗ trợ cấu trúc mật độ thay đổi và cụm không lồi | Có thể xác định noise khi trích cụm | `MinPts`, `max_eps`, quy tắc trích cụm | Thông thường cần tính lại thứ tự sau thay đổi dữ liệu | Cách cài đặt phổ biến có trường hợp xấu O(n²); phù hợp phân tích mật độ trên nhiều ngưỡng hơn DBSCAN đơn ngưỡng. |
| HDBSCAN | Mật độ phân cấp | Cụm không lồi, mật độ khác nhau | Có; có thể gán điểm không ổn định thành noise | `min_cluster_size`, `min_samples` | Thường cần dựng lại phân cấp khi dữ liệu thay đổi | Phụ thuộc triển khai; dùng cấu trúc MST/phân cấp, tốn thêm thời gian và bộ nhớ để tạo, lưu cây. |

## Tham số đang dùng và lý do

### DBSCAN và Incremental DBSCAN

- `eps` (Eps) là bán kính lân cận. Hai điểm được xem là láng giềng nếu khoảng cách Euclidean không lớn hơn Eps; code dùng điều kiện `distance <= eps` nên điểm đúng trên biên vẫn được tính.
- `min_pts` (MinPts) là số điểm tối thiểu trong lân cận để một điểm trở thành core. Lân cận tính cả chính điểm đang xét. Vì vậy `MinPts=4` có nghĩa là điểm đó cần tổng cộng ít nhất bốn điểm trong bán kính, bao gồm chính nó.
- `metric=Euclidean` được dùng xuyên suốt để bán kính có ý nghĩa nhất quán trong 2D và các không gian nhiều chiều. Với ATM, tọa độ có đơn vị km. Với luồng thẻ, các đặc trưng được chuẩn hóa trước để một đặc trưng có thang đo lớn không lấn át các đặc trưng khác.
- Bản DBSCAN OOP nhận `eps` và `min_pts`; bản Incremental DBSCAN kế thừa đúng hai tham số này. Incremental không có thêm ngưỡng mật độ riêng: nó lưu cache lân cận theo ID để khi insert/delete chỉ xác định và cập nhật vùng bị ảnh hưởng. Insert có thể làm điểm thành core, hấp thụ noise hoặc nối cụm; delete có thể làm điểm mất tính core và tách cụm.

Phần cài đặt bám theo nguồn được phân công trong sheet: DBSCAN nền theo [Ester et al. (1996)](https://cdn.aaai.org/KDD/1996/KDD96-037.pdf), Incremental DBSCAN theo [Ester et al. (1998)](https://www.vldb.org/conf/1998/p323.pdf). DBSCAN dùng stack như giả mã trong paper 1998. Insert xác định `NewCore` và `UpdSeedIns`, sau đó dùng stack để duyệt phần lõi mới và nhãn cụm cũ trong tập seed để xử lý Creation, Absorption hoặc Merge. Delete xác định `LostCore` và `UpdSeedDel`: tập seed rỗng là Removal, các seed trực tiếp nối nhau là Reduction, còn Potential Split được kiểm tra bằng BFS bắt đầu từ các seed. Paper mô tả thuật toán nhưng không kèm mã nguồn Python; phần triển khai giữ các bước và quy tắc của paper, còn class và cache lân cận là cấu trúc của dự án.

Các cấu hình mẫu là `Eps=0.5 km, MinPts=4` cho minh họa ATM 2D, `Eps=1.1, MinPts=4` cho ví dụ R02, và `Eps=0.4, MinPts=6` cho benchmark luồng thẻ sau chuẩn hóa. Chúng phục vụ các dữ liệu và mục tiêu minh họa khác nhau, không được xem là cùng một cấu hình hay kết quả tối ưu. Benchmark lấy `Eps`/`MinPts` từ `data/raw/card_stream/params.json` rồi giữ nguyên cho cả bốn phương pháp để không tinh chỉnh riêng cho một thuật toán.

### Kiểm tra độ nhạy tham số trên D0

Để xem `Eps=0.4, MinPts=6` có hợp lý trên dữ liệu benchmark hay không, đã chạy DBSCAN OOP trên 500 điểm D0 sau chuẩn hóa đóng băng. Quét `Eps` trong `[0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60]` và `MinPts` trong `[4, 6, 8]` riêng cho các lát đặc trưng 2D, 3D, 4D. Bảng dưới giữ `MinPts=6` và chọn ba Eps đại diện; Silhouette bỏ noise nên phải đọc cùng tỷ lệ noise và số cụm.

| Số chiều | Eps | Số cụm | Noise | Silhouette tự cài |
|---:|---:|---:|---:|---:|
| 2D | 0.25 | 3 | 2.2% | 0.7848 |
| 2D | **0.40** | **3** | **1.0%** | **0.7789** |
| 2D | 0.60 | 2 | 1.0% | 0.5753 |
| 3D | 0.25 | 3 | 2.2% | 0.7889 |
| 3D | **0.40** | **3** | **1.0%** | **0.7843** |
| 3D | 0.60 | 2 | 1.0% | 0.6264 |
| 4D | 0.25 | 3 | 5.0% | 0.7583 |
| 4D | **0.40** | **3** | **1.0%** | **0.7462** |
| 4D | 0.60 | 3 | 1.0% | 0.7462 |

`Eps=0.25` cho Silhouette nhỉnh hơn nhưng loại nhiều điểm noise hơn, nhất là 4D. `Eps=0.60` làm 2D và 3D nhập hai nhóm thành một cụm lớn hơn, Silhouette giảm rõ. Ở các giá trị `Eps=0.35` đến `0.50`, số cụm và các chỉ số tổng hợp gần như ổn định; `0.40` nằm trong vùng ổn định đó và giữ noise thấp. Khi giữ `Eps=0.40` rồi thử `MinPts=4, 6, 8`, cả ba chiều đều cho 3 cụm và các tỷ lệ noise/Silhouette gần nhau; vì vậy `MinPts=6` được giữ làm giá trị trung gian, không phải vì có một giá trị tối ưu duy nhất.

Đây là kiểm tra độ nhạy nhỏ trên D0 mô phỏng, không phải tìm kiếm toàn bộ không gian tham số. Kết quả giúp biện minh rằng bộ hiện tại là một cấu hình ổn định, không chứng minh nó tối ưu cho dữ liệu thật hoặc cho toàn bộ luồng cập nhật.

### Vì sao chọn DBSCAN, HDBSCAN và OPTICS

- **DBSCAN của scikit-learn** là mốc đối chứng trực tiếp cho bản tự cài DBSCAN. scikit-learn không cung cấp `IncrementalDBSCAN` hay `partial_fit` cho DBSCAN, nên adapter thêm/xóa điểm rồi fit lại toàn bộ sau mỗi lần cập nhật. Cặp này giúp kiểm tra nhãn và so chi phí cập nhật cục bộ với tính lại toàn bộ.
- **HDBSCAN** là phương pháp phân cụm mật độ phân cấp: nó khảo sát nhiều mức mật độ và chọn các cụm ổn định, phù hợp để xem kết quả khi mật độ các cụm khác nhau. HDBSCAN không có Eps tương đương DBSCAN nên đây không phải phép so cấu hình hoàn toàn giống nhau.
- **OPTICS** cùng họ mật độ với DBSCAN và giữ thứ tự/reachability để xem cấu trúc ở nhiều ngưỡng. Nó được chọn để đối chiếu một thuật toán mật độ khác, không phải vì nó là phiên bản incremental.
- Không đưa K-Means vào nhóm so sánh chính vì K-Means thuộc phương pháp phân hoạch, cần chọn số cụm và không mô hình hóa noise theo cách DBSCAN làm. So sánh chính tập trung vào các thuật toán mật độ cùng khả năng tìm cụm hình dạng bất kỳ và nhận biết noise.

### Tham số của ba đối chứng scikit-learn

| Thuật toán | Cấu hình | Lý do |
|---|---|---|
| DBSCAN | `eps=Eps`, `min_samples=MinPts`, `metric='euclidean'`, `n_jobs=1` | `min_samples` có cùng vai trò và cũng tính chính điểm đó, nên ánh xạ trực tiếp MinPts. `n_jobs=1` tránh khác biệt do chạy song song trong phép đo thời gian. |
| HDBSCAN | `min_cluster_size=6`, `min_samples=6`, `cluster_selection_method='eom'`, `metric='euclidean'`, `n_jobs=1` | `min_cluster_size` loại các nhóm nhỏ hơn ngưỡng; `min_samples` điều khiển mức độ nghiêm ngặt về mật độ. Cả hai được đặt bằng MinPts để giữ một ngưỡng tham chiếu dễ giải thích, không khẳng định tương đương toán học với DBSCAN. `eom` chọn cụm ổn định theo Excess of Mass, là cách chọn chuẩn của HDBSCAN. |
| OPTICS | `min_samples=6`, `eps=Eps`, `max_eps=Eps+1e-6`, `cluster_method='dbscan'`, `metric='euclidean'`, `n_jobs=1` | Giữ cùng ngưỡng mật độ tham chiếu; `cluster_method='dbscan'` trích nhãn tại Eps để dễ đối chiếu DBSCAN. scikit-learn yêu cầu ngưỡng trích `eps` nhỏ hơn `max_eps`, nên cộng một lượng nhỏ `1e-6`; đây là điều kiện kỹ thuật, không phải tham số được tối ưu. OPTICS vẫn phải tính thứ tự reachability trước khi trích cụm. |

Trong cấu hình HDBSCAN, `min_cluster_size=MinPts` và `min_samples=MinPts` là lựa chọn đơn giản để kiểm soát số tham số và giữ mốc mật độ dễ so sánh. Chúng không biến HDBSCAN thành DBSCAN. Kết quả HDBSCAN được đọc theo Silhouette, số cụm và tỷ lệ noise; ARI chỉ đo độ đồng thuận nhãn với bản Incremental DBSCAN.

## Benchmark thực nghiệm

Benchmark cập nhật chạy cùng D0, thứ tự insert/delete, tọa độ, metric Euclidean và `MinPts` tham chiếu qua bốn phương pháp: Incremental DBSCAN OOP tự cài, DBSCAN, HDBSCAN và OPTICS của scikit-learn. Incremental DBSCAN dùng insert/delete cục bộ; ba estimator scikit-learn được fit lại toàn bộ trên tập điểm hiện tại sau từng cập nhật.

| Phương pháp | Cấu hình trong benchmark | Cách cập nhật |
|---|---|---|
| Incremental DBSCAN OOP | `eps` và `min_pts` theo `params.json`; Euclidean | Insert/delete vùng ảnh hưởng |
| scikit-learn DBSCAN | `eps` theo `params.json`, `min_samples=MinPts`, Euclidean | Fit lại toàn bộ |
| scikit-learn HDBSCAN | `min_cluster_size=MinPts`, `min_samples=MinPts`, `cluster_selection_method='eom'`, Euclidean | Fit lại toàn bộ |
| scikit-learn OPTICS | `min_samples=MinPts`, `eps=Eps`, `max_eps=Eps+1e-6`, `cluster_method='dbscan'`, Euclidean | Fit lại toàn bộ |

Benchmark dùng cùng D0, thứ tự insert/delete, tọa độ, metric Euclidean và MinPts tham chiếu. Các giá trị `Eps=0.4, MinPts=6` là cấu hình cố định trong `params.json`, giữ nguyên giữa các chiều và thuật toán để so sánh có kiểm soát; nhóm không tuyên bố đây là bộ tham số tối ưu. `MinPts` được ánh xạ sang `min_samples`; HDBSCAN vẫn dùng tham số phân cấp riêng. OPTICS tính thứ tự reachability rồi trích cụm kiểu DBSCAN tại cùng `eps`. Cách này giúp nhãn DBSCAN và OPTICS dễ đối chiếu hơn nhưng không làm quy trình nội bộ của chúng giống nhau. Tham số được giải thích trong tài liệu chính thức của [DBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html), [HDBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html), [OPTICS](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.OPTICS.html) và [trích cụm OPTICS kiểu DBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.cluster_optics_dbscan.html).

Mặc định benchmark chạy 5 lượt trên prefix 20 cập nhật để có thể so trung vị và độ dao động thời gian mà vẫn giữ thí nghiệm ở quy mô vừa phải. Mỗi lượt dùng cùng D0 và thứ tự cập nhật; 2D, 3D, 4D lần lượt lấy 2, 3, 4 đặc trưng đầu sau chuẩn hóa đóng băng từ cùng luồng thẻ, nên đây là các lát đặc trưng lồng nhau chứ không phải ba bộ dữ liệu độc lập. Mọi chiều giữ `eps=0.4`, `MinPts=6`; seed của dữ liệu nằm trong `params.json`, còn `--seed 42` xáo thứ tự chạy các thuật toán để giảm thiên lệch do thuật toán nào luôn chạy trước. Một lượt warm-up fit D0 chạy trước nhưng không đưa vào kết quả. Thời gian fit D0 được ghi riêng với tổng thời gian insert/delete. CSV theo lượt, theo từng bước, bảng tổng hợp và hình được lưu tại `results/density_algorithm_benchmark_repeats.csv`, `results/density_algorithm_benchmark_runs.csv`, `results/density_algorithm_benchmark_summary.csv` và `report/figures/density_algorithm_benchmark.png`.

Chất lượng cụm được đọc bằng Silhouette tự cài, kèm số cụm và tỷ lệ noise vì noise bị bỏ khỏi phép tính Silhouette. ARI chỉ báo độ đồng thuận nhãn với Incremental DBSCAN, không dùng để kết luận HDBSCAN hoặc OPTICS tốt hơn. Thời gian đo phản ánh prefix 20 cập nhật của dữ liệu mô phỏng, không đại diện cho toàn bộ 300 bước hay dữ liệu giao dịch thật.

Chạy lại bằng:

```bash
python -m baseline.benchmark_density_algorithms --repeats 5 --warmups 1 --max-steps 20 --seed 42
```

## Cách đọc kết quả

- Chọn DBSCAN/Incremental DBSCAN khi cần cụm theo mật độ, hình dạng tùy ý và phát hiện noise; cần kiểm tra độ nhạy của `eps` và `MinPts`.
- OPTICS có thể xem cấu trúc mật độ qua nhiều ngưỡng; benchmark này cố định cách trích cụm DBSCAN để đối chiếu với cùng `eps`.
- Cân nhắc HDBSCAN khi mật độ thay đổi giữa các cụm và có thể chấp nhận bước dựng phân cấp.
- Incremental DBSCAN có lợi khi thay đổi dữ liệu cục bộ; nếu một cập nhật ảnh hưởng phần lớn dữ liệu thì chi phí có thể gần với chạy lại DBSCAN đầy đủ.

Phần đối chiếu tập trung vào thuật toán dựa trên mật độ; không đưa K-Means/HAC vào phần so sánh chính vì chúng thuộc nhóm phương pháp khác.
