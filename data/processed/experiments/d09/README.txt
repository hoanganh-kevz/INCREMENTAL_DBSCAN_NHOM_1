Benchmark D09 (python -m baseline.benchmark_incremental). Cột thời gian phụ thuộc máy.
Máy: macOS-27.2-arm64-arm-64bit-Mach-O, arm, Python 3.13.2, scikit-learn 1.6.1, numpy 2.5.3.
Theo n: 20 update mỗi (family, n); incremental và sklearn_refit đo mỗi update 5 lần trên bản sao sâu (trung vị), own_refit đo một lần và chỉ vài update ở n lớn (mỗi update là một lần fit đầy đủ). fit của incremental đo một lần.
Bộ nhớ: tracemalloc; retained_kb là cấp phát khi sao sâu engine, insert_peak_kb và delete_peak_kb là đỉnh của một update.
Theo m: n = 2000, mỗi update đo một lần; own_refit_each_ms_est là ước tính m x một lần fit.
