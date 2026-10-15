"""Test của TV4 (dữ liệu, tiền xử lý, baseline, trực quan hóa).

Đặt riêng trong tests/tv4/ để không trùng tên file với test của đồng đội (test_dbscan.py, test_metrics.py...) và không cần
tests/conftest.py hay src/__init__.py: gói này tự đưa thư mục gốc repo vào sys.path, nên `pytest tests/` chạy được từ gốc repo.
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
