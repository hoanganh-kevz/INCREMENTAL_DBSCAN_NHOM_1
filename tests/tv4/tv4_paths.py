"""Đường dẫn dùng chung cho test của TV4."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
TRUTH = ROOT / "data" / "truth"
