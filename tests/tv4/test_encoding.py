"""Mọi lệnh đọc/ghi file văn bản phải ghi rõ encoding="utf-8" (Windows mặc định cp1252 hoặc cp1258,
còn JSON và README có tiếng Việt). Quét AST của mọi file .py và mọi ô code của notebook trong repo."""
import ast
import json

import pytest

from .tv4_paths import ROOT

# Chỉ quét file của TV4. Không quét src/core, src/clustering, src/evaluation, src/experiments hay test của đồng đội,
# để code của họ không làm test này (và pytest chung của nhóm) đỏ vì lý do không liên quan.
OWN_GLOBS = ("data/*.py", "tests/tv4/*.py", "src/preprocessing/*.py", "src/visualization/*.py", "baseline/*.py", "tools/*.py")
READ_WRITE = {"read_text", "write_text"}


def _is_binary(call):
    modes = [a.value for a in call.args[1:2] if isinstance(a, ast.Constant)]
    modes += [k.value.value for k in call.keywords if k.arg == "mode" and isinstance(k.value, ast.Constant)]
    return any(isinstance(m, str) and "b" in m for m in modes)


def _violations(tree):
    bad = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        name = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else None
        has_enc = any(k.arg == "encoding" for k in node.keywords)
        if name in READ_WRITE and not has_enc:
            bad.append((node.lineno, name))
        # open(...) kiểu builtin hoặc Path.open(...): chỉ tính khi không phải chế độ nhị phân
        is_open = (isinstance(f, ast.Name) and name == "open") or \
                  (isinstance(f, ast.Attribute) and name == "open" and not isinstance(f.value, ast.Name)) or \
                  (isinstance(f, ast.Attribute) and name == "open" and f.value.id in ("io", "builtins"))
        if is_open and not has_enc and not _is_binary(node):
            bad.append((node.lineno, "open"))
    return bad


def _py_files():
    return sorted({p for g in OWN_GLOBS for p in ROOT.glob(g)})


@pytest.mark.parametrize("path", _py_files(), ids=lambda p: p.relative_to(ROOT).as_posix())
def test_python_file_has_explicit_encoding(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    assert _violations(tree) == [], f"{path.relative_to(ROOT)}: thiếu encoding='utf-8' ở {_violations(tree)}"


def test_notebook_cells_have_explicit_encoding():
    for nb_path in sorted((ROOT / "notebooks").glob("D0*.ipynb")):
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        for k, cell in enumerate(nb["cells"]):
            if cell["cell_type"] != "code":
                continue
            src = "".join(cell["source"])
            src = "\n".join(l for l in src.split("\n") if not l.lstrip().startswith(("%", "!")))
            assert _violations(ast.parse(src)) == [], f"{nb_path.name} ô {k}: thiếu encoding='utf-8'"
