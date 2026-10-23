"""Đo một lần cập nhật từ hai snapshot (trước, sau): UpdSeed, số điểm đổi trạng thái, phạm vi xử lý lại của phép xóa. numpy thuần, không sklearn.

Snapshot là dict ids, coords, labels, point_types (mã 3 lõi, 2 biên, 1 nhiễu) như IncrementalEngine.snapshot() hay RefitDBSCAN.snapshot().
Định nghĩa UpdSeed theo Ester et al. (1998), giống data/README_data.md mục 3:
  insert: các điểm lõi SAU update nằm trong Eps của một điểm vừa thành lõi (kể cả điểm chèn nếu nó là lõi);
  delete: các điểm lõi SAU update nằm trong Eps của một điểm vừa mất tính lõi (kể cả điểm xóa nếu nó từng là lõi).
"""
import numpy as np

from src.visualization.plotter import align_labels

CORE = 3


def _ids(snap):
    return np.asarray(snap["ids"]).tolist()


def core_ids(snap):
    return {i for i, t in zip(_ids(snap), np.asarray(snap["point_types"]).tolist()) if t == CORE}


def _coords_of(snap, wanted):
    pos = {i: k for k, i in enumerate(_ids(snap))}
    return np.asarray(snap["coords"], dtype=float)[[pos[i] for i in wanted]]


def within_eps(snap, centers, eps, candidates=None):
    """Tập id của `snap` (hoặc của candidates) cách ít nhất một điểm trong `centers` (mảng tọa độ) không quá eps, biên bao hàm."""
    ids = _ids(snap)
    coords = np.asarray(snap["coords"], dtype=float)
    centers = np.asarray(centers, dtype=float).reshape(-1, coords.shape[1] if coords.size else 1)
    if not len(ids) or not len(centers):
        return set()
    keep = np.ones(len(ids), dtype=bool) if candidates is None else np.array([i in candidates for i in ids])
    near = np.zeros(len(ids), dtype=bool)
    for c in centers:
        near |= np.sqrt(((coords - c) ** 2).sum(axis=1)) <= eps
    return {i for i, k in zip(ids, near & keep) if k}


def update_seeds(before, after, op, eps):
    """UpdSeed của một update (tập id), cùng tập điểm đổi tính lõi (vừa thành lõi khi insert, vừa mất tính lõi khi delete)."""
    cb, ca = core_ids(before), core_ids(after)
    if op == "insert":
        changed = sorted(ca - cb)
        centers = _coords_of(after, changed)
    else:
        changed = sorted(cb - ca)
        centers = _coords_of(before, changed)
    return within_eps(after, centers, eps, candidates=ca), set(changed)


def changed_points(before, after):
    """Id có mặt ở cả hai snapshot mà loại điểm hoặc cụm thay đổi. Cụm của `after` được khớp về `before` (align_labels), nên đánh số lại không tính,
    còn hai cụm nhập làm một thì các điểm của cụm bị nhập tính là đổi."""
    lb = dict(zip(_ids(before), np.asarray(before["labels"]).tolist()))
    tb = dict(zip(_ids(before), np.asarray(before["point_types"]).tolist()))
    la = align_labels(before, after).tolist()
    return {i for i, l, t in zip(_ids(after), la, np.asarray(after["point_types"]).tolist())
            if i in lb and (lb[i] != l or tb[i] != t)}


def delete_scope(before, point_id, eps):
    """Số điểm IncrementalDBSCAN.delete_point của nhóm gán lại nhãn: mọi điểm thuộc các cụm có điểm nằm trong Eps của điểm bị xóa
    (kể cả cụm của chính nó). Cách cài này đi lại cả cụm bị ảnh hưởng chứ không chỉ quanh UpdSeed."""
    labels = dict(zip(_ids(before), np.asarray(before["labels"]).tolist()))
    near = within_eps(before, _coords_of(before, [point_id]), eps)
    touched = {labels[i] for i in near if labels[i] >= 0}
    return sum(1 for l in labels.values() if l in touched)


def describe_update(before, after, op, point_id, eps):
    """Một dòng mô tả update: n_before, n_after, n_updseed, n_core_changed, n_changed (không tính điểm vừa chèn hay xóa), n_reprocessed (chỉ delete)."""
    seeds, changed_core = update_seeds(before, after, op, eps)
    row = {"n_before": len(_ids(before)), "n_after": len(_ids(after)), "n_updseed": len(seeds), "n_core_changed": len(changed_core),
           "n_changed": len(changed_points(before, after) - {point_id})}
    row["n_reprocessed"] = delete_scope(before, point_id, eps) if op == "delete" else np.nan
    return row
