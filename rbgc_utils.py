
from __future__ import annotations
import csv
import hashlib
import json
import os
import math
from pathlib import Path
import numpy as np
import scipy.sparse as sp

def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(clean(data), ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(temp, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def file_identity(path):
    path = Path(path).expanduser().resolve()
    st = path.stat()
    with path.open('rb') as f:
        head = f.read(65536)
        f.seek(max(0, st.st_size - 65536))
        tail = f.read(65536)
    return {'path': str(path), 'size': st.st_size, 'mtime_ns': st.st_mtime_ns,
            'sample_sha256': hashlib.sha256(head + tail).hexdigest()}


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', newline='', encoding='utf-8') as f:
        if keys:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(clean(rows))


def evaluate(y, pred):
    import numpy as np
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score
    from sklearn.metrics.cluster import contingency_matrix
    y = np.asarray(y).reshape(-1)
    pred = np.asarray(pred).reshape(-1)
    if len(y) != len(pred):
        raise ValueError('Prediction and label lengths differ')
    if not len(y):
        raise ValueError('Empty evaluation set')
    # Ignore explicitly unlabeled rows in evaluation only; retain them in training.
    valid = np.ones(len(y), dtype=bool)
    if np.issubdtype(y.dtype, np.number):
        valid &= np.isfinite(y) & (y >= 0)
    if not valid.any():
        return {'evaluated_rows': 0}
    yy, pp = y[valid], pred[valid]
    tab = contingency_matrix(yy, pp, sparse=True).astype(np.float64)
    row = np.asarray(tab.sum(1)).ravel()
    col = np.asarray(tab.sum(0)).ravel()
    tp = float((tab.data * (tab.data - 1) / 2).sum())
    predicted_pairs = float((col * (col - 1) / 2).sum())
    true_pairs = float((row * (row - 1) / 2).sum())
    precision = tp / predicted_pairs if predicted_pairs else 0.0
    recall = tp / true_pairs if true_pairs else 0.0
    coo = tab.tocoo()
    bp = float((coo.data**2 / col[coo.col]).sum() / len(yy))
    br = float((coo.data**2 / row[coo.row]).sum() / len(yy))
    f1 = lambda p, r: 2*p*r/(p+r) if p+r else 0.0
    return {'ARI': adjusted_rand_score(yy, pp), 'NMI': normalized_mutual_info_score(yy, pp),
            'AMI': adjusted_mutual_info_score(yy, pp), 'pair_precision': precision,
            'pair_recall': recall, 'pair_F1': f1(precision, recall),
            'bcubed_precision': bp, 'bcubed_recall': br, 'bcubed_F1': f1(bp, br),
            'evaluated_rows': len(yy), 'true_K_evaluation_only': len(np.unique(yy))}

def load_knn(path):
    # Preserve the established mainline's negative-distance convention and
    # support its legacy FAISS N x 2 x k container as well as modern keys.
    with np.load(path, allow_pickle=True) as z:
        if 'data' in z:
            value = z['data']
            value = np.asarray(value.tolist()) if value.dtype == object else value
            if value.ndim != 3 or value.shape[1] != 2:
                raise ValueError('Legacy kNN data must have shape (N,2,k)')
            return np.asarray(value[:, 0, :], np.int64), np.asarray(value[:, 1, :], np.float32)
        ik = next((k for k in ('nbrs', 'indices', 'knn_indices', 'knn_graph', 'knns') if k in z), None)
        vk = next((k for k in ('sims', 'similarities', 'knn_sims', 'distances') if k in z), None)
        if ik is None or vk is None:
            raise ValueError(f'{path}: unknown kNN keys {z.files}')
        indices = np.asarray(z[ik], np.int64)
        values = np.asarray(z[vk], np.float32)
        if vk == 'distances':
            values = -values
        if indices.shape != values.shape or indices.ndim != 2:
            raise ValueError('Invalid kNN array shapes')
        return indices, values


def input_bundle(a):
    if a.manifest:
        manifest = Path(a.manifest).expanduser().resolve()
        b = read_json(manifest)
        side = math.isqrt(int(b['D']))
        if side * side != int(b['D']):
            raise ValueError('Manifest D must correspond to a square residual.')
        a.image_size = side
        if not b.get('indices'):
            raise ValueError('Manifest must record sample-to-original row indices.')
        for key in ('labels', 'indices'):
            if b.get(key) and not Path(b[key]).is_file():
                raise FileNotFoundError(f'Missing cached {key}: {b[key]}')
        # Cache created by experiments 01/02 already has aligned, selected rows.
        b['manifest_path'] = str(manifest)
    else:
        b = dict(N=a.rows, D=a.image_size ** 2, packed=a.packed_u32,
                 reliable_knn=a.reliable_knn_npz, labels=a.labels,
                 indices=a.source_indices, dataset='custom', platform=None)
    n, d = int(b['N']), int(b['D'])
    if n < 2 or not 2 <= a.kmax <= n:
        raise ValueError('Require N >= 2 and 2 <= kmax <= N.')
    # Current packed CUDA kernel has the same aligned-image scope as v4.
    if a.image_size < 32 or a.image_size % 32 or d != a.image_size ** 2:
        raise ValueError('image_size must be a positive multiple of 32.')
    paths = {}
    for key in ('packed', 'reliable_knn', 'indices', 'labels'):
        if not b.get(key):
            continue
        path = Path(b[key]).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f'Missing existing {key}: {path}')
        b[key] = str(path)
        paths[key] = file_identity(path)
    if not b.get('packed') or not b.get('reliable_knn'):
        raise ValueError('Existing packed binary data and reliable kNN are required.')
    if Path(b['packed']).stat().st_size != n * ((d + 31) // 32) * 4:
        raise ValueError('Packed file size differs from N x ceil(D/32) uint32.')
    if b.get('indices'):
        ids = np.load(b['indices'], allow_pickle=False).reshape(-1)
        if len(ids) != n or not np.issubdtype(ids.dtype, np.integer):
            raise ValueError('Source indices must contain N integer entries.')
        if np.any(ids < 0) or len(np.unique(ids)) != n:
            raise ValueError('Source indices must be nonnegative and unique.')
        ids = ids.astype(np.int64)
    else:
        ids = np.arange(n, dtype=np.int64)
    return b, ids, paths


def directed_graph(path, n, topk):
    # Same positive-edge filtering and CSR duplicate reduction as the v4 worker.
    neighbors, similarities = load_knn(path)
    if neighbors.shape != similarities.shape or len(neighbors) != n:
        raise ValueError('kNN dimensions differ from the packed data.')
    if topk > neighbors.shape[1]:
        raise ValueError(f'Requested topk={topk}; stored={neighbors.shape[1]}.')
    neighbors, similarities = neighbors[:, :topk], similarities[:, :topk]
    rows = np.repeat(np.arange(n), topk)
    cols, values = neighbors.reshape(-1), similarities.reshape(-1)
    keep = ((cols >= 0) & (cols < n) & (cols != rows)
            & np.isfinite(values) & (values > 0))
    graph = sp.coo_matrix((values[keep], (rows[keep], cols[keep])),
                          shape=(n, n), dtype=np.float32).tocsr()
    graph.sum_duplicates()
    graph.eliminate_zeros()
    if graph.nnz == 0:
        raise ValueError('Reliable kNN has no positive finite non-self edges.')
    return graph
