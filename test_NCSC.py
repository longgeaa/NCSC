#!/usr/bin/env python3

from __future__ import annotations
import os
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.6'
import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from torch.nn import functional as F


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', '--weights', required=True,
                   help='checkpoint.pt containing both model and coarsener states.')
    p.add_argument('--packed_u32', required=True)
    p.add_argument('--reliable_knn_npz', required=True)
    p.add_argument('--rows', type=int, help='Optional; otherwise inferred from the packed file.')
    p.add_argument('--rbgc-dir', default=str(Path(__file__).resolve().parent))
    p.add_argument('--source_indices', help='Optional .npy of original IDs for the input rows.')
    p.add_argument('--labels', help='Optional evaluation labels; never used for inference.')
    p.add_argument('--label_column', type=int)
    p.add_argument('--labels_order', choices=['input', 'original'], default='input')
    p.add_argument('--topk', type=int, help='Default: the value saved in the checkpoint.')
    p.add_argument('--shuffle_seed', type=int, help='Default: the saved training shuffle seed.')
    shuffle = p.add_mutually_exclusive_group()
    shuffle.add_argument('--no_shuffle', action='store_true', default=None)
    shuffle.add_argument('--shuffle', action='store_false', dest='no_shuffle')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--out_dir', required=True)
    p.add_argument('--save_assignments', action='store_true', help='Also save S_final.npy and SC_final.npy.')
    p.add_argument('--ext_verbose', action='store_true')
    a = p.parse_args()
    if a.rows is not None and a.rows < 2:
        p.error('--rows must be at least 2.')
    if a.topk is not None and a.topk < 1:
        p.error('--topk must be positive.')
    if a.shuffle_seed is not None and a.shuffle_seed < 0:
        p.error('--shuffle_seed must be nonnegative.')
    return a


def normalized_graph(util, path, n, topk, permutation):
    directed = util.directed_graph(path, n, topk)[permutation][:, permutation].tocsr()
    message = directed.maximum(directed.T).tocsr()
    adjacency = message.astype(np.float32) + sp.identity(n, dtype=np.float32, format='csr')
    degree = np.asarray(adjacency.sum(axis=1)).reshape(-1)
    inv = np.zeros_like(degree, dtype=np.float32)
    positive = degree > 0
    inv[positive] = 1.0 / np.sqrt(degree[positive])
    scale = sp.diags(inv, dtype=np.float32)
    return (scale @ adjacency @ scale).tocsr().astype(np.float32)


def source_ids(path, n):
    if path is None:
        return np.arange(n, dtype=np.int64)
    ids = np.load(path, allow_pickle=False).reshape(-1)
    if (len(ids) != n or not np.issubdtype(ids.dtype, np.integer)
            or np.any(ids < 0) or len(np.unique(ids)) != n):
        raise ValueError('--source_indices must contain N unique nonnegative integers.')
    return ids.astype(np.int64)


def evaluation_labels(a, ids):
    if a.labels is None:
        return None
    y = np.load(a.labels, allow_pickle=False)
    if y.ndim == 2:
        column = a.label_column
        if column is None:
            if y.shape[1] != 1:
                raise ValueError('Multi-column labels require --label_column.')
            column = 0
        if not 0 <= column < y.shape[1]:
            raise ValueError('--label_column is outside the label array.')
        y = y[:, column]
    if y.ndim != 1:
        raise ValueError('Labels must be one-dimensional after column selection.')
    if a.labels_order == 'original':
        if ids.max() >= len(y):
            raise ValueError('Original IDs exceed the label array.')
        y = y[ids]
    elif len(y) != len(ids):
        raise ValueError('--labels_order input requires exactly N label rows.')
    return y


@torch.inference_mode()
def main():
    a = parse_args()
    out = Path(a.out_dir).expanduser().resolve()
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise FileExistsError(f'Choose a new, empty --out_dir: {out}')
    src = Path(a.rbgc_dir).expanduser().resolve()
    for name in ('rbgc_packed.py', 'rbgc_utils.py'):
        if not (src / name).is_file():
            raise FileNotFoundError(f'Missing dependency: {src / name}')
    sys.path.insert(0, str(src))
    packed_ops = importlib.import_module('rbgc_packed')
    util = importlib.import_module('rbgc_utils')
    device = torch.device(a.device)
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('The original packed XNOR projection requires CUDA and its build toolchain.')
    torch.cuda.set_device(device)

    checkpoint = torch.load(Path(a.checkpoint).expanduser(), map_location='cpu', weights_only=True)
    if not isinstance(checkpoint, dict) or not {'model', 'coarsener'} <= checkpoint.keys():
        raise ValueError('Use the complete checkpoint.pt saved by run_NCSC.py (model + coarsener).')
    state, params = checkpoint['model'], checkpoint.get('params', {})
    d, hidden = state['gcn.projection.weight'].shape
    kmax = state['assignment.bias'].numel()
    phi = checkpoint['coarsener']['phi']
    expected = {'gcn.skip_weight': (hidden,), 'gcn.bias': (hidden,),
                'assignment.weight': (kmax, hidden)}
    if phi.shape != (kmax, kmax) or any(tuple(state[k].shape) != shape for k, shape in expected.items()):
        raise ValueError('Checkpoint backbone and consolidation shapes do not match.')
    topk = a.topk if a.topk is not None else int(params.get('topk', 120))
    shuffle_seed = a.shuffle_seed if a.shuffle_seed is not None else int(params.get('shuffle_seed', 123))
    no_shuffle = a.no_shuffle if a.no_shuffle is not None else bool(params.get('no_shuffle', False))

    words = (d + 31) // 32
    packed_path = Path(a.packed_u32).expanduser().resolve()
    n, remainder = divmod(packed_path.stat().st_size, words * np.dtype(np.uint32).itemsize)
    if remainder or n < 2 or (a.rows is not None and n != a.rows):
        raise ValueError('Packed file size must equal N x ceil(D/32) uint32 words.')
    ids = source_ids(a.source_indices, n)
    permutation = (np.arange(n) if no_shuffle
                   else np.random.default_rng(shuffle_seed).permutation(n))
    graph = normalized_graph(util, a.reliable_knn_npz, n, topk, permutation).tocoo()
    index = torch.from_numpy(np.vstack([graph.row, graph.col]).astype(np.int64)).to(device)
    value = torch.from_numpy(graph.data.astype(np.float32)).to(device)
    a_norm = torch.sparse_coo_tensor(index, value, graph.shape, device=device).coalesce()
    packed = np.memmap(packed_path, mode='r', dtype=np.uint32, shape=(n, words))
    x = torch.from_numpy(np.ascontiguousarray(packed[permutation]).view(np.int32)).to(device)

    # Original packed forward kernel; cache sign(W) once because weights are fixed.
    ext = packed_ops.load_xnor_ext(verbose=a.ext_verbose)
    w_pack = ext.pack_weight(state['gcn.projection.weight'].to(device).contiguous())
    z = ext.xnor_forward(x, w_pack, d)
    h = F.selu(z * state['gcn.skip_weight'].to(device)
               + torch.sparse.mm(a_norm, z) + state['gcn.bias'].to(device))
    logits = F.linear(h, state['assignment.weight'].to(device), state['assignment.bias'].to(device))
    s = torch.softmax(logits, dim=1)  # Final readout uses no dropout.
    c = torch.softmax(phi.to(device), dim=1)
    pi = s @ c
    pred_internal = pi.argmax(dim=1).cpu().numpy()
    fine_internal = s.argmax(dim=1).cpu().numpy()
    pred, fine = np.empty_like(pred_internal), np.empty_like(fine_internal)
    pred[permutation], fine[permutation] = pred_internal, fine_internal

    out.mkdir(parents=True, exist_ok=True)
    np.save(out / 'prediction.npy', pred)
    np.save(out / 'prediction_S.npy', fine)
    np.save(out / 'prediction_with_original_index.npy', np.column_stack([ids, pred]))
    np.save(out / 'C_final.npy', c.cpu().numpy())
    if a.save_assignments:
        s_input = np.empty((n, kmax), dtype=np.float32)
        pi_input = np.empty_like(s_input)
        s_input[permutation], pi_input[permutation] = s.cpu().numpy(), pi.cpu().numpy()
        np.save(out / 'S_final.npy', s_input)
        np.save(out / 'SC_final.npy', pi_input)

    result = dict(checkpoint=str(Path(a.checkpoint).expanduser().resolve()),
                  N=n, D=d, hidden_channels=hidden, kmax=kmax, topk=topk,
                  shuffle_seed=shuffle_seed, no_shuffle=no_shuffle,
                  pred_K=len(np.unique(pred)), fine_K=len(np.unique(fine)),
                  prediction_order='input')
    y = evaluation_labels(a, ids)
    if y is not None:
        metrics = dict(SC=util.evaluate(y, pred), S=util.evaluate(y, fine))
        util.save_json(out / 'metrics.json', metrics)
        result.update(metrics)
    util.save_json(out / 'summary.json', result)
    print(json.dumps(util.clean(result), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
