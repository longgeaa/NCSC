
from __future__ import annotations
import math
import random
from typing import Dict
import numpy as np
import scipy.sparse as sp
import torch
from torch import nn
from torch.nn import functional as F
from rbgc_packed import PackedXNORLinear

def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def state_dict_cpu(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def normalize_message_graph(graph: sp.csr_matrix) -> sp.csr_matrix:
    a = graph.astype(np.float32) + sp.identity(graph.shape[0], dtype=np.float32, format="csr")
    degree = np.asarray(a.sum(axis=1)).reshape(-1)
    inv = np.zeros_like(degree, dtype=np.float32)
    pos = degree > 0
    inv[pos] = 1.0 / np.sqrt(degree[pos])
    s = sp.diags(inv, dtype=np.float32)
    return (s @ a @ s).tocsr().astype(np.float32)


def scipy_to_torch_sparse(matrix: sp.csr_matrix, device: str) -> torch.Tensor:
    coo = matrix.tocoo()
    idx = torch.from_numpy(np.vstack([coo.row, coo.col]).astype(np.int64)).to(device)
    val = torch.from_numpy(coo.data.astype(np.float32)).to(device)
    return torch.sparse_coo_tensor(idx, val, matrix.shape, device=device).coalesce()


def loss_graph_tensors(matrix: sp.csr_matrix, device: str):
    coo = matrix.tocoo()
    row = torch.from_numpy(coo.row.astype(np.int64)).to(device)
    col = torch.from_numpy(coo.col.astype(np.int64)).to(device)
    ew = torch.from_numpy(coo.data.astype(np.float32)).to(device)
    degree = torch.from_numpy(np.asarray(matrix.sum(axis=0)).reshape(-1).astype(np.float32)).to(device)
    return row, col, ew, degree


class DataBundle:
    def __init__(self, n, packed_x, permutation, labels_current, labels_original,
                 message_graph, loss_graph, a_norm, loss_tensors,
                 loss_sparse=None, loss_aggregation="spmm"):
        self.n = n
        self.packed_x = packed_x
        self.permutation = permutation
        self.labels_current = labels_current
        self.labels_original = labels_original
        self.message_graph = message_graph
        self.loss_graph = loss_graph
        self.a_norm = a_norm
        self.loss_tensors = loss_tensors
        self.loss_sparse = loss_sparse
        self.loss_aggregation = loss_aggregation


def glorot_vector_(p: torch.Tensor) -> None:
    lim = math.sqrt(3.0 / max(1, p.numel()))
    nn.init.uniform_(p, -lim, lim)


class BinaryGCN(nn.Module):
    def __init__(self, in_bits: int, hidden: int, grad_bits: int, ext_verbose: bool,
                 unpack_backend: str = "fused"):
        super().__init__()
        self.projection = PackedXNORLinear(in_bits, hidden, grad_bits, ext_verbose, unpack_backend)
        self.skip_weight = nn.Parameter(torch.empty(hidden))
        self.bias = nn.Parameter(torch.empty(hidden))
        glorot_vector_(self.skip_weight)
        glorot_vector_(self.bias)

    def forward(self, x: torch.Tensor, a_norm: torch.Tensor) -> torch.Tensor:
        z = self.projection(x)
        m = torch.sparse.mm(a_norm, z)
        return F.selu(z * self.skip_weight + m + self.bias)


class RBGCBackbone(nn.Module):
    def __init__(self, in_bits: int, hidden: int, k: int, grad_bits: int, dropout: float, ext_verbose: bool,
                 unpack_backend: str = "fused"):
        super().__init__()
        self.gcn = BinaryGCN(in_bits, hidden, grad_bits, ext_verbose, unpack_backend)
        self.assignment = nn.Linear(hidden, k)
        nn.init.orthogonal_(self.assignment.weight)
        nn.init.zeros_(self.assignment.bias)
        self.dropout = nn.Dropout(dropout)

    def encode(self, x: torch.Tensor, a_norm: torch.Tensor) -> torch.Tensor:
        return self.gcn(x, a_norm)

    def forward(self, x: torch.Tensor, a_norm: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.encode(x, a_norm)
        logits = self.assignment(h)
        s = torch.softmax(self.dropout(logits), dim=1)
        return {"hidden": h, "logits": logits, "assignment": s}


def aggregate_loss_graph(s, data):
    return torch.sparse.mm(data.loss_sparse, s)


def stream_seed(seed, epoch, stream):
    """Separate per-step random streams; not a cross-hardware determinism guarantee."""
    return (seed * 1000003 + epoch * 1000033 + stream * 15485863) % (2**63 - 1)


def dropout_mask(logits, a, epoch):
    if a.dropout == 0:
        return torch.ones_like(logits)
    generator = torch.Generator(device=logits.device)
    generator.manual_seed(stream_seed(a.seed, epoch, 1))
    return (torch.rand(logits.shape, device=logits.device, generator=generator) >=
            a.dropout).to(logits.dtype) / (1 - a.dropout)


def backward_aligned(loss, a, epoch):
    device = torch.device(a.device)
    index = device.index if device.index is not None else torch.cuda.current_device()
    with torch.random.fork_rng(devices=[index]):
        torch.cuda.manual_seed(stream_seed(a.seed, epoch, 3))
        loss.backward()


def balance(t):
    return t.sum(0).norm() * math.sqrt(t.shape[1]) / t.shape[0] - 1


def quotient(t, data, gamma):
    # Live differentiable quotient; no evidence masks, detach, or positive clipping.
    at = aggregate_loss_graph(t, data)
    degree = data.loss_tensors[3]
    mass = degree.sum().clamp_min(1e-30)
    observed = t.T @ at
    volume = t.T @ degree
    r = observed / mass - gamma * torch.outer(volume, volume) / mass.square()
    return observed, (r + r.T) * 0.5, mass
