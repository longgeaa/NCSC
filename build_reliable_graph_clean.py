#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image
from tqdm import tqdm


def parse_paths(path: str, col: int, root: str = "") -> List[str]:
    base = Path(root) if root else None
    out: List[str] = []
    for ln, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        fields = line.strip().split()
        if not fields:
            continue
        if col < 0 or col >= len(fields):
            raise ValueError(f"{path}:{ln}: col={col}, fields={len(fields)}")
        p = Path(fields[col])
        if not p.is_absolute() and base is not None:
            p = base / p
        out.append(str(p))
    return out


def import_prnu(repo: str):
    if repo:
        sys.path.insert(0, str(Path(repo).resolve()))
    try:
        return importlib.import_module("prnu.functions")
    except Exception as e:
        raise RuntimeError(
            "Cannot import official prnu.functions. Pass --prnu_repo /path/to/prnu-python"
        ) from e


def load_rgb(path: str, size: int) -> np.ndarray:
    with Image.open(path) as im:
        arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
    h, w = arr.shape[:2]
    if h < size or w < size:
        raise ValueError(f"image smaller than {size}: {path}, shape={arr.shape}")
    y, x = (h - size) // 2, (w - size) // 2
    return np.ascontiguousarray(arr[y:y+size, x:x+size])


def block_mean(x: np.ndarray, size: int, bs: int) -> np.ndarray:
    g = size // bs
    return x.reshape(g, bs, g, bs).mean((1, 3)).reshape(-1).astype(np.float32)


def low_texture_rank(texture: np.ndarray) -> np.ndarray:
    order = np.argsort(texture, kind="mergesort")
    rank = np.empty(len(texture), np.float32)
    rank[order] = np.arange(len(texture), dtype=np.float32)
    if len(texture) > 1:
        rank /= float(len(texture) - 1)
    return (1.0 - rank).astype(np.float32)


def build_reliability(image_paths: List[str], pf, size: int, bs: int) -> np.ndarray:
    all_q = []
    for path in tqdm(image_paths, desc="reliability"):
        rgb = load_rgb(path, size)
        qpix = (pf.inten_scale(rgb) * pf.saturation(rgb)).mean(axis=2).astype(np.float32)
        q_int = np.clip(block_mean(qpix, size, bs), 0.0, 1.0)

        gray = (0.29893602 * rgb[:, :, 0] + 0.58704307 * rgb[:, :, 1] + 0.11402090 * rgb[:, :, 2]).astype(np.float32) / 255.0
        gx = np.zeros_like(gray)
        gy = np.zeros_like(gray)
        gx[:, 1:] = gray[:, 1:] - gray[:, :-1]
        gy[1:, :] = gray[1:, :] - gray[:-1, :]
        texture = block_mean(np.sqrt(gx * gx + gy * gy), size, bs)
        q_tex = low_texture_rank(texture)
        all_q.append((q_int * q_tex).astype(np.float32))
    return np.stack(all_q, axis=0)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--residual_matrix", required=True, help="N x D float .npy in original order")
    p.add_argument("--image_list", required=True)
    p.add_argument("--image_path_col", type=int, default=0)
    p.add_argument("--image_root", default="")
    p.add_argument("--prnu_repo", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--reliability_cache", default="", help="Optional .npz cache to reuse/save q")
    p.add_argument("--topk", type=int, default=256)
    p.add_argument("--image_size", type=int, default=512)
    p.add_argument("--block_size", type=int, default=32)
    p.add_argument("--transform_batch", type=int, default=8)
    p.add_argument("--search_batch", type=int, default=128)
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--faiss_use_float16", type=int, choices=[0, 1], default=1)
    p.add_argument("--temp_dtype", choices=["float16", "float32"], default="float16")
    p.add_argument("--keep_temp", type=int, choices=[0, 1], default=0)
    a = p.parse_args()

    if a.image_size % a.block_size or a.block_size % 32:
        raise ValueError("block_size must divide image_size and be a multiple of 32")

    x = np.load(a.residual_matrix, mmap_mode="r", allow_pickle=False)
    if x.ndim != 2:
        raise ValueError("residual_matrix must be [N,D]")
    n, d = x.shape
    if d != a.image_size * a.image_size:
        raise ValueError(f"D={d} != image_size^2")

    image_paths = parse_paths(a.image_list, a.image_path_col, a.image_root)
    if len(image_paths) != n:
        raise ValueError(f"image count={len(image_paths)} != residual N={n}")

    q = None
    cache = Path(a.reliability_cache) if a.reliability_cache else None
    if cache is not None and cache.exists():
        with np.load(cache, allow_pickle=False) as z:
            q = np.asarray(z["R_bright_lowtex"], dtype=np.float32)
    if q is None:
        pf = import_prnu(a.prnu_repo)
        q = build_reliability(image_paths, pf, a.image_size, a.block_size)
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache, R_bright_lowtex=q.astype(np.float16), image_size=np.asarray([a.image_size]), block_size=np.asarray([a.block_size]))
    if q.shape[0] != n:
        raise ValueError("reliability N mismatch")

    g = a.image_size // a.block_size
    b = g * g
    if q.shape[1] != b:
        raise ValueError(f"reliability blocks={q.shape[1]} expected {b}")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".reliable_vectors.dat")
    dtype = np.float16 if a.temp_dtype == "float16" else np.float32
    mm = np.memmap(tmp, dtype=dtype, mode="w+", shape=(n, d))

    for st in tqdm(range(0, n, a.transform_batch), desc="weighted residuals"):
        en = min(n, st + a.transform_batch)
        w = np.asarray(x[st:en], np.float32).reshape(en-st, g, a.block_size, g, a.block_size)
        qq = q[st:en].reshape(en-st, g, 1, g, 1)
        v = (w * np.sqrt(np.maximum(qq, 0.0))).reshape(en-st, d)
        v /= np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
        mm[st:en] = v.astype(dtype)
    mm.flush()

    try:
        import faiss
    except Exception as e:
        raise RuntimeError("faiss-gpu is required") from e
    if not hasattr(faiss, "StandardGpuResources"):
        raise RuntimeError("GPU FAISS is required for full-dimensional 512x512 KNN")

    res = faiss.StandardGpuResources()
    cfg = faiss.GpuIndexFlatConfig()
    cfg.device = int(a.gpu_id)
    cfg.useFloat16 = bool(a.faiss_use_float16)
    index = faiss.GpuIndexFlatIP(res, d, cfg)

    for st in tqdm(range(0, n, a.search_batch), desc="FAISS add"):
        en = min(n, st + a.search_batch)
        index.add(np.asarray(mm[st:en], np.float32))

    kout = min(a.topk, n - 1)
    nbr = np.empty((n, kout), np.int64)
    sim = np.empty((n, kout), np.float32)
    kquery = min(kout + 1, n)
    for st in tqdm(range(0, n, a.search_batch), desc="FAISS search"):
        en = min(n, st + a.search_batch)
        D, I = index.search(np.asarray(mm[st:en], np.float32), kquery)
        for r, gi in enumerate(range(st, en)):
            keep = I[r] != gi
            ii, dd = I[r][keep][:kout], D[r][keep][:kout]
            if len(ii) < kout:
                raise RuntimeError("not enough neighbors after removing self")
            nbr[gi], sim[gi] = ii, dd

    np.savez_compressed(out, nbrs=nbr, sims=sim, indices=nbr, similarities=sim)
    meta = {
        "N": n, "D": d, "topk": kout,
        "formula": "W'(x)=sqrt(q_intensity*q_lowtexture)*W(x)",
        "labels_used": False,
        "faiss_use_float16": bool(a.faiss_use_float16),
        "temp_dtype": a.temp_dtype,
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))

    del mm
    if not a.keep_temp:
        try:
            tmp.unlink()
        except OSError:
            pass


if __name__ == "__main__":
    main()
