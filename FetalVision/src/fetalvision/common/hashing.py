"""Exact (SHA-256) and perceptual (pHash + dHash) duplicate detection with union-find clustering."""
from __future__ import annotations
import hashlib
from pathlib import Path
from typing import Iterable
import numpy as np
from PIL import Image
import imagehash


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def perceptual_hashes(path: str | Path, hash_size: int = 8) -> tuple[imagehash.ImageHash, imagehash.ImageHash]:
    with Image.open(path) as im:
        im = im.convert("L")
        return imagehash.phash(im, hash_size=hash_size), imagehash.dhash(im, hash_size=hash_size)


class UnionFind:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def near_duplicate_clusters(phashes: list, dhashes: list, max_phash_dist: int = 4, max_dhash_dist: int = 6,
                            groups: Iterable | None = None, within_group_only: bool = False) -> np.ndarray:
    """Cluster images whose pHash Hamming distance <= max_phash_dist AND dHash distance <= max_dhash_dist.
    Requiring both hashes to agree reduces false positives on textured ultrasound images.
    Returns an integer cluster id per image. O(n^2) but fine for <= ~20k images (64-bit ints, vectorised)."""
    n = len(phashes)
    ph = np.array([int(str(h), 16) for h in phashes], dtype=np.uint64)
    dh = np.array([int(str(h), 16) for h in dhashes], dtype=np.uint64)
    groups = np.asarray(list(groups)) if groups is not None else None
    uf = UnionFind(n)
    for i in range(n):
        if i + 1 >= n:
            break
        xp = np.bitwise_xor(ph[i + 1:], ph[i])
        xd = np.bitwise_xor(dh[i + 1:], dh[i])
        dp = _popcount64(xp)
        dd = _popcount64(xd)
        cand = np.where((dp <= max_phash_dist) & (dd <= max_dhash_dist))[0]
        for j in cand:
            jj = i + 1 + int(j)
            if within_group_only and groups is not None and groups[i] != groups[jj]:
                continue
            uf.union(i, jj)
    roots = np.array([uf.find(i) for i in range(n)])
    _, ids = np.unique(roots, return_inverse=True)
    return ids


def _popcount64(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.uint64)
    cnt = np.zeros(arr.shape, dtype=np.int32)
    for k in range(8):
        byte = ((arr >> np.uint64(8 * k)) & np.uint64(0xFF)).astype(np.uint8)
        cnt += np.unpackbits(byte[:, None], axis=1).sum(axis=1).astype(np.int32)
    return cnt
