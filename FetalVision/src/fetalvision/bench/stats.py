"""Benchmark statistics helpers (pure numpy; unit-tested)."""
from __future__ import annotations
import time
from typing import Callable, Iterable, Sequence
import numpy as np


def latency_summary(samples_ms: Sequence[float]) -> dict:
    """p50/p95/p99/mean/std/min/max (ms) + n. Empty -> Nones."""
    a = np.asarray(list(samples_ms), dtype=float)
    if a.size == 0:
        return {"n": 0, "p50_ms": None, "p95_ms": None, "p99_ms": None, "mean_ms": None, "std_ms": None, "min_ms": None, "max_ms": None}
    return {"n": int(a.size), "p50_ms": float(np.percentile(a, 50)), "p95_ms": float(np.percentile(a, 95)),
            "p99_ms": float(np.percentile(a, 99)), "mean_ms": float(a.mean()), "std_ms": float(a.std(ddof=0)),
            "min_ms": float(a.min()), "max_ms": float(a.max())}


def throughput_img_per_s(n_images: int, wall_seconds: float) -> float:
    return float(n_images / wall_seconds) if wall_seconds > 0 else float("nan")


def time_loop(fn: Callable[[object], object], inputs: Sequence[object], n_warmup: int, n_iters: int) -> tuple[list[float], float]:
    """Call fn on inputs cycled round-robin: n_warmup untimed calls, then n_iters timed calls.
    Returns (per-call latencies in ms, total wall seconds of the timed section)."""
    k = len(inputs)
    if k == 0:
        raise ValueError("no inputs")
    for i in range(n_warmup):
        fn(inputs[i % k])
    lat = []
    t_start = time.perf_counter()
    for i in range(n_iters):
        t0 = time.perf_counter()
        fn(inputs[i % k])
        lat.append((time.perf_counter() - t0) * 1000.0)
    return lat, time.perf_counter() - t_start


def peak_rss_mb() -> float | None:
    """Peak resident set size of this process in MB (ru_maxrss: bytes on macOS, kB on Linux)."""
    try:
        import resource, sys
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return float(r / 2**20) if sys.platform == "darwin" else float(r / 2**10)
    except Exception:
        return None


def current_rss_mb() -> float | None:
    try:
        import psutil, os
        return float(psutil.Process(os.getpid()).memory_info().rss / 2**20)
    except Exception:
        return None


def check_criteria(summary: dict, criteria: dict) -> dict:
    """criteria: {name: {"metric": key path 'a.b', "op": '<=' | '>=', "value": x}} -> {name: {passed, actual, ...}}."""
    out = {}
    for name, c in criteria.items():
        cur = summary
        for part in c["metric"].split("."):
            cur = cur.get(part) if isinstance(cur, dict) else None
        passed = None
        if isinstance(cur, (int, float)) and cur == cur:
            passed = cur <= c["value"] if c["op"] == "<=" else cur >= c["value"]
        out[name] = {**c, "actual": cur, "passed": passed}
    return out
