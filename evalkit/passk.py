"""Unbiased pass@k (Chen et al., 2021) plus bootstrap confidence intervals over problems."""
from __future__ import annotations

from collections import defaultdict

import numpy as np


def pass_at_k(n: int, c: int, k: int) -> float:
    """P(at least one of k samples drawn without replacement from n passes), c correct."""
    if k > n:
        raise ValueError(f"k={k} > n={n}")
    if n - c < k:
        return 1.0
    return float(1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


def counts_by_problem(results, is_pass=lambda r: r.passed, include=lambda r: True):
    """-> {problem_id: (n, c)}. `include` lets you drop samples (e.g. infra failures)."""
    n, c = defaultdict(int), defaultdict(int)
    for r in results:
        if not include(r):
            continue
        n[r.problem_id] += 1
        c[r.problem_id] += int(is_pass(r))
    return {p: (n[p], c[p]) for p in n}


def score(counts: dict, k: int, iters: int = 2000, alpha: float = 0.05, seed: int = 0) -> dict:
    per = np.array([pass_at_k(n, c, k) for n, c in counts.values() if n >= k])
    if per.size == 0:
        return {"k": k, "mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "problems": 0}
    rng = np.random.default_rng(seed)
    boots = rng.choice(per, size=(iters, per.size), replace=True).mean(axis=1)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return {"k": k, "mean": float(per.mean()), "lo": float(lo), "hi": float(hi),
            "problems": int(per.size)}
