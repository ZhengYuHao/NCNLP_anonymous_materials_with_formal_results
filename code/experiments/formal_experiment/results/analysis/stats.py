"""Statistical primitives used by the RQ1-RQ3 analysis scripts.

Implements the frozen analysis protocol: Wilson 95% intervals, paired
percentile bootstrap intervals (10,000 fixed-seed resamples, workflows as
clusters), two-sided exact McNemar tests, and Holm correction.
"""

from __future__ import annotations

import math
import random

Z95 = 1.959963984540054
BOOTSTRAP_SEED = 20260916
BOOTSTRAP_RESAMPLES = 10_000


def wilson_pct(k: int, n: int) -> tuple[float, float]:
    """Wilson 95% interval for a binomial proportion, in percent (2 decimals)."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + Z95 * Z95 / n
    center = (p + Z95 * Z95 / (2 * n)) / denom
    half = Z95 * math.sqrt(p * (1 - p) / n + Z95 * Z95 / (4 * n * n)) / denom
    return (round(max(0.0, center - half) * 100, 2), round(min(1.0, center + half) * 100, 2))


def exact_mcnemar(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value for discordant counts (b, c)."""
    d = b + c
    if d == 0:
        return 1.0
    m = min(b, c)
    total = 0
    for k in range(m + 1):
        total += math.comb(d, k)
    p = min(1.0, 2.0 * total / (2 ** d))
    return p


def holm_adjust(raw: dict[str, float]) -> dict[str, float]:
    """Holm step-down adjustment over a family of raw p-values."""
    items = sorted(raw.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted, running = {}, 0.0
    for i, (name, p) in enumerate(items):
        running = max(running, min(1.0, p * (m - i)))
        adjusted[name] = running
    return adjusted


def paired_bootstrap_ci(delta_by_task: dict[str, float], resamples: int = BOOTSTRAP_RESAMPLES,
                        seed: int = BOOTSTRAP_SEED) -> tuple[float, float]:
    """Percentile bootstrap 95% CI for mean(delta), workflows resampled jointly."""
    rng = random.Random(seed)
    values = list(delta_by_task.values())
    n = len(values)
    if n == 0:
        return (0.0, 0.0)
    stats = []
    for _ in range(resamples):
        total = 0.0
        for _ in range(n):
            total += values[rng.randrange(n)]
        stats.append(total / n)
    stats.sort()
    lo = stats[int(0.025 * resamples)]
    hi = stats[min(resamples - 1, int(0.975 * resamples))]
    return (round(lo * 100, 2), round(hi * 100, 2))


def clustered_ratio_bootstrap(per_task_num: dict[str, int], per_task_den: dict[str, int],
                             resamples: int = BOOTSTRAP_RESAMPLES,
                             seed: int = BOOTSTRAP_SEED) -> tuple[float, float]:
    """Workflow-clustered bootstrap for a ratio of totals (both recomputed)."""
    rng = random.Random(seed + 1)
    keys = sorted(set(per_task_num) | set(per_task_den))
    if not keys or sum(per_task_den.values()) == 0:
        return (0.0, 0.0)
    stats = []
    for _ in range(resamples):
        num = den = 0
        for _ in range(len(keys)):
            k = keys[rng.randrange(len(keys))]
            num += per_task_num.get(k, 0)
            den += per_task_den.get(k, 0)
        if den > 0:
            stats.append(num / den)
        else:
            stats.append(0.0)
    stats.sort()
    lo = stats[int(0.025 * resamples)]
    hi = stats[min(resamples - 1, int(0.975 * resamples))]
    return (round(lo * 100, 2), round(hi * 100, 2))


def read_jsonl(path):
    import json
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]
