"""Statistics in pure NumPy (no SciPy): Student-t CDF/quantiles via the
regularised incomplete beta function, Welch t-test, Mann-Whitney U (normal
approximation with tie correction and continuity correction), paired
bootstrap, Cohen's d, Kaplan-Meier."""
from __future__ import annotations

import math

import numpy as np


def _betacf(a, b, x, itmax=300, eps=3e-14):
    qab, qap, qam = a + b, a + 1, a - 1
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > 1e-300 else 1e-300)
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d; d = 1.0 / (d if abs(d) > 1e-300 else 1e-300)
        c = 1.0 + aa / c if abs(c) > 1e-300 else 1e300
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d; d = 1.0 / (d if abs(d) > 1e-300 else 1e-300)
        c = 1.0 + aa / c if abs(c) > 1e-300 else 1e300
        de = d * c
        h *= de
        if abs(de - 1.0) < eps:
            break
    return h


def betai(a, b, x):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbt = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x)
    bt = math.exp(lbt)
    if x < (a + 1) / (a + b + 2):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1 - x) / b


def t_cdf(t, df):
    if not df or df <= 0 or not np.isfinite(t):
        return float("nan")
    x = df / (df + t * t)
    p = 0.5 * betai(df / 2.0, 0.5, x)
    return 1.0 - p if t > 0 else p


def t_ppf(q, df):
    lo, hi = -1e3, 1e3
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if t_cdf(mid, df) < q:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def norm_cdf(z):
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def mean_ci(x, level=0.95):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    n = len(x)
    if n == 0:
        return float("nan"), float("nan"), 0
    if n == 1:
        return float(x[0]), float("nan"), 1
    half = t_ppf(0.5 + level / 2, n - 1) * x.std(ddof=1) / math.sqrt(n)
    return float(x.mean()), float(half), n


def welch(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return {"t": float("nan"), "df": float("nan"), "p": float("nan")}
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    if va + vb == 0:
        return {"t": float("nan"), "df": float("nan"), "p": 1.0 if a.mean() == b.mean() else 0.0}
    t = (a.mean() - b.mean()) / math.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    return {"t": float(t), "df": float(df), "p": float(2 * (1 - t_cdf(abs(t), df)))}


def one_sample_t(a, mu):
    a = np.asarray(a, float)
    if len(a) < 2:
        return {"t": float("nan"), "p": float("nan")}
    sd = a.std(ddof=1)
    if sd == 0:
        return {"t": float("nan"), "p": 1.0 if a.mean() == mu else 0.0}
    t = (a.mean() - mu) / (sd / math.sqrt(len(a)))
    return {"t": float(t), "p": float(2 * (1 - t_cdf(abs(t), len(a) - 1)))}


def mann_whitney(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    n1, n2 = len(a), len(b)
    allv = np.concatenate([a, b])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv))
    sv = allv[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1
        i = j + 1
    U1 = ranks[:n1].sum() - n1 * (n1 + 1) / 2
    _, counts = np.unique(allv, return_counts=True)
    N = n1 + n2
    sigma2 = n1 * n2 / 12.0 * ((N + 1) - np.sum(counts ** 3 - counts) / (N * (N - 1)))
    if sigma2 <= 0:
        return {"U": float(U1), "p": 1.0}
    z = (abs(U1 - n1 * n2 / 2.0) - 0.5) / math.sqrt(sigma2)
    return {"U": float(U1), "p": float(min(1.0, 2 * (1 - norm_cdf(max(z, 0)))))}


def paired_bootstrap(a, b, n=10_000, seed=0):
    """Paired (by index) bootstrap of mean(a - b)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    m = d[idx].mean(1)
    p = 2 * min(float(np.mean(m <= 0)), float(np.mean(m >= 0)))
    return {"mean_diff": float(d.mean()), "ci_lo": float(np.percentile(m, 2.5)),
            "ci_hi": float(np.percentile(m, 97.5)), "p": float(min(1.0, p))}


def cohen_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    sp = math.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / sp) if sp > 0 else float("nan")


def kaplan_meier(times, events):
    """times: steps; events: True if collision observed (False = censored at cap)."""
    t = np.asarray(times, float); e = np.asarray(events, bool)
    order = np.argsort(t)
    t, e = t[order], e[order]
    S, xs, ys = 1.0, [0.0], [1.0]
    n = len(t)
    for u in np.unique(t[e]):
        at_risk = np.sum(t >= u)
        d = np.sum((t == u) & e)
        S *= 1 - d / at_risk
        xs.append(float(u)); ys.append(float(S))
    return xs, ys


def compare(a, b, seed=0) -> dict:
    """All tests for a (method) vs b (baseline), both per-seed arrays (paired)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    out = {"n_a": len(a), "n_b": len(b), "mean_a": float(a.mean()), "mean_b": float(b.mean())}
    if len(b) == 1:                       # deterministic baseline (hand rule): one-sample test
        out["one_sample_t"] = one_sample_t(a, b[0])
        out["bootstrap"] = paired_bootstrap(a, np.full_like(a, b[0]), seed=seed)
        return out
    if len(a) == len(b):
        out["bootstrap"] = paired_bootstrap(a, b, seed=seed)
    out["welch"] = welch(a, b)
    out["mann_whitney"] = mann_whitney(a, b)
    out["cohen_d"] = cohen_d(a, b)
    return out
