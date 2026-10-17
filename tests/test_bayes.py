"""Tests for the Bayesian decompression-rate fit (``decompression/bayes.py``, METHODS.md sections 2-6).

Exact properties (scaling invariance, K linearity, closed-form K marginal), table fidelity, grid
convergence, coverage of the credible intervals on synthetic data, and a reporting test that writes
the posterior-versus-2023 numbers. Sullivan Ranch inputs are read from ``$NEWCOMBE_REF`` and never
copied into the repo. Numbers are dumped to ``$BAYES_NUMBERS_JSON`` (default
``decompression/.g_cache/bayes_numbers.json``, gitignored). Slow tests carry ``@pytest.mark.slow``.
"""

from __future__ import annotations

import functools
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy.integrate import quad

from decompression import bayes, cases, laws
from decompression import newcombe as nc

REF = cases.ref_root()
needs_ref = pytest.mark.skipif(
    REF is None, reason="$NEWCOMBE_REF reference folder not found"
)
KEYS = [c.key for c in cases.REGISTRY]
COVERAGE_KEYS = ["ol_with_stalling", "cpx_wht1_exterior"]
NUMBERS = Path(
    os.environ.get("BAYES_NUMBERS_JSON", bayes.CACHE_DIR / "bayes_numbers.json")
)


def record(name: str, numbers: dict) -> None:
    NUMBERS.parent.mkdir(exist_ok=True)
    data = json.loads(NUMBERS.read_text()) if NUMBERS.exists() else {}
    data[name] = numbers
    NUMBERS.write_text(json.dumps(data, indent=1))


@functools.cache
def get(key: str) -> cases.Resolved:
    assert REF is not None
    return cases.resolve(cases.BY_KEY[key], REF)


def pf_of(case: nc.Case) -> float:
    return case.Pf_fixed if case.Pf_fixed is not None else 2.5


def max_rel(a: np.ndarray, b: np.ndarray, floor: float = 1e-6) -> float:
    """Largest |a - b| / |b| over elements with |b| > floor * max|b|.

    With Pf = 0 the rim ends at H(0) = 0: the last pressure is 0 only up to rounding (~1e-12 bar), so the
    last node is ~1e-11 against a profile of ~300 and has no meaningful relative difference.
    """
    keep = np.abs(b) > floor * np.max(np.abs(b))
    return float(np.max(np.abs(a - b)[keep] / np.abs(b)[keep]))


def quantile_cdf(prob: np.ndarray, edges_lo: float, width: float, qs) -> list[float]:
    """Quantiles of a gridded marginal (cell probabilities), linear within cells."""
    cdf = np.concatenate([[0.0], np.cumsum(prob)])
    x = edges_lo + width * np.arange(len(cdf))
    return [float(np.interp(q, cdf, x)) for q in qs]


# ---------------------------------------------------------------- exact properties
@needs_ref
@pytest.mark.parametrize("key", KEYS)
def test_scaling_invariance(key):
    """(D, v_P) -> (aD, a v_P) leaves the final profile unchanged to rounding (METHODS.md section 2)."""
    r = get(key)
    c = r.case
    worst, n_ok, worst_inexact = 0.0, 0, 0.0
    for row in r.saved[:10]:
        v, P0, Kd, logDH = row[2], row[3], row[4], row[5]
        if not v > 0:
            continue
        kw: dict[str, Any] = dict(  # noqa: C408
            P0=P0, Pf=pf_of(c), radius=c.radius, Kd=Kd,
            solex_P=c.solex_P, solex_H=c.solex_H, numx=c.numx,
        )  # fmt: skip
        base = nc.forward(v, logDH=logDH, **kw)
        for a in (0.5, 2.0, 8.0, 3.7):
            scaled = nc.forward(a * v, logDH=logDH + math.log10(a), **kw)
            if base is None or scaled is None:
                continue
            rel = max_rel(scaled[1], base[1])
            if (
                a == 3.7
            ):  # log10(3.7) is rounded when added to logDH: an input error, not the scheme
                worst_inexact = max(worst_inexact, float(rel))
                continue
            worst = max(worst, float(rel))
            n_ok += 1
    assert n_ok >= 10
    record(
        f"scaling_invariance/{key}",
        {"max_rel_pow2": worst, "max_rel_a3.7": worst_inexact, "n": n_ok},
    )
    # a = 2^k scales D exactly. METHODS.md measured 1.2e-16 to 2.6e-16 (this test: <= 1e-15); x10 margin
    assert worst <= 3e-15, worst
    assert worst_inexact <= 1e-11, (
        worst_inexact
    )  # the rounded logDH input is amplified ~100x


@needs_ref
@pytest.mark.parametrize("key", KEYS)
def test_kd_linearity(key):
    """C(K) = K C(1): exact for K a power of two, rounding-level for any K (METHODS.md section 3)."""
    r = get(key)
    c = r.case
    row = r.saved[0]
    kw: dict[str, Any] = dict(  # noqa: C408
        P0=row[3], Pf=pf_of(c), radius=c.radius, logDH=row[5], solex_P=c.solex_P,
        solex_H=c.solex_H, numx=c.numx,
    )  # fmt: skip
    one = nc.forward(row[2], Kd=1.0, **kw)
    assert one is not None
    for k, tol in ((8.0, 0.0), (0.25, 0.0), (row[4], 1e-14)):
        got = nc.forward(row[2], Kd=k, **kw)
        assert got is not None
        err = max_rel(got[1], k * one[1])
        record(f"kd_linearity/{key}/K={k:g}", {"max_rel": err})
        assert err <= tol, (k, err)


def _log_integral(h, s, g, mu_K, s_K) -> float:
    """log of the integral over K of N(K; mu_K, s_K) prod N(h_i; K g_i, s_i), by adaptive quadrature."""
    h, s, g = (np.asarray(v, float) for v in (h, s, g))

    def logf(K):
        return (
            -0.5 * np.sum(((h - K * g) / s) ** 2)
            - np.sum(np.log(np.sqrt(2 * np.pi) * s))
            - 0.5 * ((K - mu_K) / s_K) ** 2
            - math.log(np.sqrt(2 * np.pi) * s_K)
        )

    Ks = np.linspace(mu_K - 10 * s_K, mu_K + 10 * s_K, 4001)
    m = max(logf(K) for K in Ks)
    val, _ = quad(
        lambda K: math.exp(logf(K) - m), mu_K - 10 * s_K, mu_K + 10 * s_K,
        epsabs=1e-13, epsrel=1e-12, limit=400, points=[float(Ks[np.argmax([logf(K) for K in Ks])])],
    )  # fmt: skip
    return m + math.log(val)


def test_kd_marginal_matches_quadrature():
    """Closed-form K marginal equals numerical integration over K to 1e-8 (METHODS.md 4.2)."""
    rng = np.random.default_rng(1)
    worst = 0.0
    for mu_K, s_K, scale in ((9.0, 0.5, 3.0), (90.0, 5.0, 0.04), (120.0, 5.0, 0.03)):
        for _ in range(6):
            n = int(rng.integers(3, 12))
            g = rng.uniform(0.05, 1.0, n) * scale
            s = rng.uniform(0.5, 2.0, n) * 0.5 * scale * mu_K / 9.0
            K_true = rng.normal(mu_K, s_K)
            h = K_true * g + rng.normal(0, s)
            exact = float(bayes.kd_marginal_loglike(h, s, g, mu_K, s_K))
            worst = max(worst, abs(exact - _log_integral(h, s, g, mu_K, s_K)))
    record("kd_marginal_quadrature", {"max_abs_diff": worst})
    assert worst < 1e-8, worst


def test_kd_marginal_refuses_mass_at_negative_k():
    with pytest.raises(AssertionError):
        bayes.kd_marginal_loglike([1.0], [1.0], [[0.5]], 1.0, 1.0)


# ---------------------------------------------------------------- table fidelity
@needs_ref
def test_g_table_matches_forward():
    """Table entries equal direct forward runs written independently (different v_P, so also property 1)."""
    r = get("ol_with_stalling")
    c = r.case
    table = bayes.g_table(
        c,
        numx=81,
        n_tau=9,
        n_P0=3,
        n_Pf=2,
        tau_range=(-1.5, 0.5),
        n_jobs=1,
        cache=False,
    )
    assert table.g.shape == (9, 3, 2, len(c.dist))
    worst = 0.0
    v_P = 7.3
    for i in (0, 3, 8):
        for j in (0, 2):
            for k in (0, 1):
                tau, P0, Pf = 10 ** table.log10_tau[i], table.P0[j], table.Pf[k]
                D = tau * c.radius**2 * v_P / (P0 - Pf)  # um^2/s
                res = nc.forward(
                    v_P, P0=P0, Pf=Pf, radius=c.radius, logDH=math.log10(D * 1e-12),
                    Kd=1.0, solex_P=c.solex_P, solex_H=c.solex_H, numx=81,
                )  # fmt: skip
                assert res is not None
                want = nc.matlab_spline(res[0], res[1], np.abs(c.dist))
                worst = max(worst, float(np.max(np.abs(table.g[i, j, k] / want - 1))))
    record("g_table_vs_forward", {"max_rel": worst})
    assert worst < 1e-12, worst


@needs_ref
def test_g_table_cache_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(bayes, "CACHE_DIR", tmp_path)
    c = get("ol_with_stalling").case
    kw: dict[str, Any] = {
        "numx": 41,
        "n_tau": 4,
        "n_P0": 2,
        "n_Pf": 2,
        "tau_range": (-1.0, 0.0),
        "n_jobs": 1,
    }
    first = bayes.g_table(c, **kw)
    second = bayes.g_table(c, **kw)
    assert not first.from_cache and second.from_cache
    assert np.array_equal(first.g, second.g)
    assert list(tmp_path.glob("*.npz"))


# ---------------------------------------------------------------- grid convergence
@needs_ref
@pytest.mark.slow
def test_grid_convergence():
    """Doubling the tau grid and numx moves the log tau posterior quantiles by < 0.01."""
    c = get("ol_with_stalling").case
    rng_tau = (-1.0, 0.2)
    kw: dict[str, Any] = {"tau_range": rng_tau, "n_P0": 9, "n_Pf": 3, "cache": False}
    base_numx = 4 * (c.numx - 1) + 1
    qs = (0.025, 0.16, 0.5, 0.84, 0.975)
    out = {}
    for name, n_tau, numx in (
        ("base", 200, base_numx),
        ("refined", 400, 2 * (base_numx - 1) + 1),
    ):
        t = bayes.g_table(c, n_tau=n_tau, numx=numx, **kw)
        p = bayes.grid_posterior(t)
        lo = rng_tau[0]
        out[name] = quantile_cdf(p.marginal("tau"), lo, t.dlog_tau, qs)
    diff = np.abs(np.array(out["base"]) - np.array(out["refined"]))
    record("grid_convergence", {**out, "max_abs_diff": float(diff.max())})
    assert diff.max() < 0.01, (out, diff)


# ---------------------------------------------------------------- coverage on synthetic data
def _simulate(table: bayes.GTable, rng: np.random.Generator):
    """One synthetic data set from the prior (truth off-grid, forward at the table resolution)."""
    c = table.case
    lt = rng.uniform(
        table.log10_tau[0] - table.dlog_tau / 2,
        table.log10_tau[-1] + table.dlog_tau / 2,
    )
    P0 = rng.uniform(c.Pmin, c.Pmax)
    Pf = c.Pf_fixed if c.Pf_fixed is not None else rng.uniform(c.Pfmin, c.Pfmax)
    K = rng.normal(c.Kd_mean, c.Kd_sd)
    g = bayes.g_direct(c, lt, P0, Pf, table.numx)
    h = K * g + rng.normal(0.0, c.one_sig, len(g))
    return h, {"log10_tau": lt, "P0": P0, "Pf": Pf, "K": K}


@needs_ref
@pytest.mark.slow
@pytest.mark.parametrize("key", COVERAGE_KEYS)
def test_coverage(key):
    """The 68 and 95 % intervals contain the truth at their nominal rates (within 3 binomial sigma)."""
    c = get(key).case
    table = bayes.g_table(c)  # default grid, cached
    n_sets, n_draw = 200, 4000
    rng = np.random.default_rng(20261017)
    hits = {
        q: {lev: 0 for lev in (0.68, 0.95)} for q in ("log10_tau", "log10_v_P", "K")
    }
    for _ in range(n_sets):
        h, truth = _simulate(table, rng)
        post = bayes.grid_posterior(table, h=h)
        smp = bayes.sample_rates(post, n_draw, rng)
        # truth for v_P: an independent draw of the 2023 D prior, as the sampler marginalises it
        one = bayes.Samples(
            c, np.array([10 ** truth["log10_tau"]]), np.array([truth["P0"]]), np.array([truth["Pf"]]),
            np.array([truth["K"]]), np.array([c.T_C + 273.0]), np.array([c.radius]), np.array([2800.0]),
            np.array([rng.random()]),
        )  # fmt: skip
        draws = {
            "log10_tau": (np.log10(smp.tau), truth["log10_tau"]),
            "K": (smp.K, truth["K"]),
            "log10_v_P": (
                np.log10(smp.v_P_bar_s(None)),
                float(np.log10(one.v_P_bar_s(None))[0]),
            ),
        }
        for q, (x, t) in draws.items():
            for lev in (0.68, 0.95):
                lo, hi = np.quantile(x, [(1 - lev) / 2, (1 + lev) / 2])
                hits[q][lev] += int(lo <= t <= hi)
    rep = {}
    for q, by_lev in hits.items():
        for lev, k in by_lev.items():
            p_hat = k / n_sets
            sig = math.sqrt(lev * (1 - lev) / n_sets)
            rep[f"{q}@{lev}"] = {"coverage": p_hat, "binom_sigma": sig, "n": n_sets}
    record(f"coverage/{key}", rep)
    for name, v in rep.items():
        lev = float(name.split("@")[1])
        assert abs(v["coverage"] - lev) <= 3 * v["binom_sigma"], (key, name, v)


# ---------------------------------------------------------------- agreement with 2023 (reporting)
@needs_ref
@pytest.mark.slow
@pytest.mark.parametrize("key", KEYS)
def test_agreement_with_2023_reports(key):
    """Posterior v_P (2023 law and priors) vs the saved 2023 MC fits. Writes numbers; no agreement assert."""
    r = get(key)
    out = bayes.fit_case(r, laws=laws.applicable_laws(key))
    record(f"agreement_2023/{key}", out)
    med = out["summary_2023_law"]["v_P_bar_s"]["median"]
    assert math.isfinite(med) and med > 0
