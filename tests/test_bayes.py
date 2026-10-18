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
    assert table.profile.shape == (9, 3, 2, 81)
    g_tab = table.g_at(np.abs(c.dist) / c.radius)
    assert g_tab.shape == (9, 3, 2, len(c.dist))
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
                worst = max(worst, float(np.max(np.abs(g_tab[i, j, k] / want - 1))))
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
    assert np.array_equal(first.profile, second.profile)
    assert list(tmp_path.glob("*.npz"))


# ---------------------------------------------------------------- L as a posterior dimension
def _small_table(c: nc.Case, **kw) -> bayes.GTable:
    args: dict[str, Any] = {
        "numx": 61, "n_tau": 8, "n_P0": 2, "n_Pf": 2, "tau_range": (-1.5, 0.5),
        "n_jobs": 1, "cache": False,
    }  # fmt: skip
    return bayes.g_table(c, **{**args, **kw})


@needs_ref
@pytest.mark.parametrize(
    ("key", "rim", "scale"),
    [
        ("ol_with_stalling", False, 0.9),
        ("ol_with_stalling", False, 1.15),
        ("cpx_wht1_exterior", True, 0.95),
        ("cpx_wht1_exterior", True, 1.1),
    ],
)
def test_table_read_at_xi_of_L_equals_direct_forward_with_radius_L(key, rim, scale):
    """G read at xi_i(L) equals a direct forward run with half-width L, for both distance conventions."""
    r = get(key)
    c = r.case
    assert (r.spec.dist_b != 0) == rim  # the registry convention under test
    L = scale * c.radius
    table = _small_table(c)
    xi = bayes.xi_at_L(c, L, rim)
    assert np.all(xi <= 1)
    g_tab = table.g_at(xi)
    # independent positions in um from the centre
    d_abs = np.abs(c.dist)
    x_um = np.abs(L - (c.radius - d_abs)) if rim else d_abs
    worst = 0.0
    v_P = 3.1
    for i in (0, 4, 7):
        for j in (0, 1):
            for k in (0, 1):
                tau, P0, Pf = 10 ** table.log10_tau[i], table.P0[j], table.Pf[k]
                D = tau * L**2 * v_P / (P0 - Pf)
                res = nc.forward(
                    v_P, P0=P0, Pf=Pf, radius=L, logDH=math.log10(D * 1e-12), Kd=1.0,
                    solex_P=c.solex_P, solex_H=c.solex_H, numx=table.numx,
                )  # fmt: skip
                assert res is not None
                want = nc.matlab_spline(res[0], res[1], x_um)
                worst = max(worst, float(np.max(np.abs(g_tab[i, j, k] / want - 1))))
    record(f"table_read_vs_direct_L/{key}/L={scale}", {"max_rel": worst})
    assert worst < 1e-9, worst


@needs_ref
@pytest.mark.parametrize("key", ["ol_with_stalling", "cpx_wht1_exterior"])
def test_fixed_L_equals_fixed_L_path(key):
    """With L fixed, the posterior is the one built from the table read exactly as ``misfit`` reads the model."""
    r = get(key)
    c = r.case
    rim = r.spec.dist_b != 0
    table = _small_table(c)
    post = bayes.grid_posterior(
        table, rim_referenced=rim, P0=table.P0, Pf=table.Pf
    )  # posterior cells = table nodes: no P0/Pf interpolation
    assert post.prob.shape[0] == 1 and post.lprior.name == "fixed"
    x_grid = nc.matlab_colon(0.0, c.radius / (table.numx - 1), c.radius)
    h, s = c.h2o_meas, np.full(len(c.dist), c.one_sig)
    worst = 0.0
    for i, j, k in ((0, 0, 0), (3, 1, 0), (7, 1, 1), (5, 0, 1)):
        g_misfit = nc.matlab_spline(x_grid, table.profile[i, j, k], np.abs(c.dist))
        ll = float(bayes.kd_marginal_loglike(h, s, g_misfit, c.Kd_mean, c.Kd_sd))
        worst = max(worst, abs(ll - float(post.ell[0, i, j, k])))
    record(f"fixed_L_path/{key}", {"max_abs_ll": worst})
    assert worst < 1e-8, worst
    # both conventions coincide at L = L0, and an explicit fixed prior is the default
    other = bayes.grid_posterior(
        table, rim_referenced=not rim, P0=table.P0, Pf=table.Pf
    )
    assert np.allclose(other.prob, post.prob, rtol=0, atol=1e-12)


@needs_ref
def test_L_prior_invalid_nodes_have_zero_probability():
    """L that puts a measured point outside the crystal has zero likelihood (opx WHT1 exterior, L x 0.8)."""
    r = get("opx_wht1_exterior")
    c = r.case
    table = _small_table(c)
    lp = bayes.l_prior(c, "uniform", 0.2, n=9)
    post = bayes.grid_posterior(table, lprior=lp, rim_referenced=r.spec.dist_b != 0)
    assert not post.valid_L.all() and post.valid_L.any()
    assert np.all(post.marginal("L")[~post.valid_L] == 0.0)
    assert abs(post.prob.sum() - 1.0) < 1e-12
    smp = bayes.sample_rates(post, 2000, np.random.default_rng(0))
    assert smp.L.min() >= lp.L[post.valid_L].min() - lp.width.max()
    assert post.bounds()["valid_L_fraction"] < 1.0


# ---------------------------------------------------------------- grid convergence
def v_quantiles(post: bayes.Posterior, qs) -> list[float]:
    """Quantiles of log10 v_P (MPa/s) from the gridded marginal, linear within cells."""
    cdf = np.cumsum(post.marginal("v"))
    return [float(np.interp(q, cdf, post.log10_v_MPa)) for q in qs]


@needs_ref
@pytest.mark.slow
def test_grid_convergence():
    """Doubling the tau and v_P grids and numx moves the log10 v_P posterior quantiles by < 0.01."""
    c = get("ol_with_stalling").case
    kw: dict[str, Any] = {
        "tau_range": (-1.5, 1.0),
        "n_P0": 3,
        "n_Pf": 2,
        "cache": False,
    }
    base_numx = 2 * (c.numx - 1) + 1
    qs = (0.025, 0.16, 0.5, 0.84, 0.975)
    out = {}
    for name, n_tau, n_v, numx in (
        ("base", 200, 175, base_numx),
        ("refined", 400, 350, 2 * (base_numx - 1) + 1),
    ):
        t = bayes.g_table(c, n_tau=n_tau, numx=numx, **kw)
        out[name] = v_quantiles(bayes.grid_posterior(t, n_v=n_v), qs)
    diff = np.abs(np.array(out["base"]) - np.array(out["refined"]))
    record("grid_convergence", {**out, "max_abs_diff": float(diff.max())})
    assert diff.max() < 0.01, (out, diff)


# ---------------------------------------------------------------- coverage on synthetic data
def _simulate(table: bayes.GTable, rng: np.random.Generator):
    """One synthetic data set drawn from the priors (physical parameters, truth off-grid).

    tau outside the table is clamped to its nearest end, the same assumption the posterior makes.
    """
    c = table.case
    log_v_bar = rng.uniform(*bayes.V_RANGE_MPA) + bayes.LOG10_BAR_PER_MPA
    lo, hi = c.logDH_bounds()
    log_d = 12.0 + rng.uniform(lo, hi)
    P0 = rng.uniform(c.Pmin, c.Pmax)
    Pf = c.Pf_fixed if c.Pf_fixed is not None else rng.uniform(c.Pfmin, c.Pfmax)
    K = rng.normal(c.Kd_mean, c.Kd_sd)
    lt = log_d + np.log10(P0 - Pf) - 2 * np.log10(c.radius) - log_v_bar
    lt_used = float(np.clip(lt, table.log10_tau[0], table.log10_tau[-1]))
    g = bayes.g_direct(c, lt_used, P0, Pf, table.numx)
    h = K * g + rng.normal(0.0, c.one_sig, len(g))
    return h, {"log10_v_bar": log_v_bar, "K": K, "in_table": lt == lt_used}


@needs_ref
@pytest.mark.slow
@pytest.mark.parametrize("key", COVERAGE_KEYS)
def test_coverage(key):
    """The 68 and 95 % intervals contain the truth at their nominal rates (within 3 binomial sigma)."""
    c = get(key).case
    table = bayes.g_table(c)  # default grid, cached
    n_sets, n_draw = 200, 4000
    rng = np.random.default_rng(20261017)
    hits = {q: {lev: 0 for lev in (0.68, 0.95)} for q in ("log10_v_P", "K")}
    n_bounded = n_in_table = 0
    for _ in range(n_sets):
        h, truth = _simulate(table, rng)
        post = bayes.grid_posterior(table, h=h)
        smp = bayes.sample_rates(post, n_draw, rng)
        n_bounded += post.bounds()["flag"] == "bounded"
        n_in_table += int(truth["in_table"])
        draws = {
            "log10_v_P": (smp.log10_v_P_bar_s, truth["log10_v_bar"]),
            "K": (smp.K, truth["K"]),
        }
        for q, (x, t) in draws.items():
            for lev in (0.68, 0.95):
                lo, hi = np.quantile(x, [(1 - lev) / 2, (1 + lev) / 2])
                hits[q][lev] += int(lo <= t <= hi)
    rep: dict[str, Any] = {"n_bounded": n_bounded, "n_truth_tau_in_table": n_in_table}
    for q, by_lev in hits.items():
        for lev, k in by_lev.items():
            sig = math.sqrt(lev * (1 - lev) / n_sets)
            rep[f"{q}@{lev}"] = {
                "coverage": k / n_sets,
                "binom_sigma": sig,
                "n": n_sets,
            }
    record(f"coverage/{key}", rep)
    for name, v in rep.items():
        if "@" in name:
            lev = float(name.split("@")[1])
            assert abs(v["coverage"] - lev) <= 3 * v["binom_sigma"], (key, name, v)


# ---------------------------------------------------------------- one-sided bounds (METHODS.md 4.6)
@needs_ref
def test_flat_profile_gives_lower_bound_on_v_P():
    """A profile with no rim loss (too fast to resolve) must come out as a lower bound on v_P, not a median."""
    c = get("ol_with_stalling").case
    table = bayes.g_table(
        c, n_tau=120, n_P0=3, n_Pf=2, tau_range=(-2.5, 1.0), cache=False, n_jobs=1
    )
    rng = np.random.default_rng(3)
    # the table's smallest tau: the profile is still at its initial value everywhere except the rim node
    g_flat = bayes.g_direct(c, table.log10_tau[0], c.Pmax, c.Pfmax, table.numx)
    h = c.Kd_mean * g_flat + rng.normal(0.0, c.one_sig, len(g_flat))
    post = bayes.grid_posterior(table, h=h)
    b = post.bounds()
    smp = bayes.sample_rates(post, 20000, rng)
    out = bayes.summarize(smp, b)
    record("flat_profile", {"bounds": b, "summary": out["v_P_MPa_s"]})
    assert b["flag"] == "lower" and b["low_tau_edge"] and not b["high_tau_edge"], b
    assert "median" not in out["v_P_MPa_s"] and out["v_P_MPa_s"]["flag"] == "lower"
    # and the bound is informative: 95 % of the posterior is above it, well above the prior floor
    assert out["v_P_MPa_s"]["bound95"] > 10 ** bayes.V_RANGE_MPA[0] * 100


@needs_ref
def test_resolved_profile_is_bounded_and_interpolation_consistent():
    """The real olivine profile is bounded, and the gridded posterior is normalised."""
    c = get("ol_with_stalling").case
    table = bayes.g_table(
        c, n_tau=120, n_P0=3, n_Pf=2, tau_range=(-2.5, 1.0), cache=False, n_jobs=1
    )
    post = bayes.grid_posterior(table)
    assert post.bounds()["flag"] == "bounded"
    assert abs(post.prob.sum() - 1.0) < 1e-12
    assert abs(post.marginal("v").sum() - 1.0) < 1e-12


# ---------------------------------------------------------------- laws
@needs_ref
@pytest.mark.parametrize("key", KEYS)
def test_registry_dh_matches_bracket_key(key):
    """Each case's MATLAB DHlo/DHhi pair is the `*_2023_bracket` its key maps to."""
    c = get(key).case
    br = laws.BRACKETS[laws.bracket_key(key)]
    assert (c.dh_lo, c.dh_hi) == (br.lo, br.hi)
    assert br.log10_bounds(c.T_C + 273) == c.logDH_bounds()


def test_named_law_values_at_1200c():
    """log10 D(1200 C) of the named laws against the METHODS.md section 5 table."""
    want = {
        "ol_ferriss2018_a": -10.01,
        "opx_carpenter2003_001": -10.50,
        "opx_carpenter2003_100": -11.10,
        "opx_demersroberge2021_100": -10.04,
        "opx_demersroberge2021_010": -10.21,
        "opx_demersroberge2021_001": -9.95,
        "cpx_2023_exterior": -10.26,
        "gb_demouchy": -8.72,
    }
    for k, v in want.items():
        assert abs(laws.LAWS[k].log10_at(1473.0) - v) < 0.006, k
    anchor = laws.cpx_ferriss2016_tetal(laws.TETAL_REF)
    assert anchor.log10_at(1473.0) == pytest.approx(
        laws.CPX_2023_EXTERIOR.log10_at(1473.0), abs=1e-12
    )


def test_law_prior_nodes():
    c = nc.Case(
        dist=np.array([1.0]), h2o_meas=np.array([1.0]), solex_P=np.array([0.0, 1.0]),
        solex_H=np.array([0.0, 1.0]),
    )  # fmt: skip
    dp = bayes.law_prior(laws.OL_FERRISS2018_A, c, n=11)
    # T 1453..1493 K: log D -10.07 .. -9.95 (m^2/s) -> um^2/s
    assert dp.log10_D.min() > 12 - 10.08 and dp.log10_D.max() < 12 - 9.94
    assert dp.width.sum() == pytest.approx(
        laws.OL_FERRISS2018_A.log10_at(1493.0) - laws.OL_FERRISS2018_A.log10_at(1453.0),
        abs=1e-9,
    )


# ---------------------------------------------------------------- agreement with 2023 (reporting)
@needs_ref
@pytest.mark.slow
@pytest.mark.parametrize("key", KEYS)
def test_agreement_with_2023_reports(key):
    """Posterior v_P (2023 law and priors) vs the saved 2023 MC fits. Writes numbers; no agreement assert."""
    r = get(key)
    out = bayes.fit_case(r, laws=laws.applicable_laws(key))
    record(f"agreement_2023/{key}", out)
    v = out["summary_2023_bracket"][
        "v_P_bar_s"
    ]  # median, or the one-sided bound if flagged
    val = v.get("median", v.get("bound95"))
    assert val is not None and math.isfinite(val) and val > 0
