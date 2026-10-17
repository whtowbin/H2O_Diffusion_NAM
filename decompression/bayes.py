"""Bayesian decompression-rate fit (METHODS.md section 4).

The profile depends on (D, v_P) only through tau = D (P0 - Pf) / (v_P L^2) (property 1) and is linear
in the partition coefficient K (property 2). So ``G(|d_i|/L; tau, P0, Pf)`` is tabulated once per case
and degassing path with K = 1 (``g_table``), K is integrated out in closed form
(``kd_marginal_loglike``), the (tau, P0, Pf) posterior is evaluated exactly on the grid
(``grid_posterior``), and T, L and the diffusion law are applied afterwards (``sample_rates``).

Units: lengths um, time s, pressure bar, D um^2/s inside the solver and m^2/s in ``DiffLaw``.
"""

from __future__ import annotations

import hashlib
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path

import numpy as np
from scipy.stats import norm

from decompression.cases import Resolved
from decompression.newcombe import Case, DiffLaw, forward, matlab_spline

CACHE_DIR = Path(__file__).parent / ".g_cache"  # gitignored
TABLE_VERSION = 1  # bump when the table definition changes (invalidates the cache)
G_GRAV = 9.81  # m/s^2
BAR_PER_MPA = 10.0
PA_PER_BAR = 1e5


# --------------------------------------------------------------------------------------
# G table
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class GTable:
    """``g[i_tau, i_P0, i_Pf, i_data]`` = profile / K at |d_i| for cell-centre (tau, P0, Pf)."""

    case: Case
    log10_tau: np.ndarray
    P0: np.ndarray
    Pf: np.ndarray
    g: np.ndarray
    numx: int
    build_seconds: float = 0.0
    from_cache: bool = False

    @property
    def dlog_tau(self) -> float:
        return float(self.log10_tau[1] - self.log10_tau[0])

    @property
    def dP0(self) -> float:
        return float(self.P0[1] - self.P0[0]) if len(self.P0) > 1 else 0.0

    @property
    def dPf(self) -> float:
        return float(self.Pf[1] - self.Pf[0]) if len(self.Pf) > 1 else 0.0


def _cells(lo: float, hi: float, n: int) -> np.ndarray:
    """Centres of n equal cells spanning [lo, hi] (midpoint rule for a uniform prior)."""
    return lo + (np.arange(n) + 0.5) * (hi - lo) / n


def _pf_cells(case: Case, n_Pf: int) -> np.ndarray:
    if case.Pf_fixed is not None:
        return np.array([float(case.Pf_fixed)])
    return _cells(case.Pfmin, case.Pfmax, n_Pf)


def _profile_at_data(
    log10_tau, P0, Pf, radius, numx, dist_abs, sP, sH
) -> np.ndarray | None:
    """Direct forward run with v_P = 1 bar/s, K = 1 and D = tau L^2 / (P0 - Pf); g at the data."""
    D = 10.0**log10_tau * radius**2 / (P0 - Pf)  # um^2/s
    res = forward(
        1.0, P0=P0, Pf=Pf, radius=radius, logDH=float(np.log10(D / 1e12)), Kd=1.0,
        solex_P=sP, solex_H=sH, numx=numx,
    )  # fmt: skip
    if res is None:
        return None
    return matlab_spline(res[0], res[1], dist_abs)


def g_direct(
    case: Case, log10_tau: float, P0: float, Pf: float, numx: int, path=None
) -> np.ndarray:
    """One table entry computed directly with ``forward`` (used by the table, tests and simulations)."""
    sP, sH = _path_arrays(case, path)
    g = _profile_at_data(
        log10_tau, P0, Pf, case.radius, numx, np.abs(case.dist), sP, sH
    )
    if g is None:
        raise ValueError(f"forward returned no profile at log10 tau = {log10_tau}")
    return g


def _path_arrays(case: Case, path) -> tuple[np.ndarray, np.ndarray]:
    if path is None:
        return np.asarray(case.solex_P, float), np.asarray(case.solex_H, float)
    return np.asarray(path[0], float), np.asarray(path[1], float)


def _g_block(args) -> np.ndarray:
    lts, P0, Pf, radius, numx, dist_abs, sP, sH = args
    out = np.empty((len(lts), len(dist_abs)))
    for k, lt in enumerate(lts):
        g = _profile_at_data(lt, P0, Pf, radius, numx, dist_abs, sP, sH)
        if g is None:
            raise ValueError(f"forward returned no profile at log10 tau = {lt}")
        out[k] = g
    return out


def _table_key(case: Case, sP, sH, numx, lts, P0s, Pfs) -> str:
    h = hashlib.sha256(f"gtable-v{TABLE_VERSION}|{case.radius!r}|{numx}".encode())
    for a in (np.abs(case.dist), sP, sH, lts, P0s, Pfs):
        h.update(np.ascontiguousarray(a, dtype=float).tobytes())
        h.update(b"|")
    return h.hexdigest()[:24]


def g_table(
    case: Case,
    path: tuple[np.ndarray, np.ndarray] | None = None,
    *,
    numx: int | None = None,
    n_tau: int = 400,
    n_P0: int = 25,
    n_Pf: int = 5,
    tau_range: tuple[float, float] | None = None,
    n_jobs: int | None = None,
    cache: bool = True,
    refresh: bool = False,
) -> GTable:
    """Tabulate G on the log10 tau x P0 x Pf grid (K = 1) with the explicit solver.

    ``path`` is (P bar, H2O wt%) of the degassing path (default: the case's own). ``numx`` defaults to
    4 (case.numx - 1) + 1. The cache (gitignored, keyed by a hash of the inputs) holds only G, never data;
    ``refresh`` rebuilds and overwrites an existing entry (for timing).
    """
    t0 = time.perf_counter()
    numx = 4 * (case.numx - 1) + 1 if numx is None else numx
    sP, sH = _path_arrays(case, path)
    lo, hi = tau_range if tau_range is not None else auto_tau_range(case, (sP, sH))
    lts = _cells(lo, hi, n_tau)
    P0s = _cells(case.Pmin, case.Pmax, n_P0)
    Pfs = _pf_cells(case, n_Pf)
    key = _table_key(case, sP, sH, numx, lts, P0s, Pfs)
    file = CACHE_DIR / f"{key}.npz"
    if cache and not refresh and file.exists():
        with np.load(file) as z:
            g = z["g"]
        return GTable(
            case, lts, P0s, Pfs, g, numx, time.perf_counter() - t0, from_cache=True
        )

    dist_abs = np.abs(case.dist)
    jobs = [
        (lts, p0, pf, case.radius, numx, dist_abs, sP, sH) for p0 in P0s for pf in Pfs
    ]
    n_jobs = os.cpu_count() or 1 if n_jobs is None else n_jobs
    if n_jobs > 1:
        # one thread per worker: threaded BLAS/Accelerate inside 12 workers oversubscribes the cores ~6x
        for var in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMBA_NUM_THREADS",
        ):
            os.environ[var] = "1"
        with ProcessPoolExecutor(n_jobs, mp_context=get_context("spawn")) as pool:
            blocks = list(pool.map(_g_block, jobs, chunksize=1))
    else:
        blocks = [_g_block(j) for j in jobs]
    g = np.empty((n_tau, len(P0s), len(Pfs), len(dist_abs)))
    idx = 0
    for j in range(len(P0s)):
        for k in range(len(Pfs)):
            g[:, j, k, :] = blocks[idx]
            idx += 1
    if cache:
        CACHE_DIR.mkdir(exist_ok=True)
        np.savez_compressed(file, g=g)
    return GTable(case, lts, P0s, Pfs, g, numx, time.perf_counter() - t0)


def auto_tau_range(
    case: Case,
    path: tuple[np.ndarray, np.ndarray] | None = None,
    *,
    lo: float = -4.0,
    hi: float = 2.5,
    step: float = 0.1,
    delta: float = 12.0,
    pad: float = 0.1,
    sigma_floor: float = 0.02,
) -> tuple[float, float]:
    """log10 tau interval where the data can be fit (a coarse scan at the corners and middle of the priors).

    Keeps every tau whose K-marginal log-likelihood is within ``delta`` of the best, with the data
    uncertainty inflated to at least ``sigma_floor`` of each value so a coarse step cannot step over a
    narrow peak, then pads by ``pad`` decades. A heuristic: check ``Posterior.edge_mass``.
    """
    sP, sH = _path_arrays(case, path)
    Pfs = [case.Pf_fixed] if case.Pf_fixed is not None else [case.Pfmin, case.Pfmax]
    combos = [(p0, pf) for p0 in (case.Pmin, case.Pmax) for pf in Pfs]
    combos.append((0.5 * (case.Pmin + case.Pmax), 0.5 * (Pfs[0] + Pfs[-1])))
    lts = np.arange(lo, hi + 1e-9, step)
    dist_abs = np.abs(case.dist)
    h = np.asarray(case.h2o_meas, float)
    s = np.maximum(case.one_sig, sigma_floor * np.abs(h))
    g = np.array(
        [
            [_g_coarse(lt, p0, pf, case, dist_abs, sP, sH) for lt in lts]
            for p0, pf in combos
        ]
    )  # (combo, tau, data)
    ll = kd_marginal_loglike(h, s, g, case.Kd_mean, case.Kd_sd)
    ll = np.where(
        np.isnan(ll), -np.inf, ll
    )  # tau too small for the solver (< 10 steps)
    keep = ll >= ll.max() - delta
    ok = np.where(keep.any(axis=0))[0]
    lo_k, hi_k = lts[ok.min()] - pad, lts[ok.max()] + pad
    return float(max(lo, lo_k)), float(min(hi, hi_k))


def _g_coarse(lt, p0, pf, case, dist_abs, sP, sH):
    # below tau ~ 3e-3 the 2023 grid needs < 10 time steps (the solver refuses): use the fine grid there
    numx = case.numx if lt >= -2.5 else 4 * (case.numx - 1) + 1
    g = _profile_at_data(lt, p0, pf, case.radius, numx, dist_abs, sP, sH)
    return np.full(len(dist_abs), np.nan) if g is None else g


# --------------------------------------------------------------------------------------
# K marginal and grid posterior
# --------------------------------------------------------------------------------------
def kd_marginal_loglike(h, s, g, mu_K: float, s_K: float, *, return_ab: bool = False):
    """log p(h | g) with K ~ N(mu_K, s_K^2) integrated out (METHODS.md 4.2); g has the data on its last axis."""
    # prior mass at K < 0 must be negligible (the closed form integrates K over the whole line)
    mass_neg = float(norm.cdf(-mu_K / s_K))
    assert mass_neg < 1e-9, (
        f"prior mass at K<0 is {mass_neg:.2e}: closed form not valid"
    )
    h = np.asarray(h, float)
    s = np.broadcast_to(np.asarray(s, float), h.shape)
    g = np.asarray(g, float)
    w = 1.0 / s**2
    a = np.sum(g * g * w, axis=-1) + 1.0 / s_K**2
    b = np.sum(h * g * w, axis=-1) + mu_K / s_K**2
    ll = (
        -0.5 * (np.sum(h * h * w) + mu_K**2 / s_K**2 - b * b / a)
        - 0.5 * np.log(a * s_K**2)
        - np.sum(np.log(np.sqrt(2.0 * np.pi) * s))
    )
    return (ll, a, b) if return_ab else ll


@dataclass(frozen=True)
class Posterior:
    table: GTable
    loglike: np.ndarray  # (tau, P0, Pf)
    prob: np.ndarray  # normalised cell probabilities
    a: np.ndarray
    b: np.ndarray  # K | cell ~ N(b/a, 1/a)
    mu_K: float
    s_K: float

    def marginal(self, axis: str) -> np.ndarray:
        keep = {"tau": 0, "P0": 1, "Pf": 2}[axis]
        other = tuple(i for i in range(3) if i != keep)
        return self.prob.sum(axis=other)

    def edge_mass(self, n: int = 3) -> float:
        """Posterior mass in the n outermost tau cells at either end of the grid (should be ~0)."""
        m = self.marginal("tau")
        return float(m[:n].sum() + m[-n:].sum())


def grid_posterior(
    table: GTable,
    h=None,
    s=None,
    *,
    mu_K: float | None = None,
    s_K: float | None = None,
    sigma_m: float = 0.0,
) -> Posterior:
    """Exact posterior on the grid; defaults from the case (data, one_sig, Kd prior); s_i^2 = sigma_i^2 + sigma_m^2."""
    c = table.case
    h = np.asarray(c.h2o_meas if h is None else h, float)
    s = np.broadcast_to(np.asarray(c.one_sig if s is None else s, float), h.shape)
    mu_K = c.Kd_mean if mu_K is None else mu_K
    s_K = c.Kd_sd if s_K is None else s_K
    s_tot = np.sqrt(s**2 + sigma_m**2)
    ll, a, b = kd_marginal_loglike(h, s_tot, table.g, mu_K, s_K, return_ab=True)
    p = np.exp(ll - ll.max())  # uniform priors in log tau, P0, Pf: equal cell weights
    return Posterior(table, ll, p / p.sum(), a, b, mu_K, s_K)


# --------------------------------------------------------------------------------------
# From tau to rates
# --------------------------------------------------------------------------------------
def log10_D_m2s(law: DiffLaw, T_K) -> np.ndarray:
    """Vectorised ``DiffLaw.log10_at`` (same operator order)."""
    T_K = np.asarray(T_K, float)
    return np.log10(law.A * np.exp(-law.Ea / (8.314 * T_K)) * law.mult)


@dataclass(frozen=True)
class Samples:
    """Posterior draws; T, L, rho and the 2023 D draw are independent of the data (METHODS.md 4.4)."""

    case: Case
    tau: np.ndarray
    P0: np.ndarray
    Pf: np.ndarray
    K: np.ndarray
    T_K: np.ndarray
    L: np.ndarray  # um
    rho: np.ndarray  # kg/m^3
    u01: np.ndarray  # uniform draw for the 2023 log D prior

    def log10_D_um2s(self, law: DiffLaw | None) -> np.ndarray:
        """law=None: the 2023 prior, log D uniform between the case's T-20 and T+20 laws."""
        if law is None:
            lo, hi = self.case.logDH_bounds()
            return (
                12.0 + hi - self.u01 * (hi - lo)
            )  # master_script_test.m: logDH = hi - rand (hi - lo)
        return 12.0 + log10_D_m2s(law, self.T_K)

    def v_P_bar_s(self, law: DiffLaw | None = None) -> np.ndarray:
        """v_P = D (P0 - Pf) / (tau L^2), bar/s."""
        return (
            10.0 ** self.log10_D_um2s(law)
            * (self.P0 - self.Pf)
            / (self.tau * self.L**2)
        )

    def v_P_MPa_s(self, law: DiffLaw | None = None) -> np.ndarray:
        return self.v_P_bar_s(law) / BAR_PER_MPA

    def u_m_s(self, law: DiffLaw | None = None) -> np.ndarray:
        """Ascent speed v_P / (rho g)."""
        return self.v_P_bar_s(law) * PA_PER_BAR / (self.rho * G_GRAV)


def sample_rates(
    post: Posterior,
    n: int,
    rng: np.random.Generator,
    *,
    T_halfwidth: float = 20.0,
    T_dist: str = "uniform",
    L_rel_sd: float = 0.0,
    rho: tuple[float, float] = (2800.0, 100.0),
    jitter: bool = True,
) -> Samples:
    """Draw (tau, P0, Pf) from the grid posterior (uniform within a cell if ``jitter``), K | cell, and T, L, rho.

    T ~ U(T0 -/+ T_halfwidth) K or N(T0, T_halfwidth) (``T_dist``), T0 = T_C + 273 as in the MATLAB.
    L ~ N(L0, (L_rel_sd L0)^2) truncated positive; the default keeps the 2023 value.
    The diffusion law is applied afterwards by ``Samples.v_P_bar_s(law)``.
    """
    t = post.table
    flat = rng.choice(post.prob.size, size=n, p=post.prob.ravel())
    i, j, k = np.unravel_index(flat, post.prob.shape)

    def draw(centre, width):
        return centre + (rng.random(n) - 0.5) * width if jitter else centre

    lt = draw(t.log10_tau[i], t.dlog_tau)
    P0 = draw(t.P0[j], t.dP0)
    Pf = draw(t.Pf[k], t.dPf)
    a, b = post.a[i, j, k], post.b[i, j, k]
    K = b / a + rng.standard_normal(n) / np.sqrt(a)
    T0 = t.case.T_C + 273.0
    if T_dist == "uniform":
        T_K = T0 + rng.uniform(-T_halfwidth, T_halfwidth, n)
    elif T_dist == "normal":
        T_K = T0 + rng.normal(0.0, T_halfwidth, n)
    else:
        raise ValueError(f"T_dist must be 'uniform' or 'normal', got {T_dist!r}")
    L = t.case.radius * np.clip(1.0 + L_rel_sd * rng.standard_normal(n), 0.05, None)
    rho_s = rng.normal(rho[0], rho[1], n)
    return Samples(t.case, 10.0**lt, P0, Pf, K, T_K, L, rho_s, rng.random(n))


def _interval(x: np.ndarray) -> dict[str, float]:
    q = np.quantile(x, [0.025, 0.16, 0.5, 0.84, 0.975])
    return dict(
        zip(("lo95", "lo68", "median", "hi68", "hi95"), map(float, q), strict=True)
    )


def summarize(
    samples: Samples, law: DiffLaw | None = None
) -> dict[str, dict[str, float]]:
    """Median and 68/95 % central intervals of log10 v_P, v_P (MPa/s, bar/s), u (m/s), K, P0, Pf, log10 tau."""
    vp = samples.v_P_bar_s(law)
    return {
        "log10_v_P_bar_s": _interval(np.log10(vp)),
        "v_P_MPa_s": _interval(vp / BAR_PER_MPA),
        "v_P_bar_s": _interval(vp),
        "u_m_s": _interval(vp * PA_PER_BAR / (samples.rho * G_GRAV)),
        "K": _interval(samples.K),
        "P0_bar": _interval(samples.P0),
        "Pf_bar": _interval(samples.Pf),
        "log10_tau": _interval(np.log10(samples.tau)),
    }


# --------------------------------------------------------------------------------------
# One case end to end (used by the results script and the reporting test)
# --------------------------------------------------------------------------------------
def saved_2023_summary(saved: np.ndarray) -> dict[str, float]:
    """Median and IQR of the saved 2023 MC fits (column 2 = fitted dP/dt, bar/s)."""
    v = saved[:, 2]
    q1, med, q3 = np.quantile(v, [0.25, 0.5, 0.75])
    return {"median": float(med), "q1": float(q1), "q3": float(q3), "n": len(v)}


def fit_case(
    res: Resolved,
    *,
    laws: dict[str, DiffLaw] | None = None,
    n_samples: int = 40000,
    seed: int = 0,
    cache: bool = True,
    n_jobs: int | None = None,
    **table_kw,
) -> dict:
    """Table, posterior, 2023-law summary, 2023 MC summary and v_P medians under ``laws``, with timings."""
    case = res.case
    table = g_table(case, cache=cache, n_jobs=n_jobs, **table_kw)
    t0 = time.perf_counter()
    post = grid_posterior(table)
    rng = np.random.default_rng(seed)
    smp = sample_rates(post, n_samples, rng)
    summary = summarize(smp, None)
    law_summary = {
        name: _interval(smp.v_P_bar_s(law)) for name, law in (laws or {}).items()
    }
    return {
        "key": res.spec.key,
        "tau_range": [float(table.log10_tau[0]), float(table.log10_tau[-1])],
        "grid": list(table.g.shape[:3]),
        "numx_fine": table.numx,
        "table_seconds": table.build_seconds,
        "table_from_cache": table.from_cache,
        "posterior_seconds": time.perf_counter() - t0,
        "edge_mass": post.edge_mass(),
        "summary_2023_law": summary,
        "mc_2023": saved_2023_summary(res.saved),
        "law_v_P_bar_s": law_summary,
    }
