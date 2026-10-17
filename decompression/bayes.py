"""Bayesian decompression-rate fit (METHODS.md section 4).

The profile depends on (D, v_P) only through tau = D (P0 - Pf) / (v_P L^2) (property 1) and is linear
in the partition coefficient K (property 2). So ``G(|d_i|/L; tau, P0, Pf)`` is tabulated once per case
and degassing path with K = 1 (``g_table``) and K is integrated out in closed form
(``kd_marginal_loglike``). The priors sit on the physical parameters, log10 v_P (MPa/s), log D, P0 and
Pf; ``grid_posterior`` evaluates the posterior on a grid in log v_P x log D nodes x (P0, Pf) nodes
with the K-marginal log-likelihood interpolated in log tau. Profiles that do not bound the rate are
flagged as one-sided (METHODS.md 4.6). L is fixed (METHODS.md 2, consequence 3).

Units: lengths um, time s, pressure bar, D um^2/s inside the solver and m^2/s in ``DiffLaw``;
v_P is bar/s except where a name says MPa/s.
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
from scipy.interpolate import CubicSpline
from scipy.stats import norm

from decompression.cases import Resolved
from decompression.newcombe import Case, DiffLaw, forward, matlab_spline

CACHE_DIR = Path(__file__).parent / ".g_cache"  # gitignored
TABLE_VERSION = 3  # bump when the table definition changes (invalidates the cache)
G_GRAV = 9.81  # m/s^2
BAR_PER_MPA = 10.0
PA_PER_BAR = 1e5


# --------------------------------------------------------------------------------------
# G table
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class GTable:
    """``profile[i_tau, i_P0, i_Pf, i_x]`` = C(xi_x, t_tot) / K on the whole solver grid xi = x / L in [0, 1].

    log10 tau, P0 and Pf nodes include both ends (the posterior interpolates the profile linearly in P0 and Pf). The table does not depend on L or on
    where the data sit: ``g_at(xi)`` reads it at any positions (METHODS.md 4.4).
    """

    case: Case
    log10_tau: np.ndarray
    P0: np.ndarray
    Pf: np.ndarray
    profile: np.ndarray
    numx: int
    build_seconds: float = 0.0
    from_cache: bool = False

    @property
    def xi(self) -> np.ndarray:
        return np.linspace(0.0, 1.0, self.numx)

    def g_at(self, xi) -> np.ndarray:
        """Profile / K at positions ``xi`` (not-a-knot cubic spline, as ``misfit`` reads the model): (tau, P0, Pf, len(xi))."""
        return CubicSpline(
            self.xi, self.profile, axis=-1, bc_type="not-a-knot", extrapolate=True
        )(np.asarray(xi, float))

    @property
    def dlog_tau(self) -> float:
        return float(self.log10_tau[1] - self.log10_tau[0])  # nodes are equally spaced


def _cells(lo: float, hi: float, n: int) -> np.ndarray:
    """Centres of n equal cells spanning [lo, hi] (midpoint rule for a uniform prior)."""
    return lo + (np.arange(n) + 0.5) * (hi - lo) / n


def _pf_nodes(case: Case, n_Pf: int, centres: bool) -> np.ndarray:
    if case.Pf_fixed is not None:
        return np.array([float(case.Pf_fixed)])
    if centres:
        return _cells(case.Pfmin, case.Pfmax, n_Pf)
    return np.linspace(case.Pfmin, case.Pfmax, n_Pf)


def _interp_matrix(nodes: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Linear-interpolation weights (len(x), len(nodes)); the identity when x equals the nodes."""
    if len(nodes) == 1:
        return np.ones((len(x), 1))
    idx = np.clip(np.searchsorted(nodes, x) - 1, 0, len(nodes) - 2)
    t = (x - nodes[idx]) / (nodes[idx + 1] - nodes[idx])
    w = np.zeros((len(x), len(nodes)))
    w[np.arange(len(x)), idx] = 1.0 - t
    w[np.arange(len(x)), idx + 1] = t
    return w


def _profile_full(
    log10_tau, P0, Pf, radius, numx, sP, sH
) -> tuple[np.ndarray, np.ndarray] | None:
    """Direct forward run with v_P = 1 bar/s, K = 1 and D = tau L^2 / (P0 - Pf): (x um, C)."""
    D = 10.0**log10_tau * radius**2 / (P0 - Pf)  # um^2/s
    return forward(
        1.0, P0=P0, Pf=Pf, radius=radius, logDH=float(np.log10(D / 1e12)), Kd=1.0,
        solex_P=sP, solex_H=sH, numx=numx,
    )  # fmt: skip


def g_direct(
    case: Case,
    log10_tau: float,
    P0: float,
    Pf: float,
    numx: int,
    path=None,
    *,
    radius: float | None = None,
    x_um=None,
) -> np.ndarray:
    """Profile / K at ``x_um`` (default |case.dist|) from one direct ``forward`` run with half-width ``radius``
    (default case.radius): used by tests and simulations."""
    sP, sH = _path_arrays(case, path)
    radius = case.radius if radius is None else radius
    x_um = np.abs(case.dist) if x_um is None else x_um
    res = _profile_full(log10_tau, P0, Pf, radius, numx, sP, sH)
    if res is None:
        raise ValueError(f"forward returned no profile at log10 tau = {log10_tau}")
    return matlab_spline(res[0], res[1], x_um)


def _path_arrays(case: Case, path) -> tuple[np.ndarray, np.ndarray]:
    if path is None:
        return np.asarray(case.solex_P, float), np.asarray(case.solex_H, float)
    return np.asarray(path[0], float), np.asarray(path[1], float)


def _g_block(args) -> np.ndarray:
    lts, P0, Pf, radius, numx, sP, sH = args
    out = np.empty((len(lts), numx))
    for k, lt in enumerate(lts):
        res = _profile_full(lt, P0, Pf, radius, numx, sP, sH)
        if res is None:
            raise ValueError(f"forward returned no profile at log10 tau = {lt}")
        out[k] = res[1]
    return out


def _table_key(sP, sH, numx, lts, P0s, Pfs) -> str:
    h = hashlib.sha256(f"gtable-v{TABLE_VERSION}|{numx}".encode())
    for a in (sP, sH, lts, P0s, Pfs):
        h.update(np.ascontiguousarray(a, dtype=float).tobytes())
        h.update(b"|")
    return h.hexdigest()[:24]


def solver_tau_floor(numx: int) -> float:
    """Smallest log10 tau the solver accepts at ``numx`` nodes (it needs >= 10 time steps), with a margin."""
    return float(np.log10(4.05 / (numx - 1) ** 2) + 0.08)


def default_tau_range(numx: int, hi: float = 2.5) -> tuple[float, float]:
    """Table range: from the solver floor to ``hi`` (cost grows with tau; ~ numx^2 tau steps per solve).

    This is a numerical limit, not a likelihood-based choice: the priors imply tau up to ~10^3.6, and
    ``Posterior.bounds`` flags the cases whose likelihood is still high at an edge.
    """
    return solver_tau_floor(numx), hi


def g_table(
    case: Case,
    path: tuple[np.ndarray, np.ndarray] | None = None,
    *,
    numx: int | None = None,
    n_tau: int = 400,
    n_P0: int = 5,
    n_Pf: int | None = None,
    tau_range: tuple[float, float] | None = None,
    n_jobs: int | None = None,
    cache: bool = True,
    refresh: bool = False,
    n_chunks: int = 6,
) -> GTable:
    """Tabulate G on the log10 tau x P0 x Pf grid (K = 1) with the explicit solver.

    ``path`` is (P bar, H2O wt%) of the degassing path (default: the case's own). ``numx`` defaults to
    2 (case.numx - 1) + 1 (4x moves log v_P quantiles by <= 0.0017 dex on three cases). The cache (gitignored, keyed by a hash of the inputs) holds only G, never data;
    ``refresh`` rebuilds and overwrites an existing entry (for timing). ``n_Pf`` defaults to 2 when the
    Pf prior spans <= 10 bar and 6 otherwise.
    """
    t0 = time.perf_counter()
    numx = 2 * (case.numx - 1) + 1 if numx is None else numx
    sP, sH = _path_arrays(case, path)
    if (
        n_Pf is None
    ):  # 2 nodes suffice for Pf in 0-5 bar; 0-200/400 bar needs more (log v_P moves 0.03-0.07 dex at 2)
        n_Pf = 2 if case.Pfmax - case.Pfmin <= 10 else 6
    lo, hi = tau_range if tau_range is not None else default_tau_range(numx)
    if lo < solver_tau_floor(numx) - 0.08:
        raise ValueError(f"log10 tau {lo} is below the solver floor at numx={numx}")
    lts = np.linspace(lo, hi, n_tau)
    P0s = np.linspace(
        case.Pmin, case.Pmax, n_P0
    )  # nodes include both ends; the posterior interpolates
    Pfs = _pf_nodes(case, n_Pf, centres=False)
    key = _table_key(sP, sH, numx, lts, P0s, Pfs)
    file = CACHE_DIR / f"{key}.npz"
    if cache and not refresh and file.exists():
        with np.load(file) as z:
            prof = z["profile"]
        return GTable(
            case, lts, P0s, Pfs, prof, numx, time.perf_counter() - t0, from_cache=True
        )

    # interleaved tau subsets per (P0, Pf) node balance the load (cost grows with tau)
    chunks = [np.arange(c, n_tau, n_chunks) for c in range(n_chunks)]
    jobs = [
        (lts[idx], p0, pf, case.radius, numx, sP, sH)
        for p0 in P0s
        for pf in Pfs
        for idx in chunks
    ]
    n_jobs = os.cpu_count() or 1 if n_jobs is None else n_jobs
    if n_jobs > 1:
        # one thread per worker: threaded BLAS/Accelerate inside 12 workers oversubscribes the cores
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
    prof = np.empty((n_tau, len(P0s), len(Pfs), numx))
    it = iter(blocks)
    for j in range(len(P0s)):
        for k in range(len(Pfs)):
            for idx in chunks:
                prof[idx, j, k, :] = next(it)
    if cache:
        CACHE_DIR.mkdir(exist_ok=True)
        np.savez_compressed(file, profile=prof)
    return GTable(case, lts, P0s, Pfs, prof, numx, time.perf_counter() - t0)


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


# --------------------------------------------------------------------------------------
# Priors on D
# --------------------------------------------------------------------------------------
def log10_D_m2s(law: DiffLaw, T_K) -> np.ndarray:
    """Vectorised ``DiffLaw.log10_at`` (same operator order)."""
    T_K = np.asarray(T_K, float)
    return np.log10(law.A * np.exp(-law.Ea / (8.314 * T_K)) * law.mult)


@dataclass(frozen=True)
class DPrior:
    """log10 D (um^2/s) at equal-probability nodes; ``width`` is the log10 D extent of each node's cell."""

    name: str
    log10_D: np.ndarray
    width: np.ndarray


def bracket_prior(case: Case, n: int = 21, name: str = "2023_bracket") -> DPrior:
    """The 2023 prior: log10 D uniform between the case's DHlo (T - 20 K) and DHhi (T + 20 K) laws."""
    lo, hi = case.logDH_bounds()  # log10 m^2/s
    edges = np.linspace(lo, hi, n + 1)
    return DPrior(name, 12.0 + 0.5 * (edges[1:] + edges[:-1]), np.diff(edges))


def law_prior(
    law: DiffLaw, case: Case, n: int = 21, T_halfwidth: float = 20.0, name: str = ""
) -> DPrior:
    """A named law with T ~ U(T0 - T_halfwidth, T0 + T_halfwidth) K, T0 = T_C + 273 as in the MATLAB."""
    T0 = case.T_C + 273.0
    edges = T0 + np.linspace(-T_halfwidth, T_halfwidth, n + 1)
    logd = 12.0 + log10_D_m2s(law, edges)
    centres = 12.0 + log10_D_m2s(law, 0.5 * (edges[1:] + edges[:-1]))
    return DPrior(name, centres, np.abs(np.diff(logd)))


# --------------------------------------------------------------------------------------
# Prior on L and where the data sit in xi
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class LPrior:
    """Half-width L (um) at equal-probability nodes; ``width`` is each node's cell width (0 when fixed)."""

    name: str
    L: np.ndarray
    width: np.ndarray


def l_fixed(case: Case) -> LPrior:
    """The 2023 value, fixed (reproduces 2023)."""
    return LPrior("fixed", np.array([float(case.radius)]), np.array([0.0]))


def l_prior(case: Case, kind: str = "uniform", rel: float = 0.2, n: int = 11) -> LPrior:
    """L ~ U(L0 (1 - rel), L0 (1 + rel)) or N(L0, (rel L0)^2) (cells truncated at 4 sigma), L0 = case.radius."""
    if kind == "uniform":
        edges = case.radius * (1.0 + rel * np.linspace(-1.0, 1.0, n + 1))
    elif kind == "normal":
        z = np.clip(norm.ppf(np.linspace(0.0, 1.0, n + 1)), -4.0, 4.0)
        edges = case.radius * (1.0 + rel * z)
    else:
        raise ValueError(f"kind must be 'uniform' or 'normal', got {kind!r}")
    return LPrior(f"{kind}_{rel:g}", 0.5 * (edges[1:] + edges[:-1]), np.diff(edges))


def xi_at_L(case: Case, L: float, rim_referenced: bool) -> np.ndarray:
    """Positions xi_i in the table for half-width L (METHODS.md 4.4).

    Centre-referenced distances (registry ``dist_b == 0``): xi = |d| / L. Rim-referenced
    (``dist_b != 0``): the data sit at d_rim = L0 - |dist| from the rim, xi = 1 - d_rim / L, where
    L0 is the 2023 half-width the registry's ``dist`` was built with; a point past the centre
    (d_rim > L) is mirrored, |xi|. At L = L0 both are |dist| / L0.
    """
    d = np.abs(np.asarray(case.dist, float))
    if rim_referenced:
        return np.abs(
            1.0 - (case.radius - d) / L
        )  # a point past the centre mirrors (symmetric slab)
    return d / L


# --------------------------------------------------------------------------------------
# Posterior on (L, log v_P, log D, P0, Pf)
# --------------------------------------------------------------------------------------
V_RANGE_MPA = (-3.0, 4.0)  # default prior: log10 v_P in MPa/s, i.e. 1e-4 to 1e3 MPa/s
LOG10_BAR_PER_MPA = 1.0


@dataclass(frozen=True)
class Posterior:
    """Normalised posterior probabilities on cells [L node, log v_P, log D node, P0 cell, Pf cell]."""

    table: GTable
    dprior: DPrior
    lprior: LPrior
    P0: np.ndarray  # posterior cell centres (the table is interpolated onto them)
    Pf: np.ndarray
    dP0: float
    dPf: float
    log10_v_MPa: np.ndarray  # cell centres
    prob: np.ndarray
    valid_L: (
        np.ndarray
    )  # False: a data point falls outside the crystal (xi > 1), zero likelihood
    ell: (
        np.ndarray
    )  # table K-marginal log-likelihood (L, tau, P0, Pf), -inf where invalid
    a: np.ndarray
    b: np.ndarray  # K | cell ~ N(b/a, 1/a), same shape as ell
    mu_K: float
    s_K: float

    @property
    def dv(self) -> float:
        return float(self.log10_v_MPa[1] - self.log10_v_MPa[0])

    def marginal(self, axis: str) -> np.ndarray:
        keep = {"L": 0, "v": 1, "D": 2, "P0": 3, "Pf": 4}[axis]
        return self.prob.sum(axis=tuple(i for i in range(5) if i != keep))

    def prior_log10_tau_range(self) -> tuple[float, float]:
        """Range of log10 tau the priors can reach (cell edges of log v_P, log D and the valid L; full P0, Pf ranges)."""
        c = self.table.case
        pf_lo = c.Pf_fixed if c.Pf_fixed is not None else c.Pfmin
        pf_hi = c.Pf_fixed if c.Pf_fixed is not None else c.Pfmax
        v_lo = self.log10_v_MPa[0] - self.dv / 2 + LOG10_BAR_PER_MPA
        v_hi = self.log10_v_MPa[-1] + self.dv / 2 + LOG10_BAR_PER_MPA
        d_lo = float(np.min(self.dprior.log10_D - self.dprior.width / 2))
        d_hi = float(np.max(self.dprior.log10_D + self.dprior.width / 2))
        L_ok = self.lprior.L[self.valid_L]
        w_ok = self.lprior.width[self.valid_L]
        L_lo = float(np.min(L_ok - w_ok / 2))
        L_hi = float(np.max(L_ok + w_ok / 2))
        lo = d_lo + np.log10(c.Pmin - pf_hi) - 2.0 * np.log10(L_hi) - v_hi
        hi = d_hi + np.log10(c.Pmax - pf_lo) - 2.0 * np.log10(L_lo) - v_lo
        return float(lo), float(hi)

    def bounds(self, within: float = 2.0) -> dict:
        """One-sided detection (METHODS.md 4.6).

        ``low_tau_edge`` / ``high_tau_edge`` are True when the prior can reach beyond that table edge and the
        K-marginal log-likelihood there is within ``within`` of its maximum over the reachable table nodes
        (all valid L). Small tau = fast ascent (a lower bound on v_P), large tau = slow ascent (an upper bound).
        """
        lt = self.table.log10_tau
        p_lo, p_hi = self.prior_log10_tau_range()
        reach = (lt >= p_lo) & (lt <= p_hi)
        top = float(self.ell[:, reach].max())
        low = bool(p_lo < lt[0] and self.ell[:, 0].max() >= top - within)
        high = bool(p_hi > lt[-1] and self.ell[:, -1].max() >= top - within)
        flag = {
            (False, False): "bounded",
            (True, False): "lower",
            (False, True): "upper",
            (True, True): "unconstrained",
        }[(low, high)]
        return {
            "flag": flag,
            "low_tau_edge": low,
            "high_tau_edge": high,
            "prior_log10_tau": [p_lo, p_hi],
            "table_log10_tau": [float(lt[0]), float(lt[-1])],
            "valid_L_fraction": float(self.valid_L.mean()),
        }


def grid_posterior(
    table: GTable,
    h=None,
    s=None,
    *,
    dprior: DPrior | None = None,
    lprior: LPrior | None = None,
    rim_referenced: bool = False,
    v_range: tuple[float, float] = V_RANGE_MPA,
    n_v: int = 350,
    n_P0: int = 15,
    n_Pf: int = 6,
    P0=None,
    Pf=None,
    mu_K: float | None = None,
    s_K: float | None = None,
    sigma_m: float = 0.0,
) -> Posterior:
    """Posterior of METHODS.md 4.3 on a grid in L x log10 v_P (MPa/s) x log D nodes x (P0, Pf) nodes.

    For each L the table is read at xi_i(L) (``xi_at_L``; ``rim_referenced`` = registry ``dist_b != 0``);
    a value of L that puts a point outside the crystal (xi > 1) has zero likelihood. The K-marginal log-likelihood and
    the K posterior terms (a, b) are interpolated linearly in log10 tau (clamped at the table ends), with
    log tau = log D + log (P0 - Pf) - 2 log L - log v_P. The profile table (nodes at the ends of the P0 and Pf
    ranges) is interpolated linearly onto ``n_P0`` x ``n_Pf`` prior cells (or onto explicit ``P0``/``Pf`` values). Defaults: the case's data, ``one_sig``, Kd prior,
    the 2023 D bracket and L fixed at the 2023 value. s_i^2 = sigma_i^2 + sigma_m^2.
    """
    c = table.case
    h = np.asarray(c.h2o_meas if h is None else h, float)
    s = np.broadcast_to(np.asarray(c.one_sig if s is None else s, float), h.shape)
    mu_K = c.Kd_mean if mu_K is None else mu_K
    s_K = c.Kd_sd if s_K is None else s_K
    dprior = bracket_prior(c) if dprior is None else dprior
    lprior = l_fixed(c) if lprior is None else lprior
    s_tot = np.sqrt(s**2 + sigma_m**2)

    if P0 is None:
        P0 = _cells(c.Pmin, c.Pmax, n_P0)
        dP0 = (c.Pmax - c.Pmin) / n_P0
    else:
        P0, dP0 = np.asarray(P0, float), 0.0
    if Pf is None:
        Pf = _pf_nodes(c, n_Pf, centres=True)
        dPf = 0.0 if c.Pf_fixed is not None else (c.Pfmax - c.Pfmin) / n_Pf
    else:
        Pf, dPf = np.asarray(Pf, float), 0.0
    W0, Wf = _interp_matrix(table.P0, P0), _interp_matrix(table.Pf, Pf)

    nL, nT = len(lprior.L), len(table.log10_tau)
    tab_shape = (nL, nT, len(P0), len(Pf))
    ell_L = np.full(tab_shape, -np.inf)
    a_L, b_L = np.ones(tab_shape), np.ones(tab_shape)
    valid = np.zeros(nL, bool)
    for l, L in enumerate(lprior.L):
        xi = xi_at_L(c, float(L), rim_referenced)
        if np.any(xi > 1.0):
            continue
        valid[l] = True
        g = np.einsum(
            "qk,tpkd->tpqd", Wf, np.einsum("pj,tjkd->tpkd", W0, table.g_at(xi))
        )
        ell_L[l], a_L[l], b_L[l] = kd_marginal_loglike(
            h, s_tot, g, mu_K, s_K, return_ab=True
        )
    if not valid.any():
        raise ValueError("no L node keeps every data point inside the crystal")

    edges = np.linspace(v_range[0], v_range[1], n_v + 1)
    v = 0.5 * (edges[1:] + edges[:-1])
    ell5 = np.full((nL, n_v, len(dprior.log10_D), len(P0), len(Pf)), -np.inf)
    for l in np.flatnonzero(valid):
        L2 = 2.0 * np.log10(lprior.L[l])
        for j, p0 in enumerate(P0):
            for k, pf in enumerate(Pf):
                # log tau = log D + log (P0 - Pf) - 2 log L - log v_P (bar/s), as (v, D)
                x = (
                    dprior.log10_D[None, :]
                    + np.log10(p0 - pf)
                    - L2
                    - (v[:, None] + LOG10_BAR_PER_MPA)
                )
                ell5[l, :, :, j, k] = np.interp(x, table.log10_tau, ell_L[l, :, j, k])
    p = np.exp(
        ell5 - ell5.max()
    )  # uniform priors on log v_P, the D, L, P0 and Pf nodes
    return Posterior(
        table,
        dprior,
        lprior,
        P0,
        Pf,
        dP0,
        dPf,
        v,
        p / p.sum(),
        valid,
        ell_L,
        a_L,
        b_L,
        mu_K,
        s_K,
    )


# --------------------------------------------------------------------------------------
# Draws and summaries
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Samples:
    log10_v_P_bar_s: np.ndarray
    log10_D_um2s: np.ndarray
    P0: np.ndarray
    Pf: np.ndarray
    K: np.ndarray
    rho: np.ndarray  # kg/m^3
    log10_tau: np.ndarray
    L: np.ndarray  # um

    @property
    def v_P_bar_s(self) -> np.ndarray:
        return 10.0**self.log10_v_P_bar_s

    @property
    def v_P_MPa_s(self) -> np.ndarray:
        return self.v_P_bar_s / BAR_PER_MPA

    @property
    def u_m_s(self) -> np.ndarray:
        """Ascent speed v_P / (rho g)."""
        return self.v_P_bar_s * PA_PER_BAR / (self.rho * G_GRAV)


def sample_rates(
    post: Posterior,
    n: int,
    rng: np.random.Generator,
    *,
    rho: tuple[float, float] = (2800.0, 100.0),
    jitter: bool = True,
) -> Samples:
    """Draw cells from the posterior (uniform within a cell if ``jitter``), K | cell, and rho."""
    t = post.table
    flat = rng.choice(post.prob.size, size=n, p=post.prob.ravel())
    l, i, d, j, k = np.unravel_index(flat, post.prob.shape)

    def draw(centre, width):
        return centre + (rng.random(n) - 0.5) * width if jitter else centre

    logv = draw(post.log10_v_MPa[i], post.dv) + LOG10_BAR_PER_MPA
    logd = draw(post.dprior.log10_D[d], post.dprior.width[d])
    L = draw(post.lprior.L[l], post.lprior.width[l])
    P0 = draw(post.P0[j], post.dP0)
    Pf = draw(post.Pf[k], post.dPf)
    # K | cell: (a, b) interpolated at the cell's node log tau, per (L, P0, Pf) table slice
    node_lt = (
        post.dprior.log10_D[d]
        + np.log10(post.P0[j] - post.Pf[k])
        - 2.0 * np.log10(post.lprior.L[l])
        - (post.log10_v_MPa[i] + LOG10_BAR_PER_MPA)
    )
    a, b = np.empty(n), np.empty(n)
    key = (l * len(post.P0) + j) * len(post.Pf) + k
    for u in np.unique(key):
        m = key == u
        a[m] = np.interp(node_lt[m], t.log10_tau, post.a[l[m][0], :, j[m][0], k[m][0]])
        b[m] = np.interp(node_lt[m], t.log10_tau, post.b[l[m][0], :, j[m][0], k[m][0]])
    K = b / a + rng.standard_normal(n) / np.sqrt(a)
    lt = logd + np.log10(P0 - Pf) - 2.0 * np.log10(L) - logv
    return Samples(logv, logd, P0, Pf, K, rng.normal(rho[0], rho[1], n), lt, L)


def _interval(x: np.ndarray) -> dict[str, float]:
    q = np.quantile(x, [0.025, 0.16, 0.5, 0.84, 0.975])
    return dict(
        zip(("lo95", "lo68", "median", "hi68", "hi95"), map(float, q), strict=True)
    )


def _rate_summary(x: np.ndarray, flag: str) -> dict:
    """Median and intervals if the rate is bounded; else the 95 % one-sided bound and the flag (4.6)."""
    if flag == "bounded":
        return {"flag": flag, **_interval(x)}
    if flag == "lower":  # 95 % of the posterior lies above this
        return {"flag": flag, "bound95": float(np.quantile(x, 0.05))}
    if flag == "upper":
        return {"flag": flag, "bound95": float(np.quantile(x, 0.95))}
    return {"flag": flag}


def summarize(samples: Samples, bounds: dict | None = None) -> dict[str, dict]:
    """Median and 68/95 % intervals of log10 v_P, v_P (MPa/s, bar/s), u (m/s), K, P0, Pf, log10 tau.

    The rate quantities carry ``flag`` ("bounded", "lower", "upper", "unconstrained"); when it is not
    "bounded" they hold the one-sided 95 % bound (``bound95``) instead of a median.
    """
    flag = "bounded" if bounds is None else bounds["flag"]
    out = {
        "log10_v_P_bar_s": _rate_summary(samples.log10_v_P_bar_s, flag),
        "v_P_MPa_s": _rate_summary(samples.v_P_MPa_s, flag),
        "v_P_bar_s": _rate_summary(samples.v_P_bar_s, flag),
        "u_m_s": _rate_summary(samples.u_m_s, flag),
    }
    for name, x in (
        ("K", samples.K),
        ("P0_bar", samples.P0),
        ("Pf_bar", samples.Pf),
        ("log10_tau", samples.log10_tau),
        ("L_um", samples.L),
    ):
        out[name] = _interval(x)
    return out


# --------------------------------------------------------------------------------------
# One case end to end (used by the results script and the reporting test)
# --------------------------------------------------------------------------------------
def saved_2023_summary(saved: np.ndarray) -> dict[str, float]:
    """Median and IQR of the saved 2023 MC fits (column 2 = fitted dP/dt, bar/s)."""
    v = saved[:, 2]
    q1, med, q3 = np.quantile(v, [0.25, 0.5, 0.75])
    return {"median": float(med), "q1": float(q1), "q3": float(q3), "n": len(v)}


def posterior_summary(
    post: Posterior, n_samples: int, seed: int
) -> tuple[dict, dict, float]:
    """(summary, bounds, seconds) for one posterior, with ``n_samples`` draws."""
    t0 = time.perf_counter()
    bounds = post.bounds()
    smp = sample_rates(post, n_samples, np.random.default_rng(seed))
    return summarize(smp, bounds), bounds, time.perf_counter() - t0


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
    """Table, posterior under the 2023 bracket, 2023 MC summary, and posteriors under each named law.

    Named laws use T ~ U(T0 +/- 20) K; each is a separate (cheap) posterior on the same table.
    """
    case = res.case
    rim = res.spec.dist_b != 0  # registry convention: rim-referenced distances
    table = g_table(case, cache=cache, n_jobs=n_jobs, **table_kw)
    t0 = time.perf_counter()
    post = grid_posterior(table, rim_referenced=rim)
    summary, bounds, _ = posterior_summary(post, n_samples, seed)
    post_seconds = time.perf_counter() - t0
    t1 = time.perf_counter()
    law_out = {}
    for name, law in (laws or {}).items():
        lp = grid_posterior(
            table, dprior=law_prior(law, case, name=name), rim_referenced=rim
        )
        s_l, b_l, _ = posterior_summary(lp, n_samples, seed)
        law_out[name] = {"summary": s_l, "bounds": b_l}
    t2 = time.perf_counter()
    l_out = {}
    for kind, rel in (
        ("uniform", 0.2),
    ):  # L-prior sensitivity: 2023 bracket, L ~ U(L0 +/- 20 %)
        lpost = grid_posterior(
            table, lprior=l_prior(case, kind, rel), rim_referenced=rim
        )
        s_l, b_l, _ = posterior_summary(lpost, n_samples, seed)
        l_out[f"{kind}_{rel:g}"] = {"summary": s_l, "bounds": b_l}
    return {
        "key": res.spec.key,
        "tau_range": [float(table.log10_tau[0]), float(table.log10_tau[-1])],
        "grid": list(table.profile.shape[:3]),
        "numx_fine": table.numx,
        "table_seconds": table.build_seconds,
        "table_from_cache": table.from_cache,
        "posterior_seconds": post_seconds,
        "law_posteriors_seconds": t2 - t1,
        "l_posteriors_seconds": time.perf_counter() - t2,
        "L_sensitivity": l_out,
        "bounds": bounds,
        "summary_2023_bracket": summary,
        "mc_2023": saved_2023_summary(res.saved),
        "laws": law_out,
    }
