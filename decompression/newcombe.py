"""Faithful Python port of the Newcombe-style H-in-olivine decompression Monte Carlo fit.

Ported line for line from ``olivineMC.m`` / ``master_script_test.m`` (MATLAB) with no physics
changes. The MATLAB sources and the Sullivan Ranch data are not part of this repository; tests
read them from ``$NEWCOMBE_REF``.

Units: lengths in um, time in s, pressure in bar, concentrations in ppm (olivine) / wt% (melt).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numba import njit
from scipy.interpolate import CubicSpline

EPS = float(np.finfo(float).eps)
CHUNK = (
    1_000_000  # time steps per block; blocking only bounds memory, values are unchanged
)


# --------------------------------------------------------------------------------------
# MATLAB primitives
# --------------------------------------------------------------------------------------
def _tfloor(x: float, ct: float) -> float:
    """Tolerant floor (Hagerty's FL5), as used for the colon operator's element count."""
    q = 1.0
    rmax = q / (2.0 - ct)
    t1 = 1.0 + math.floor(x)
    t1 = (ct / q) * (-t1 if t1 < 0.0 else t1)
    t1 = min(rmax, t1)
    t1 = max(ct, t1)
    t1 = math.floor(x + t1)
    if x <= 0.0 or (t1 - x) < rmax:
        return t1
    return t1 - 1.0


def _teq(u: float, v: float, ct: float = 3.0 * EPS) -> bool:
    return abs(u - v) < max(abs(u), abs(v)) * ct


def colon_n_final(a: float, step: float, b: float) -> tuple[int, float]:
    """Number of elements and last element of the MATLAB range ``a:step:b``."""
    if (
        step == 0.0
        or math.isnan(step)
        or math.isnan(a)
        or math.isnan(b)
        or (b > a and step < 0.0)
        or (b < a and step > 0.0)
    ):
        return 0, a
    if a == b:
        return 1, a
    ct = 3.0 * EPS
    tmp = _tfloor((b - a + step) / step, ct)
    n = int(tmp) if tmp > 0.0 else 0
    # accept the neighbouring element count if it lands on the limit (Octave's rule)
    if not _teq(a + (n - 1) * step, b):
        if _teq(a + (n - 2) * step, b):
            n -= 1
        elif _teq(a + n * step, b):
            n += 1
    if n <= 0:
        return 0, a
    final = a + (n - 1) * step
    if (step > 0.0 and final >= b) or (step < 0.0 and final <= b):
        final = b
    return n, final


def colon_slice(
    a: float, step: float, n: int, final: float, lo: int, hi: int
) -> np.ndarray:
    """Elements ``lo:hi`` (python indices) of the range ``a:step:b`` with n elements."""
    k = np.arange(lo, hi, dtype=float)
    out = a + k * step
    if hi == n and hi > lo:
        out[-1] = final
    if lo == 0 and hi > lo:
        out[0] = a
    return out


def matlab_colon(a: float, step: float, b: float) -> np.ndarray:
    """MATLAB ``a:step:b`` including its floating-point end tolerance."""
    n, final = colon_n_final(a, step, b)
    return colon_slice(a, step, n, final, 0, n)


def matlab_spline(x, y, xi) -> np.ndarray:
    """MATLAB ``spline(x, y, xi)``: not-a-knot cubic with extrapolation.

    n = 2 gives a line and n = 3 a parabola through the points (what not-a-knot reduces to,
    and what MATLAB documents). x is sorted first.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    order = np.argsort(x, kind="stable")
    x, y = x[order], y[order]
    return CubicSpline(x, y, bc_type="not-a-knot", extrapolate=True)(
        np.asarray(xi, dtype=float)
    )


# --------------------------------------------------------------------------------------
# Forward model (olivineMC.m)
# --------------------------------------------------------------------------------------
@njit(cache=True)
def _advance_numba(c, bc, r):
    n = c.shape[0]
    cur = c.copy()
    nxt = np.empty_like(cur)
    for j in range(bc.shape[0]):
        for i in range(1, n - 1):
            nxt[i] = cur[i] + r * (cur[i + 1] - 2.0 * cur[i] + cur[i - 1])
        nxt[n - 1] = bc[j]
        nxt[0] = nxt[1]
        cur, nxt = nxt, cur
    return cur


def _advance_numpy(c: np.ndarray, bc: np.ndarray, r: float) -> np.ndarray:
    """Vectorised-over-x reference of the same loop (same arithmetic order)."""
    cur = c.copy()
    nxt = np.empty_like(cur)
    for b in bc:
        nxt[1:-1] = cur[1:-1] + r * (cur[2:] - 2.0 * cur[1:-1] + cur[:-2])
        nxt[-1] = b
        nxt[0] = nxt[1]
        cur, nxt = nxt, cur
    return cur


def forward(
    dPbydt: float,
    *,
    P0: float,
    Pf: float,
    radius: float,
    logDH: float,
    Kd: float,
    solex_P,
    solex_H,
    numx: int = 41,
    use_numba: bool = True,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Explicit FTCS diffusion with a degassing boundary; returns (x, C_final) or None.

    None corresponds to olivineMC returning 1e6 (too many or too few time steps).
    """
    advance = _advance_numba if use_numba else _advance_numpy
    DH2O = 1e12 * 10.0**logDH  # um^2/s
    with np.errstate(all="ignore"):
        t_tot = np.float64(P0 - Pf) / np.float64(dPbydt)
        dx = radius / (numx - 1)
        numt_f = np.ceil((2 * DH2O * t_tot) / (0.9 * (dx * dx)) + 1)
    if numt_f > 10**8 or numt_f < 10:
        return None
    if not np.isfinite(numt_f):  # NaN: MATLAB would fall through and then error
        raise ValueError("non-finite number of time steps")
    numt = int(numt_f)
    t_tot = float(t_tot)
    dt = t_tot / (numt - 1)
    dP = -dt * dPbydt
    n_p, p_final = colon_n_final(P0, dP, Pf)
    if n_p < numt:
        raise IndexError(f"Pvec has {n_p} elements but {numt} time steps are needed")
    delta_term = DH2O * dt / (dx * dx)

    c = np.empty(0)
    for lo in range(0, numt, CHUNK):
        hi = min(lo + CHUNK, numt)
        pvec = colon_slice(P0, dP, n_p, p_final, lo, hi)
        bc = matlab_spline(solex_P, solex_H, pvec) * Kd
        if lo == 0:
            c = np.full(numx, bc[0])
            bc = bc[1:]
        c = advance(c, bc, delta_term)

    x = matlab_colon(0.0, dx, radius)
    if len(x) != numx:
        raise ValueError("x grid length differs from numx")
    return x, c


def misfit(
    x, *, h2o, dist, P0, Pf, radius, logDH, Kd, solex_P, solex_H, numx: int = 41
) -> float:
    """olivineMC(x0): mean squared residual; x[1] is ignored, as in the MATLAB."""
    res = forward(
        float(x[0]), P0=P0, Pf=Pf, radius=radius, logDH=logDH, Kd=Kd,
        solex_P=solex_P, solex_H=solex_H, numx=numx,
    )  # fmt: skip
    if res is None:
        return 1000000.0
    xg, cfin = res
    model = matlab_spline(xg, cfin, np.abs(dist))
    return float(np.sum((np.asarray(h2o) - model) ** 2) / len(model))


# --------------------------------------------------------------------------------------
# fminsearch (MathWorks implementation of Lagarias et al. 1998)
# --------------------------------------------------------------------------------------
@dataclass
class FminResult:
    x: np.ndarray
    fval: float
    exitflag: int
    n_iter: int
    n_fev: int


def _eps_of(v: float) -> float:
    return float(np.spacing(abs(v)))


def fminsearch(
    f: Callable[[np.ndarray], float],
    x0,
    tolx: float = 1e-4,
    tolfun: float = 1e-4,
    maxiter: int | None = None,
    maxfev: int | None = None,
) -> FminResult:
    xin = np.asarray(x0, dtype=float).ravel()
    n = xin.size
    maxiter = 200 * n if maxiter is None else maxiter
    maxfev = 200 * n if maxfev is None else maxfev
    rho, chi, psi, sigma = 1.0, 2.0, 0.5, 0.5

    v = np.zeros((n, n + 1))
    fv = np.zeros(n + 1)
    v[:, 0] = xin
    fv[0] = f(xin.copy())
    usual_delta, zero_term_delta = 0.05, 0.00025
    for j in range(n):
        y = xin.copy()
        y[j] = (1 + usual_delta) * y[j] if y[j] != 0 else zero_term_delta
        v[:, j + 1] = y
        fv[j + 1] = f(y.copy())
    order = np.argsort(fv, kind="stable")
    fv, v = fv[order], v[:, order]
    itercount = 1
    func_evals = n + 1

    while func_evals < maxfev and itercount < maxiter:
        if np.max(np.abs(fv[0] - fv[1:])) <= max(
            tolfun, 10 * _eps_of(fv[0])
        ) and np.max(np.abs(v[:, 1:] - v[:, [0]])) <= max(
            tolx, 10 * _eps_of(np.max(v[:, 0]))
        ):
            break
        xbar = np.sum(v[:, :n], axis=1) / n
        xr = (1 + rho) * xbar - rho * v[:, -1]
        fxr = f(xr.copy())
        func_evals += 1
        if fxr < fv[0]:
            xe = (1 + rho * chi) * xbar - rho * chi * v[:, -1]
            fxe = f(xe.copy())
            func_evals += 1
            if fxe < fxr:
                v[:, -1], fv[-1] = xe, fxe
            else:
                v[:, -1], fv[-1] = xr, fxr
        elif fxr < fv[n - 1]:
            v[:, -1], fv[-1] = xr, fxr
        else:
            shrink = False
            if fxr < fv[-1]:
                xc = (1 + psi * rho) * xbar - psi * rho * v[:, -1]
                fxc = f(xc.copy())
                func_evals += 1
                if fxc <= fxr:
                    v[:, -1], fv[-1] = xc, fxc
                else:
                    shrink = True
            else:
                xcc = (1 - psi) * xbar + psi * v[:, -1]
                fxcc = f(xcc.copy())
                func_evals += 1
                if fxcc < fv[-1]:
                    v[:, -1], fv[-1] = xcc, fxcc
                else:
                    shrink = True
            if shrink:
                for j in range(1, n + 1):
                    v[:, j] = v[:, 0] + sigma * (v[:, j] - v[:, 0])
                    fv[j] = f(v[:, j].copy())
                func_evals += n
        order = np.argsort(fv, kind="stable")
        fv, v = fv[order], v[:, order]
        itercount += 1

    exitflag = 1 if (func_evals < maxfev and itercount < maxiter) else 0
    return FminResult(v[:, 0].copy(), float(fv[0]), exitflag, itercount, func_evals)


# --------------------------------------------------------------------------------------
# Case, row replay and Monte Carlo driver (master_script_test.m)
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class DiffLaw:
    """D = A * exp(-Ea / (R * T_K)) * mult, in m^2/s, with the MATLAB operator order."""

    A: float
    Ea: float
    mult: float = 1.0

    def log10_at(self, T_K: float) -> float:
        return math.log10(self.A * math.exp(-self.Ea / (8.314 * T_K)) * self.mult)


@dataclass
class Case:
    dist: np.ndarray
    h2o_meas: np.ndarray
    solex_P: np.ndarray
    solex_H: np.ndarray
    radius: float = 290 / 2
    T_C: float = 1200.0
    Pmin: float = 5000.0
    Pmax: float = 6700.0
    Pfmin: float = 0.0
    Pfmax: float = 5.0
    logdPbydtmin: float = -1.0
    logdPbydtmax: float = 1.0
    Kd_mean: float = 9.0
    Kd_sd: float = 0.5
    one_sig: float = 0.25
    tolx: float = 1.0
    tolfun: float = 1.0
    numx: int = 41
    # DHhi is evaluated at T_K + 20 and DHlo at T_K - 20 (master_script_test*.m), each with its own law
    dh_hi: DiffLaw = DiffLaw(10**-5.4, 130000.0)
    dh_lo: DiffLaw = DiffLaw(10**-5.4, 130000.0)
    # a case whose olivineMC.m hard-codes ``Pf = 0`` overrides the global Pf drawn by the master script
    Pf_fixed: float | None = None

    def logDH_bounds(self) -> tuple[float, float]:
        """(logDHlo, logDHhi) for T_K -/+ 20 K."""
        T_K = self.T_C + 273
        return self.dh_lo.log10_at(T_K - 20), self.dh_hi.log10_at(T_K + 20)


def _read_numeric(path: Path) -> np.ndarray:
    return np.loadtxt(path, dtype=float, ndmin=2)


def load_case(ref_dir: str | Path) -> Case:
    """Load the Sullivan Ranch inputs from the reference folder."""
    ref = Path(ref_dir)
    prof = _read_numeric(ref / "Sullivan_Ranch_Ol_2.txt")
    solex = _read_numeric(ref / "VesiCal1200_VaporBubble_H2O_only.txt")
    return Case(
        dist=prof[:, 0], h2o_meas=prof[:, 1], solex_P=solex[:, 0], solex_H=solex[:, 1]
    )


def load_saved_results(path: str | Path) -> np.ndarray:
    return np.loadtxt(path, delimiter=",", dtype=float, ndmin=2)


def _misfit_for(case: Case, h2o, P0, Pf, logDH, Kd) -> Callable[[np.ndarray], float]:
    if case.Pf_fixed is not None:
        Pf = case.Pf_fixed

    def f(x: np.ndarray) -> float:
        return misfit(
            x, h2o=h2o, dist=case.dist, P0=P0, Pf=Pf, radius=case.radius, logDH=logDH, Kd=Kd,
            solex_P=case.solex_P, solex_H=case.solex_H, numx=case.numx,
        )  # fmt: skip

    return f


def fit_row(row, case: Case, Pf: float = 2.5) -> FminResult:
    """Replay one saved MATLAB run: [fval, x0orig, xfit, P0, Kd, logDH, H2O...]."""
    row = np.asarray(row, dtype=float)
    x0orig, P0, Kd, logDH, h2o = row[1], row[3], row[4], row[5], row[6:]
    f = _misfit_for(case, h2o, P0, Pf, logDH, Kd)
    return fminsearch(f, [x0orig, P0], tolx=case.tolx, tolfun=case.tolfun)


def run_mc(
    case: Case, rng: np.random.Generator, p: int = 100
) -> tuple[np.ndarray, np.ndarray]:
    """Mirror of master_script_test.m. Returns (final_results[p, 6 + len(H2O)], Pf[p])."""
    logDHlo, logDHhi = case.logDH_bounds()
    m = len(case.h2o_meas)
    out = np.zeros((p, 6 + m))
    pf_all = np.zeros(p)
    for k in range(p):
        h2o = case.h2o_meas + rng.normal(0.0, case.one_sig, size=m)  # addnoise
        while True:  # "crazy" rejection loop
            logDH = logDHhi - rng.random() * (logDHhi - logDHlo)
            Kd = rng.normal(case.Kd_mean, case.Kd_sd)
            logdp = (
                rng.random() * (case.logdPbydtmax - case.logdPbydtmin)
                + case.logdPbydtmin
            )
            x0 = np.array([10.0**logdp, 0.0])
            P0 = rng.random() * (case.Pmax - case.Pmin) + case.Pmin
            Pf = rng.random() * (case.Pfmax - case.Pfmin) + case.Pfmin
            x0[1] = P0
            melt_in = matlab_spline(case.solex_P, case.solex_H, P0)
            if float(melt_in) * Kd > np.max(h2o):
                break
        f = _misfit_for(case, h2o, P0, Pf, logDH, Kd)
        res = fminsearch(f, x0, tolx=case.tolx, tolfun=case.tolfun)
        out[k, :6] = [res.fval, x0[0], res.x[0], P0, Kd, logDH]
        out[k, 6:] = h2o
        pf_all[k] = Pf
    return out, pf_all
