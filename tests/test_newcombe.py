"""Tests for the Newcombe decompression port.

The MATLAB sources and Sullivan Ranch data are read from ``$NEWCOMBE_REF`` at test time and are
never copied into the repository. Octave drivers are written to a temporary directory.
Set ``PORT_NUMBERS_JSON`` to a path to have each test dump its measured numbers there.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import brentq

from decompression import newcombe as nc

DEFAULT_REF = (
    "~/Projects/Ch3_Xenolith_SIMS_Clean/data/recovered/M14_ModelCode/Decompression modeling/"
    "Megan's Updated Code 2023 (Feb Pre-Defense-Hotter 1200 C) Archive"
)
CASE_SUBDIR = "Sullivan Ranch Olivine/H_in_ol_MCfitting Sullivan Ranch ol _No Stalling"
SAVED = "final_results_Sullivan_Ranch_Ol_2_Feb24_23.txt"


def _case_dir() -> Path | None:
    ref = Path(os.environ.get("NEWCOMBE_REF", DEFAULT_REF)).expanduser() / CASE_SUBDIR
    return ref if (ref / "olivineMC.m").is_file() else None


CASE_DIR = _case_dir()
HAVE_OCTAVE = shutil.which("octave") is not None
needs_ref = pytest.mark.skipif(
    CASE_DIR is None, reason="$NEWCOMBE_REF reference folder not found"
)
needs_octave = pytest.mark.skipif(not HAVE_OCTAVE, reason="octave is not installed")


def record(name: str, numbers: dict) -> None:
    path = os.environ.get("PORT_NUMBERS_JSON")
    if not path:
        return
    p = Path(path)
    data = json.loads(p.read_text()) if p.exists() else {}
    data[name] = numbers
    p.write_text(json.dumps(data, indent=1))


def run_octave(script: str, tmp_path: Path, timeout: int = 3000) -> str:
    f = tmp_path / "driver.m"
    f.write_text(script)
    out = subprocess.run(
        ["octave", "--no-gui", "--quiet", str(f)],
        capture_output=True, text=True, timeout=timeout, cwd=tmp_path, check=False,
    )  # fmt: skip
    if out.returncode != 0:
        raise RuntimeError(f"octave failed:\n{out.stderr[-2000:]}")
    return out.stdout


def result_lines(stdout: str, tag: str) -> list[list[float]]:
    return [
        [float(v) for v in ln.split()[1:]]
        for ln in stdout.splitlines()
        if ln.startswith(tag + " ")
    ]


@pytest.fixture(scope="module")
def case() -> nc.Case:
    assert CASE_DIR is not None
    return nc.load_case(CASE_DIR)


# ---------------------------------------------------------------- T1
@needs_octave
def test_t1_colon_and_spline_vs_octave(tmp_path):
    rng = np.random.default_rng(0)
    colons = [
        (0.0, 0.1, 1.0),
        (0.0, 0.1, 0.3),
        (0.0, 0.2, 1.0),
        (1.0, -0.1, 0.0),
        (0.0, 145 / 40, 145.0),
    ]
    colons += [
        (6000.0, -0.3, 2.5),
        (6000.0, -0.3, 0.0),
        (5000.0, -0.7, 0.0),
        (0.0, 1 / 3, 10.0),
    ]
    colons += [
        (0.1, 0.1, 0.7),
        (0.0, 0.7, 0.7),
        (5.0, 1.0, 5.0),
        (5.0, 1.0, 4.0),
        (0.0, 1e-3, 1.0),
    ]
    for _ in range(300):  # exactly the construction used by olivineMC
        P0, Pf = rng.uniform(5000, 6700), rng.uniform(0, 5)
        numt = math.ceil(10 ** rng.uniform(1, 6))
        dt = ((P0 - Pf) / rng.uniform(0.1, 10)) / (numt - 1)
        colons.append((P0, -dt * rng.uniform(0.1, 10), Pf))
    for _ in range(100):  # the x grid, other radii
        r = rng.uniform(50, 300)
        colons.append((0.0, r / 40, r))

    splines = []
    for _ in range(60):
        n = int(rng.integers(2, 12))
        x = rng.permutation(np.cumsum(rng.uniform(0.2, 3.0, n)))
        y = rng.normal(size=n) * 5
        xi = rng.uniform(x.min() - 3, x.max() + 3, 15)
        splines.append((x, y, xi))
    splines.append(
        (
            np.array([14.0, -28, -86, -115]),
            np.array([7.87, 7.38, 5.34, 4.43]),
            np.linspace(0, 115, 17),
        )
    )

    lines = ["fid = 1;"]
    colons = [(float(a), float(s), float(b)) for a, s, b in colons]
    for i, (a, s, b) in enumerate(colons):
        lines.append(
            f"r = {a!r}:{s!r}:{b!r}; if isempty(r), r = [NaN NaN]; end; printf('COL %d %.17g %.17g\\n', numel(r), r(1), r(end));"
        )
    for i, (x, y, xi) in enumerate(splines):
        fm = lambda v: "[" + " ".join(f"{float(t)!r}" for t in v) + "]"
        lines.append(
            f"yy = spline({fm(x)}, {fm(y)}, {fm(xi)}); printf('SPL'); printf(' %.17g', yy); printf('\\n');"
        )
    out = run_octave("\n".join(lines) + "\n", tmp_path)

    cols = result_lines(out, "COL")
    assert len(cols) == len(colons)
    worst_c = 0.0
    for (a, s, b), (n_oct, first, last) in zip(colons, cols, strict=True):
        n, _ = nc.colon_n_final(a, s, b)
        if n == 0:
            assert math.isnan(first)
            continue
        assert n == int(n_oct), (a, s, b)
        v = nc.matlab_colon(a, s, b)
        assert v[0] == first
        # last element can differ by 1 ulp of max(|a|,|b|) (Octave build may fuse multiply-add)
        worst_c = max(worst_c, abs(v[-1] - last) / max(abs(a), abs(b), 1e-300))
    assert worst_c < 1e-12

    spls = result_lines(out, "SPL")
    worst_s = 0.0
    for (x, y, xi), yy in zip(splines, spls, strict=True):
        mine = nc.matlab_spline(x, y, xi)
        # relative to the largest value of that spline, so near-zero crossings do not dominate
        worst_s = max(worst_s, float(np.max(np.abs(mine - yy)) / np.max(np.abs(yy))))
    assert worst_s < 1e-12
    record(
        "T1",
        {
            "n_colon": len(colons),
            "n_spline": len(splines),
            "colon_max_rel": worst_c,
            "spline_max_rel": worst_s,
        },
    )


# ---------------------------------------------------------------- T2
T2_DRIVER = """
addpath('{ref}');
global H2O dist radius Solex_P Solex_H logDH Kd P0 Pf
S = dlmread('{solex}'); Solex_P = S(:,1); Solex_H = S(:,2);
dist = dlmread('{dist}'); dist = dist(:); radius = {radius!r};
Q = dlmread('{params}');
for k = 1:size(Q,1)
  dp = Q(k,1); P0 = Q(k,2); Pf = Q(k,3); logDH = Q(k,4); Kd = Q(k,5); H2O = Q(k,6:end)';
  v = olivineMC([dp, P0]);
  printf('\\nT2 %.17g\\n', v);
end
"""


def _t2_params(case: nc.Case) -> np.ndarray:
    rng = np.random.default_rng(42)
    lo, hi = case.logDH_bounds()
    rows = []
    for _ in range(24):
        logdp = rng.uniform(
            -0.3, 1.0
        )  # slower rates are covered in the long-run set below
        rows.append(
            [10**logdp, rng.uniform(5000, 6700), rng.uniform(0, 5), rng.uniform(lo, hi),
             rng.normal(9, 0.5), *(case.h2o_meas + rng.normal(0, 0.25, 4))]
        )  # fmt: skip
    base = rows[0][1:]
    rows.append([1e-4, *base])  # numt > 1e8 -> 1e6
    rows.append([1e9, *base])  # numt < 10 -> 1e6
    rows.append(
        [10.0, 6700.0, 0.0, hi, 9.5, *case.h2o_meas]
    )  # edge: Pf = 0, max P0, fastest
    return np.array(rows)


@needs_ref
@needs_octave
def test_t2_forward_vs_octave_olivineMC(tmp_path, case):
    params = _t2_params(case)
    np.savetxt(tmp_path / "params.csv", params, delimiter=",", fmt="%.17g")
    np.savetxt(
        tmp_path / "solex.csv",
        np.c_[case.solex_P, case.solex_H],
        delimiter=",",
        fmt="%.17g",
    )
    np.savetxt(tmp_path / "dist.csv", case.dist, fmt="%.17g")
    script = T2_DRIVER.format(
        ref=str(CASE_DIR).replace("'", "''"), solex=tmp_path / "solex.csv", dist=tmp_path / "dist.csv",
        radius=case.radius, params=tmp_path / "params.csv",
    )  # fmt: skip
    out = run_octave(script, tmp_path)
    oct_vals = np.array([r[0] for r in result_lines(out, "T2")])
    assert len(oct_vals) == len(params) >= 20

    mine = []
    for r in params:
        f = nc._misfit_for(case, r[5:], r[1], r[2], r[3], r[4])
        mine.append(f(np.array([r[0], r[1]])))
    mine = np.array(mine)
    rel = np.abs(mine - oct_vals) / np.abs(oct_vals)
    # also: numpy-loop variant of the kernel is bit-identical to the numba one
    for r in params[:6]:
        kw = {
            "P0": r[1],
            "Pf": r[2],
            "radius": case.radius,
            "logDH": r[3],
            "Kd": r[4],
            "solex_P": case.solex_P,
            "solex_H": case.solex_H,
        }
        a, b = (
            nc.forward(r[0], use_numba=True, **kw),
            nc.forward(r[0], use_numba=False, **kw),
        )
        assert a is not None and b is not None
        assert np.array_equal(a[1], b[1])
    record(
        "T2",
        {
            "n": len(params),
            "max_rel": float(rel.max()),
            "n_sentinel_1e6": int(np.sum(oct_vals == 1e6)),
            "min_val": float(oct_vals.min()),
            "max_val": float(oct_vals[oct_vals < 1e6].max()),
        },
    )
    assert rel.max() < 1e-9, (rel, mine, oct_vals)

    # Octave forward twice is identical (baseline control)
    np.savetxt(tmp_path / "params.csv", params[:3], delimiter=",", fmt="%.17g")
    out2 = run_octave(script, tmp_path)
    assert result_lines(out2, "T2") == result_lines(out, "T2")[:3]


# ---------------------------------------------------------------- T3
def test_t3_fminsearch_sanity():
    rosen = lambda x: 100.0 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2
    r = nc.fminsearch(
        rosen, [-1.2, 1.0], tolx=1e-4, tolfun=1e-4, maxiter=400, maxfev=400
    )
    assert r.exitflag == 1
    assert np.allclose(r.x, [1, 1], atol=1e-3)
    assert r.fval < 1e-6
    # MATLAB documents ~85 iterations / 159 evaluations for this call (TolX=TolFun=1e-4)
    assert 70 <= r.n_iter <= 100 and 140 <= r.n_fev <= 180

    calls = []

    def quad(x):
        calls.append(np.array(x, dtype=float))
        return (x[0] - 3.0) ** 2  # dummy second dimension

    q = nc.fminsearch(quad, [0.5, 7.0], tolx=1e-4, tolfun=1e-4)
    assert q.x[0] == pytest.approx(3.0, abs=1e-3)
    # initial simplex: x0 and x0 with component j scaled by 1.05
    assert np.array_equal(calls[0], [0.5, 7.0])
    assert np.allclose(calls[1], [0.525, 7.0]) and np.allclose(calls[2], [0.5, 7.35])
    # zero component -> 0.00025
    z = []
    nc.fminsearch(
        lambda x: z.append(x.copy()) or float(np.sum(x**2)),
        [0.0, 1.0],
        maxiter=1,
        maxfev=3,
    )
    assert np.array_equal(z[1], [0.00025, 1.0])
    record(
        "T3",
        {
            "rosen_x": r.x.tolist(),
            "rosen_fval": r.fval,
            "rosen_iter": r.n_iter,
            "rosen_fev": r.n_fev,
            "quad_x": q.x.tolist(),
            "quad_fev": q.n_fev,
        },
    )


def test_fminsearch_bit_identical_repeat():
    f = lambda x: float(np.sin(x[0]) + 0.1 * x[0] ** 2 + 0.01 * x[1] ** 2)
    a = nc.fminsearch(f, [2.0, 1.0], tolx=1e-6, tolfun=1e-6)
    b = nc.fminsearch(f, [2.0, 1.0], tolx=1e-6, tolfun=1e-6)
    assert np.array_equal(a.x, b.x) and a.fval == b.fval and a.n_fev == b.n_fev


# ---------------------------------------------------------------- T4
def _perturb(row: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Perturb inputs within +/- half a unit of their 5 significant-figure rounding."""
    out = row.copy()
    for j in [1, 3, 4, 5, *range(6, len(row))]:
        v = row[j]
        half = 0.5 * 10.0 ** (math.floor(math.log10(abs(v))) - 4)
        out[j] = v + rng.uniform(-1, 1) * half
    return out


def _rel(a, b):
    return abs(a - b) / abs(b)


@needs_ref
def test_t4_replay_of_saved_matlab_rows(case):
    assert CASE_DIR is not None
    saved = nc.load_saved_results(CASE_DIR / SAVED)
    assert saved.shape == (100, 10)

    # --- baseline-against-itself: two identical runs are bit-identical
    a, b = nc.fit_row(saved[0], case), nc.fit_row(saved[0], case)
    assert np.array_equal(a.x, b.x) and a.fval == b.fval and a.n_fev == b.n_fev

    # --- control: 10 rows, 3 Pf values x 3 input perturbations each, vs the unperturbed Pf=2.5 run
    rng = np.random.default_rng(7)
    base = [nc.fit_row(r, case, 2.5) for r in saved]
    ctrl_rows = list(range(0, 100, 10))
    ctrl = []
    for i in ctrl_rows:
        dps, fvs = [], []
        for pf in (0.0, 2.5, 5.0):
            for _ in range(3):
                res = nc.fit_row(_perturb(saved[i], rng), case, pf)
                dps.append(res.x[0])
                fvs.append(res.fval)
        ctrl.append({
            "row": i,
            "dp_max_rel": max(_rel(v, base[i].x[0]) for v in dps),
            "fval_max_rel": max(_rel(v, base[i].fval) for v in fvs),
        })  # fmt: skip
    tol_dp = max(c["dp_max_rel"] for c in ctrl)
    tol_fv = max(c["fval_max_rel"] for c in ctrl)

    # --- replay vs saved MATLAB values
    rel_dp = np.array([_rel(base[i].x[0], saved[i, 2]) for i in range(100)])
    rel_fv = np.array([_rel(base[i].fval, saved[i, 0]) for i in range(100)])
    out_dp = np.flatnonzero(rel_dp > tol_dp)
    out_fv = np.flatnonzero(rel_fv > tol_fv)

    # --- optimizer-independent: misfit at the SAVED fitted dP/dt over Pf in [0, 5] (Pf was not
    # saved). A faithful forward model must reproduce the saved fval for some Pf in the prior.
    pf_grid = np.linspace(0.0, 5.0, 26)
    chk = {}
    bracketed, best_rel, pf_hat = [], [], []
    for r in saved:
        vals = np.array(
            [
                nc._misfit_for(case, r[6:], r[3], pf, r[5], r[4])(
                    np.array([r[2], r[3]])
                )
                for pf in pf_grid
            ]
        )
        bracketed.append(
            bool(vals.min() * (1 - 1e-3) <= r[0] <= vals.max() * (1 + 1e-3))
        )
        k = int(np.argmin(np.abs(vals - r[0])))
        best_rel.append(abs(vals[k] - r[0]) / r[0])
        pf_hat.append(float(pf_grid[k]))
        chk.setdefault("pf0", []).append(_rel(vals[0], r[0]))
        chk.setdefault("pf2.5", []).append(_rel(vals[13], r[0]))
        chk.setdefault("pf5", []).append(_rel(vals[-1], r[0]))
    best_rel = np.array(best_rel)

    # --- recovered-Pf replay: misfit(saved dP/dt; Pf) = saved fval can have two roots in Pf (the
    # misfit is not monotone in Pf). Refit at every root; "best root" is the one whose replay is
    # closest to the saved result, so it is an upper bound on agreement, not an unbiased estimate.
    pf_roots, rec_dp, rec_fv, rec_dp_first = [], [], [], []
    for i, r in enumerate(saved):
        g = lambda pf, r=r: (
            nc._misfit_for(case, r[6:], r[3], pf, r[5], r[4])(np.array([r[2], r[3]]))
            - r[0]
        )
        gv = np.array([g(pf) for pf in pf_grid])
        flips = np.flatnonzero(np.sign(gv[:-1]) * np.sign(gv[1:]) < 0)
        roots = [brentq(g, pf_grid[k], pf_grid[k + 1]) for k in flips] or [pf_hat[i]]
        fits = [nc.fit_row(r, case, float(pf_i)) for pf_i in roots]
        d_dp = [_rel(f.x[0], r[2]) for f in fits]
        k = int(np.argmin(d_dp))
        pf_roots.append([float(v) for v in roots])
        rec_dp.append(d_dp[k])
        rec_dp_first.append(d_dp[0])
        rec_fv.append(_rel(fits[k].fval, r[0]))
    rec_dp, rec_fv, rec_dp_first = (
        np.array(rec_dp),
        np.array(rec_fv),
        np.array(rec_dp_first),
    )

    record("T4", {
        "control": ctrl, "tol_dp": tol_dp, "tol_fval": tol_fv,
        "dp_median_rel": float(np.median(rel_dp)), "dp_max_rel": float(rel_dp.max()),
        "fval_median_rel": float(np.median(rel_fv)), "fval_max_rel": float(rel_fv.max()),
        "rows_outside_dp": out_dp.tolist(), "rows_outside_fval": out_fv.tolist(),
        "rel_dp": rel_dp.tolist(), "rel_fval": rel_fv.tolist(),
        "exitflags": [int(b.exitflag) for b in base], "n_fev": [int(b.n_fev) for b in base],
        "py_dp": [float(b.x[0]) for b in base], "py_fval": [float(b.fval) for b in base],
        "at_saved_fixed_pf": {k: {"median": float(np.median(v)), "max": float(np.max(v))} for k, v in chk.items()},
        "recovered_pf": {
            "pf_roots": pf_roots, "rel_dp_first_root": rec_dp_first.tolist(),
            "dp_median_first_root": float(np.median(rec_dp_first)), "dp_max_first_root": float(rec_dp_first.max()),
            "n_rows_two_roots": int(sum(len(v) == 2 for v in pf_roots)), "rel_dp": rec_dp.tolist(), "rel_fval": rec_fv.tolist(),
            "dp_median": float(np.median(rec_dp)), "dp_max": float(rec_dp.max()),
            "fval_median": float(np.median(rec_fv)), "fval_max": float(rec_fv.max()),
            "n_dp_below_1e-3": int(np.sum(rec_dp < 1e-3)), "n_dp_below_1e-4": int(np.sum(rec_dp < 1e-4)), "n_dp_outside_control_tol": int(np.sum(rec_dp > tol_dp)),
        },
        "n_bracketed_by_pf_0_to_5": int(sum(bracketed)), "best_grid_rel_median": float(np.median(best_rel)),
        "best_grid_rel_max": float(best_rel.max()), "pf_hat": pf_hat, "bracketed": bracketed,
    })  # fmt: skip
    assert sum(bracketed) >= 95
    assert (
        float(np.max(rec_dp)) < 5e-3
    )  # recovered-Pf replay (best root) agrees to ~5 s.f.
    assert np.median(rel_dp) <= tol_dp
