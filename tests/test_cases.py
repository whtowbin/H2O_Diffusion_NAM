"""Tests for the Sullivan Ranch decompression cases registered in ``decompression/cases.py``.

Per case: (a) the registry agrees with the MATLAB it was read from; (b) the forward model equals
Octave running that case's unchanged ``olivineMC.m``; (c) replay of every saved ``final_results`` row
with the control-tolerance method of ``test_newcombe.py::test_t4_replay_of_saved_matlab_rows``.
MATLAB sources and data are read from ``$NEWCOMBE_REF`` and never copied into the repo.
Set ``CASES_NUMBERS_JSON`` to a path to have each test dump its measured numbers there.
Slow tests (replays) carry ``@pytest.mark.slow``; skip them with ``-m "not slow"``.
"""

from __future__ import annotations

import functools
import json
import math
import os
import shutil
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import brentq
from test_newcombe import result_lines, run_octave

from decompression import cases
from decompression import newcombe as nc

REF = cases.ref_root()
needs_ref = pytest.mark.skipif(
    REF is None, reason="$NEWCOMBE_REF reference folder not found"
)
needs_octave = pytest.mark.skipif(
    shutil.which("octave") is None, reason="octave is not installed"
)
KEYS = [c.key for c in cases.REGISTRY]
REPLAY_KEYS = [c.key for c in cases.REGISTRY if c.replay]
DONE_CASE = "Sullivan Ranch Olivine/H_in_ol_MCfitting Sullivan Ranch ol _No Stalling"


def record(name: str, numbers: dict) -> None:
    path = os.environ.get("CASES_NUMBERS_JSON")
    if not path:
        return
    p = Path(path)
    data = json.loads(p.read_text()) if p.exists() else {}
    data[name] = numbers
    p.write_text(json.dumps(data, indent=1))


@functools.cache
def get(key: str) -> cases.Resolved:
    assert REF is not None
    return cases.resolve(cases.BY_KEY[key], REF)


# ---------------------------------------------------------------- registry vs MATLAB
@needs_ref
@pytest.mark.parametrize("key", KEYS)
def test_registry_matches_matlab_sources(key):
    assert REF is not None
    assert cases.check_against_sources(cases.BY_KEY[key], REF) == []


@needs_ref
@pytest.mark.parametrize("key", KEYS)
def test_saved_rows_consistent_with_registry(key):
    """Hard bounds the saved rows must satisfy if the registry describes the run that made them."""
    r = get(key)
    s, a = r.spec, r.saved
    assert a.shape == (100, 6 + len(r.case.h2o_meas))
    lo, hi = r.case.logDH_bounds()
    out = {
        "P0_in_prior": bool(
            np.all((a[:, 3] >= s.Pmin * 0.999) & (a[:, 3] <= s.Pmax * 1.001))
        ),
        "logDH_in_bounds": bool(
            np.all((a[:, 5] >= min(lo, hi) - 1e-3) & (a[:, 5] <= max(lo, hi) + 1e-3))
        ),
        "x0orig_in_prior": bool(
            np.all(
                (a[:, 1] >= 10**s.logdPbydtmin * 0.999)
                & (a[:, 1] <= 10**s.logdPbydtmax * 1.001)
            )
        ),
        "Kd_median": float(np.median(a[:, 4])),
        "Kd_sd": float(np.std(a[:, 4], ddof=1)),
        "P0_min": float(a[:, 3].min()),
        "P0_max": float(a[:, 3].max()),
        "noise_sd": float(np.std(a[:, 6:] - r.case.h2o_meas, ddof=1)),
        "logDH_range": [float(a[:, 5].min()), float(a[:, 5].max())],
        "logDH_bounds": [lo, hi],
    }
    # saved noisy profile = measured profile + N(0, one_sig): column means within 4 standard errors
    z = (a[:, 6:].mean(0) - r.case.h2o_meas) / (s.one_sig / math.sqrt(len(a)))
    out["max_abs_z_profile"] = float(np.max(np.abs(z)))
    record(f"{key}/saved_vs_registry", out)
    assert out["logDH_in_bounds"] and out["x0orig_in_prior"]
    assert out["max_abs_z_profile"] < 4.5
    # the rejection test of the master script: Kd * melt(P0) > max(noisy H2O) for every saved row
    melt = nc.matlab_spline(r.case.solex_P, r.case.solex_H, a[:, 3])
    assert np.all(melt * a[:, 4] > a[:, 6:].max(1))


@needs_ref
@pytest.mark.parametrize("key", KEYS)
def test_saved_inputs_vs_script_priors(key):
    """Does the preserved master script's rejection sampler reproduce the saved Kd and P0? (recorded, not asserted)

    Draws (noisy profile, Kd ~ N, P0 ~ U) as master_script_test*.m does, keeps draws with
    Kd * melt(P0) > max(noisy profile), and compares with the saved columns by two-sample KS.
    A tiny p means the saved file was made with other settings than the preserved script.
    """
    from scipy.stats import ks_2samp

    r = get(key)
    s, c, a = r.spec, r.case, r.saved
    rng = np.random.default_rng(5)
    n = 200_000
    h = c.h2o_meas + rng.normal(0.0, s.one_sig, (n, len(c.h2o_meas)))
    kd = rng.normal(s.Kd_mean, s.Kd_sd, n)
    p0 = rng.random(n) * (s.Pmax - s.Pmin) + s.Pmin
    ok = nc.matlab_spline(c.solex_P, c.solex_H, p0) * kd > h.max(1)
    out = {
        "n_accepted": int(ok.sum()),
        "ks_p_Kd": float(ks_2samp(a[:, 4], kd[ok]).pvalue),
        "ks_p_P0": float(ks_2samp(a[:, 3], p0[ok]).pvalue),
    }
    record(f"{key}/saved_vs_script_priors", out)
    assert out["n_accepted"] > 1000


@needs_ref
def test_nostall_file_in_stalling_folder_is_the_done_case():
    r = get("ol_nostall_file_in_stalling_folder")
    done = next(Path(REF or ".").glob(DONE_CASE + "/final_results*.txt"))
    assert r.results.read_bytes() == done.read_bytes()
    assert (r.folder / "olivineMC.m").read_bytes() == (
        done.parent / "olivineMC.m"
    ).read_bytes()


# ---------------------------------------------------------------- forward model vs Octave
DRIVER = """
addpath('{model_dir}');
global H2O dist radius Solex_P Solex_H logDH Kd P0 Pf
S = dlmread('{solex}'); Solex_P = S(:,1); Solex_H = S(:,2);
dist = dlmread('{dist}'); dist = dist(:); radius = {radius!r};
Q = dlmread('{params}');
for k = 1:size(Q,1)
  dp = Q(k,1); P0 = Q(k,2); Pf = Q(k,3); logDH = Q(k,4); Kd = Q(k,5); H2O = Q(k,6:end)';
  v = olivineMC([dp, P0]);
  printf('\\nFWD %.17g\\n', v);
end
"""


def _numt(r: cases.Resolved, dp, P0, Pf, logDH) -> float:
    pf = r.case.Pf_fixed if r.case.Pf_fixed is not None else Pf
    dx = r.case.radius / (r.case.numx - 1)
    return math.ceil(2 * 1e12 * 10**logDH * ((P0 - pf) / dp) / (0.9 * dx * dx) + 1)


def forward_params(r: cases.Resolved, n_real: int = 8) -> np.ndarray:
    """Rows [dP/dt, P0, Pf, logDH, Kd, H2O...]: saved-row inputs with dP/dt varied, plus edge and sentinel sets."""
    rng = np.random.default_rng(1234)
    c, rows = r.case, []
    for i in rng.choice(len(r.saved), n_real, replace=False):
        s = r.saved[i]
        dp = s[2] * math.exp(rng.uniform(math.log(0.5), math.log(2.0)))
        pf = rng.uniform(c.Pfmin, c.Pfmax)
        while _numt(r, dp, s[3], pf, s[5]) > 3e4:  # keep the Octave loop affordable
            dp *= 2.0
        rows.append([dp, s[3], pf, s[5], s[4], *s[6:]])
    s = r.saved[0]  # edge: P0 at its maximum, Pf at its minimum, the saved fitted rate
    rows.append([s[2], c.Pmax, c.Pfmin, s[5], s[4], *c.h2o_meas])
    rows.append([1e-9, *rows[0][1:]])  # numt > 1e8 -> 1e6
    rows.append([1e12, *rows[0][1:]])  # numt < 10 -> 1e6
    return np.array(rows)


@needs_ref
@needs_octave
@pytest.mark.slow
@pytest.mark.parametrize(
    "key", REPLAY_KEYS
)  # the byte-copy case shares olivineMC.m with the done case (T2)
def test_forward_vs_octave_unchanged_olivineMC(key, tmp_path):
    r = get(key)
    params = forward_params(r)
    np.savetxt(tmp_path / "params.csv", params, delimiter=",", fmt="%.17g")
    np.savetxt(
        tmp_path / "solex.csv",
        np.c_[r.case.solex_P, r.case.solex_H],
        delimiter=",",
        fmt="%.17g",
    )
    np.savetxt(tmp_path / "dist.csv", r.case.dist, fmt="%.17g")
    script = DRIVER.format(
        model_dir=str(r.folder).replace("'", "''"), solex=tmp_path / "solex.csv", dist=tmp_path / "dist.csv",
        radius=r.case.radius, params=tmp_path / "params.csv",
    )  # fmt: skip
    out = run_octave(script, tmp_path)
    oct_vals = np.array([v[0] for v in result_lines(out, "FWD")])
    assert len(oct_vals) == len(params) >= 8

    mine = np.array([
        nc._misfit_for(r.case, p[5:], p[1], p[2], p[3], p[4])(np.array([p[0], p[1]])) for p in params
    ])  # fmt: skip
    rel = np.abs(mine - oct_vals) / np.abs(oct_vals)
    sentinel = oct_vals == 1e6
    record(f"{key}/forward_vs_octave", {
        "n": len(params), "max_rel": float(rel.max()), "n_sentinel_1e6": int(sentinel.sum()),
        "min_val": float(oct_vals[~sentinel].min()), "max_val": float(oct_vals[~sentinel].max()),
    })  # fmt: skip
    assert sentinel.sum() == 2  # exactly the two constructed sentinel sets
    assert rel.max() < 1e-9, (rel, mine, oct_vals)


# ---------------------------------------------------------------- replay of the saved rows
def _perturb(row: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Perturb inputs within +/- half a unit of their 5 significant-figure rounding."""
    out = row.copy()
    for j in [1, 3, 4, 5, *range(6, len(row))]:
        v = row[j]
        if v != 0:
            out[j] = v + rng.uniform(-1, 1) * 0.5 * 10.0 ** (
                math.floor(math.log10(abs(v))) - 4
            )
    return out


def _rel(a, b):
    return abs(a - b) / abs(b)


def _pf_roots(case: nc.Case, row: np.ndarray, grid: np.ndarray) -> list[float]:
    """Pf values with misfit(saved dP/dt; Pf) = saved fval (can be two; none -> the closest grid point)."""

    def g(pf: float) -> float:
        f = nc._misfit_for(case, row[6:], row[3], pf, row[5], row[4])
        return f(np.array([row[2], row[3]])) - row[0]

    gv = np.array([g(pf) for pf in grid])
    flips = np.flatnonzero(np.sign(gv[:-1]) * np.sign(gv[1:]) < 0)
    roots = [brentq(g, grid[k], grid[k + 1], xtol=1e-9 * grid[-1]) for k in flips]
    return roots or [float(grid[np.argmin(np.abs(gv))])]


@needs_ref
@pytest.mark.slow
@pytest.mark.parametrize("key", REPLAY_KEYS)
def test_replay_of_saved_matlab_rows(key):
    r = get(key)
    case, saved = r.case, r.saved
    n = len(saved)
    pf_mid = 0.5 * (case.Pfmin + case.Pfmax)

    # baseline-against-itself: two identical runs are bit-identical
    a, b = nc.fit_row(saved[0], case, pf_mid), nc.fit_row(saved[0], case, pf_mid)
    assert np.array_equal(a.x, b.x) and a.fval == b.fval and a.n_fev == b.n_fev

    base = [nc.fit_row(row, case, pf_mid) for row in saved]

    # control: 10 rows, Pf over the prior x perturbed inputs, vs the unperturbed mid-Pf run
    # (Pf is known for a fixed-Pf case, so there the Pf values collapse and more perturbations are used)
    rng = np.random.default_rng(7)
    pfs, reps = (
        ((pf_mid,), 9)
        if case.Pf_fixed is not None
        else ((case.Pfmin, pf_mid, case.Pfmax), 3)
    )
    ctrl = []
    for i in range(0, n, 10):
        dps, fvs = [], []
        for pf in pfs:
            for _ in range(reps):
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

    rel_dp = np.array([_rel(base[i].x[0], saved[i, 2]) for i in range(n)])
    rel_fv = np.array([_rel(base[i].fval, saved[i, 0]) for i in range(n)])
    out_dp, out_fv = np.flatnonzero(rel_dp > tol_dp), np.flatnonzero(rel_fv > tol_fv)

    # optimizer-independent: misfit at the SAVED fitted dP/dt, Pf scanned over its prior
    # (for a fixed-Pf case this is one number per row and must equal the saved fval)
    pf_grid = (
        [case.Pf_fixed]
        if case.Pf_fixed is not None
        else list(np.linspace(case.Pfmin, case.Pfmax, 26))
    )
    bracketed, best_rel = [], []
    for row in saved:
        vals = np.array([
            nc._misfit_for(case, row[6:], row[3], pf, row[5], row[4])(np.array([row[2], row[3]])) for pf in pf_grid
        ])  # fmt: skip
        bracketed.append(
            bool(vals.min() * (1 - 1e-3) <= row[0] <= vals.max() * (1 + 1e-3))
        )
        best_rel.append(float(np.min(np.abs(vals - row[0])) / row[0]))
    best_rel = np.array(best_rel)

    # recovered-Pf replay (only where Pf was not saved): solve misfit(saved dP/dt; Pf) = saved fval for Pf,
    # refit at each root. "best root" is the root whose replay is closest to the saved result, so it is an
    # upper bound on agreement; "first root" is an unbiased but arbitrary choice.
    rec = None
    if case.Pf_fixed is None:
        grid = np.linspace(case.Pfmin, case.Pfmax, 41)
        roots = [_pf_roots(case, row, grid) for row in saved]
        fits = [
            [nc.fit_row(row, case, pf) for pf in rts]
            for row, rts in zip(saved, roots, strict=True)
        ]
        d_dp = [
            [_rel(f.x[0], row[2]) for f in fs]
            for row, fs in zip(saved, fits, strict=True)
        ]
        k_best = [int(np.argmin(d)) for d in d_dp]
        r_best = np.array([d[k] for d, k in zip(d_dp, k_best, strict=True)])
        r_first = np.array([d[0] for d in d_dp])
        v_best = np.array(
            [
                _rel(fs[k].fval, row[0])
                for fs, k, row in zip(fits, k_best, saved, strict=True)
            ]
        )
        rec = {
            "n_rows_two_roots": int(sum(len(r_) == 2 for r_ in roots)),
            "dp_median_best": float(np.median(r_best)), "dp_max_best": float(r_best.max()),
            "dp_median_first": float(np.median(r_first)), "dp_max_first": float(r_first.max()),
            "fval_median_best": float(np.median(v_best)), "fval_max_best": float(v_best.max()),
            "rows_outside_dp_best": np.flatnonzero(r_best > tol_dp).tolist(),
            "rows_outside_fval_best": np.flatnonzero(v_best > tol_fv).tolist(),
            "rel_dp_best": r_best.tolist(), "rel_fval_best": v_best.tolist(),
        }  # fmt: skip

    record(f"{key}/replay", {
        "rows": n, "pf_used": pf_mid, "tol_dp": tol_dp, "tol_fval": tol_fv,
        "dp_median_rel": float(np.median(rel_dp)), "dp_max_rel": float(rel_dp.max()),
        "fval_median_rel": float(np.median(rel_fv)), "fval_max_rel": float(rel_fv.max()),
        "rows_outside_dp": out_dp.tolist(), "rows_outside_fval": out_fv.tolist(),
        "control": ctrl, "exitflags": [int(b.exitflag) for b in base],
        "n_fev_min": min(int(b.n_fev) for b in base), "n_fev_max": max(int(b.n_fev) for b in base),
        "n_bracketed": int(sum(bracketed)), "saved_dp_misfit_rel_best_median": float(np.median(best_rel)),
        "saved_dp_misfit_rel_best_max": float(best_rel.max()),
        "rel_dp": rel_dp.tolist(), "rel_fval": rel_fv.tolist(), "recovered_pf": rec,
    })  # fmt: skip
    assert np.median(rel_dp) <= tol_dp
