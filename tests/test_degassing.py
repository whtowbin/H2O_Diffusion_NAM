"""Tests for `decompression.degassing` (VESIcal degassing paths as the melt-H2O boundary).

Feb 2023 VESIcal exports, melt-inclusion compositions and the Newcombe reference folder are read at
test time from the paths below (override with $DEGASSING_EXPORTS, $MI_CSV, $NEWCOMBE_REF) and are never
copied into the repository. Set ``DEGASSING_NUMBERS_JSON`` to dump the measured numbers to a file.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from decompression import degassing as dg
from decompression import newcombe as nc

CH3 = Path("~/Projects/Ch3_Xenolith_SIMS_Clean/data").expanduser()
EXPORTS = Path(
    os.environ.get(
        "DEGASSING_EXPORTS", CH3 / "recovered/M09_Degassing/Degassing Models"
    )
).expanduser()
MI_CSV = Path(
    os.environ.get("MI_CSV", CH3 / "compiled/csv/T3_MeltInclusions.csv")
).expanduser()
REF = Path(
    os.environ.get(
        "NEWCOMBE_REF",
        CH3 / "recovered/M14_ModelCode/Decompression modeling/"
        "Megan's Updated Code 2023 (Feb Pre-Defense-Hotter 1200 C) Archive",
    )
).expanduser()
CASE_DIR = (
    REF / "Sullivan Ranch Olivine/H_in_ol_MCfitting Sullivan Ranch ol _No Stalling"
)
SAVED = CASE_DIR / "final_results_Sullivan_Ranch_Ol_2_Feb24_23.txt"
SOLEX_TXT = CASE_DIR / "VesiCal1200_VaporBubble_H2O_only.txt"

# Export file -> VESIcal model name. Prefix-less files are MagmaSat and cannot be regenerated here.
EXPORT_FILES = {
    "Dixon": "Dixon_GCB_Closed_Degassing_1200C_Highest_Measured_Inclusion.xlsx",
    "ShishkinaIdealMixing": "Shishkina_GCB_Closed_Degassing_1200C_Highest_Measured_Inclusion.xlsx",
    "IaconoMarziano": "IaconoMarziano_GCB_Closed_Degassing_1200C_Highest_Measured_Inclusion.xlsx",
}
# Anhydrous-normalised oxide columns of T3_MeltInclusions.csv.
NORM_COLS = {
    f"{o.upper()}normANHYD": o
    for o in (
        "SiO2",
        "TiO2",
        "Al2O3",
        "Fe2O3",
        "FeO",
        "MnO",
        "MgO",
        "CaO",
        "Na2O",
        "K2O",
        "P2O5",
    )
}


def record(name: str, numbers: dict) -> None:
    path = os.environ.get("DEGASSING_NUMBERS_JSON")
    if not path:
        return
    p = Path(path)
    data = json.loads(p.read_text()) if p.exists() else {}
    data[name] = numbers
    p.write_text(json.dumps(data, indent=1))


needs_exports = pytest.mark.skipif(
    not (EXPORTS.is_dir() and MI_CSV.is_file()),
    reason="Feb 2023 exports or MI csv not found",
)
needs_ref = pytest.mark.skipif(
    not (SOLEX_TXT.is_file() and SAVED.is_file()),
    reason="$NEWCOMBE_REF reference folder not found",
)


def gcb_mean_composition() -> dict[str, float]:
    """Composition the Feb 2023 'Highest_Measured_Inclusion' runs started from.

    Mean over the GCB inclusions of the anhydrous-normalised oxides (``*normANHYD`` columns of
    T3_MeltInclusions.csv), with H2O = max of the GCB H2O column (wt%) and CO2 = max of the GCB CO2
    column (ppm) truncated to 4 decimals in wt%. Found by fitting: the Dixon model sees SiO2 only, and
    the export is reproduced to ~2e-4 wt% H2O at SiO2 = mean(SIO2normANHYD); no single inclusion does.
    """
    t3 = pd.read_csv(MI_CSV, encoding="utf-8-sig")
    g = t3[t3.Locality == "GCB"]
    comp = {oxide: float(g[col].mean()) for col, oxide in NORM_COLS.items()}
    comp["H2O"] = float(g["H2O"].max())
    comp["CO2"] = math.floor(float(g["CO2"].max()) / 1e4 * 1e4) / 1e4
    return comp


def load_export(model: str) -> pd.DataFrame:
    return pd.read_excel(EXPORTS / EXPORT_FILES[model], index_col=0)


def regenerate(
    model: str, comp: dict, ref: pd.DataFrame, cache_dir=None
) -> pd.DataFrame:
    return dg.degassing_path(
        comp, temperature_C=1200, model=model, start_pressure=float(ref.Pressure_bars.iloc[0]),
        final_pressure=float(ref.Pressure_bars.iloc[-1]), steps=len(ref), cache_dir=cache_dir,
    )  # fmt: skip


def h2o_on(ref_pressure: np.ndarray, df: pd.DataFrame) -> np.ndarray:
    p, h = df.Pressure_bars.to_numpy(), df.H2O_liq.to_numpy()
    return np.interp(ref_pressure, p[::-1], h[::-1])


def diff_stats(h_new: np.ndarray, h_ref: np.ndarray) -> dict:
    d = np.abs(h_new - h_ref)
    return {"max_abs": float(d.max()), "max_rel": float((d / h_ref).max())}


# ------------------------------------------------------------------ unit tests
def test_magmasat_raises_clear_error():
    with pytest.raises(RuntimeError, match="ThermoEngine"):
        dg.degassing_path(
            {"SiO2": 46, "H2O": 3, "CO2": 0.3}, temperature_C=1200, model="MagmaSat"
        )


def test_init_vapor_rejected_for_python_models():
    with pytest.raises(NotImplementedError):
        dg.degassing_path(
            {"SiO2": 46, "H2O": 3, "CO2": 0.3},
            temperature_C=1200,
            model="Dixon",
            init_vapor=1.0,
        )


def test_cache_key_depends_on_every_input():
    comp = {"SiO2": 46.0, "H2O": 3.0, "CO2": 0.3}

    def key(c=comp, **change) -> str:
        args = {
            "temperature_C": 1200.0, "model": "Dixon", "fractionate_vapor": 0.0, "init_vapor": 0.0,
            "final_pressure": 1.0, "steps": 50, "start_pressure": "saturation", **change,
        }  # fmt: skip
        return dg.cache_key(
            c, temperature_C=args["temperature_C"], model=args["model"],
            fractionate_vapor=args["fractionate_vapor"], init_vapor=args["init_vapor"],
            final_pressure=args["final_pressure"], steps=args["steps"],
            start_pressure=args["start_pressure"],
        )  # fmt: skip

    base = key()
    # case-insensitive oxide names; non-oxide analytes are ignored
    assert key({"SIO2": 46.0, "h2o": 3.0, "CO2": 0.3, "S": 1400.0}) == base
    for change in (
        {"temperature_C": 1100.0}, {"model": "ShishkinaIdealMixing"}, {"fractionate_vapor": 0.5},
        {"final_pressure": 2.0}, {"steps": 51}, {"start_pressure": 5000.0},
    ):  # fmt: skip
        assert key(**change) != base, change
    assert key({**comp, "SiO2": 46.0001}) != base


def test_cache_roundtrip_and_hit_skips_vesical(tmp_path, monkeypatch):
    comp = {
        "SiO2": 46.0, "TIO2": 2.0, "AL2O3": 15.0, "FEO": 8.0, "MGO": 7.0, "CAO": 11.0,
        "NA2O": 3.0, "K2O": 1.5, "H2O": 3.0, "CO2": 0.3, "S": 1400.0,
    }  # fmt: skip

    def run() -> pd.DataFrame:
        return dg.degassing_path(
            comp, temperature_C=1200, model="Dixon", start_pressure=3000.0, final_pressure=100.0,
            steps=20, cache_dir=tmp_path,
        )  # fmt: skip

    first = run()
    assert len(list(tmp_path.glob("*.csv"))) == 1
    monkeypatch.setitem(
        sys.modules, "VESIcal", None
    )  # a recompute would fail on import
    pd.testing.assert_frame_equal(first, run(), check_exact=True)


def test_fractionate_vapor_changes_the_path(tmp_path):
    comp = {
        "SiO2": 46.0, "TIO2": 2.0, "AL2O3": 15.0, "FEO": 8.0, "MGO": 7.0, "CAO": 11.0,
        "NA2O": 3.0, "K2O": 1.5, "H2O": 3.0, "CO2": 0.3,
    }  # fmt: skip
    ends = {}
    for frac in (0.0, 0.5, 1.0):
        df = dg.degassing_path(
            comp, temperature_C=1200, model="Dixon", start_pressure=3000.0, final_pressure=100.0,
            steps=40, fractionate_vapor=frac, cache_dir=tmp_path,
        )  # fmt: skip
        ends[frac] = float(df.H2O_liq.iloc[-1])
    assert len(list(tmp_path.glob("*.csv"))) == 3  # distinct cache entries
    # open-system removal of the vapour keeps more H2O in the melt at the end: monotone in fractionation
    assert ends[0.0] < ends[0.5] < ends[1.0]


def test_boundary_arrays_order_and_anchor():
    df = pd.DataFrame(
        {"Pressure_bars": [3000.0, 2000.0, 1.0], "H2O_liq": [3.0, 2.5, 0.1]}
    )
    p, h = dg.boundary_arrays(df)
    assert p.tolist() == [3000.0, 2000.0, 1.0, 0.0] and h.tolist() == [
        3.0,
        2.5,
        0.1,
        0.0,
    ]
    p, h = dg.boundary_arrays(df.iloc[::-1])
    assert p.tolist() == [3000.0, 2000.0, 1.0, 0.0]
    p, h = dg.boundary_arrays(df, anchor_zero=False)
    assert p.tolist() == [3000.0, 2000.0, 1.0]


@needs_ref
def test_boundary_arrays_match_txt_layout():
    txt = np.loadtxt(SOLEX_TXT)
    assert txt[-1].tolist() == [0.0, 0.0] and np.all(np.diff(txt[:-1, 0]) < 0)
    export = pd.read_excel(EXPORTS / "GCB_Closed_Degassing_1200C_CO2 Highest Bubble_Corrected.xlsx",
                           index_col=0)  # fmt: skip
    p, h = dg.boundary_arrays(export)
    assert p.shape == txt[:, 0].shape
    # txt files carry 8 significant figures
    np.testing.assert_allclose(p, txt[:, 0], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(h, txt[:, 1], rtol=1e-6, atol=1e-8)


# ------------------------------------------------------------------ T1: regenerate Feb 2023 paths
@needs_exports
def test_identified_start_composition_dixon_fit():
    """SiO2 of the identified composition is the only thing Dixon sees; no single T3 inclusion fits."""
    comp = gcb_mean_composition()
    ref = load_export("Dixon")
    # the volatiles are the export's own first row
    assert (comp["H2O"], comp["CO2"]) == (ref.H2O_liq.iloc[0], ref.CO2_liq.iloc[0])
    t3 = pd.read_csv(MI_CSV, encoding="utf-8-sig")
    best_single = min(
        diff_stats(
            h2o_on(ref.Pressure_bars.to_numpy(),
                   regenerate("Dixon", {**comp, "SiO2": s}, ref, cache_dir=None)),
            ref.H2O_liq.to_numpy(),
        )["max_abs"]
        for s in t3[t3.Locality == "GCB"].SIO2
    )  # fmt: skip
    mean_fit = diff_stats(
        h2o_on(ref.Pressure_bars.to_numpy(), regenerate("Dixon", comp, ref)),
        ref.H2O_liq.to_numpy(),
    )
    record(
        "identify_dixon",
        {"best_single_inclusion_raw_SiO2_max_abs": best_single, **mean_fit},
    )
    assert mean_fit["max_abs"] < 1e-3 < best_single


@needs_exports
@pytest.mark.parametrize("model", list(EXPORT_FILES))
def test_regenerate_feb2023_export(model, tmp_path):
    comp = gcb_mean_composition()
    ref = load_export(model)
    p_ref, h_ref = ref.Pressure_bars.to_numpy(), ref.H2O_liq.to_numpy()

    a = regenerate(model, comp, ref)
    # control 1: the identical call twice (cache off) is bit-identical
    b = regenerate(model, comp, ref)
    pd.testing.assert_frame_equal(a, b, check_exact=True)
    # control 2: sensitivity to rounding of the input composition (2 decimals, then +0.5 % on all oxides)
    rounded = {k: round(v, 2) for k, v in comp.items()}
    plus = {
        k: v * (1 + 0.005) if k not in ("H2O", "CO2") else v for k, v in comp.items()
    }
    sens = {
        "round_2dp": diff_stats(h2o_on(p_ref, regenerate(model, rounded, ref)), h2o_on(p_ref, a)),
        "oxides_plus_0.5pct": diff_stats(h2o_on(p_ref, regenerate(model, plus, ref)), h2o_on(p_ref, a)),
    }  # fmt: skip
    # control 3: independent +-2 % jitter of every oxide (4 seeded draws): spread of the misfit to the
    # saved path if the true composition differed from the identified one by that much
    rng = np.random.default_rng(3)
    jit = []
    for _ in range(4):
        c = {
            k: v * (1 + rng.uniform(-0.02, 0.02)) if k not in ("H2O", "CO2") else v
            for k, v in comp.items()
        }
        jit.append(
            diff_stats(h2o_on(p_ref, regenerate(model, c, ref)), h_ref)["max_abs"]
        )
    sens["jitter_2pct_vs_saved_max_abs_range"] = [min(jit), max(jit)]
    # the comparison itself: regenerated path vs saved export, on the saved pressures
    stats = diff_stats(h2o_on(p_ref, a), h_ref)
    same_grid = diff_stats(a.H2O_liq.to_numpy(), h_ref)
    co2 = float(
        np.max(
            np.abs(
                np.interp(p_ref, a.Pressure_bars[::-1], a.CO2_liq[::-1]) - ref.CO2_liq
            )
        )
    )
    record(
        f"regenerate_{model}",
        {"vs_saved": stats, "row_by_row": same_grid, "max_abs_CO2_liq": co2, "control_sensitivity": sens,
         "vesical": dg.vesical_version(), "n_rows": len(ref)},
    )  # fmt: skip
    print(f"\n{model}: vs saved {stats}; controls {sens}; CO2_liq max abs {co2:.2e}")
    # Regression pins, not agreement claims: Dixon reproduces the export; Shishkina and
    # IaconoMarziano differ by more than composition rounding can explain (see DEGASSING_REPORT.md).
    assert (
        stats["max_abs"]
        < {"Dixon": 1e-3, "ShishkinaIdealMixing": 0.05, "IaconoMarziano": 0.3}[model]
    )


# ------------------------------------------------------------------ T2: plug into the forward model
def dixon_boundary(cache_dir) -> tuple[np.ndarray, np.ndarray]:
    df = dg.degassing_path(
        gcb_mean_composition(), temperature_C=1200, model="Dixon", start_pressure=7000.0,
        final_pressure=1.0, steps=200, cache_dir=cache_dir,
    )  # fmt: skip
    return dg.boundary_arrays(df)


def misfit_for(case: nc.Case, row: np.ndarray, solex_P, solex_H):
    """Misfit closure for one saved row (Pf = 2.5, as in the port's replay)."""
    h2o, P0, Kd, logDH = row[6:], float(row[3]), float(row[4]), float(row[5])

    def f(x):
        return nc.misfit(
            x, h2o=h2o, dist=case.dist, P0=P0, Pf=2.5, radius=case.radius, logDH=logDH, Kd=Kd,
            solex_P=solex_P, solex_H=solex_H, numx=case.numx,
        )  # fmt: skip

    return f


@needs_exports
@needs_ref
def test_dixon_path_runs_through_forward_and_misfit(tmp_path):
    solex_P, solex_H = dixon_boundary(tmp_path)
    case = nc.load_case(CASE_DIR)
    row = nc.load_saved_results(SAVED)[0]
    res = nc.forward(
        float(row[2]), P0=float(row[3]), Pf=2.5, radius=case.radius, logDH=float(row[5]),
        Kd=float(row[4]), solex_P=solex_P, solex_H=solex_H, numx=case.numx,
    )  # fmt: skip
    assert res is not None and np.all(np.isfinite(res[1]))
    m = misfit_for(case, row, solex_P, solex_H)(np.array([row[2], row[3]]))
    assert math.isfinite(m) and m != 1e6
    # a second call is served from the cache and is identical
    p2, h2 = dixon_boundary(tmp_path)
    assert np.array_equal(solex_P, p2) and np.array_equal(solex_H, h2)
    assert len(list(tmp_path.glob("*.csv"))) == 1


@needs_exports
@needs_ref
def test_fitted_dpdt_magmasat_txt_vs_dixon_path(tmp_path):
    """Numbers only: refit three saved MATLAB rows with the saved MagmaSat path and a Dixon path."""
    dixon_P, dixon_H = dixon_boundary(tmp_path)
    case = nc.load_case(CASE_DIR)
    saved = nc.load_saved_results(SAVED)
    out = []
    for i in (0, 40, 80):
        row = saved[i]
        P0 = float(row[3])
        fits = {}
        for name, (sp, sh) in {
            "magmasat_txt": (case.solex_P, case.solex_H),
            "dixon": (dixon_P, dixon_H),
        }.items():
            res = nc.fminsearch(
                misfit_for(case, row, sp, sh), [row[1], P0], tolx=1.0, tolfun=1.0
            )
            assert math.isfinite(res.fval)
            fits[name] = {"dPdt": float(res.x[0]), "fval": float(res.fval)}
        out.append({
            "row": i, "P0": P0, "Kd": float(row[4]), "saved_dPdt": float(row[2]), "saved_fval": float(row[0]),
            **fits,
            "ratio_dixon_over_magmasat": fits["dixon"]["dPdt"] / fits["magmasat_txt"]["dPdt"],
            "melt_H2O_at_P0_magmasat": float(nc.matlab_spline(case.solex_P, case.solex_H, P0)),
            "melt_H2O_at_P0_dixon": float(nc.matlab_spline(dixon_P, dixon_H, P0)),
        })  # fmt: skip
    record("dpdt_magmasat_vs_dixon", {"rows": out})
    for o in out:
        print(f"\n{o}")
