"""Registry of the Sullivan Ranch decompression cases (Newcombe-style fits), one ``CaseSpec`` each.

Every numeric field carries a comment with the MATLAB file and line it was read from. The MATLAB
sources, the profiles and the saved ``final_results`` files are NOT in this repository (the repo is
public); they are found under ``$NEWCOMBE_REF`` at run time. Profile, solubility and results file
names are not stored here: they are read from the ``load`` lines of the master script and by globbing
the case folder (see ``resolve``). ``check_against_sources`` re-parses the MATLAB and returns every
field that disagrees with this registry.

Known unknowns, common to all cases:
- ``Pf`` (final pressure, drawn per run from U(Pfmin, Pfmax)) is not saved in ``final_results``,
  except for the case whose ``olivineMC.m`` hard-codes ``Pf = 0``.
- The saved files hold 5 significant figures (``dlmwrite`` default).
- Where a saved file's statistics contradict the preserved master script, ``notes`` says so; the
  replay uses the preserved script's ``TolFun``/``TolX`` and ``Pf`` prior because nothing else is known.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from decompression.newcombe import Case, DiffLaw, load_saved_results

DEFAULT_REF = (
    "~/Projects/Ch3_Xenolith_SIMS_Clean/data/recovered/M14_ModelCode/Decompression modeling/"
    "Megan's Updated Code 2023 (Feb Pre-Defense-Hotter 1200 C) Archive"
)


def ref_root() -> Path | None:
    """The reference folder ($NEWCOMBE_REF or the default), or None if it does not exist."""
    root = Path(os.environ.get("NEWCOMBE_REF", DEFAULT_REF)).expanduser()
    return root if root.is_dir() else None


@dataclass(frozen=True)
class CaseSpec:
    key: str
    description: str
    folder: str  # under $NEWCOMBE_REF; its olivineMC.m is the forward model
    script_folder: str  # folder whose master_script_test*.m produced the saved results
    radius: float  # um
    dist_a: int  # dist = dist_a * x_file + dist_b * radius
    dist_b: int
    Pmin: float
    Pmax: float
    Pfmin: float
    Pfmax: float
    logdPbydtmin: float
    logdPbydtmax: float
    Kd_mean: float
    Kd_sd: float
    one_sig: float
    tolfun: float
    tolx: float
    numx: int
    dh_hi: DiffLaw
    dh_lo: DiffLaw
    Pf_fixed: float | None = None
    T_C: float = 1200.0  # master_script_test*.m `T_C = 1200;` in every case
    replay: bool = True  # False: results file is a byte copy of another case's
    notes: str = ""


REGISTRY: tuple[CaseSpec, ...] = (
    CaseSpec(
        key="ol_nostall_file_in_stalling_folder",
        description="Sullivan Ranch olivine, shallow-stalling folder, noStalling results file (same bytes as the No Stalling case)",
        folder="Sullivan Ranch Olivine/H_in_ol_MCfitting Sullivan Ranch ol _shallow stalling ",  # under $NEWCOMBE_REF; olivineMC.m in this folder is the model
        script_folder="Sullivan Ranch Olivine/H_in_ol_MCfitting Sullivan Ranch ol _No Stalling",  # folder of master_script_test.m that matches the saved results (see notes)
        radius=290 / 2,  # master_script_test.m:29  (um)
        dist_a=1,
        dist_b=0,  # master_script_test.m:28: dist = dist_a * x_file + dist_b * radius (abs() is taken in olivineMC.m)
        Pmin=5000.0,
        Pmax=6700.0,  # master_script_test.m:34, :33
        Pfmin=0.0,
        Pfmax=5.0,  # master_script_test.m:37, :36  (Pf is NOT saved)
        logdPbydtmin=-1.0,
        logdPbydtmax=1.0,  # master_script_test.m:42, :41
        Kd_mean=9.0,
        Kd_sd=0.5,  # master_script_test.m:77
        one_sig=0.25,  # master_script_test.m:65
        tolfun=1.0,
        tolx=1.0,  # master_script_test.m:122 (TolFun, TolX)
        numx=41,  # olivineMC.m:52
        dh_hi=DiffLaw(10**-5.4, 130000.0),  # master_script_test.m:47
        dh_lo=DiffLaw(10**-5.4, 130000.0),  # master_script_test.m:48
        replay=False,
        notes="Results file is byte-identical to the done No Stalling case, so it is not replayed again.",
    ),
    CaseSpec(
        key="ol_with_stalling",
        description="Sullivan Ranch olivine, shallow stalling (P0 150-300 bar), with_Stalling results file",
        folder="Sullivan Ranch Olivine/H_in_ol_MCfitting Sullivan Ranch ol _shallow stalling ",  # under $NEWCOMBE_REF; olivineMC.m in this folder is the model
        script_folder="Sullivan Ranch Olivine/H_in_ol_MCfitting Sullivan Ranch ol _shallow stalling ",  # folder of master_script_test.m that matches the saved results (see notes)
        radius=290 / 2,  # master_script_test.m:29  (um)
        dist_a=1,
        dist_b=0,  # master_script_test.m:28: dist = dist_a * x_file + dist_b * radius (abs() is taken in olivineMC.m)
        Pmin=150.0,
        Pmax=300.0,  # master_script_test.m:34, :33
        Pfmin=0.0,
        Pfmax=5.0,  # master_script_test.m:37, :36  (Pf is NOT saved)
        logdPbydtmin=-1.0,
        logdPbydtmax=1.0,  # master_script_test.m:42, :41
        Kd_mean=9.0,
        Kd_sd=0.5,  # master_script_test.m:77
        one_sig=0.25,  # master_script_test.m:65
        tolfun=1.0,
        tolx=1.0,  # master_script_test.m:122 (TolFun, TolX)
        numx=41,  # olivineMC.m:52
        dh_hi=DiffLaw(10**-5.4, 130000.0),  # master_script_test.m:47
        dh_lo=DiffLaw(10**-5.4, 130000.0),  # master_script_test.m:48
        notes="Pf is not saved (prior U(0, 5)).",
    ),
    CaseSpec(
        key="opx_wht1_exterior",
        description="WHT1 exterior orthopyroxene",
        folder="Sullivan Ranch OPX/H_in_ol_MCfitting Sullivan Ranch WHT1 Exterior OPX",  # under $NEWCOMBE_REF; olivineMC.m in this folder is the model
        script_folder="Sullivan Ranch OPX/H_in_ol_MCfitting Sullivan Ranch WHT1 Exterior OPX",  # folder of master_script_test.m that matches the saved results (see notes)
        radius=360 / 2,  # master_script_test.m:29  (um)
        dist_a=-1,
        dist_b=0,  # master_script_test.m:28: dist = dist_a * x_file + dist_b * radius (abs() is taken in olivineMC.m)
        Pmin=6000.0,
        Pmax=6700.0,  # master_script_test.m:34, :33
        Pfmin=0.0,
        Pfmax=5.0,  # master_script_test.m:37, :36  (Pf is NOT saved)
        logdPbydtmin=-2.0,
        logdPbydtmax=0.0,  # master_script_test.m:42, :41
        Kd_mean=90.0,
        Kd_sd=5.0,  # master_script_test.m:77
        one_sig=20.0,  # master_script_test.m:65
        tolfun=0.1,
        tolx=0.1,  # master_script_test.m:122 (TolFun, TolX)
        numx=61,  # olivineMC.m:52
        dh_hi=DiffLaw(10**-4.788, 161000.0),  # master_script_test.m:47
        dh_lo=DiffLaw(10**-6.6, 127000.0),  # master_script_test.m:48
        notes="Pf not saved (U(0, 5)). Saved Kd does not fit the script's N(90, 5) (KS p 1e-19); a truncated N(97.5, 5) fits (KS p 0.5, inference): the saved file comes from a run with a different Kd mean.",
    ),
    CaseSpec(
        key="opx_wht1_exterior_higher_kd",
        description="WHT1 exterior orthopyroxene, higher Kd",
        folder="Sullivan Ranch OPX/H_in_ol_MCfitting Sullivan Ranch WHT1 Exterior OPX Higher Kd",  # under $NEWCOMBE_REF; olivineMC.m in this folder is the model
        script_folder="Sullivan Ranch OPX/H_in_ol_MCfitting Sullivan Ranch WHT1 Exterior OPX Higher Kd",  # folder of master_script_test_Higher_Kd.m that matches the saved results (see notes)
        radius=360 / 2,  # master_script_test_Higher_Kd.m:29  (um)
        dist_a=-1,
        dist_b=0,  # master_script_test_Higher_Kd.m:28: dist = dist_a * x_file + dist_b * radius (abs() is taken in olivineMC.m)
        Pmin=6000.0,
        Pmax=6700.0,  # master_script_test_Higher_Kd.m:34, :33
        Pfmin=0.0,
        Pfmax=5.0,  # master_script_test_Higher_Kd.m:37, :36  (Pf is NOT saved)
        logdPbydtmin=-2.0,
        logdPbydtmax=0.0,  # master_script_test_Higher_Kd.m:42, :41
        Kd_mean=120.0,
        Kd_sd=5.0,  # master_script_test_Higher_Kd.m:77
        one_sig=20.0,  # master_script_test_Higher_Kd.m:65
        tolfun=0.1,
        tolx=0.1,  # master_script_test_Higher_Kd.m:122 (TolFun, TolX)
        numx=61,  # olivineMC.m:52
        dh_hi=DiffLaw(10**-4.788, 161000.0),  # master_script_test_Higher_Kd.m:47
        dh_lo=DiffLaw(10**-6.6, 127000.0),  # master_script_test_Higher_Kd.m:48
        notes="Pf not saved (U(0, 5)).",
    ),
    CaseSpec(
        key="opx_wht1_interior",
        description="WHT1 interior orthopyroxene (OPX2)",
        folder="Sullivan Ranch OPX/H_in_ol_MCfitting Sullivan Ranch WHT1 Interior OPX ",  # under $NEWCOMBE_REF; olivineMC.m in this folder is the model
        script_folder="Sullivan Ranch OPX/H_in_ol_MCfitting Sullivan Ranch WHT1 Interior OPX ",  # folder of master_script_test.m that matches the saved results (see notes)
        radius=320 / 2,  # master_script_test.m:27  (um)
        dist_a=-1,
        dist_b=1,  # master_script_test.m:31: dist = dist_a * x_file + dist_b * radius (abs() is taken in olivineMC.m)
        Pmin=6000.0,
        Pmax=6700.0,  # master_script_test.m:35, :34
        Pfmin=0.0,
        Pfmax=10.0,  # master_script_test.m:38, :37  (Pf is NOT saved)
        logdPbydtmin=-2.0,
        logdPbydtmax=0.0,  # master_script_test.m:43, :42
        Kd_mean=90.0,
        Kd_sd=5.0,  # master_script_test.m:78
        one_sig=20.0,  # master_script_test.m:66
        tolfun=0.1,
        tolx=0.1,  # master_script_test.m:123 (TolFun, TolX)
        numx=61,  # olivineMC.m:52
        dh_hi=DiffLaw(10**-4.788, 161000.0),  # master_script_test.m:48
        dh_lo=DiffLaw(10**-6.6, 127000.0),  # master_script_test.m:49
        notes="Pf not saved (U(0, 10)). Saved Kd does not fit the script's N(90, 5) (KS p 4e-14); N(95, 6) fits (KS p 0.8, inference).",
    ),
    CaseSpec(
        key="opx_gcb_interior",
        description="GCB interior orthopyroxene folder (data and saved rows are the WHT1 exterior OPX profile)",
        folder="Sullivan Ranch OPX/H_in_ol_MCfitting Sullivan Ranch GCB Interior OPX  ",  # under $NEWCOMBE_REF; olivineMC.m in this folder is the model
        script_folder="Sullivan Ranch OPX/H_in_ol_MCfitting Sullivan Ranch GCB Interior OPX  ",  # folder of master_script_test.m that matches the saved results (see notes)
        radius=360 / 2,  # master_script_test.m:29  (um)
        dist_a=-1,
        dist_b=0,  # master_script_test.m:28: dist = dist_a * x_file + dist_b * radius (abs() is taken in olivineMC.m)
        Pmin=6000.0,
        Pmax=6700.0,  # master_script_test.m:34, :33
        Pfmin=0.0,
        Pfmax=5.0,  # master_script_test.m:37, :36  (Pf is NOT saved)
        logdPbydtmin=-2.0,
        logdPbydtmax=0.0,  # master_script_test.m:42, :41
        Kd_mean=90.0,
        Kd_sd=5.0,  # master_script_test.m:77
        one_sig=20.0,  # master_script_test.m:65
        tolfun=0.1,
        tolx=0.1,  # master_script_test.m:122 (TolFun, TolX)
        numx=61,  # olivineMC.m:52
        dh_hi=DiffLaw(10**-4.788, 161000.0),  # master_script_test.m:47
        dh_lo=DiffLaw(10**-6.6, 127000.0),  # master_script_test.m:48
        Pf_fixed=0.0,  # olivineMC.m:48: Pf = 0 overrides the global Pf
        notes="olivineMC.m hard-codes Pf = 0, so Pf is known (the master's random Pf is overwritten). Saved Kd does not fit the script's N(90, 5) (KS p 1e-145); N(120, 10) fits (KS p 0.97, inference). Same noisy profile as the WHT1 exterior OPX files (largest column-mean |z| 1.3).",
    ),
    CaseSpec(
        key="cpx_wht1_exterior",
        description="WHT1 exterior clinopyroxene (CPX1 Exterior2)",
        folder="Sullivan ranch Cpx/H_in_ol_MCfitting Sullivan Ranch WHT1 Exterior clinopyroxene",  # under $NEWCOMBE_REF; olivineMC.m in this folder is the model
        script_folder="Sullivan ranch Cpx/H_in_ol_MCfitting Sullivan Ranch WHT1 Exterior clinopyroxene",  # folder of master_script_test.m that matches the saved results (see notes)
        radius=115,  # master_script_test.m:29  (um)
        dist_a=-1,
        dist_b=1,  # master_script_test.m:30: dist = dist_a * x_file + dist_b * radius (abs() is taken in olivineMC.m)
        Pmin=6000.0,
        Pmax=6700.0,  # master_script_test.m:35, :34
        Pfmin=0.0,
        Pfmax=5.0,  # master_script_test.m:38, :37  (Pf is NOT saved)
        logdPbydtmin=-2.0,
        logdPbydtmax=1.0,  # master_script_test.m:43, :42
        Kd_mean=120.0,
        Kd_sd=5.0,  # master_script_test.m:78
        one_sig=20.0,  # master_script_test.m:66
        tolfun=0.1,
        tolx=0.1,  # master_script_test.m:123 (TolFun, TolX)
        numx=31,  # olivineMC.m:52
        dh_hi=DiffLaw(10**-6.71, 100000.0),  # master_script_test.m:48
        dh_lo=DiffLaw(10**-6.71, 130000.0),  # master_script_test.m:49
        notes="Pf not saved (U(0, 5)). master_script_test.m is dated Apr 8 2023, after the results file (Feb 25 2023).",
    ),
    CaseSpec(
        key="cpx_wht1_interior1",
        description="WHT1 interior clinopyroxene 1",
        folder="Sullivan ranch Cpx/H_in_ol_MCfitting Sullivan Ranch WHT1 Interior clinopyroxene1_Check Diffusivity",  # under $NEWCOMBE_REF; olivineMC.m in this folder is the model
        script_folder="Sullivan ranch Cpx/H_in_ol_MCfitting Sullivan Ranch WHT1 Interior clinopyroxene1_Check Diffusivity",  # folder of master_script_test.m that matches the saved results (see notes)
        radius=113,  # master_script_test.m:27  (um)
        dist_a=-1,
        dist_b=1,  # master_script_test.m:30: dist = dist_a * x_file + dist_b * radius (abs() is taken in olivineMC.m)
        Pmin=6000.0,
        Pmax=6700.0,  # master_script_test.m:36, :35
        Pfmin=0.0,
        Pfmax=400.0,  # master_script_test.m:39, :38  (Pf is NOT saved)
        logdPbydtmin=-2.0,
        logdPbydtmax=2.0,  # master_script_test.m:44, :43
        Kd_mean=120.0,
        Kd_sd=5.0,  # master_script_test.m:79
        one_sig=20.0,  # master_script_test.m:67
        tolfun=1.0,
        tolx=0.1,  # master_script_test.m:124 (TolFun, TolX)
        numx=61,  # olivineMC.m:52
        dh_hi=DiffLaw(10**-4.788, 161000.0),  # master_script_test.m:49
        dh_lo=DiffLaw(10**-6.6, 127000.0),  # master_script_test.m:50
        notes="Pf not saved (U(0, 400)). Saved P0 goes down to 4146 bar (script Pmin 6000, KS p 6e-40) and Kd median is 143 (script N(120, 5), KS p 5e-103); N(142.5, 7.5) with Pmin about 4100 fits (inference). Not from the preserved script.",
    ),
    CaseSpec(
        key="cpx_wht1_interior2",
        description="WHT1 interior clinopyroxene 2",
        folder="Sullivan ranch Cpx/H_in_ol_MCfitting Sullivan Ranch WHT1 Interior clinopyroxene2_Check Diffusivity ",  # under $NEWCOMBE_REF; olivineMC.m in this folder is the model
        script_folder="Sullivan ranch Cpx/H_in_ol_MCfitting Sullivan Ranch WHT1 Interior clinopyroxene2_Check Diffusivity ",  # folder of master_script_test.m that matches the saved results (see notes)
        radius=134,  # master_script_test.m:28  (um)
        dist_a=-1,
        dist_b=1,  # master_script_test.m:29: dist = dist_a * x_file + dist_b * radius (abs() is taken in olivineMC.m)
        Pmin=6000.0,
        Pmax=6700.0,  # master_script_test.m:35, :34
        Pfmin=0.0,
        Pfmax=400.0,  # master_script_test.m:38, :37  (Pf is NOT saved)
        logdPbydtmin=-2.0,
        logdPbydtmax=2.0,  # master_script_test.m:43, :42
        Kd_mean=120.0,
        Kd_sd=5.0,  # master_script_test.m:78
        one_sig=20.0,  # master_script_test.m:66
        tolfun=0.1,
        tolx=0.1,  # master_script_test.m:123 (TolFun, TolX)
        numx=61,  # olivineMC.m:23
        dh_hi=DiffLaw(10**-4.788, 161000.0),  # master_script_test.m:48
        dh_lo=DiffLaw(10**-6.6, 127000.0),  # master_script_test.m:49
        notes="Pf not saved (U(0, 400)). Saved Kd does not fit the script's N(120, 5) (KS p 1e-17); N(122.5, 7.5) fits (KS p 0.8, inference).",
    ),
    CaseSpec(
        key="opx_grain_boundary",
        description="WHT1 all orthopyroxene rims, grain-boundary profile (radius 50 mm)",
        folder="H_in_ol_MCfitting Sullivan Ranch WHT1 All opx Grain Bound ",  # under $NEWCOMBE_REF; olivineMC.m in this folder is the model
        script_folder="H_in_ol_MCfitting Sullivan Ranch WHT1 All opx Grain Bound ",  # folder of master_script_test.m that matches the saved results (see notes)
        radius=(50 * 1000 * 2) / 2,  # master_script_test.m:28  (um)
        dist_a=1,
        dist_b=-1,  # master_script_test.m:29: dist = dist_a * x_file + dist_b * radius (abs() is taken in olivineMC.m)
        Pmin=6000.0,
        Pmax=6700.0,  # master_script_test.m:35, :34
        Pfmin=0.0,
        Pfmax=200.0,  # master_script_test.m:38, :37  (Pf is NOT saved)
        logdPbydtmin=-3.0,
        logdPbydtmax=1.8,  # master_script_test.m:43, :42
        Kd_mean=90.0,
        Kd_sd=5.0,  # master_script_test.m:78
        one_sig=20.0,  # master_script_test.m:66
        tolfun=1.0,
        tolx=0.1,  # master_script_test.m:123 (TolFun, TolX)
        numx=61,  # olivineMC.m:52
        dh_hi=DiffLaw(10**-6.2, 71000.0, 1000.0),  # master_script_test.m:48
        dh_lo=DiffLaw(10**-6.2, 71000.0),  # master_script_test.m:49
        notes="Pf not saved (U(0, 200)). Radius 50000 um, D_hi carries an extra x1000 (DHhi line).",
    ),
)
BY_KEY = {c.key: c for c in REGISTRY}


# --------------------------------------------------------------------------------------
# Reading the MATLAB
# --------------------------------------------------------------------------------------
def matlab_active_lines(path: Path) -> list[tuple[int, str]]:
    """(1-based line number, text) of code lines: CRLF handled, %{ %} blocks and % comments removed."""
    text = path.read_text(encoding="latin-1").replace("\r\n", "\n").replace("\r", "\n")
    out, depth = [], 0
    for i, line in enumerate(text.split("\n"), 1):
        if line.strip() == "%{":
            depth += 1
        elif line.strip() == "%}":
            depth = max(0, depth - 1)
        elif not depth:
            code = re.sub(r"%.*$", "", line).strip()
            if code:
                out.append((i, code))
    return out


def _find(lines: list[tuple[int, str]], pattern: str) -> re.Match[str] | None:
    for _, text in lines:
        if m := re.search(pattern, text):
            return m
    return None


def _num(expr: str) -> float:
    """Evaluate a plain MATLAB numeric expression such as ``(50*1000*2)/2``."""
    if not re.fullmatch(r"[\d.\s+\-*/()eE]+", expr):
        raise ValueError(f"not a plain numeric expression: {expr!r}")
    return float(eval(expr, {"__builtins__": {}}))


def _scalar(lines: list[tuple[int, str]], name: str) -> float | None:
    m = _find(lines, rf"^{name}\s*=\s*([^;]+);")
    return _num(m.group(1)) if m else None


def _law(lines: list[tuple[int, str]], which: str) -> DiffLaw | None:
    m = _find(
        lines,
        rf"^DH{which}\s*=\s*10\^(-?[\d.]+)\*exp\(-(\d+)/\(8\.314\*T_{which}\)\)(?:\*(\d+))?;",
    )
    if not m:
        return None
    return DiffLaw(10 ** float(m.group(1)), float(m.group(2)), float(m.group(3) or 1))


def _dist_coeffs(lines: list[tuple[int, str]]) -> tuple[int, int] | None:
    m = _find(lines, r"^dist\s*=\s*([^;]+);")
    if not m:
        return None
    s = m.group(1).replace(" ", "")
    neg, minus_r = s.startswith("-"), "-radius" in s
    return (-1 if neg else 1, (1 if neg else -1) if minus_r else 0)


def script_path(spec: CaseSpec, ref: Path) -> Path:
    found = sorted((ref / spec.script_folder).glob("master_script_test*.m"))
    if len(found) != 1:
        raise FileNotFoundError(
            f"expected one master_script_test*.m in {spec.script_folder}: {found}"
        )
    return found[0]


def check_against_sources(spec: CaseSpec, ref: Path) -> list[str]:
    """Re-parse the MATLAB and list every registry field that disagrees with it."""
    ms = matlab_active_lines(script_path(spec, ref))
    om = matlab_active_lines(ref / spec.folder / "olivineMC.m")
    bad: list[str] = []

    def eq(name: str, got, want) -> None:
        if got != want:
            bad.append(f"{name}: registry {want!r} vs MATLAB {got!r}")

    for f, m in [("radius", "radius"), ("T_C", "T_C"), ("Pmin", "Pmin"), ("Pmax", "Pmax"),
                 ("Pfmin", "Pfmin"), ("Pfmax", "Pfmax"), ("logdPbydtmin", "logdPbydtmin"),
                 ("logdPbydtmax", "logdPbydtmax"), ("one_sig", "one_sig")]:  # fmt: skip
        eq(f, _scalar(ms, m), getattr(spec, f))
    kd = _find(ms, r"Kd\s*=\s*normrnd\(([^,]+),\s*([^)]+)\)")
    eq(
        "Kd",
        (_num(kd.group(1)), _num(kd.group(2))) if kd else None,
        (spec.Kd_mean, spec.Kd_sd),
    )
    opt = _find(ms, r"'TolFun'\s*,\s*([\d.]+)\s*,\s*'TolX'\s*,\s*([\d.]+)")
    eq(
        "tol",
        (float(opt.group(1)), float(opt.group(2))) if opt else None,
        (spec.tolfun, spec.tolx),
    )
    eq("dh_hi", _law(ms, "hi"), spec.dh_hi)
    eq("dh_lo", _law(ms, "lo"), spec.dh_lo)
    eq("dist", _dist_coeffs(ms), (spec.dist_a, spec.dist_b))
    nx = _scalar(om, "numx")
    eq("numx", int(nx) if nx is not None else None, spec.numx)
    eq("Pf_fixed", _scalar(om, "Pf"), spec.Pf_fixed)
    return bad


# --------------------------------------------------------------------------------------
# Resolving a case to data
# --------------------------------------------------------------------------------------
@dataclass
class Resolved:
    spec: CaseSpec
    case: Case
    saved: np.ndarray  # [fval, x0orig(1), x(1), P0, Kd, logDH, H2O_noisy...]
    folder: Path
    profile: Path
    solex: Path
    results: Path


def resolve(spec: CaseSpec, ref: Path) -> Resolved:
    """Build the ``Case`` and load the saved rows; file names come from the master script / glob."""
    folder = ref / spec.folder
    loads = [m.group(1) for _, t in matlab_active_lines(script_path(spec, ref))
             if (m := re.match(r"load\s+(\S+)", t))]  # fmt: skip
    profile = next(n for n in loads if not n.startswith("VesiCal"))
    solex = next(n for n in loads if n.startswith("VesiCal"))
    prof = np.loadtxt(folder / profile, ndmin=2)
    sol = np.loadtxt(folder / solex, ndmin=2)
    # the results file is the final_results* of width 6 + n_profile; if several, the one whose P0 lies in the prior
    cands = [(p, load_saved_results(p)) for p in sorted(folder.glob("final_results*"))]
    cands = [(p, a) for p, a in cands if a.shape[1] == 6 + len(prof)]
    if len(cands) > 1:
        cands = [
            (p, a)
            for p, a in cands
            if spec.Pmin * 0.999 <= a[:, 3].min() and a[:, 3].max() <= spec.Pmax * 1.001
        ]
    if len(cands) != 1:
        raise FileNotFoundError(
            f"{spec.key}: expected one matching final_results file, got {[c[0].name for c in cands]}"
        )
    results, saved = cands[0]
    case = Case(
        dist=spec.dist_a * prof[:, 0] + spec.dist_b * spec.radius, h2o_meas=prof[:, 1],
        solex_P=sol[:, 0], solex_H=sol[:, 1], radius=spec.radius, T_C=spec.T_C,
        Pmin=spec.Pmin, Pmax=spec.Pmax, Pfmin=spec.Pfmin, Pfmax=spec.Pfmax,
        logdPbydtmin=spec.logdPbydtmin, logdPbydtmax=spec.logdPbydtmax,
        Kd_mean=spec.Kd_mean, Kd_sd=spec.Kd_sd, one_sig=spec.one_sig, tolx=spec.tolx,
        tolfun=spec.tolfun, numx=spec.numx, dh_hi=spec.dh_hi, dh_lo=spec.dh_lo, Pf_fixed=spec.Pf_fixed,
    )  # fmt: skip
    return Resolved(
        spec, case, saved, folder, folder / profile, folder / solex, results
    )
