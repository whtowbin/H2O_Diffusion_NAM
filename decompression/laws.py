"""Named H diffusion laws (METHODS.md section 5) as ``DiffLaw`` objects.

``DiffLaw(A, Ea, mult)`` is D = A exp(-Ea / (8.314 T)) * mult in m^2/s with Ea in J/mol.
The Sullivan Ranch Tet-Al values are read at run time from the Ch3 compiled EMPA table; no data
values are stored in this repository.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from decompression.newcombe import DiffLaw

EMPA_CSV = Path(
    "~/Projects/Ch3_Xenolith_SIMS_Clean/data/compiled/csv/T2_EMPA.csv"
).expanduser()

# Ferriss et al. 2016 fig. 12 slope of log10 D against log10 Tet-Al (decompression/PORT_PLAN.md,
# from the docstring of the repo's DH2O_Mantle_Cpx); METHODS.md section 5 rounds it to -1.0485.
FERRISS2016_SLOPE = -1.04854545
TETAL_REF = 0.07  # Al(IV) per 6 O of the 2023 exterior-cpx estimate

OL_FERRISS2018_A = DiffLaw(10**-5.4, 130000.0)  # = 2023 olivine runs = DH2O_Ol
OPX_CARPENTER2003_001 = DiffLaw(10**-4.788, 161000.0)  # Carpenter 2003 thesis, San Carlos opx [001] (2023 DHhi)
OPX_CARPENTER2003_100 = DiffLaw(10**-6.6, 127000.0)  # Carpenter 2003 thesis, [100] (2023 DHlo)
# Demers-Roberge et al. 2021 (GCA, 10.1016/j.gca.2021.04.005), Table 3 and abstract:
# isobaric Arrhenius log D = log D0 - Ea / (2.303 R T), ~DQFM-1, Tanzanian opx, m^2/s, Ea kJ/mol.
# [100]: log D0 -5.11, Ea 139; [010]: -4.93, 149; [001]: -5.52, 125.
# (Their 2.303 is ln 10; the exp form below is the same law.)
OPX_DEMERSROBERGE2021_100 = DiffLaw(10**-5.11, 139000.0)
OPX_DEMERSROBERGE2021_010 = DiffLaw(10**-4.93, 149000.0)
OPX_DEMERSROBERGE2021_001 = DiffLaw(10**-5.52, 125000.0)
CPX_2023_EXTERIOR = DiffLaw(
    10**-6.71, 100000.0
)  # 2023 estimate for 0.07 Al(IV), after Ferriss 2016
GB_DEMOUCHY = DiffLaw(10**-6.2, 71000.0)  # effective grain boundary
GB_DEMOUCHY_X1000 = DiffLaw(10**-6.2, 71000.0, 1000.0)  # the 2023 sensitivity run

LAWS: dict[str, DiffLaw] = {
    "ol_ferriss2018_a": OL_FERRISS2018_A,
    "opx_carpenter2003_001": OPX_CARPENTER2003_001,
    "opx_carpenter2003_100": OPX_CARPENTER2003_100,
    "opx_demersroberge2021_100": OPX_DEMERSROBERGE2021_100,
    "opx_demersroberge2021_010": OPX_DEMERSROBERGE2021_010,
    "opx_demersroberge2021_001": OPX_DEMERSROBERGE2021_001,
    "cpx_2023_exterior": CPX_2023_EXTERIOR,
    "gb_demouchy": GB_DEMOUCHY,
    "gb_demouchy_x1000": GB_DEMOUCHY_X1000,
}


def cpx_ferriss2016_tetal(
    tetal: float,
    anchor: DiffLaw = CPX_2023_EXTERIOR,
    tetal_ref: float = TETAL_REF,
    slope: float = FERRISS2016_SLOPE,
) -> DiffLaw:
    """log10 D = anchor + slope * (log10 TetAl - log10 TetAl_ref): same Ea, A scaled by (TetAl/ref)^slope."""
    if tetal <= 0:
        raise ValueError(f"Tet-Al must be positive, got {tetal}")
    factor = 10.0 ** (slope * (math.log10(tetal) - math.log10(tetal_ref)))
    return DiffLaw(anchor.A, anchor.Ea, anchor.mult * factor)


def sullivan_ranch_tetal(
    sample: str = "AZ18WHT1", csv: Path = EMPA_CSV, grain: str = "Primary"
) -> float:
    """Tet-Al of the ``grain`` cpx analysis of ``sample`` from T2_EMPA.csv (header on line 2)."""
    df = pd.read_csv(csv, header=1, encoding="utf-8-sig")
    # first of the duplicated "Tet Al" columns
    tet = next(c for c in df.columns if c == "Tet Al" or c.startswith("Tet Al."))
    sel = df[
        (df["Phase"] == "cpx")
        & (df["Sample"] == sample)
        & df["Grain ID"].astype(str).str.contains(grain)
    ]
    if len(sel) != 1:
        raise LookupError(f"expected one {grain} cpx row for {sample}, got {len(sel)}")
    return float(sel[tet].iloc[0])


@dataclass(frozen=True)
class Bracket:
    """The 2023 D prior: log10 D uniform between ``lo`` at T - 20 K and ``hi`` at T + 20 K (METHODS.md section 5)."""

    lo: DiffLaw
    hi: DiffLaw

    def log10_bounds(self, T_K: float) -> tuple[float, float]:
        return self.lo.log10_at(T_K - 20), self.hi.log10_at(T_K + 20)


# from each master_script_test*.m (DHhi, DHlo lines)
BRACKETS: dict[str, Bracket] = {
    "ol_2023_bracket": Bracket(OL_FERRISS2018_A, OL_FERRISS2018_A),
    "opx_2023_bracket": Bracket(OPX_CARPENTER2003_100, OPX_CARPENTER2003_001),
    "cpx_2023_bracket": Bracket(DiffLaw(10**-6.71, 130000.0), CPX_2023_EXTERIOR),
    "gb_2023_bracket": Bracket(GB_DEMOUCHY, GB_DEMOUCHY_X1000),
}


def bracket_key(case_key: str) -> str:
    """The `*_2023_bracket` key of a registry case (the interior cpx runs used the opx bracket)."""
    if case_key.startswith("ol_"):
        return "ol_2023_bracket"
    if case_key == "opx_grain_boundary":
        return "gb_2023_bracket"
    if case_key.startswith(("opx_", "cpx_wht1_interior")):
        return "opx_2023_bracket"
    if case_key.startswith("cpx_"):
        return "cpx_2023_bracket"
    raise KeyError(case_key)


def applicable_laws(case_key: str) -> dict[str, DiffLaw]:
    """The named single laws that apply to a registry case, by mineral; cpx adds the Tet-Al law."""

    def pick(*names: str) -> dict[str, DiffLaw]:
        return {n: LAWS[n] for n in names}

    if case_key.startswith("ol_"):
        return pick("ol_ferriss2018_a")
    if case_key == "opx_grain_boundary":
        return pick("gb_demouchy")
    if case_key.startswith("opx_"):
        return pick(
            "opx_carpenter2003_001",
            "opx_carpenter2003_100",
            "opx_demersroberge2021_100",
            "opx_demersroberge2021_010",
            "opx_demersroberge2021_001",
        )
    if case_key.startswith("cpx_"):  # all three are WHT1 (one primary cpx analysis)
        return {
            **pick("cpx_2023_exterior"),
            "cpx_ferriss2016_tetal": cpx_ferriss2016_tetal(sullivan_ranch_tetal()),
        }
    raise KeyError(case_key)
