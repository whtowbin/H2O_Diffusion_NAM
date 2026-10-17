"""VESIcal degassing paths as the melt-H2O boundary condition for `newcombe.forward`.

`degassing_path` wraps `VESIcal.calculate_degassing_path` (VESIcal 1.2.12 API) with an on-disk cache;
`boundary_arrays` turns the result into the `solex_P` [bar] / `solex_H` [wt% H2O in melt] arrays the
forward model takes (descending pressure, as in the Feb 2023 `VesiCal1200_*_H2O_only.txt` exports).

The MagmaSat model needs ThermoEngine, which is not a dependency here; asking for it raises.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import warnings
from importlib import metadata
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

CACHE_DIR = Path(__file__).parent / ".degassing_cache"  # gitignored

# Oxide names VESIcal expects (wt%) -> accepted spellings are matched case-insensitively.
_OXIDES = (
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
    "Cr2O3",
    "NiO",
    "CoO",
    "H2O",
    "CO2",
)
_CANON = {o.upper(): o for o in _OXIDES}
# Non-oxide analytes (S, Cl, ...) and other columns are ignored.


def vesical_version() -> str:
    return metadata.version("VESIcal")


def _canonical_composition(composition: dict) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, val in composition.items():
        canon = _CANON.get(str(key).strip().upper())
        if canon is None:
            continue
        out[canon] = float(val)
    for needed in ("SiO2", "H2O", "CO2"):
        if needed not in out:
            raise ValueError(f"composition is missing {needed}")
    return out


def _check_model(model: str) -> None:
    if model.lower() == "magmasat" and importlib.util.find_spec("thermoengine") is None:
        raise RuntimeError(
            "model='MagmaSat' requires ThermoEngine, which is not installed (it is not on PyPI and "
            "is not a dependency of this project). Use another VESIcal model ('Dixon', "
            "'IaconoMarziano', 'ShishkinaIdealMixing', 'Liu', 'MooreWater', ...) or load a saved "
            "MagmaSat path; there is no silent fallback."
        )


def cache_key(
    composition: dict,
    *,
    temperature_C,
    model,
    fractionate_vapor,
    init_vapor,
    final_pressure,
    steps,
    start_pressure,
) -> str:
    payload = {
        "composition": sorted(_canonical_composition(composition).items()),
        "temperature_C": float(temperature_C),
        "model": model,
        "fractionate_vapor": float(fractionate_vapor),
        "init_vapor": float(init_vapor),
        "final_pressure": float(final_pressure),
        "steps": int(steps),
        "start_pressure": start_pressure
        if isinstance(start_pressure, str)
        else float(start_pressure),
        "vesical": vesical_version(),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:24]


def degassing_path(
    composition: dict,
    *,
    temperature_C: float,
    model: str,
    fractionate_vapor: float = 0.0,
    init_vapor: float = 0.0,
    final_pressure: float = 1.0,
    steps: int = 200,
    start_pressure: str | float = "saturation",
    cache_dir: Path | None = CACHE_DIR,
) -> pd.DataFrame:
    """Degassing path from VESIcal as a DataFrame.

    Columns: Pressure_bars, H2O_liq, CO2_liq (wt%), H2O_fl, CO2_fl (mole fraction), FluidProportion_wt.
    `composition`: oxide wt% incl. H2O and CO2 (keys matched case-insensitively; S, Cl etc. ignored).
    `fractionate_vapor`: 0 closed, 1 open, 0.5 half-open. `start_pressure`: 'saturation' or a bar value
    (VESIcal then starts there, so the first rows are fluid-free when above saturation).
    `init_vapor` (initial fluid wt%) exists only in the MagmaSat implementation, so non-zero values are
    rejected for the other models. Results are cached on disk keyed by all inputs and the VESIcal version.
    """
    _check_model(model)
    if init_vapor and model.lower() != "magmasat":
        raise NotImplementedError(
            "init_vapor is only supported by the MagmaSat degassing path"
        )
    comp = _canonical_composition(composition)
    key = cache_key(
        comp,
        temperature_C=temperature_C,
        model=model,
        fractionate_vapor=fractionate_vapor,
        init_vapor=init_vapor,
        final_pressure=final_pressure,
        steps=steps,
        start_pressure=start_pressure,
    )
    cache_file = None if cache_dir is None else Path(cache_dir) / f"{key}.csv"
    if cache_file is not None and cache_file.exists():
        return pd.read_csv(cache_file, float_precision="round_trip")  # ty: ignore[no-matching-overload]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # VESIcal import warns about ThermoEngine
        import VESIcal as v

        sample = v.Sample(comp)
        kwargs: dict = {
            "sample": sample,
            "model": model,
            "temperature": temperature_C,
            "pressure": start_pressure,
            "fractionate_vapor": fractionate_vapor,
            "final_pressure": final_pressure,
            "steps": steps,
            "silence_warnings": True,
        }
        if model.lower() == "magmasat":
            kwargs["init_vapor"] = init_vapor
        df = v.calculate_degassing_path(**kwargs).result
    df = pd.DataFrame(df).reset_index(drop=True)
    if bool(np.isnan(df.to_numpy(dtype=float)).any()):
        raise ValueError(
            f"VESIcal {model} degassing path contains NaN (solver failure) for this input"
        )
    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        buf = StringIO()
        df.to_csv(buf, index=False)
        tmp = cache_file.with_suffix(".tmp")
        tmp.write_text(buf.getvalue())
        tmp.replace(cache_file)
    return df


def boundary_arrays(
    path_df: pd.DataFrame, *, anchor_zero: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """(solex_P [bar], solex_H [wt% H2O in melt]) in the order of the Feb 2023 txt files.

    Those files are the VESIcal path (descending pressure) followed by an anchor row (0, 0), so a Pf
    between 0 and the last path pressure is still covered by the spline. `anchor_zero=False` omits it.
    """
    p = path_df["Pressure_bars"].to_numpy(dtype=float)
    h = path_df["H2O_liq"].to_numpy(dtype=float)
    if p[0] < p[-1]:
        p, h = p[::-1], h[::-1]
    if anchor_zero:
        p, h = np.append(p, 0.0), np.append(h, 0.0)
    return p, h
