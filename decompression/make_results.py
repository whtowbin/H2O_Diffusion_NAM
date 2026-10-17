"""Run the Bayesian fit on every registered case and write ``RESULTS.md`` (numbers only).

Reads the unpublished Sullivan Ranch inputs from ``$NEWCOMBE_REF`` and Tet-Al from the Ch3 EMPA table;
the output holds unpublished results, so the file must not be pushed to the public repo.
Usage: ``uv run python -m decompression.make_results [--cached]`` (default rebuilds every G table to time it;
``--cached`` reuses the tables and the build times of the previous run stored in ``.g_cache/results.json``).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from decompression import bayes, cases, laws
from decompression import newcombe as nc

OUT = Path(__file__).parent / "RESULTS.md"
N_SAMPLES = 40000
SEED = 0

HEADER = """# Posterior decompression rates (Bayesian fit) versus the 2023 Monte Carlo

**Values come from unpublished data. Do not push this file to the public repo without Henry's OK.**

Model: METHODS.md. Data: the 2023 profiles per case. sigma_i = case `one_sig` (SIMS 1 sigma), sigma_m = 0.
Priors: the 2023 per-case priors (K ~ N(Kd_mean, Kd_sd), P0 ~ U(Pmin, Pmax), Pf ~ U(Pfmin, Pfmax) or the fixed Pf,
log10 tau uniform over the tabulated range); D: log10 D uniform between the 2023 T-20 K and T+20 K laws
(table 1) or a named law with T ~ U(1473 +/- 20) K (table 4); L fixed at the 2023 value; rho = 2800 +/- 100 kg/m3.
Grid: 400 log10 tau x 25 P0 x 5 Pf cell centres (Pf: 1 when fixed), solver numx = 4 (numx_2023 - 1) + 1;
{n} posterior draws (cells jittered uniformly), seed {seed}. The tau range is set per case by
`bayes.auto_tau_range` (listed in table 3); `edge mass` is the posterior mass in the 3 outermost tau cells per end.
"""


def fmt(x: float) -> str:
    return f"{x:.3g}"


def ci(d: dict) -> str:
    return f"{fmt(d['median'])} ({fmt(d['lo68'])}-{fmt(d['hi68'])})"


def time_2023_mc(res: cases.Resolved, n_rows: int = 3) -> float:
    """Seconds for one 2023-style fminsearch fit (mean of ``n_rows`` saved rows, one core, 2023 grid)."""
    pf = res.spec.Pf_fixed if res.spec.Pf_fixed is not None else 2.5
    ts = []
    for row in res.saved[:n_rows]:
        t0 = time.perf_counter()
        nc.fit_row(row, res.case, Pf=pf)
        ts.append(time.perf_counter() - t0)
    return float(np.mean(ts))


def main() -> None:
    cached = "--cached" in sys.argv
    ref = cases.ref_root()
    assert ref is not None, "reference folder not found"
    store = Path(__file__).parent / ".g_cache" / "results.json"
    old = (
        {r["key"]: r for r in json.loads(store.read_text())}
        if cached and store.exists()
        else {}
    )
    rows = []
    for spec in cases.REGISTRY:
        res = cases.resolve(spec, ref)
        row = bayes.fit_case(
            res, laws=laws.applicable_laws(spec.key), n_samples=N_SAMPLES, seed=SEED,
            cache=True, refresh=not cached,
        )  # fmt: skip
        if spec.key in old:  # keep the build time measured when the table was made
            row["table_seconds"], row["table_from_cache"] = (
                old[spec.key]["table_seconds"],
                False,
            )
        row["mc_2023_fit_seconds"] = (
            old[spec.key]["mc_2023_fit_seconds"]
            if spec.key in old and "mc_2023_fit_seconds" in old[spec.key]
            else time_2023_mc(res)
        )
        rows.append(row)
        print(spec.key, "done", flush=True)
    (Path(__file__).parent / ".g_cache").mkdir(exist_ok=True)
    (Path(__file__).parent / ".g_cache" / "results.json").write_text(
        json.dumps(rows, indent=1)
    )

    lines = [HEADER.format(n=N_SAMPLES, seed=SEED)]
    lines += [
        "## 1. v_P (bar/s), 2023 law and priors: posterior vs saved 2023 MC fits\n",
        "| case | post median | 68 % | 95 % | 2023 median | 2023 IQR | post/2023 | u median (m/s) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        v, mc, u = (
            r["summary_2023_law"]["v_P_bar_s"],
            r["mc_2023"],
            r["summary_2023_law"]["u_m_s"],
        )
        lines.append(
            f"| {r['key']} | {fmt(v['median'])} | {fmt(v['lo68'])}-{fmt(v['hi68'])} | {fmt(v['lo95'])}-{fmt(v['hi95'])} "
            f"| {fmt(mc['median'])} | {fmt(mc['q1'])}-{fmt(mc['q3'])} | {v['median'] / mc['median']:.2f} | {fmt(u['median'])} |"
        )
    lines += [
        "\n## 2. Other posteriors, median (68 %)\n",
        "| case | K | P0 (bar) | Pf (bar) | log10 tau |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        s = r["summary_2023_law"]
        lines.append(
            f"| {r['key']} | {ci(s['K'])} | {ci(s['P0_bar'])} | {ci(s['Pf_bar'])} | {ci(s['log10_tau'])} |"
        )
    lines += [
        "\n## 3. Runtime (12-core Apple M-series, spawn pool) and grid\n",
        "Table build is one-off per case and path (12 workers, includes `auto_tau_range`), reusable for any data, sigma, priors or law.",
        "The 2023-style column is 100 x the mean time of `newcombe.fit_row` over 3 saved rows (numba solver, 2023 grid).",
        "Measured once (ol_with_stalling, cpx_wht1_exterior): a table with 5 P0 x 2 Pf cells instead of 25 x 5 builds in 1.2 s",
        "(vs 24 s and 22 s) and moves the log10 tau posterior quantiles by <= 0.004 (olivine) and <= 1e-4 (cpx); not the default.\n",
        "| case | log10 tau range | table build (s) | posterior + 40k draws (s) | 2023-style MC, 100 fits, 1 core (s) | edge mass |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['key']} | {r['tau_range'][0]:.2f} to {r['tau_range'][1]:.2f} | {r['table_seconds']:.1f}"
            f"{' (cache)' if r['table_from_cache'] else ''} | {r['posterior_seconds']:.2f} | {100 * r['mc_2023_fit_seconds']:.0f} | {r['edge_mass']:.1e} |"
        )
    lines += [
        "\n## 4. v_P (bar/s) under each applicable law, median (68 %); same (tau, P0, Pf) draws (property 1)\n",
        "| case | law | v_P | v_P / 2023-law median |",
        "|---|---|---|---|",
    ]
    for r in rows:
        base = r["summary_2023_law"]["v_P_bar_s"]["median"]
        for name, d in r["law_v_P_bar_s"].items():
            lines.append(
                f"| {r['key']} | {name} | {ci(d)} | {d['median'] / base:.3g} |"
            )
    OUT.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
