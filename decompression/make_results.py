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

Model: METHODS.md (corrected 2026-10-17). Data: the 2023 profiles per case. sigma_i = case `one_sig` (SIMS 1 sigma), sigma_m = 0.
Priors on physical parameters: log10 v_P ~ U(-3, 4) with v_P in MPa/s (= 1e-4 to 1e3 MPa/s = 1e-3 to 1e4 bar/s; 1 MPa = 10 bar);
log10 D uniform between the case's 2023 T-20 K and T+20 K laws (the `*_2023_bracket`), or a named law with T ~ U(1473 +/- 20) K (table 4);
K ~ N(Kd_mean, Kd_sd); P0 ~ U(Pmin, Pmax); Pf ~ U(Pfmin, Pfmax) or the fixed Pf; L fixed at the 2023 value (table 5: L ~ U(L0 +/- 20 %), a posterior dimension read from the same table at xi_i(L); L with a point outside the crystal has zero likelihood); rho = 2800 +/- 100 kg/m3.
Grid: posterior on 700 log v_P cells x 41 log D nodes x 5 P0 x (2 or 6) Pf nodes, K-marginal log-likelihood interpolated in log tau;
G table: 400 log10 tau nodes from the solver floor (>= 10 time steps) to {hi}, 5 P0 x 2 Pf cell centres (Pf: 1 when fixed; 6 when the Pf prior spans > 10 bar, the three interior-cpx and grain-boundary cases),
solver numx = 2 (numx_2023 - 1) + 1; 11 L nodes when L varies; {n} posterior draws (cells jittered uniformly), seed {seed}.
The priors reach log tau up to about 3.6; the table stops at {hi} (cost grows with tau), so the likelihood is held at its edge value beyond.
Flags (METHODS 4.6): `lower` / `upper` = the likelihood is within 2 log-units of its maximum at the low-tau / high-tau table edge that the
prior can reach, so the data bound v_P from one side only; the table then shows the 95 % one-sided bound (`>` or `<`) instead of a median.
"""


def fmt(x: float) -> str:
    return f"{x:.3g}"


def rate(d: dict, key_med: str = "median") -> str:
    """Median, or the one-sided bound with its flag."""
    if d["flag"] == "bounded":
        return fmt(d[key_med])
    if d["flag"] == "lower":
        return f"> {fmt(d['bound95'])} (lower)"
    if d["flag"] == "upper":
        return f"< {fmt(d['bound95'])} (upper)"
    return "unconstrained"


def rate_ci(d: dict, lo: str, hi: str) -> str:
    return f"{fmt(d[lo])}-{fmt(d[hi])}" if d["flag"] == "bounded" else "-"


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

    hi = rows[0]["tau_range"][1]
    lines = [HEADER.format(n=N_SAMPLES, seed=SEED, hi=f"log10 tau = {hi:g}")]
    lines += [
        "## 1. v_P (bar/s), 2023 bracket and priors on physical parameters: posterior vs saved 2023 MC fits\n",
        "| case | flag | post median or 95 % bound | 68 % | 95 % | 2023 median | 2023 IQR | post/2023 | u median (m/s) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        S = r["summary_2023_bracket"]
        v, mc, u = S["v_P_bar_s"], r["mc_2023"], S["u_m_s"]
        ratio = f"{v['median'] / mc['median']:.2f}" if v["flag"] == "bounded" else "-"
        lines.append(
            f"| {r['key']} | {v['flag']} | {rate(v)} | {rate_ci(v, 'lo68', 'hi68')} | {rate_ci(v, 'lo95', 'hi95')} "
            f"| {fmt(mc['median'])} | {fmt(mc['q1'])}-{fmt(mc['q3'])} | {ratio} | {rate(u)} |"
        )
    lines += [
        "\n## 2. Other posteriors, median (68 %)\n",
        "| case | K | P0 (bar) | Pf (bar) | log10 tau |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        s_ = r["summary_2023_bracket"]
        lines.append(
            f"| {r['key']} | {ci(s_['K'])} | {ci(s_['P0_bar'])} | {ci(s_['Pf_bar'])} | {ci(s_['log10_tau'])} |"
        )
    lines += [
        "\n## 3. Runtime (12-core Apple M-series, spawn pool) and grid\n",
        "Table build is one-off per case and path (12 workers), reusable for any data, sigma, priors, law or temperature range.",
        "Posterior = grid posterior + 40k draws + summary for the 2023 bracket. The 2023-style column is 100 x the mean time of",
        "`newcombe.fit_row` over 3 saved rows (numba solver, 2023 grid, one core). The machine was shared, so times are upper bounds.\n",
        "| case | log10 tau table | table build (s) | posterior (s) | 2023-style MC, 100 fits, 1 core (s) |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['key']} | {r['tau_range'][0]:.2f} to {r['tau_range'][1]:.2f} | {r['table_seconds']:.1f}"
            f"{' (cache)' if r['table_from_cache'] else ''} | {r['posterior_seconds']:.2f} | {100 * r['mc_2023_fit_seconds']:.0f} |"
        )
    lines += [
        "\n## 4. v_P (bar/s) under each prior on D (separate posterior per row; named laws T ~ U(1473 +/- 20) K)\n",
        "| case | D prior | flag | v_P median or bound | 68 % | ratio to bracket |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        bk = laws.bracket_key(r["key"])
        base = r["summary_2023_bracket"]["v_P_bar_s"]
        entries = [(bk, base)] + [
            (n, d["summary"]["v_P_bar_s"]) for n, d in r["laws"].items()
        ]
        for name, v in entries:
            ratio = (
                f"{v['median'] / base['median']:.3g}"
                if v["flag"] == "bounded" == base["flag"]
                else "-"
            )
            lines.append(
                f"| {r['key']} | {name} | {v['flag']} | {rate(v)} | {rate_ci(v, 'lo68', 'hi68')} | {ratio} |"
            )
    lines += [
        "\n## 5. L-prior sensitivity: v_P (bar/s), 2023 bracket, L fixed vs L ~ U(L0 +/- 20 %)\n",
        "Valid L = fraction of the 11 L nodes that keep every data point inside the crystal.\n",
        "| case | L0 (um) | L fixed: v_P | L ~ U(+/-20 %): flag | v_P median or bound | 68 % | ratio to fixed | L posterior median (68 %) | valid L |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        base = r["summary_2023_bracket"]
        sens = r["L_sensitivity"]["uniform_0.2"]
        v, v0 = sens["summary"]["v_P_bar_s"], base["v_P_bar_s"]
        ratio = (
            f"{v['median'] / v0['median']:.3g}"
            if v["flag"] == "bounded" == v0["flag"]
            else "-"
        )
        lines.append(
            f"| {r['key']} | {fmt(base['L_um']['median'])} | {rate(v0)} | {v['flag']} | {rate(v)} | {rate_ci(v, 'lo68', 'hi68')} "
            f"| {ratio} | {ci(sens['summary']['L_um'])} | {sens['bounds']['valid_L_fraction']:.2f} |"
        )
    OUT.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
