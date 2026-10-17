# Plan: decompression rates for the paper, once the inputs are decided

**Goal.** Produce the paper's decompression-rate and ascent-speed results (with full uncertainties and a sensitivity table) for every
Sullivan Ranch H profile, from Table 1 data, using the Bayesian fit in `decompression/METHODS.md`.

**Approach.** (1) Write every modelling choice into one scenario file, so results are reproducible from it. (2) Build one G table per
distinct degassing path × pressure history × initial-condition mode (METHODS §7), cached. (3) Evaluate the posterior for every
profile × diffusion law × scenario in seconds by reusing those tables, and write the results and figures to the private Ch3 repo.

**Out of scope.** 3-D or spherical geometry and concentration-dependent D (new solvers, METHODS §7.2 last rows); Cima / Vulcan's
Throne profiles (no profiles measured); publishing the port (gate G4); changing Table 1 values.

**Status (2026-10-17).** Port, cases, VESIcal paths and the first Bayesian fit are on branch `decompression-port` (local, not pushed).
A rework implementing the corrected METHODS §4 (priors on physical parameters, one-sided bounds, L as a posterior dimension) is in
progress; interface names below are those in `decompression/bayes.py` at that rework and must be re-checked when it lands.

## Gate 0 — decisions needed before Task 1 (Henry)
Each answer goes into the scenario file (Task 1). Question numbers refer to `Ch3_Xenolith_SIMS_Clean/docs/OPEN_QUESTIONS.md`.

| # | decision | options | default if undecided |
|---|---|---|---|
| G0.1 | opx diffusion law (Q20) | Carpenter 2003 bracket / Demers-Roberge 2021 per axis / both | both, reported side by side |
| G0.2 | cpx law anchor (Q21) | 2023 anchor / docstring Kunlun+3.3 / Kunlun Table 4; ±0.7 log sensitivity | 2023 anchor + sensitivity |
| G0.3 | degassing paths (Q22) | MagmaSat (needs ThermoEngine container) + Iacono-Marziano main; Dixon/Shishkina brackets | 2023 MagmaSat txt + Iacono-Marziano + Dixon |
| G0.4 | olivine/melt K treatment (Q23) | constant N(9, 0.5) / wider prior 3–11 / path-dependent K(P) shape | constant, wider prior as sensitivity |
| G0.5 | priors on P0, Pf, T, K per phase (Q26) | 2023 scripts / inferred from saved rows / new | 2023 scripts |
| G0.6 | which profiles: Table 1 grains/profiles per phase, incl. what replaces the mislabeled "GCB interior OPX" (Q25) | list of `(sample, phase, grain, profile)` | all profiles fitted in 2023 (by grain) |
| G0.7 | half-width L per grain and its prior | measured section half-width ± allowance (off-centre cut, 3-D) | 2023 values fixed + ±20 % sensitivity |
| G0.8 | pressure histories | linear only / linear + stall (depth, duration ranges) | linear + the 2023 stalling scenario |
| G0.9 | initial condition | equilibrium with melt at P0 / free mantle value c0 (§7.3) | equilibrium; c0-free as sensitivity |
| G0.10 | σ per point | Table 1 `1 sigma s.e.` / 2023 `one_sig` (Herasil SD) / both + σ_m | 2023 `one_sig`, Table 1 σ as sensitivity |

Gates that block specific tasks: **G1** Docker/ThermoEngine for MagmaSat paths (Task 2b); **G2** Henry's answers to G0.1–G0.10
(Task 6 final run); **G3** text-vs-data note items that change profiles (T1-170, T1-076 in `TEXT_DATA_INCONSISTENCIES.md`) before
Task 6; **G4** permission from M. Newcombe and Henry before any of the port goes public.

## Tasks

### Task 1 — Scenario file and profile loader
- Creates `decompression/scenario.py`, `decompression/profiles.py`, `scenarios/paper_2026.toml` (scenario file holds names and
  choices only, never data values), `tests/test_scenario.py`, `tests/test_profiles.py`.
- Interfaces:
  - `@dataclass Scenario(name, profiles: list[ProfileRef], laws: dict[str, list[str]], paths: list[str], histories: list[str],
    ic_mode: Literal["equilibrium", "free_c0"], K_prior: dict, P0_prior, Pf_prior, T_prior, L_prior: dict, sigma_mode, vP_range)`
  - `load_scenarios(path: Path) -> list[Scenario]` (validates law keys against `laws.applicable_laws`, path keys against Task 2).
  - `@dataclass ProfileRef(sample, phase, grain, profile, exclude_flagged=True)`;
    `load_profile(ref, t1_csv=CH3_T1_CSV) -> Profile(dist_rim_um, h2o_ppm, sigma_ppm, t1_rowids)` reading Ch3 `T1_SIMS_NAMS.csv` at run
    time (column `Min. Dist. Rim (µm)`, rim-referenced; header line 1).
- Tests: `test_load_scenarios_roundtrip`; `test_unknown_law_key_raises`; `test_profile_matches_matlab_txt` (WHT1 exterior opx and
  the olivine profile equal the 2023 MATLAB txt values after the rim/centre conversion, excluding flagged points);
  `test_flagged_points_excluded` (T1-104…T1-233 set); `test_missing_distance_raises` (T1-229, T1-240 have no distance).
- Verify: `uv run pytest tests/test_scenario.py tests/test_profiles.py -q`.

### Task 2 — Degassing paths, including effective paths
- Changes `decompression/degassing.py`; creates `decompression/paths.py`, `tests/test_paths.py`.
- Interfaces: `PATHS: dict[str, PathSpec]` registry (`"magmasat_2023_vb"` = the 2023 txt, `"iacono_closed_1200"`, `"dixon_closed_1200"`,
  …); `build_path(key) -> (P_bar, H_wt)`; `effective_path(P, H, f_of_P) -> (P, f·H)` for K(P) shapes (METHODS §7.2).
- Tests: `test_magmasat_2023_txt_loads` (bit-equal to the txt); `test_dixon_regenerates_2023` (existing ≤ 3e-4 wt%);
  `test_effective_path_constant_f_equals_rescale` (f ≡ c gives the same posterior as K → cK).
- Task 2b (gate G1): MagmaSat paths in the ThermoEngine container; `test_magmasat_regenerates_2023_txt` (tolerance from a rounding control).
- Verify: `uv run pytest tests/test_paths.py -q`.

### Task 3 — Pressure histories
- Changes `decompression/newcombe.py` (`forward` gains `history: Callable[[np.ndarray], np.ndarray] | None = None`, default linear,
  bit-identical when None); creates `decompression/histories.py`, `tests/test_histories.py`.
- Interfaces: `linear(s)`, `stall(p_stall_frac, t_stall_frac)(s)` returning `π(s)` with `π(0) = 0`, `π(1) = 1`, monotone.
- Tests: `test_linear_history_bit_identical` (all existing Octave/replay tests still pass unchanged); `test_stall_reproduces_2023_stalling_case`
  (if the 2023 stalling run is expressible as a history; otherwise document why and test against a direct two-stage forward);
  `test_history_monotone_validation`.
- Verify: `uv run pytest -q -m "not slow"` then the slow replay tests once.

### Task 4 — Superposition for a free initial value (METHODS §7.3)
- Changes `decompression/bayes.py`; creates `tests/test_superposition.py`.
- Interfaces: `a_table(...)` (rim-driven part `A`, same axes as `GTable`), `b_table(log_tau_grid, numx)` (initial-decay part, τ only);
  `kd_c0_marginal_loglike(h, s, A_i, B_i, mu, Sigma)` (2 × 2 closed form).
- Tests: `test_equilibrium_superposition_equals_G` (`A + H(P0) B = G` to 1e-12); `test_superposition_equals_direct_forward` (uniform
  `c0 ≠ K H(P0)` vs a direct solve); `test_two_coefficient_marginal_matches_quadrature` (2-D quadrature, 1e-8).
- Verify: `uv run pytest tests/test_superposition.py -q`.

### Task 5 — Table store and builder CLI
- Creates `decompression/tables.py`, `decompression/__main__.py` (subcommands `build-tables`, `run`), `tests/test_tables.py`.
- Interfaces: `table_key(path_key, history_key, ic_part, numx, tau_grid, P0_nodes, Pf_nodes) -> str`;
  `get_table(...) -> GTable` (build or load from the gitignored cache); `python -m decompression build-tables scenarios/paper_2026.toml`
  builds exactly the distinct tables the scenarios need and prints their count and build time.
- Tests: `test_cache_hit_identical`; `test_key_changes_with_path_and_history`; `test_phases_share_table` (olivine and opx with the
  same path/history/ranges resolve to one key).
- Verify: `uv run pytest tests/test_tables.py -q`; `uv run python -m decompression build-tables scenarios/paper_2026.toml`.

### Task 6 — Posterior runs and results (needs G2, G3)
- Creates `decompression/run.py`; writes results to `~/Projects/Ch3_Xenolith_SIMS_Clean/results/decompression/` (private repo;
  never to this public repo): `posteriors.parquet` (one row per profile × law × path × history × scenario variant: quantiles of
  log10 v_P, v_P [MPa/s], u [m/s], K, P0, Pf, L, flags) and `summary.md`.
- Interface: `python -m decompression run scenarios/paper_2026.toml --out <dir>`; seeded (`--seed`), so reruns are bit-identical.
- Tests: `test_end_to_end_synthetic_recovery` (synthetic profiles through the whole CLI recover truth within 2σ); `test_reproduces_results_md`
  (2023 settings give the numbers in `decompression/RESULTS.md` to rounding); `test_one_sided_flag_propagates`.
- Verify: `uv run pytest tests/test_run.py -q`; run the CLI; DuckDB check that every profile × law × path row exists:
  `uv run --with duckdb python -c "import duckdb; print(duckdb.sql(\"select count(*), count(distinct profile) from '<dir>/posteriors.parquet'\"))"`.

### Task 7 — Sensitivity table for the paper
- Creates `decompression/sensitivity.py` → `<dir>/sensitivity.md` and `.csv`: for each profile, the median log v_P under the base
  scenario and its shift for each one-at-a-time change (law, path, K treatment, L ±20 %, T ±20 K, σ mode, history, c0-free).
- Test: `test_sensitivity_rows_complete`; `test_law_shift_equals_D_ratio_when_prior_flat` (property 1 check on real output).
- Verify: `uv run pytest tests/test_sensitivity.py -q`; open `sensitivity.md`.

### Task 8 — Figures (private Ch3 repo)
- Creates `~/Projects/Ch3_Xenolith_SIMS_Clean/scripts/figures/fig11_posterior.py` (data + posterior-predictive band per profile,
  as the new Fig 11) and `fig12_rates.py` (v_P / ascent speed per profile and phase with 68/95 % intervals and one-sided bounds marked,
  coloured by law; the new Fig 12); outputs to `figures/regenerated/`.
- Verification: run both scripts; view the PNGs; numbers on the figures match `summary.md` (spot-check 3 profiles).

### Task 9 — Methods text and supplement
- Creates `~/Projects/Ch3_Xenolith_SIMS_Clean/docs/methods_decompression.md`: paper-ready methods (model, the two exact properties,
  priors table with sources, validation summary) and a supplement table of all inputs per profile, generated from the scenario file.
- Verification: every number in the text is read from `summary.md`/`sensitivity.csv` by a small script (no hand-typed values).

## Requirement → task check
Full uncertainties → Tasks 4, 6 (posterior incl. L, T, K, c0, P0, Pf); every Table 1 profile → Task 1 (G0.6); alternative laws →
Tasks 6–7; alternative degassing paths and K(P) → Task 2; stalling/multi-stage → Task 3; speed via reuse → Task 5; reproducibility →
Tasks 1, 5, 6 (scenario file + seeded runs); privacy → results only in the private repo, data read at run time; paper outputs → Tasks 7–9.
