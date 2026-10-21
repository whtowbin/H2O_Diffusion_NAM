# Plan: decompression rates for the paper, once the inputs are decided

**Goal.** Produce the paper's decompression-rate and ascent-speed results (with full uncertainties and a sensitivity table) for every
Sullivan Ranch H profile, from Table 1 data, using the Bayesian fit in `decompression/METHODS.md`.

**Approach.** (1) Write every modelling choice into one scenario file, so results are reproducible from it. (2) Build one G table per
distinct degassing path × pressure history × initial-condition mode (METHODS §7), cached. (3) Evaluate the posterior for every
profile × diffusion law × scenario in seconds by reusing those tables, and write the results and figures to the private Ch3 repo.

**Out of scope.** 3-D or spherical geometry and concentration-dependent D (new solvers, METHODS §7.2 last rows); Cima / Vulcan's
Throne profiles (no profiles measured); publishing the port (gate G4); changing Table 1 values.

**Status (2026-10-17; pushed 2026-10-18).** Port, cases, VESIcal paths and the first Bayesian fit are on branch `decompression-port` (public on GitHub since 2026-10-18).
A rework implementing the corrected METHODS §4 (priors on physical parameters, one-sided bounds, L as a posterior dimension) is in
progress; interface names below are those in `decompression/bayes.py` at that rework and must be re-checked when it lands.

## Update 2026-10-20 — what changed since the plan was written
Sources: Ch3 repo `docs/STARTING_MELT_PF2016_2026-10-20.md`, `docs/MIMIC_PEC_AND_SOLUBILITY_2026-10-18.md`, `docs/OPEN_QUESTIONS.md`
(Q18, Q22–Q31 context blocks), `docs/MIMIC_IMPROVEMENTS_2026-10-20.md` (pending).
- **The 2023 starting melt is identified.** The 2023 MagmaSat path (`VesiCal1200_VaporBubble*.txt`, 1200 °C) started from the
  "GCB Lava Whole Rock" row of the 2023 MiMiC input = Plank & Forsyth (2016) Table S5 **AZ09-UK01 (GC2 lava)**, Fe3+/FeT 0.17
  (P&F S3 gives 0.167 for GC2), with H2O 3.36 wt% (2023 max, Ol23) and CO2 8529 ppm (2023 max CO2vbg, ol7). MagmaSat
  reproduces its first rows (6757 bar vs 6756.1; melt CO2 within 2.3 ppm). The measured-CO2 path used 3296 ppm.
- **The inclusions are not that lava plus or minus olivine.** 0 of 15 olivine-addition fits reach the lava within 2σ (Kd 0.27–0.33):
  the inclusions are low in FeOT, MnO, MgO and high in CaO, Al2O3, Na2O, K2O, P2O5, and olivine-independent ratios differ
  (TiO2/Al2O3, K2O/TiO2). Readings: uncorrected Fe loss (MiMiC FEOTI blank) and/or a different magma batch.
- **Effect on the starting pressure is small for the 2023 recipe:** lava + 2026 maxima (Ol23 H2O 3.549, ol7 CO2vbg 8529) gives
  6821 bar (+1.0 %); Fe3+/FeT 0.20–0.21 adds 1.4–1.8 %; pairing matters most: Ol23's own volatiles give 5157 bar (−24 %).
- **Gates:** G1 resolved (MagmaSat runs in the ThermoEngine container under Colima, 20–120 s per point; 2023 path rows reproduced);
  G4 resolved (Henry approved public pushes of this branch, 2026-10-18).
- **Other inputs found since:** MiMiC's entrapment temperature is `Ti` (1131–1198 °C for the key inclusions), below the 1200 °C
  path; the Feb 2023 code set used T = 1200 °C everywhere, the July set 1125 °C for olivine, opx exterior, all-rims and cpx exterior
  only (Q18). The summary workbook's "WHT1 Exterior OPX" rows come from the Pf = 0 run in the "GCB Interior OPX" folder (median
  0.26 MPa/s) rather than the Exterior folder's later run (0.51 MPa/s) (Q25). Saved Kd sit 2–6 prior SD above the script priors
  (Q26). The Fig 11 caption gives olivine Kd 0.009, text and scripts 0.0009 (Q23). Solubility-model spread at the inclusions: total
  CO2 −12 to +17 %, entrapment P e.g. Ol23 311–519 MPa (Q22, Q29). T1-076 has C and S 7× and 14× the rest of its profile
  (Q13; likely contaminated); T1-170 analysis in Q12.

## Gate 0 — decisions needed before Task 1 (Henry)
Each answer goes into the scenario file (Task 1). Question numbers refer to `Ch3_Xenolith_SIMS_Clean/docs/OPEN_QUESTIONS.md`.

| # | decision | options | default if undecided |
|---|---|---|---|
| G0.1 | (reframed 2026-10-17) pyroxene D is **inferred** from the olivine rate (METHODS §8); which experimental opx values to compare against (Q20) | Carpenter 2003 / Demers-Roberge 2021 per axis | **decided 2026-10-17:** Carpenter [100]–[001] range as the main comparison (orientations unknown; thesis citation OK), Demers-Roberge per axis alongside |
| G0.2 | (reframed) which cpx references to compare against (Q21) | Ferriss 2016 Kunlun / mantle-cpx estimate / 2023 anchor | all three as comparison |
| G0.3 | degassing paths (Q22) | MagmaSat (needs ThermoEngine container) + Iacono-Marziano main; Dixon/Shishkina brackets | 2023 MagmaSat txt + Iacono-Marziano + Dixon |
| G0.4 | olivine/melt K treatment (Q23) | constant N(9, 0.5) / wider prior 3–11 / path-dependent K(P) shape | constant, wider prior as sensitivity |
| G0.5 | priors on P0, Pf, T, K per phase (Q26) | 2023 scripts / inferred from saved rows / new | 2023 scripts |
| G0.6 | which profiles: Table 1 grains/profiles per phase, incl. what replaces the mislabeled "GCB interior OPX" (Q25) | list of `(sample, phase, grain, profile)` | all profiles fitted in 2023 (by grain) |
| G0.7 | half-width L per grain and its prior | measured section half-width ± allowance (off-centre cut, 3-D) | 2023 values fixed + ±20 % sensitivity. Note: with L ± 20 %, the opx WHT1 exterior posterior moves L to ≈ 173 µm and v_P × 1.5 (only 73 % of L nodes keep all points inside the crystal at the 2023 L) — check that grain's half-width first |
| G0.8 | pressure histories | linear only / linear + stall (depth, duration ranges) | linear + the 2023 stalling scenario |
| G0.9 | initial condition | equilibrium with melt at P0 / free mantle value c0 (§7.3) | equilibrium; c0-free as sensitivity |
| G0.10 | σ per point | Table 1 `1 sigma s.e.` / 2023 `one_sig` (Herasil SD) / both + σ_m | 2023 `one_sig`, Table 1 σ as sensitivity |
| G0.11 (new 2026-10-20) | starting-melt major elements for the degassing path, **per volcano** (Sullivan Ranch = GC2/GCB vent; Cima and Vulcan's Throne are different volcanoes with their own starting melts and are never compared with the GC2 lava; Henry 2026-10-20). Sullivan Ranch options: | AZ09-UK01 lava as in 2023 / the starting inclusion's own MiMiC-corrected composition / lava olivine-corrected to Fo90 (P&F S7; −4 % Psat) / Fe-loss-corrected inclusion (after the MiMiC review) | AZ09-UK01 lava, Fe3+/FeT 0.17 (reproduces 2023; shown to be a liquid, no accumulation, Ch3 `docs/LAVA_VS_INCLUSIONS_LLD_2026-10-20.md`); Fe-loss + PEC-corrected inclusion mean as the alternative bracket (−9.8 % Psat); lava + 5 % olivine −1.5 %; scoria +11.8 % |
| G0.12 (new) | volatile pairing at the start | max H2O and max CO2vbg from different inclusions (2023 recipe, 6821 bar with 2026 values) / one inclusion's own H2O–CO2 pair (Ol23: 5157 bar) / envelope over the top-3 (Q27) | 2023 recipe with 2026 values; single-inclusion pair as sensitivity |
| G0.13 (new) | degassing-path temperature | 1200 °C (Feb 2023) / 1125 °C (July 2023 set) / MiMiC Ti of the starting inclusion (Ol23 1166 °C) | 1200 °C; 1166 °C as sensitivity (Psat moves ≤ 4 % between Ti and 1250 °C, Q30) |
| G0.14 (new) | which saved exterior-opx run counts as "2023" for replay tests (Q25) | GCB-folder file (Pf = 0, used by the summary workbook) / Exterior-folder file (later) | GCB-folder file (matches the workbook) |

Gates that block specific tasks: ~~**G1** Docker/ThermoEngine for MagmaSat paths (Task 2b)~~ resolved 2026-10-20 (Colima);
**G2** Henry's answers to G0.1–G0.14 (Task 6 final run; every other task builds with the defaults above, so an answer is a
scenario-file edit); **G3** text-vs-data note items that change profiles (T1-170, T1-076 in `TEXT_DATA_INCONSISTENCIES.md`) before
Task 6; ~~**G4** permission before the port goes public~~ resolved 2026-10-18 (Henry approved public pushes of this branch);
**G5 (new)** the MiMiC method changes Henry accepts from Ch3 `docs/MIMIC_IMPROVEMENTS_2026-10-20.md` before Task 1b's priors
are final. Measured on the key inclusions (2026-10-20): (a) Fe-loss correction, per volcano: Sullivan Ranch FEOTI 11.6–11.9 wt% from the GC2 olivine control line; Cima and Vulcan's Throne need their own (olivine control line through MO10-16 and AZ09-UK02/UK03 respectively; not yet done; their vent whole rocks give FEOTI 10.38 and 10.02) (LLD study; the WGC array is too low, it would imply Fe gain for 4 of 11 inclusions) — H2O median −6 %, CO2vbg median +5.6 %; with FEOTI = vent whole rock H2O drops 3–7 % (Ol23
3.549 → 3.337 wt%) and raises CO2vbg 5–9 % — the source of FEOTI is Henry's call; (b) two MiMiC code issues to fix in our patched
copy first: resampled FEOTI never used (its uncertainty is not propagated) and the Fe-Mg exchange divisor 2.298 vs 1.7825 molar
(≤ +2 % CO2vbg); (c) report MC median and 16–84 / 2.5–97.5 % intervals with no-bubble counts instead of draw 0 ± SD after the
modified-z filter (Ol23 Pvbg 418, 95 % 307–643 MPa); (d) bubble CO2: Duan & Zhang (2006) H2O-CO2 EOS at equilibrium vapor XCO2 (Ch3 `docs/BUBBLE_EOS_2026-10-20.md`)
raises CO2vbg −1 to +8 % and the SR start to 7049 bar (+3.9 %); a pure-CO2 bubble is an upper bound (+55 % Ol23, start 8399 bar); (e) Fe3+/FeT 0.167 ± 0.025 for Sullivan Ranch (P&F S3) instead of the hard-coded 0.2 (Ol23 H2O −1.3 %,
CO2vbg +3–4 %); (f) flag inclusions out of Fe-Mg equilibrium with the host (dFo > +2: AZ09GCB ol6, mo1016 06b_a) and treat a blank
host Fo as an error; (g) no-bubble draws should get P and Ti from the corrected melt. Ford vs Toplis Kd (H2O −2.5 to −4.8 %,
CO2vbg +5 to +14 %) stays a quoted systematic. Task 1b uses the median and percentile intervals from (c).

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

### Task 1b — Eruption priors from the 2026 MiMiC rerun (Henry, 2026-10-18)
- Input: `Ch3_Xenolith_SIMS_Clean/data/mimic_2026/summary.csv` (PEC + bubble-corrected H2O, CO2, Pvbg/Pglass with MC 1σ per inclusion;
  SR from `ExampleInput.csv` Sep 2025, Cima/VT from `…updated2025_1.csv` with **assumed** SR-relative uncertainties).
- Creates `decompression/eruptions.py`, `tests/test_eruptions.py`.
- Interface: `eruption_prior(name) -> EruptionPrior(H2O0, CO2_0, composition, Psat)` per eruption (Sullivan Ranch, Cima, Vulcan's Throne):
  the starting melt for the degassing path = the highest-H2O inclusion (or the top-3 envelope, a scenario option), its CO2
  (measured or bubble-corrected: scenario option) and its MC uncertainty; `Psat` from VESIcal for that melt sets the P0 prior.
- Scenarios this feeds (Task 2/3): closed vs 50 % open degassing; solubility model (MagmaSat via container, Iacono-Marziano, Dixon);
  measured vs bubble-corrected CO2; linear vs stalling ascent.
- Update 2026-10-20: each eruption has its own starting melt; a prior never borrows another volcano's composition (Sullivan Ranch: GC2
  lava AZ09-UK01 / scoria AZ09-GCB; Cima: vent 16 MO10-16; Vulcan's Throne: AZ09-UK02 tephra / AZ09-UK03 lava; all P&F 2016 S5),
  and `test_prior_composition_is_own_volcano` checks it.
- Update 2026-10-20: `EruptionPrior` also carries `composition_source` (G0.11: `"lava_AZ09-UK01"`, `"inclusion"`, `"lava_Fo90"`,
  (Sullivan Ranch values; Cima/VT use `"wr_MO10-16"` / `"wr_AZ09-UK02"` and their own inclusions),
  `"inclusion_feloss"`), `fe3_fet`, `volatile_pairing` (G0.12) and `T_path` (G0.13). The Sullivan Ranch lava composition is read
  at run time from Ch3 `data/recovered/M20_PlankForsyth2016/Plank&Forsyth-Supp-Tables.xlsx` (S5, AZ09-UK01) — never copied here.
  Inclusions flagged by the MiMiC equilibrium check (AZ09GCB GC2_ol6, mo1016 06b_a; Q31) are excluded from the max-H2O pick unless
  the scenario says otherwise; the Cima maximum then becomes 06a_a S-corrected (1.33 wt%).
- Tests: `test_prior_uses_max_h2o_inclusion` (picks the documented inclusion per eruption); `test_assumed_uncertainty_flag_propagates`
  (Cima/VT priors carry the "assumed uncertainty" flag into results); `test_psat_matches_vesical_direct`;
  `test_lava_recipe_reproduces_2023_start` (AZ09-UK01, Fe3+/FeT 0.17, H2O 3.35925, CO2 8476.074 ppm, 1200 °C → MagmaSat Psat
  within 0.5 % of 6756.1 bar; marked `slow`, needs the container); `test_pem_flagged_inclusions_excluded`; `test_prior_composition_is_own_volcano` (every eruption's prior and FEOTI come from that
  volcano's own whole rock or inclusions, never from another volcano).
- Verify: `uv run pytest tests/test_eruptions.py -q`.

### Task 2 — Degassing paths, including effective paths
- Changes `decompression/degassing.py`; creates `decompression/paths.py`, `tests/test_paths.py`.
- Interfaces: `PATHS: dict[str, PathSpec]` registry (`"magmasat_2023_vb"` = the 2023 txt, `"iacono_closed_1200"`, `"dixon_closed_1200"`,
  …); `build_path(key) -> (P_bar, H_wt)`; `effective_path(P, H, f_of_P) -> (P, f·H)` for K(P) shapes (METHODS §7.2).
- Tests: `test_magmasat_2023_txt_loads` (bit-equal to the txt); `test_dixon_regenerates_2023` (existing ≤ 3e-4 wt%);
  `test_effective_path_constant_f_equals_rescale` (f ≡ c gives the same posterior as K → cK).
- Task 2b (G1 resolved): MagmaSat paths in the ThermoEngine container; `test_magmasat_regenerates_2023_txt` (tolerance from a rounding
  control; the starting-melt study already reproduces rows 0–3 of both 1200 °C paths). Path keys follow the G0.11–G0.13 options,
  e.g. `"magmasat_lava017_2023pair_1200"`, `"magmasat_lava017_2026pair_1200"`, `"magmasat_ol23_ownpair_1166"`; note that VESIcal
  starts its grid above saturation, so the first written row (6756.1 bar) is the second grid point, not Psat.
- Task 2c (new): solubility-model paths for the same starts (Iacono-Marziano, Dixon, Shishkina, MafiCH) so G0.3 can be decided on
  their effect on v_P; reuse `Ch3/scripts/mimic/solswap_patch.py` composition handling only for the MiMiC side.
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
  (2023 settings give the numbers in `decompression/RESULTS.md` to rounding); `test_one_sided_flag_propagates`;
  `test_posterior_mass_above_tau_edge_below_1e-3` (every real run: posterior mass at log τ > 2.5 < 1e-3, else the run extends that
  table to log τ 3.5 and records it — the edge clamp is only approximate, METHODS §4.3).
- Verify: `uv run pytest tests/test_run.py -q`; run the CLI; DuckDB check that every profile × law × path row exists:
  `uv run --with duckdb python -c "import duckdb; print(duckdb.sql(\"select count(*), count(distinct profile) from '<dir>/posteriors.parquet'\"))"`.

### Task 6b — Pyroxene diffusivity from the olivine rate (METHODS §8)
- Changes `decompression/bayes.py`; creates `tests/test_dpx.py`; output `<dir>/dpx.md` and rows in `posteriors.parquet`.
- Interface: `infer_D_from_rate(px_case, px_table, ol_samples: Samples, logD_range=(-14, -8)) -> Posterior` (log D_px posterior and
  T-corrected log D at 1200 °C), using the olivine posterior samples of `(v_P, P0, Pf)` as the prior.
- Tests: `test_dpx_recovers_synthetic_D` (synthetic px profile made with known D and the olivine rate → posterior covers D);
  `test_dpx_matches_ratio_when_ol_sharp` (a delta-function olivine rate reproduces `log D_used + log v_ol − log v_px`);
  `test_dpx_requires_shared_path` (mismatched P0/Pf ranges raise).
- Verify: `uv run pytest tests/test_dpx.py -q`; `dpx.md` lists log D_px (median, 68/95 %) per profile next to the chapter values
  (cpx ≈ −11.4, opx ≥ −11.15) and the experimental laws: opx against the Carpenter [100]–[001] band (`laws.OPX_CARPENTER2003_100`,
  `laws.OPX_CARPENTER2003_001`, evaluated over the T prior) with Demers-Roberge per axis; cpx against the Ferriss values in G0.2.

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
Eruption-specific starting melts (incl. the identified 2023 lava start, G0.11–G0.13) → Task 1b, 2b; pyroxene D from the olivine rate → Task 6b; full uncertainties → Tasks 4, 6 (posterior incl. L, T, K, c0, P0, Pf); every Table 1 profile → Task 1 (G0.6); alternative laws →
Tasks 6–7; alternative degassing paths and K(P) → Task 2; stalling/multi-stage → Task 3; speed via reuse → Task 5; reproducibility →
Tasks 1, 5, 6 (scenario file + seeded runs); privacy → results only in the private repo, data read at run time; paper outputs → Tasks 7–9.
