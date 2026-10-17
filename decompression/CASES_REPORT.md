# Sullivan Ranch decompression cases: port report

Registry `decompression/cases.py`, tests `tests/test_cases.py` (slow tests: `-m "not slow"` skips them). Octave 11.3 stands in for MATLAB. Numbers below are from one run, `CASES_NUMBERS_JSON` dump.

## Which script made which saved file
All saved files have the `master_script_test*.m` layout `[fval, x0orig(1), x(1), P0, Kd, logDH, H2O_noisy...]` (6 + number of profile points columns, 100 rows, `dlmwrite` 5 s.f.). The older `master_script.m` layout (`[fval x0orig x0 ...]`, p = 20) does not fit any file. Every script writes its output as `final_results_Seguam_ol1_7Mar19.txt`; the files were renamed afterwards.

| case | script used | evidence from the saved rows |
|---|---|---|
| ol, noStalling file (stalling folder) | `master_script_test.m` of the No Stalling folder (Feb 24 2023: P0 U(5000, 6700)) | file is byte-identical to the done No Stalling case; the stalling folder's own script (Feb 26) has P0 U(150, 300) |
| ol, with_Stalling | `master_script_test.m` of the stalling folder (Feb 26 2023: Pmin 150, Pmax 300) | P0 150.03 to 298.21; Kd and logDH fit the script (KS below); file and script both dated Feb 26 |
| OPX WHT1 exterior | `master_script_test.m` (Feb 25) | logDH and x0orig ranges fit; Kd does NOT fit N(90, 5) |
| OPX WHT1 exterior higher Kd | `master_script_test_Higher_Kd.m` (Feb 28) | everything fits, incl. Kd N(120, 5) |
| OPX WHT1 interior | `master_script_test.m` (Feb 25) | logDH and x0orig ranges fit; Kd does NOT fit N(90, 5) |
| OPX GCB interior | `master_script_test.m` (Feb 25) of the GCB folder, olivineMC.m with `Pf = 0` | profile = WHT1 exterior OPX (below); Kd does NOT fit N(90, 5); `Pf = 0` and numx = 61 reproduce the saved fvals to 9e-5 (median) |
| CPX WHT1 exterior | `master_script_test.m` (dated Apr 8 2023, after the results of Feb 25) | every statistic fits, incl. Kd N(120, 5) |
| CPX WHT1 interior 1 | `master_script_test.m` (Feb 25) | P0 goes down to 4146 bar (script Pmin 6000) and Kd median 143: saved file is from an earlier or edited script |
| CPX WHT1 interior 2 | `master_script_test.m` (Feb 25) | logDH and x0orig fit; Kd does NOT fit N(120, 5) |
| OPX grain boundary | `master_script_test.m` (Feb 26) | everything fits, incl. Kd N(90, 5) |

GCB folder: its only profile file is byte-identical to the WHT1 exterior OPX profile and the 6 noisy columns average 96.6, 209.7, 272.2, 307.7, 322.8, 317.3 against the profile 96, 207, 273, 308, 321, 315 (largest |z| = 1.34 with se = 20/sqrt(100)). So the file belongs to the WHT1 exterior OPX profile, not to a separate GCB profile; it differs from the "WHT1 exterior OPX" file in Kd (median 119.6, sd 9.0 against 101.6, 4.5), i.e. it is a third, separate run.

## Where the MATLAB differs from the reference (beyond parameters)
- `olivineMC.m`: only `numx` (41 olivine, 31 CPX exterior, 61 all OPX and CPX interior and grain-boundary cases) and the position of the `delta_term` print differ from the reference. FD scheme, boundary conditions, geometry, diffusivity law and the `numt` rule are unchanged, so no new forward-model code was needed.
- GCB interior OPX `olivineMC.m` line 48: `Pf = 0;` is active. `Pf` is a global, so this overwrites the master script's random `Pf`. Ported as `Case.Pf_fixed`. (Archived copies of other folders also have it; the Higher Kd, exterior and interior OPX copies do not.)
- Master scripts: the diffusivity law differs per bound (`DHhi` at T+20 K and `DHlo` at T-20 K use different Arrhenius parameters in the OPX and CPX cases; ported as `Case.dh_hi`, `Case.dh_lo`, `DiffLaw`). Grain boundary: `DHhi` has an extra `*1000` that `DHlo` lacks, so logDH spans -8.75 to -5.68 (3 decades); radius = 50 mm = 50000 um; `dist = x - radius`.
- Profile geometry: `dist` is `x`, `-x`, `-(x - radius)` or `x - radius` depending on the case (`dist_a`, `dist_b` in the registry); `olivineMC.m` takes `abs(dist)`.
- Different `Pfmax` (5, 10, 200, 400 bar), `Kd` priors, `one_sig` (20 ppm for OPX/CPX), `TolFun`/`TolX` (1/1, 0.1/0.1, 1/0.1).

## Forward model against Octave running the unchanged `olivineMC.m`
11 parameter sets per case (8 from saved rows with dP/dt varied by x0.5 to x2, one P0 = Pmax / Pf = Pfmin edge, one numt > 1e8 and one numt < 10 sentinel, both returning 1e6); numt capped at 3e4 to keep Octave affordable. Threshold 1e-9. The byte-copy olivine case shares its `olivineMC.m` with the done case (T2) and is not rerun.

| case | max relative difference |
|---|---|
| ol_nostall_file_in_stalling_folder | same `olivineMC.m` bytes as the done case (T2: 3.4e-15) |
| ol_with_stalling | 2.02e-15 |
| opx_wht1_exterior | 2.73e-15 |
| opx_wht1_exterior_higher_kd | 3.25e-15 |
| opx_wht1_interior | 1.75e-15 |
| opx_gcb_interior | 4.53e-15 |
| cpx_wht1_exterior | 9.74e-16 |
| cpx_wht1_interior1 | 1.92e-15 |
| cpx_wht1_interior2 | 6.54e-16 |
| opx_grain_boundary | 1.04e-14 |

## Replay of the saved rows (Pf = middle of its prior, 5 s.f. inputs)
Control: rows 0, 10, ..., 90; each refit with Pf over {Pfmin, mid, Pfmax} x 3 inputs perturbed within the 5 s.f. rounding (GCB: 9 perturbations, Pf is known); tolerance = largest deviation from the unperturbed run. The Pf prior comes from the preserved script; Pf is not saved. rel = |python - saved| / |saved|.

| case | rows | dP/dt median | dP/dt max | fval median | fval max | tol dP/dt | tol fval | rows outside dP/dt | rows outside fval | fits not converged |
|---|---|---|---|---|---|---|---|---|---|---|
| ol_nostall_file_in_stalling_folder | 100 | byte copy of the done case (see PORT_REPORT.md T4) | | | | | | | | |
| ol_with_stalling | 100 | 2.03% | 4.59% | 3.14% | 57.9% | 6.3% | 50.3% | 0 [] | 1 [78] | 0 |
| opx_wht1_exterior | 100 | 0.517% | 0.738% | 1.77% | 8.65% | 0.726% | 3.94% | 1 [95] | 4 [11, 70, 75, 82] | 0 |
| opx_wht1_exterior_higher_kd | 100 | 0.255% | 0.72% | 0.896% | 6.9% | 0.748% | 9.97% | 0 [] | 0 [] | 0 |
| opx_wht1_interior | 100 | 0.473% | 0.89% | 0.116% | 0.466% | 1.06% | 0.469% | 0 [] | 0 [] | 0 |
| opx_gcb_interior | 100 | 0.0554% | 0.125% | 0.00927% | 0.0465% | 0.119% | 0.0454% | 2 [28, 39] | 1 [14] | 0 |
| cpx_wht1_exterior | 100 | 0.279% | 0.538% | 0.127% | 1.58% | 0.562% | 0.66% | 0 [] | 6 [29, 49, 74, 83, 91, 92] | 0 |
| cpx_wht1_interior1 | 100 | 39.3% | 334% | 16% | 122% | 479% | 129% | 0 [] | 0 [] | 0 |
| cpx_wht1_interior2 | 100 | 27.6% | 60% | 31.2% | 122% | 103% | 138% | 0 [] | 0 [] | 0 |
| opx_grain_boundary | 100 | 20.3% | 59.8% | 30.6% | 341% | 188% | 242% | 0 [] | 2 [47, 51] | 0 |

Power caveat: where Pf has a wide prior (CPX interior 400 bar, grain boundary 200 bar) the control tolerance is 100% to 480%, so this test cannot see a discrepancy below that. The recovered-Pf replay below is the sharper check.

## Recovered-Pf replay (extra, as in PORT_REPORT.md)
Solve misfit(saved dP/dt; Pf) = saved fval for Pf on a 41-point grid of the prior (misfit is not monotone in Pf, some rows have two roots), refit at each root. "best root" is closest to the saved result, so it is an upper bound on agreement; "first root" is not selected by the answer. Not defined for GCB (Pf = 0 known); there the optimizer-independent check below is direct.

| case | rows with 2 roots | dP/dt median (best) | dP/dt max (best) | fval median (best) | fval max (best) | dP/dt median (first) | dP/dt max (first) | rows outside tol dP/dt (best) | rows outside tol fval (best) |
|---|---|---|---|---|---|---|---|---|---|
| ol_with_stalling | 47 | 2.98e-04 | 9.73e-03 | 2.01e-06 | 9.51e-04 | 5.12e-04 | 9.37e-02 | [] | [] |
| opx_wht1_exterior | 1 | 6.21e-04 | 7.44e-03 | 1.11e-05 | 4.80e-04 | 6.21e-04 | 7.44e-03 | [51] | [] |
| opx_wht1_exterior_higher_kd | 3 | 5.52e-04 | 3.93e-03 | 4.33e-06 | 2.92e-04 | 5.52e-04 | 3.93e-03 | [] | [] |
| opx_wht1_interior | 0 | 6.73e-04 | 1.39e-02 | 7.54e-07 | 3.44e-04 | 6.73e-04 | 1.39e-02 | [56] | [] |
| cpx_wht1_exterior | 1 | 6.76e-04 | 6.49e-03 | 4.39e-06 | 3.35e-04 | 6.76e-04 | 6.49e-03 | [73] | [] |
| cpx_wht1_interior1 | 10 | 9.49e-04 | 8.50e-03 | 4.18e-07 | 2.43e-05 | 9.49e-04 | 8.50e-03 | [] | [] |
| cpx_wht1_interior2 | 11 | 6.14e-04 | 1.48e-03 | 5.23e-07 | 3.95e-06 | 6.22e-04 | 9.18e-02 | [] | [] |
| opx_grain_boundary | 11 | 7.85e-05 | 1.04e-03 | 1.71e-08 | 1.15e-04 | 8.03e-05 | 2.05e-01 | [] | [] |

## Optimizer-independent check
misfit at the SAVED fitted dP/dt, over the Pf prior (one value for GCB), against the saved fval. `bracketed`: saved fval lies inside the range of misfit over Pf; `best`: smallest relative difference over the Pf grid (26 points; GCB: the single Pf = 0 value, i.e. a direct forward-model check against real MATLAB output).

| case | bracketed | best median | best max |
|---|---|---|---|
| ol_with_stalling | 100/100 | 9.22e-04 | 1.87e-02 |
| opx_wht1_exterior | 100/100 | 9.53e-05 | 4.81e-04 |
| opx_wht1_exterior_higher_kd | 100/100 | 4.50e-04 | 2.84e-03 |
| opx_wht1_interior | 100/100 | 4.84e-05 | 1.81e-04 |
| opx_gcb_interior | 100/100 | 9.32e-05 | 5.28e-04 |
| cpx_wht1_exterior | 100/100 | 6.13e-05 | 4.50e-04 |
| cpx_wht1_interior1 | 100/100 | 6.27e-03 | 3.06e-02 |
| cpx_wht1_interior2 | 100/100 | 9.88e-03 | 4.72e-02 |
| opx_grain_boundary | 100/100 | 1.25e-02 | 9.29e-02 |

## Saved inputs against the preserved script's priors
Master-script rejection sampler re-run (200000 draws of noisy profile, Kd, P0; keep Kd x melt(P0) > max noisy H2O), two-sample KS against the saved columns. Small p means the saved file was not made with the preserved settings. Saved P0 / logDH / x0orig were also checked against the script's bounds (`test_saved_rows_consistent_with_registry`).

| case | KS p, Kd | KS p, P0 | saved Kd median (sd) | saved P0 range | logDH saved / script bounds |
|---|---|---|---|---|---|
| ol_nostall_file_in_stalling_folder | 0.43 | 0.17 | 9.0 (0.4) | 5026 to 6679 | -10.07 to -9.95 / -10.07 to -9.95 |
| ol_with_stalling | 0.68 | 0.53 | 9.1 (0.4) | 150 to 298 | -10.07 to -9.95 / -10.07 to -9.95 |
| opx_wht1_exterior | 1e-19 | 0.038 | 101.6 (4.5) | 6016 to 6700 | -11.16 to -10.45 / -11.17 to -10.42 |
| opx_wht1_exterior_higher_kd | 0.94 | 0.62 | 120.3 (5.0) | 6000 to 6675 | -11.16 to -10.42 / -11.17 to -10.42 |
| opx_wht1_interior | 4.2e-14 | 0.72 | 100.3 (3.6) | 6005 to 6696 | -11.16 to -10.43 / -11.17 to -10.42 |
| opx_gcb_interior | 1.5e-145 | 0.77 | 119.6 (9.0) | 6017 to 6699 | -11.16 to -10.42 / -11.17 to -10.42 |
| cpx_wht1_exterior | 0.91 | 0.85 | 120.8 (4.9) | 6008 to 6699 | -11.37 to -10.21 / -11.38 to -10.21 |
| cpx_wht1_interior1 | 4.8e-103 | 5.8e-40 | 143.1 (7.1) | 4146 to 6689 | -11.16 to -10.43 / -11.17 to -10.42 |
| cpx_wht1_interior2 | 1.3e-17 | 0.054 | 130.1 (4.4) | 6006 to 6694 | -11.15 to -10.43 / -11.17 to -10.42 |
| opx_grain_boundary | 0.96 | 0.073 | 89.8 (5.0) | 6001 to 6687 | -8.71 to -5.69 / -8.75 to -5.68 |


## Mismatches, unknowns and hypotheses
1. **Pf is not saved** (any case except GCB). Replay with Pf mid-prior is therefore limited by Pf: the control tolerance on dP/dt, which is set by Pf and rounding, is 0.7% to 1.1% (OPX and CPX exterior, interior OPX, P0 about 6000 bar, Pf 0 to 10), 6% (stalling olivine), 188% (grain boundary, Pf up to 200 bar) and 103% to 479% (CPX interior, Pf up to 400 bar). fval is Pf-dominated everywhere.
2. **Rows outside the control tolerance** (dP/dt: 3 rows; fval: 14 rows, table above). All are marginal or Pf-driven: dP/dt rows exceed the tolerance by 2% to 5% of it (OPX exterior row 95: 0.738% vs 0.726%; GCB rows 28, 39: 0.125% vs 0.119%), and the tolerance is the maximum over only 10 control rows. fval rows are the Pf-dominated ones (olivine row 78, OPX exterior 4 rows, CPX exterior 6 rows, grain boundary 2 rows) plus GCB row 14 (0.0465% vs 0.0454%, marginal). With Pf recovered from the saved fval, every fval is within 9.5e-4 relative (stalling olivine) and within 4.8e-4 in all other cases. Three dP/dt rows remain outside the control tolerance in the recovered-Pf replay (OPX exterior row 51, 0.744% vs 0.726%; OPX interior row 56, 1.39% vs 1.06%; CPX exterior row 73, 0.649% vs 0.562%). Hypothesis: `TolX = 0.1` is an absolute tolerance on dP/dt (a resolution of 0.1/dP/dt, i.e. 0.3% to 3% for dP/dt of 3 to 30), so a 5 s.f. rounding difference that changes one Nelder-Mead step can move the stopping point by that much. Not tested further.
3. **GCB case** (Pf = 0 known, no Pf freedom): dP/dt median 0.055%, max 0.125%; fval median 0.0093%, max 0.047%; misfit at the saved dP/dt equals the saved fval to 9e-5 (median), 5.3e-4 (max). This is the cleanest check of the forward model and optimizer against real MATLAB output outside the done case; it also confirms numx = 61 and `Pf = 0`.
4. **Saved files not made by the preserved scripts** (Kd, and for CPX interior 1 also P0, differ from the script; KS table above). Replay is unaffected for Kd and P0 (saved per row), but `TolFun`, `TolX` and the Pf prior are taken from the preserved script, and the saved run may have used others. The agreement above (recovered-Pf dP/dt within 1e-3 median) suggests the tolerances are right; the Pf prior is only constrained by the recovered Pf values lying inside it. Inferred Kd priors (KS best fit on a coarse grid, not facts): OPX exterior N(97.5, 5), OPX interior N(95, 6), GCB N(120, 10), CPX interior 1 N(142.5, 7.5) with Pmin about 4100, CPX interior 2 N(122.5, 7.5).
5. **Optimizer-independent check on the wide-Pf cases** (CPX interior, grain boundary): the best match over a 26-point Pf grid is only 0.6% to 1.3% median because Pf steps of 8 to 16 bar are coarse; the recovered-Pf root solve (41-point bracketing then Brent) removes that and gives fval within 2.4e-5 (CPX interior 1), 4e-6 (CPX interior 2), 1.2e-4 (grain boundary).
6. **Root ambiguity**: misfit is not monotone in Pf; up to 47 of 100 rows (stalling olivine) have two roots. "first root" is wrong for some rows (max dP/dt difference 9% to 21% for stalling olivine, CPX interior 2, grain boundary), "best root" is selected using the answer and is an upper bound.
7. **Archived MATLAB** (not ported): copies in `Archived/` and `... archived` folders use other numx (25, 41) and some have `Pf = 0`; skipped as instructed.
8. The grain-boundary case runs the same 1-D Cartesian FTCS with radius 50 mm and dx = 833 um; the port equals Octave to 1.0e-14 there, but the physical meaning of that geometry (and the extra x1000 on DHhi) is the MATLAB author's, not examined.
9. One earlier run of the (since de-duplicated) olivine forward test gave a max relative difference of exactly 0.0 where later runs of identical inputs gave 4.9e-15. Python is bit-identical across processes and fresh numba compiles (checked); Octave twice in one process is identical. Treated as ulp-level Octave variation.

## Reproduce
`uv run pytest tests/test_cases.py -m "not slow"` (registry, saved-row and prior checks, seconds). Everything: `uv run pytest` (slow replays and Octave forward runs, about 10 min for this file). `CASES_NUMBERS_JSON=path` dumps every number in this report. Run times: replay 6 s to 75 s per case, Octave forward 3 s to 55 s per case.
