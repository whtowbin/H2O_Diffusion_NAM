# Port of the Newcombe-style decompression MC fitting (MATLAB) to Python

Branch `decompression-port`, local only. This repo is PUBLIC: do not push, and do not commit the MATLAB code
(M. Newcombe's) or the unpublished Sullivan Ranch profiles. Reference files are read from
`$NEWCOMBE_REF` (default `~/Projects/Ch3_Xenolith_SIMS_Clean/data/recovered/M14_ModelCode/Decompression modeling/Megan's Updated Code 2023 (Feb Pre-Defense-Hotter 1200 C) Archive`).

Step 1 (this plan): a faithful port, proven equal to MATLAB. Step 2 (later, separate change): swap in Ferriss
diffusivities / the repo's finite-difference engine and report how the fitted dP/dt moves.

## Reference case
`Sullivan Ranch Olivine/H_in_ol_MCfitting Sullivan Ranch ol _No Stalling/` — produced by `master_script_test.m`
(not `master_script.m`) on Feb 24 2023 → `final_results_Sullivan_Ranch_Ol_2_Feb24_23.txt` (100 rows, 10 cols, `dlmwrite` 5 s.f.):
`[fval, x0orig(1), x(1), P0, Kd, logDH, H2O_noisy(1..4)]`. Inputs: `Sullivan_Ranch_Ol_2.txt` (dist µm, H2O ppm),
`VesiCal1200_VaporBubble_H2O_only.txt` (P bar, H2O wt%), radius = 290/2, T = 1200 °C,
D = 10^-5.4·exp(-130000/(8.314·T_K)) m²/s with T ± 20 K prior, Kd ~ N(9, 0.5), P0 ~ U(5000, 6700), Pf ~ U(0, 5) (Pf NOT saved),
logdPbydt ~ U(-1, 1), optimset TolFun = 1, TolX = 1, x0 = [dPbydt, P0] (2-D; olivineMC ignores x0(2) — keep the dummy dimension,
it changes the Nelder–Mead path).

## Module `decompression/newcombe.py`
1. `matlab_colon(a, step, b)` — MATLAB colon semantics incl. its floating-point end tolerance (length = floor((b-a)/step + tol) + 1,
   with MATLAB's rule; elements a + k·step, last element snapped as MATLAB does). Test against Octave on awkward cases.
2. `matlab_spline(x, y, xi)` — not-a-knot cubic spline with extrapolation (= `scipy.interpolate.CubicSpline(bc_type='not-a-knot')`
   after sorting x); handle n = 2/3 points as MATLAB does (linear / parabola).
3. `forward(dPbydt, *, P0, Pf, radius, logDH, Kd, solex_P, solex_H, numx)` — line-for-line port of `olivineMC.m`:
   D = 1e12·10^logDH (µm²/s); t_tot = (P0-Pf)/dPbydt; dx = radius/(numx-1); numt = ceil(2·D·t_tot/(0.9·dx²) + 1);
   return None if numt > 1e8 or numt < 10; dt = t_tot/(numt-1); Pvec = matlab_colon(P0, -dt·dPbydt, Pf);
   BC = spline(solex_P, solex_H, Pvec)·Kd; C0 = BC[0] everywhere; explicit FTCS with interior update from the previous step,
   then C[-1] = BC[j+1], C[0] = C[1]. Vectorise the inner loop (same arithmetic order: C[i] + r·(C[i+1] - 2C[i] + C[i-1])).
   If len(Pvec) < numt, MATLAB would raise; raise too.
4. `misfit(x, data...)` = mean((H2O - spline(x_grid, C_final, |dist|))²); 1e6 when forward returns None.
5. `fminsearch(f, x0, tolx, tolfun, maxiter=200·n, maxfev=200·n)` — exact port of MATLAB's fminsearch (Lagarias et al. 1998
   as implemented by MathWorks): initial simplex x0 plus x0 with component j scaled by 1.05 (0.00025 if zero); rho 1, chi 2,
   psi 0.5, sigma 0.5; sort with MATLAB's stable ordering; expansion accepted if fe < fr; outside/inside contraction; shrink;
   termination `max|fv1 - fv_k| <= max(TolFun, 10·eps(fv1))` and `max|v_k - v_1| <= max(TolX, 10·eps(max(v1)))`.
   Return x, fval, exitflag, n_iter, n_fev.
6. `fit_row(row, case)` replays one saved MATLAB run (given x0orig, P0, Kd, logDH, noisy H2O, Pf); `run_mc(case, rng)` is the
   Python MC driver (numpy Generator, seeded) mirroring master_script_test.m, incl. the "crazy" rejection loop.

## Tests `tests/test_newcombe.py` (pytest; skip with a clear reason if `$NEWCOMBE_REF` or `octave` is missing)
- T1 colon + spline vs Octave: random and edge inputs; exact (colon length equal; values rel < 1e-12).
- T2 forward model vs Octave running the UNCHANGED `olivineMC.m` from the reference folder (set the globals via a small driver
  .m written to a temp dir, `octave --no-gui --quiet --eval`): ≥ 20 parameter sets spanning the priors; misfit rel < 1e-9.
- T3 fminsearch port vs MATLAB documentation behaviour on Rosenbrock and a 1-D quadratic with a dummy 2nd dimension
  (sanity; Octave's fminsearch is a different algorithm, so do NOT compare against it).
- T4 replay of the 100 saved MATLAB rows: for each row run `fit_row` with Pf = 2.5 (mid-prior, unknown) and the 5 s.f. inputs;
  compare fitted dP/dt and fval to the saved values. Control first: for 10 rows, re-run with inputs perturbed within their
  5 s.f. rounding and Pf ∈ {0, 5}; the spread sets the tolerance. Report the per-row relative differences, the control spread,
  and how many rows fall outside it. Also re-evaluate `misfit` at the SAVED fitted dP/dt and compare with the saved fval
  (optimizer-independent check of the forward model against real MATLAB output).
- A baseline-against-itself control: two identical Python runs give bit-identical results; Octave forward twice identical.

Report: a short `decompression/PORT_REPORT.md` with the numbers from T1–T4 (not a narrative).

## Decisions (Henry, 2026-10-17)
- Olivine: the 2023 MATLAB fits (`master_script_test.m`) already use D = 10^-5.4·exp(-130 kJ/RT) = `DH2O_Ol` (Ferriss 2018).
  Swapping in the Python olivine law changes nothing; no olivine "step 2" refit needed for D.
- cpx: use Ferriss 2016 with a per-grain tetrahedral-Al correction (fixes `DH2O_Mantle_Cpx`, whose body returns the olivine law).
  Form: log10 D(T, TetAl) = logD0_ref − Ea/(ln10·R·T) + s·(log10 TetAl − log10 TetAl_ref), s = −1.04854545 (Ferriss 2016 fig. 12,
  from the docstring). Default anchor = the 2023 exterior-cpx estimate "for mantle cpx with 0.07 Al(IV) based on Ferriss 2016":
  logD0_ref = −6.71, Ea = 100 kJ/mol, TetAl_ref = 0.07. The docstring's Kunlun-diopside + 10^3.3 anchor needs Ferriss 2016 constants
  that are not in the project: expose the anchor as parameters and note it as open. TetAl per grain from Ch3 T2_EMPA "Tet Al"
  (Sullivan Ranch has one primary-cpx analysis per xenolith, so effectively per sample). Report the 2023 law (baseline) and the
  Tet-Al law side by side per cpx case.
- Interior cpx "Check Diffusivity" runs (Carpenter 2003 opx law): deliberate sensitivity test; port as-is, label as sensitivity.
- opx: keep the 2023 Carpenter 2003 [001] law (no Python opx law yet).
- MagmaSat: later (needs ThermoEngine container). Other VESIcal models now.
