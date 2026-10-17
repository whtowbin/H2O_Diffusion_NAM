# Posterior decompression rates (Bayesian fit) versus the 2023 Monte Carlo

**Values come from unpublished data. Do not push this file to the public repo without Henry's OK.**

Model: METHODS.md. Data: the 2023 profiles per case. sigma_i = case `one_sig` (SIMS 1 sigma), sigma_m = 0.
Priors: the 2023 per-case priors (K ~ N(Kd_mean, Kd_sd), P0 ~ U(Pmin, Pmax), Pf ~ U(Pfmin, Pfmax) or the fixed Pf,
log10 tau uniform over the tabulated range); D: log10 D uniform between the 2023 T-20 K and T+20 K laws
(table 1) or a named law with T ~ U(1473 +/- 20) K (table 4); L fixed at the 2023 value; rho = 2800 +/- 100 kg/m3.
Grid: 400 log10 tau x 25 P0 x 5 Pf cell centres (Pf: 1 when fixed), solver numx = 4 (numx_2023 - 1) + 1;
40000 posterior draws (cells jittered uniformly), seed 0. The tau range is set per case by
`bayes.auto_tau_range` (listed in table 3); `edge mass` is the posterior mass in the 3 outermost tau cells per end.

## 1. v_P (bar/s), 2023 law and priors: posterior vs saved 2023 MC fits

| case | post median | 68 % | 95 % | 2023 median | 2023 IQR | post/2023 | u median (m/s) |
|---|---|---|---|---|---|---|---|
| ol_nostall_file_in_stalling_folder | 1.5 | 1.3-1.74 | 1.15-2 | 1.53 | 1.41-1.7 | 0.98 | 5.48 |
| ol_with_stalling | 1.91 | 1.55-2.44 | 1.3-3.22 | 1.91 | 1.63-2.25 | 1.00 | 6.95 |
| opx_wht1_exterior | 6.85 | 3.71-12.6 | 2.44-19.4 | 5.05 | 2.9-8.13 | 1.36 | 24.9 |
| opx_wht1_exterior_higher_kd | 2.52 | 1.39-4.55 | 0.984-6.51 | 2.68 | 1.57-4 | 0.94 | 9.18 |
| opx_wht1_interior | 26.1 | 13.5-50.6 | 8.25-91.9 | 10.4 | 6.69-15.3 | 2.52 | 95 |
| opx_gcb_interior | 6.88 | 3.72-12.7 | 2.46-19.5 | 2.6 | 1.73-3.76 | 2.64 | 25 |
| cpx_wht1_exterior | 6.72 | 2.66-17 | 1.72-26.3 | 6.49 | 3.76-14.1 | 1.04 | 24.5 |
| cpx_wht1_interior1 | 297 | 145-619 | 76.4-1.24e+03 | 23.9 | 9.47-70.3 | 12.43 | 1.08e+03 |
| cpx_wht1_interior2 | 11.4 | 6.08-21.4 | 3.86-34.3 | 12.2 | 7.95-19.9 | 0.94 | 41.5 |
| opx_grain_boundary | 0.11 | 0.00967-1.24 | 0.00307-3.99 | 0.246 | 0.0453-1.2 | 0.45 | 0.401 |

## 2. Other posteriors, median (68 %)

| case | K | P0 (bar) | Pf (bar) | log10 tau |
|---|---|---|---|---|
| ol_nostall_file_in_stalling_folder | 9.04 (8.54-9.53) | 5.86e+03 (5.28e+03-6.43e+03) | 3.21 (1.39-4.47) | 1.26 (1.19-1.32) |
| ol_with_stalling | 9.04 (8.55-9.53) | 230 (185-276) | 2.81 (1.04-4.34) | -0.257 (-0.431--0.124) |
| opx_wht1_exterior | 96 (92.4-99.8) | 6.36e+03 (6.12e+03-6.59e+03) | 2.44 (0.764-4.18) | -0.333 (-0.458--0.219) |
| opx_wht1_exterior_higher_kd | 120 (115-125) | 6.35e+03 (6.11e+03-6.59e+03) | 2.45 (0.776-4.18) | 0.101 (0.0105-0.181) |
| opx_wht1_interior | 90.1 (87.5-92.9) | 6.35e+03 (6.11e+03-6.59e+03) | 4.97 (1.57-8.36) | -0.806 (-0.984--0.654) |
| opx_gcb_interior | 96 (92.4-99.9) | 6.36e+03 (6.11e+03-6.59e+03) | 0 (0-0) | -0.335 (-0.46--0.221) |
| cpx_wht1_exterior | 120 (115-125) | 6.35e+03 (6.11e+03-6.59e+03) | 2.53 (0.82-4.23) | 0.0613 (-0.0243-0.137) |
| cpx_wht1_interior1 | 119 (116-121) | 6.35e+03 (6.11e+03-6.59e+03) | 215 (74.3-344) | -1.58 (-1.8--1.38) |
| cpx_wht1_interior2 | 126 (122-130) | 6.35e+03 (6.11e+03-6.59e+03) | 356 (327-386) | -0.323 (-0.466--0.191) |
| opx_grain_boundary | 89.6 (83.7-95.1) | 6.35e+03 (6.11e+03-6.59e+03) | 171 (136-191) | 0.169 (-0.0784-0.34) |

## 3. Runtime (12-core Apple M-series, spawn pool) and grid

Table build is one-off per case and path (12 workers, includes `auto_tau_range`), reusable for any data, sigma, priors or law.
The 2023-style column is 100 x the mean time of `newcombe.fit_row` over 3 saved rows (numba solver, 2023 grid).
Measured once (ol_with_stalling, cpx_wht1_exterior): a table with 5 P0 x 2 Pf cells instead of 25 x 5 builds in 1.2 s
(vs 24 s and 22 s) and moves the log10 tau posterior quantiles by <= 0.004 (olivine) and <= 1e-4 (cpx); not the default.

| case | log10 tau range | table build (s) | posterior + 40k draws (s) | 2023-style MC, 100 fits, 1 core (s) | edge mass |
|---|---|---|---|---|---|
| ol_nostall_file_in_stalling_folder | 0.90 to 1.60 | 724.6 | 0.01 | 19 | 8.9e-09 |
| ol_with_stalling | -1.10 to 0.20 | 23.8 | 0.01 | 2 | 4.2e-09 |
| opx_wht1_exterior | -1.10 to 0.20 | 64.5 | 0.01 | 26 | 5.2e-08 |
| opx_wht1_exterior_higher_kd | -0.50 to 0.50 | 147.3 | 0.01 | 4 | 5.5e-08 |
| opx_wht1_interior | -4.00 to -0.10 | 15.9 | 0.01 | 19 | 2.2e-05 |
| opx_gcb_interior | -1.10 to 0.20 | 16.1 | 0.01 | 9 | 4.7e-08 |
| cpx_wht1_exterior | -0.50 to 0.40 | 22.2 | 0.01 | 2 | 1.4e-07 |
| cpx_wht1_interior1 | -4.00 to -0.30 | 12.5 | 0.01 | 13 | 4.5e-05 |
| cpx_wht1_interior2 | -1.10 to 0.40 | 86.1 | 0.01 | 3 | 3.3e-07 |
| opx_grain_boundary | -1.50 to 1.00 | 205.7 | 0.01 | 6 | 3.4e-07 |

## 4. v_P (bar/s) under each applicable law, median (68 %); same (tau, P0, Pf) draws (property 1)

| case | law | v_P | v_P / 2023-law median |
|---|---|---|---|
| ol_nostall_file_in_stalling_folder | ol_ferriss2018_a | 1.51 (1.31-1.74) | 1 |
| ol_with_stalling | ol_ferriss2018_a | 1.91 (1.55-2.45) | 1 |
| opx_wht1_exterior | opx_carpenter2003_001 | 13.4 (10.1-18.2) | 1.96 |
| opx_wht1_exterior | opx_demersroberge2021_100 | 38.6 (29.2-52) | 5.63 |
| opx_wht1_exterior | opx_demersroberge2021_010 | 25.8 (19.5-34.8) | 3.77 |
| opx_wht1_exterior | opx_demersroberge2021_001 | 47 (35.7-63.3) | 6.87 |
| opx_wht1_exterior_higher_kd | opx_carpenter2003_001 | 4.96 (4-6.22) | 1.97 |
| opx_wht1_exterior_higher_kd | opx_demersroberge2021_100 | 14.2 (11.6-17.8) | 5.64 |
| opx_wht1_exterior_higher_kd | opx_demersroberge2021_010 | 9.51 (7.71-11.9) | 3.77 |
| opx_wht1_exterior_higher_kd | opx_demersroberge2021_001 | 17.3 (14.2-21.6) | 6.87 |
| opx_wht1_interior | opx_carpenter2003_001 | 50.4 (35-76.9) | 1.93 |
| opx_wht1_interior | opx_demersroberge2021_100 | 145 (101-220) | 5.55 |
| opx_wht1_interior | opx_demersroberge2021_010 | 96.9 (67.3-147) | 3.71 |
| opx_wht1_interior | opx_demersroberge2021_001 | 177 (123-268) | 6.77 |
| opx_gcb_interior | opx_carpenter2003_001 | 13.5 (10.2-18.3) | 1.96 |
| opx_gcb_interior | opx_demersroberge2021_100 | 38.7 (29.4-52.2) | 5.63 |
| opx_gcb_interior | opx_demersroberge2021_010 | 25.9 (19.6-35) | 3.77 |
| opx_gcb_interior | opx_demersroberge2021_001 | 47.2 (35.9-63.5) | 6.87 |
| cpx_wht1_exterior | cpx_2023_exterior | 23.1 (19.2-28.3) | 3.44 |
| cpx_wht1_exterior | cpx_ferriss2016_tetal | 22.4 (18.6-27.4) | 3.34 |
| cpx_wht1_interior1 | cpx_2023_exterior | 1.02e+03 (625-1.7e+03) | 3.43 |
| cpx_wht1_interior1 | cpx_ferriss2016_tetal | 988 (605-1.65e+03) | 3.33 |
| cpx_wht1_interior2 | cpx_2023_exterior | 38.9 (28.6-54.4) | 3.41 |
| cpx_wht1_interior2 | cpx_ferriss2016_tetal | 37.7 (27.7-52.7) | 3.31 |
| opx_grain_boundary | gb_demouchy | 0.00321 (0.00216-0.00567) | 0.0292 |
| opx_grain_boundary | gb_demouchy_x1000 | 3.21 (2.16-5.67) | 29.2 |
