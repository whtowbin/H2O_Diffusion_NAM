# Posterior decompression rates (Bayesian fit) versus the 2023 Monte Carlo

**Values come from unpublished data. Do not push this file to the public repo without Henry's OK.**

Model: METHODS.md (corrected 2026-10-17). Data: the 2023 profiles per case. sigma_i = case `one_sig` (SIMS 1 sigma), sigma_m = 0.
Priors on physical parameters: log10 v_P ~ U(-3, 4) with v_P in MPa/s (= 1e-4 to 1e3 MPa/s = 1e-3 to 1e4 bar/s; 1 MPa = 10 bar);
log10 D uniform between the case's 2023 T-20 K and T+20 K laws (the `*_2023_bracket`), or a named law with T ~ U(1473 +/- 20) K (table 4);
K ~ N(Kd_mean, Kd_sd); P0 ~ U(Pmin, Pmax); Pf ~ U(Pfmin, Pfmax) or the fixed Pf; L fixed at the 2023 value (table 5: L ~ U(L0 +/- 20 %), a posterior dimension read from the same table at xi_i(L); L with a point outside the crystal has zero likelihood); rho = 2800 +/- 100 kg/m3.
Grid: posterior on 700 log v_P cells x 41 log D nodes x 5 P0 x (2 or 6) Pf nodes, K-marginal log-likelihood interpolated in log tau;
G table: 400 log10 tau nodes from the solver floor (>= 10 time steps) to log10 tau = 2.5, 5 P0 x 2 Pf cell centres (Pf: 1 when fixed; 6 when the Pf prior spans > 10 bar, the three interior-cpx and grain-boundary cases),
solver numx = 2 (numx_2023 - 1) + 1; 11 L nodes when L varies; 40000 posterior draws (cells jittered uniformly), seed 0.
The priors reach log tau up to about 3.6; the table stops at log10 tau = 2.5 (cost grows with tau), so the likelihood is held at its edge value beyond.
Flags (METHODS 4.6): `lower` / `upper` = the likelihood is within 2 log-units of its maximum at the low-tau / high-tau table edge that the
prior can reach, so the data bound v_P from one side only; the table then shows the 95 % one-sided bound (`>` or `<`) instead of a median.

## 1. v_P (bar/s), 2023 bracket and priors on physical parameters: posterior vs saved 2023 MC fits

| case | flag | post median or 95 % bound | 68 % | 95 % | 2023 median | 2023 IQR | post/2023 | u median (m/s) |
|---|---|---|---|---|---|---|---|---|
| ol_nostall_file_in_stalling_folder | bounded | 1.51 | 1.31-1.75 | 1.15-2.01 | 1.53 | 1.41-1.7 | 0.99 | 5.52 |
| ol_with_stalling | bounded | 1.93 | 1.56-2.45 | 1.31-3.21 | 1.91 | 1.63-2.25 | 1.01 | 7.02 |
| opx_wht1_exterior | bounded | 6.9 | 3.72-12.6 | 2.46-19.6 | 5.05 | 2.9-8.13 | 1.37 | 25.1 |
| opx_wht1_exterior_higher_kd | bounded | 2.54 | 1.4-4.54 | 0.991-6.54 | 2.68 | 1.57-4 | 0.95 | 9.26 |
| opx_wht1_interior | bounded | 26.2 | 13.5-50.3 | 8.26-92.7 | 10.4 | 6.69-15.3 | 2.53 | 95.6 |
| opx_gcb_interior | bounded | 6.93 | 3.74-12.6 | 2.46-19.7 | 2.6 | 1.73-3.76 | 2.66 | 25.2 |
| cpx_wht1_exterior | bounded | 6.78 | 2.69-16.8 | 1.73-26.4 | 6.49 | 3.76-14.1 | 1.05 | 24.8 |
| cpx_wht1_interior1 | bounded | 299 | 145-615 | 77.3-1.25e+03 | 23.9 | 9.47-70.3 | 12.50 | 1.09e+03 |
| cpx_wht1_interior2 | bounded | 11.4 | 6.06-21.1 | 3.87-34.3 | 12.2 | 7.95-19.9 | 0.94 | 41.6 |
| opx_grain_boundary | bounded | 0.199 | 0.0262-1.45 | 0.0117-4.41 | 0.246 | 0.0453-1.2 | 0.81 | 0.725 |

## 2. Other posteriors, median (68 %)

| case | K | P0 (bar) | Pf (bar) | log10 tau |
|---|---|---|---|---|
| ol_nostall_file_in_stalling_folder | 9.03 (8.53-9.52) | 5.85e+03 (5.27e+03-6.42e+03) | 3.27 (1.47-4.49) | 1.25 (1.19-1.31) |
| ol_with_stalling | 9.03 (8.55-9.52) | 230 (185-276) | 2.82 (1.04-4.34) | -0.262 (-0.431--0.127) |
| opx_wht1_exterior | 96 (92.5-99.8) | 6.35e+03 (6.11e+03-6.59e+03) | 2.45 (0.767-4.17) | -0.335 (-0.459--0.218) |
| opx_wht1_exterior_higher_kd | 120 (115-124) | 6.35e+03 (6.11e+03-6.59e+03) | 2.46 (0.783-4.17) | 0.1 (0.00917-0.182) |
| opx_wht1_interior | 90.1 (87.5-92.8) | 6.35e+03 (6.11e+03-6.59e+03) | 5.02 (1.62-8.38) | -0.806 (-0.985--0.653) |
| opx_gcb_interior | 96 (92.5-99.8) | 6.36e+03 (6.12e+03-6.59e+03) | 0 (0-0) | -0.336 (-0.461--0.22) |
| cpx_wht1_exterior | 120 (115-125) | 6.35e+03 (6.11e+03-6.59e+03) | 2.52 (0.826-4.21) | 0.0596 (-0.0269-0.138) |
| cpx_wht1_interior1 | 119 (116-121) | 6.35e+03 (6.11e+03-6.58e+03) | 216 (73.4-344) | -1.58 (-1.8--1.37) |
| cpx_wht1_interior2 | 126 (122-130) | 6.35e+03 (6.11e+03-6.59e+03) | 361 (335-388) | -0.322 (-0.465--0.188) |
| opx_grain_boundary | 89.2 (83.2-94.8) | 6.35e+03 (6.11e+03-6.59e+03) | 172 (138-191) | 0.148 (-0.112-0.334) |

## 3. Runtime (12-core Apple M-series, spawn pool) and grid

Table build is one-off per case and path (12 workers), reusable for any data, sigma, priors, law or temperature range.
Posterior = grid posterior + 40k draws + summary for the 2023 bracket. The 2023-style column is 100 x the mean time of
`newcombe.fit_row` over 3 saved rows (numba solver, 2023 grid, one core). The machine was shared, so times are upper bounds.

| case | log10 tau table | table build (s) | posterior (s) | 2023-style MC, 100 fits, 1 core (s) |
|---|---|---|---|---|
| ol_nostall_file_in_stalling_folder | -3.12 to 2.50 | 10.8 | 0.09 | 26 |
| ol_with_stalling | -3.12 to 2.50 | 13.3 | 0.10 | 3 |
| opx_wht1_exterior | -3.47 to 2.50 | 31.9 | 0.10 | 30 |
| opx_wht1_exterior_higher_kd | -3.47 to 2.50 | 31.9 | 0.11 | 5 |
| opx_wht1_interior | -3.47 to 2.50 | 31.6 | 0.10 | 21 |
| opx_gcb_interior | -3.47 to 2.50 | 17.0 | 0.05 | 13 |
| cpx_wht1_exterior | -2.87 to 2.50 | 7.5 | 0.10 | 3 |
| cpx_wht1_interior1 | -3.47 to 2.50 | 89.3 | 0.11 | 14 |
| cpx_wht1_interior2 | -3.47 to 2.50 | 89.4 | 0.10 | 3 |
| opx_grain_boundary | -3.47 to 2.50 | 89.3 | 0.12 | 6 |

## 4. v_P (bar/s) under each prior on D (separate posterior per row; named laws T ~ U(1473 +/- 20) K)

| case | D prior | flag | v_P median or bound | 68 % | ratio to bracket |
|---|---|---|---|---|---|
| ol_nostall_file_in_stalling_folder | ol_2023_bracket | bounded | 1.51 | 1.31-1.75 | 1 |
| ol_nostall_file_in_stalling_folder | ol_ferriss2018_a | bounded | 1.52 | 1.31-1.75 | 1 |
| ol_with_stalling | ol_2023_bracket | bounded | 1.93 | 1.56-2.45 | 1 |
| ol_with_stalling | ol_ferriss2018_a | bounded | 1.93 | 1.56-2.46 | 1 |
| opx_wht1_exterior | opx_2023_bracket | bounded | 6.9 | 3.72-12.6 | 1 |
| opx_wht1_exterior | opx_carpenter2003_001 | bounded | 13.5 | 10.1-18.2 | 1.95 |
| opx_wht1_exterior | opx_demersroberge2021_100 | bounded | 38.7 | 29.2-51.9 | 5.61 |
| opx_wht1_exterior | opx_demersroberge2021_010 | bounded | 25.9 | 19.5-34.8 | 3.75 |
| opx_wht1_exterior | opx_demersroberge2021_001 | bounded | 47.2 | 35.7-63.1 | 6.85 |
| opx_wht1_exterior_higher_kd | opx_2023_bracket | bounded | 2.54 | 1.4-4.54 | 1 |
| opx_wht1_exterior_higher_kd | opx_carpenter2003_001 | bounded | 4.96 | 4-6.22 | 1.96 |
| opx_wht1_exterior_higher_kd | opx_demersroberge2021_100 | bounded | 14.2 | 11.6-17.8 | 5.62 |
| opx_wht1_exterior_higher_kd | opx_demersroberge2021_010 | bounded | 9.53 | 7.72-11.9 | 3.76 |
| opx_wht1_exterior_higher_kd | opx_demersroberge2021_001 | bounded | 17.4 | 14.2-21.6 | 6.85 |
| opx_wht1_interior | opx_2023_bracket | bounded | 26.2 | 13.5-50.3 | 1 |
| opx_wht1_interior | opx_carpenter2003_001 | bounded | 50.6 | 34.9-76.5 | 1.93 |
| opx_wht1_interior | opx_demersroberge2021_100 | bounded | 145 | 101-219 | 5.53 |
| opx_wht1_interior | opx_demersroberge2021_010 | bounded | 97.1 | 67.2-147 | 3.7 |
| opx_wht1_interior | opx_demersroberge2021_001 | bounded | 177 | 123-267 | 6.74 |
| opx_gcb_interior | opx_2023_bracket | bounded | 6.93 | 3.74-12.6 | 1 |
| opx_gcb_interior | opx_carpenter2003_001 | bounded | 13.5 | 10.2-18.2 | 1.95 |
| opx_gcb_interior | opx_demersroberge2021_100 | bounded | 38.8 | 29.3-52.1 | 5.61 |
| opx_gcb_interior | opx_demersroberge2021_010 | bounded | 26 | 19.6-34.9 | 3.75 |
| opx_gcb_interior | opx_demersroberge2021_001 | bounded | 47.4 | 35.9-63.4 | 6.84 |
| cpx_wht1_exterior | cpx_2023_bracket | bounded | 6.78 | 2.69-16.8 | 1 |
| cpx_wht1_exterior | cpx_2023_exterior | bounded | 23.2 | 19.3-28.4 | 3.42 |
| cpx_wht1_exterior | cpx_ferriss2016_tetal | bounded | 22.5 | 18.7-27.5 | 3.32 |
| cpx_wht1_interior1 | opx_2023_bracket | bounded | 299 | 145-615 | 1 |
| cpx_wht1_interior1 | cpx_2023_exterior | bounded | 1.02e+03 | 621-1.68e+03 | 3.42 |
| cpx_wht1_interior1 | cpx_ferriss2016_tetal | bounded | 990 | 602-1.63e+03 | 3.32 |
| cpx_wht1_interior2 | opx_2023_bracket | bounded | 11.4 | 6.06-21.1 | 1 |
| cpx_wht1_interior2 | cpx_2023_exterior | bounded | 38.8 | 28.3-54 | 3.4 |
| cpx_wht1_interior2 | cpx_ferriss2016_tetal | bounded | 37.7 | 27.5-52.3 | 3.3 |
| opx_grain_boundary | gb_2023_bracket | bounded | 0.199 | 0.0262-1.45 | 1 |
| opx_grain_boundary | gb_demouchy | bounded | 0.0138 | 0.0109-0.0213 | 0.0692 |

## 5. L-prior sensitivity: v_P (bar/s), 2023 bracket, L fixed vs L ~ U(L0 +/- 20 %)

Valid L = fraction of the 11 L nodes that keep every data point inside the crystal.

| case | L0 (um) | L fixed: v_P | L ~ U(+/-20 %): flag | v_P median or bound | 68 % | ratio to fixed | L posterior median (68 %) | valid L |
|---|---|---|---|---|---|---|---|---|
| ol_nostall_file_in_stalling_folder | 145 | 1.51 | bounded | 1.38 | 1.17-1.62 | 0.909 | 151 (145-157) | 1.00 |
| ol_with_stalling | 145 | 1.93 | bounded | 1.77 | 1.39-2.37 | 0.917 | 148 (143-154) | 1.00 |
| opx_wht1_exterior | 180 | 6.9 | bounded | 10.3 | 5.22-20 | 1.49 | 173 (170-177) | 0.73 |
| opx_wht1_exterior_higher_kd | 180 | 2.54 | bounded | 2.62 | 1.43-4.81 | 1.03 | 179 (173-184) | 0.73 |
| opx_wht1_interior | 160 | 26.2 | bounded | 26.1 | 13.4-50.4 | 0.994 | 159 (138-181) | 1.00 |
| opx_gcb_interior | 180 | 6.93 | bounded | 10.2 | 5.2-20 | 1.47 | 174 (170-177) | 0.73 |
| cpx_wht1_exterior | 115 | 6.78 | bounded | 6.61 | 2.63-16.6 | 0.975 | 124 (111-133) | 1.00 |
| cpx_wht1_interior1 | 113 | 299 | bounded | 298 | 145-622 | 0.997 | 113 (97.5-128) | 1.00 |
| cpx_wht1_interior2 | 134 | 11.4 | bounded | 11.4 | 6.05-21.2 | 0.999 | 142 (124-155) | 1.00 |
| opx_grain_boundary | 5e+04 | 0.199 | bounded | 0.203 | 0.0262-1.55 | 1.02 | 4.67e+04 (4.17e+04-5.46e+04) | 1.00 |
