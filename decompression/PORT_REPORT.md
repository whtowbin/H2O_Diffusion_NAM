# Port report: numbers from T1-T4

Run: `uv run pytest` -> `5 passed in 279.71s`. `uvx ruff check decompression tests`: All checks passed.
`uvx ty check --python .venv decompression tests`: All checks passed.
Reference: `$NEWCOMBE_REF`, Sullivan Ranch "No Stalling" case. Octave 11.3 is the only available proxy for MATLAB.

## Deviations from the plan (no physics changes)
- Inner FTCS loop is a numba `njit` kernel (new runtime dependency `numba`) because the numpy loop is slow for ~1e6 steps. A numpy-loop variant with the same arithmetic order is kept (`use_numba=False`); T2 checks the two are bit-identical on 6 parameter sets.
- Time steps are processed in blocks of 1e6 to bound memory; values are unchanged.
- Colon end rule: the plan's "length = floor(...)" rule is not sufficient. A first port of the Octave rule (tolerant floor, then drop the last element if `a+(n-1)*step` misses `b`) returned numt-1 elements for ~60% of random olivineMC-style cases, which would make MATLAB error. The final port includes Octave's neighbour check (accept n-1 or n+1 if that element lands on the limit, 3*eps relative to max(|u|,|v|)). It matches Octave's element count on all cases. MATLAB's own rule was not testable here.
- Octave's last colon element can differ from Python's by 1 ulp of max(|a|,|b|) (probably fused multiply-add in the Octave build), so T1 compares values relative to max(|a|,|b|).

## T1 colon and spline vs Octave
- 414 colon cases (awkward fixed cases, 300 olivineMC-style Pvec cases, 100 x-grids): element counts equal in all. Max relative difference of the last element, relative to max(|a|,|b|): 8.4e-17.
- 61 spline cases (n = 2..11 unsorted points, extrapolated xi, plus the 4-point Sullivan Ranch profile): max difference relative to the largest spline value: 1.1e-14.

## T2 forward model vs Octave running the unchanged olivineMC.m
- 27 parameter sets (24 across the priors, dP/dt 0.5-10; dP/dt = 1e-4 and 1e9 sentinel cases; Pf = 0, P0 = 6700 edge). 2 return the 1e6 sentinel in both.
- Misfit range 0.238 to 121.5. Max relative difference: **3.4e-15** (threshold 1e-9).
- Octave forward run twice on 3 sets: identical. Two identical Python fits: bit-identical.

## T3 fminsearch sanity
- Rosenbrock from [-1.2, 1]: x = [1.000022, 1.000042], fval 8.2e-10, 85 iterations, 159 evaluations (the MathWorks documentation example reports 85 / 159).
- 1-D quadratic with a dummy second dimension: x1 = 3.0; initial simplex is x0, x0 with each component scaled by 1.05 (0.00025 if zero), checked.

## T4 replay of the 100 saved MATLAB rows (Pf = 2.5, 5 s.f. inputs)
Control: rows 0, 10, ..., 90; for each, Pf in {0, 2.5, 5} x 3 inputs perturbed within +/- half a unit of the 5th significant figure, compared with the unperturbed Pf = 2.5 run. Tolerance = max over the 10 rows.
- Control spread: dP/dt 2.7%-3.8% per row, tolerance **3.85%**. fval 5.5%-50% per row, tolerance **50.1%** (Pf-dominated: the fval depends strongly on Pf, which was not saved).
- Replay vs saved, fitted dP/dt: median relative difference **1.43%**, max **3.69%**, rows outside tolerance: **0**.
- Replay vs saved, fval: median **6.1%**, max **55.1%**, rows outside tolerance: **1** (row 27: 55.1% vs 50.1%; next largest rows 10 and 61 at 32% and 31%).
- All 100 fits converged by the tolerance test (none stopped on maxiter/maxfev); evaluations 76-192.

Optimizer-independent check (misfit at the SAVED dP/dt, Pf scanned over [0, 5]): the saved fval lies within the range of misfit(Pf in [0,5]) for **100/100** rows. Best grid match (Pf step 0.2): median 0.23%, max 2.3%. At a fixed Pf the median relative difference is 21% (Pf 0), 6.1% (2.5), 9.8% (5).

Recovered-Pf replay (extra, not in the plan): solve misfit(saved dP/dt; Pf) = saved fval for Pf, refit with that Pf. The misfit is not monotone in Pf, 35 rows have two roots.
- Root choice is not known, so "best root" (closest to saved) is an upper bound on agreement. dP/dt: median **1.4e-4**, max **1.1e-3**; 95/100 rows < 1e-3; 42 < 1e-4. fval: median 2.3e-6, max 2.2e-4. Rows outside the control tolerance: 0. Rows 7, 8, 44, 59, 77 are at 1.0e-3 to 1.1e-3.
- Using the first root only: median 1.5e-4, max 5.1% (rows 43, 31, 62, 84, 95, 70 off by 0.8%-5%, since the first root is the wrong one for those rows).
