# Decompression rates from H profiles: model, exact reductions, and the Bayesian fit

This note explains the diffusion model used to fit decompression rates (dP/dt) from hydrogen profiles in olivine and pyroxene,
proves two exact properties of it, and uses them to replace the 2023 Monte-Carlo-of-fminsearch with a posterior that is
faster, reproducible and carries every uncertainty explicitly. Each claim lists the test that checks it.

Notation: `C(x, t)` H2O in the crystal (ppm), `x ∈ [0, L]` distance from the crystal centre (µm), `L` = half-width ("radius" in
the MATLAB), `D` H diffusivity (µm²/s), `P(t)` melt pressure (bar), `H(P)` H2O dissolved in the melt along the degassing path
(wt%), `K` mineral/melt partition coefficient (the MATLAB `Kd`, ×10⁻⁴ absorbed into units), `P0`, `Pf` start and end pressures,
`v_P = dP/dt` decompression rate (bar/s, positive), `h_i` measured H2O at distance `d_i`, `σ_i` its 1σ uncertainty.

## 1. The model (as in the 2023 MATLAB, `olivineMC.m`)

Pressure falls linearly, `P(t) = P0 − v_P t`, for a total time `t_tot = (P0 − Pf)/v_P`. The crystal is a 1-D slab with a no-flux
centre and a rim held in equilibrium with the degassing melt:

    ∂C/∂t = D ∂²C/∂x²,            0 < x < L, 0 < t ≤ t_tot
    ∂C/∂x (0, t) = 0               (symmetry at the centre)
    C(L, t) = K · H(P(t))          (rim in equilibrium with the melt)
    C(x, 0) = K · H(P0)            (crystal starts in equilibrium at P0)

The model profile is `C(x, t_tot)`, compared with the data at `|d_i|`. `D` follows an Arrhenius law `D(T) = D0 exp(−Ea/RT)`.
The MATLAB solves this with explicit finite differences (FTCS) on `numx` nodes with `numt = ceil(2 D t_tot / (0.9 Δx²) + 1)` steps;
the Python port reproduces it to ≤ 1e-14 relative (Octave tests, `tests/test_newcombe.py`, `tests/test_cases.py`).

Note: this is a slab, not a sphere (no 2/r term), and 1-D. Mutch et al. (2024) show 1-D fits can under-estimate H loss by ~30 %
for 3-D crystals; that is a model limitation, not something the reductions below change.

## 2. Exact property 1: D and dP/dt only enter as one dimensionless time

**Claim.** For a given degassing path `H`, start and end pressures `P0, Pf` and half-width `L`, the final profile depends on `D`
and `v_P` only through

    τ = D · t_tot / L² = D (P0 − Pf) / (v_P L²).

**Proof (continuous).** Rescale `ξ = x/L ∈ [0, 1]` and `s = t/t_tot ∈ [0, 1]`. Then `∂/∂t = (1/t_tot) ∂/∂s` and
`∂²/∂x² = (1/L²) ∂²/∂ξ²`, so the equation becomes

    ∂C/∂s = τ ∂²C/∂ξ²,   ∂C/∂ξ(0, s) = 0,   C(1, s) = K · H(P0 − (P0 − Pf) s),   C(ξ, 0) = K · H(P0).

`D`, `v_P` and `L` have disappeared except through `τ`; the boundary path is a function of `s`, `P0`, `Pf` only. Hence
`C(ξ, 1) = K · G(ξ; τ, P0, Pf)` for a function `G` that does not depend on `D`, `v_P` or `L` separately. ∎

**Proof (the discrete scheme is invariant too).** Replace `(D, v_P)` by `(aD, a v_P)` for any `a > 0`. Then `D t_tot` is unchanged,
so `numt` is unchanged; `Δt = t_tot/(numt − 1)` becomes `Δt/a`, so the FTCS number `r = D Δt/Δx²` is unchanged; the pressure
step `ΔP = v_P Δt` is unchanged, so the same pressures, hence the same boundary values, are used at every step. Every arithmetic
operation is the same up to the rounding of `r` and `ΔP`. ∎
**Checked:** `test_scaling_invariance` — max relative difference ≤ 3e-16 on saved rows of all 10 cases (measured 2026-10-17:
1.2e-16 to 2.6e-16).

**Consequences.**
1. The data constrain `τ`, not `v_P` and `D` separately. Any decompression rate is

       v_P = D(T) (P0 − Pf) / (τ L²).

2. Changing the diffusion law rescales the fitted rate exactly: `v_P,new = v_P,old · D_new/D_old` (same `τ`, same path). The 2023
   fits can be converted to any new diffusivity (Ferriss 2016 cpx with tetrahedral Al, Demers-Roberge 2021 opx, …) without refitting.
   (fminsearch's absolute `TolX` makes the 2023 optima approximate at the 1–3 % level; the posterior below has no such error.)
3. `L` is **not** a pure rescaling for real data. The measured distances `d_i` are fixed in µm, so changing `L` moves the
   sampling points `ξ_i = d_i/L` as well as the `τ ↔ v_P` conversion. (Only if the profile were measured in fractions of the
   half-width would `v_P ∝ 1/L²` hold.) `L` must therefore be a dimension of the fit (section 4.4) — which costs no extra
   diffusion runs, because the table holds the whole profile in `ξ`.
   *Corrected 2026-10-17: an earlier version said `L` could be propagated after the fit; the build review showed that is wrong.*
4. Temperature only acts through `D(T)` (with a fixed path). It can be propagated after the fit **when the prior on `log v_P` is
   flat over the region the data allow**; otherwise it must be integrated together with `v_P` (section 4.3).

## 3. Exact property 2: the partition coefficient only scales the profile

**Claim.** `C(·; K) = K · C(·; K = 1)`.

**Proof.** The diffusion equation and the no-flux condition are linear and homogeneous; the initial and rim conditions are both
proportional to `K`. So `K · C(·; 1)` satisfies the problem with coefficient `K`, and the solution is unique. In the discrete scheme
every update is a linear combination of previous values and boundary values, and the cubic spline used to read the model at the
data points is linear in the node values. ∎
**Checked:** `test_kd_linearity` — exact (0.0 measured) on all cases.

**Consequence.** Given `G`, the likelihood is a quadratic in `K`, so `K` can be integrated out analytically (section 4.2) instead of
being sampled.

## 4. The Bayesian fit (new)

### 4.1 Likelihood and priors
For data `h_i ± σ_i` at `d_i` (σ_i: SIMS 1σ, optionally inflated by a model-error term σ_m: `s_i² = σ_i² + σ_m²`),

    h_i = K · g_i(τ, P0, Pf) + ε_i,   ε_i ~ N(0, s_i²),   g_i = G(|d_i|/L; τ, P0, Pf).

Parameters and priors. The physical parameters carry the priors; `τ` is a derived quantity (corrected 2026-10-17: the first
version put an unspecified uniform prior directly on `log τ`, which made poorly constrained cases depend on an arbitrary range):
- `log10 v_P ~ U(a, b)`, with `(a, b)` a stated physical range (default −3 to +4, i.e. 10⁻⁴–10³ MPa/s). The 2023 runs only *started*
  fminsearch in `U(−1, 1)` (olivine) — that was a starting range, not a prior.
- `log10 D`: the case's diffusion prior. For the 2023 cases this is uniform between the "hi" and "lo" laws (section 5); for a named
  law, `T ~ U(T0 − 20, T0 + 20)` K mapped through that law.
- `K ~ N(μ_K, s_K²)` (per case, e.g. N(9, 0.5) for the olivine runs).
- `P0 ~ U(Pmin, Pmax)`, `Pf ~ U(Pfmin, Pfmax)` (per case).
- `L`: fixed at the 2023 value by default (see 4.4 for an optional L dimension).
- Diffusion law: one of the named laws or brackets in section 5 (a model choice, reported side by side, not averaged).

These induce the prior on `log τ = log D + log(P0 − Pf) − 2 log L − log v_P` (a convolution of the `log v_P` and `log D` priors).

### 4.2 Integrating out K (closed form)
With `a = Σ g_i²/s_i² + 1/s_K²` and `b = Σ h_i g_i/s_i² + μ_K/s_K²`, the posterior of `K` given `(τ, P0, Pf)` is
`N(b/a, 1/a)` and the marginal log-likelihood is

    log p(h | τ, P0, Pf) = −½ [ Σ h_i²/s_i² + μ_K²/s_K² − b²/a ] − ½ log(a s_K²) − Σ log(√(2π) s_i).

(Derivation: complete the square in `K` in `−½[Σ (h_i − K g_i)²/s_i² + (K − μ_K)²/s_K²]` and integrate the Gaussian.)
The prior mass at `K < 0` is negligible for the priors used (μ_K/s_K ≥ 10); the code asserts this.
**Checked:** `test_kd_marginal_matches_quadrature` — closed form vs numerical integration over K to 1e-8.

### 4.3 Posterior
Because `G` does not depend on the data, it is tabulated once per case and degassing path on a grid of `log10 τ` (400 points over the
full range implied by the priors in 4.1, not a range chosen from the likelihood), `P0` (default 5) and `Pf` (default 2), on a spatial
grid 4× finer than the 2023 one (2023 grids carry 0.2–0.8 % spatial error vs a fine reference). From the table, the K-marginal
log-likelihood `ℓ(log τ, P0, Pf)` (4.2) is computed for a data set in ~0.01 s. The posterior over the physical parameters is

    p(log v_P, log D, P0, Pf | h) ∝ exp ℓ(log D + log(P0 − Pf) − 2 log L − log v_P, P0, Pf) · p(log v_P) p(log D) p(P0) p(Pf),

evaluated on a grid in `log v_P` × samples of `log D` × the `(P0, Pf)` nodes, with `ℓ` interpolated in `log τ`. This is exact up to
grid resolution and has no MCMC noise. When `p(log v_P)` is flat wherever `ℓ` is non-negligible, it reduces to "fit `τ`, then
multiply by `D(P0 − Pf)/L²`" (section 2, consequence 4).
**Cost (measured 2026-10-17, 12 cores, machine loaded):** table build 12–725 s per case (cost grows with `τ`; the slowest case has
`τ ≈ 18`); posterior 0.01 s. A 2023-style 100-fit MC is 2–26 s per case. The table is therefore *slower* for a single fit and pays
off when reused — new priors, σ, diffusion laws, temperatures, or more profiles on the same geometry. The P0 × Pf grid was reduced from
25 × 5 to 5 × 2 after measuring that this moves log τ quantiles by ≤ 0.004 (olivine) and 1e-4 (cpx) and cuts the build ~20×.
*Corrected 2026-10-17: an earlier version called the table "cheap" without measuring it.*
**Checked:** `test_grid_convergence` (posterior quantiles of log τ change < 0.01 when the τ grid is doubled and numx doubled);
`test_g_table_matches_forward` (table entries equal direct forward runs).

### 4.4 Decompression rate, ascent speed, and the half-width
`v_P` comes directly from the posterior in 4.3. Ascent speed `u = v_P / (ρ g)` with `v_P` in Pa/s, `ρ` the density of the
overlying column (default 2800 ± 100 kg/m³, a documented input) and `g = 9.81 m/s²`.
`L` is a posterior dimension, not a post-hoc factor (section 2, consequence 3), and it needs no extra diffusion runs: the table
stores the whole dimensionless profile `G(ξ; τ, P0, Pf)`, so a value of `L` only changes where the data are read from it —
`ξ_i = |d_i|/L` for centre-referenced distances, `ξ_i = 1 − d_rim,i/L` for rim-referenced ones — and the conversion
`v_P = D (P0 − Pf)/(τ L²)`. Values of `L` that put a measured point outside the crystal (`ξ_i > 1`) have zero likelihood. Prior: the
2023 value fixed (default, reproduces 2023), or a measured half-width with an allowance for off-centre sectioning and 3-D shape.

*Measured 2026-10-17 (single fminsearch fits, measured distances held fixed): the fitted rate responds to `L` × 0.8 / × 1.25 by
× 1.87 / × 0.52 (olivine, centre-referenced) and × 1.30 / × 0.86 (cpx exterior, rim-referenced), vs × 1.56 / × 0.64 for a pure `L²`
rule; for opx WHT1 exterior, `L` × 0.8 puts the outermost point outside the crystal. The response lies between `L⁰` (diffusion front
far from the centre: only √(Dt) in µm is recorded) and `L²` (core depleted: `Dt/L²` is recorded), and the misfit itself changes with
`L` (opx WHT1 interior: 736 at `L`, 270 at 1.25 `L`), so the data carry information on `L`.*
Report: median and 68/95 % intervals of `log10 v_P`, `v_P` (MPa/s) and `u` (m/s), plus the posterior of `K`, `P0`, `Pf`.

### 4.6 Profiles that do not bound the rate
If `ℓ` stays within 2 log-units of its maximum all the way to the fast-ascent end (small `τ`: too little diffusion to resolve), the
data give only a lower bound on `v_P` (and the median is set by the prior's upper limit). Likewise at the slow end. The code reports
such cases as one-sided bounds (95 % limit) and flags them, instead of a median. In the first build this applied to `opx_wht1_interior`
and `cpx_wht1_interior1`, the two cases that disagreed most with 2023.

### 4.5 What is different from the 2023 Monte Carlo
The 2023 procedure added noise to the data, drew `K` and `D` from priors, and ran fminsearch (absolute TolX 0.1–1 bar/s) 20–100
times. The spread of those optima mixes (i) the D prior — which, by property 1, is just a rescaling, (ii) the noise, and (iii)
optimizer stopping error; it is not a posterior. The new fit gives the posterior directly, separates the sources, and is exact on
its grid. Agreement with the 2023 numbers is expected for the centre of the distribution and is tested (section 6).

## 5. Diffusion laws (log10 D at 1200 °C in m²/s)

| key | law | source | log D(1200 °C) |
|---|---|---|---|
| `ol_ferriss2018_a` | 10^-5.4 exp(−130 kJ/RT) | Ferriss et al. 2018, ‖a (= 2023 olivine runs, = `DH2O_Ol`) | −10.01 |
| `opx_carpenter2003_001` | 10^-4.788 exp(−161 kJ/RT) | Carpenter 2003 thesis [001] (2023 opx runs) | −10.50 |
| `opx_demersroberge2021_{100,010,001}` | per Demers-Roberge et al. 2021 GCA | literature review 2026-10-17 (Q20) | −10.04 / −10.21 / −9.95 |
| `cpx_2023_exterior` | 10^-6.71 exp(−100 kJ/RT) | 2023 estimate for 0.07 Al(IV) "based on Ferriss 2016" | −10.26 |
| `cpx_ferriss2016_tetal` | anchor + s·(log10 TetAl − log10 TetAl_ref), s = −1.0485 | Ferriss et al. 2016 fig. 12 slope; anchor default = `cpx_2023_exterior` at TetAl_ref = 0.07 (Q21) | depends on TetAl |
| `gb_demouchy` | 10^-6.2 exp(−71 kJ/RT) (×1000 in a 2023 sensitivity run) | Demouchy, effective grain boundary | −8.72 |

**The 2023 priors are brackets, not single laws** (from each `master_script_test.m`, lines 45–49): `log D` was drawn uniformly
between `DHhi` at T+20 K and `DHlo` at T−20 K, and for opx and cpx those are *different* laws:
- olivine: the same law at both ends (temperature only), log D −10.07 to −9.95;
- opx: Carpenter [001] at T+20 K (−10.42) to Carpenter [100], 10^-6.6 exp(−127 kJ/RT), at T−20 K (−11.17) — an orientation + temperature bracket;
- cpx exterior: 10^-6.71 with Ea = 100 kJ at T+20 K to Ea = 130 kJ at T−20 K — an activation-energy + temperature bracket.
Ratios in the law table are relative to these brackets (keys `*_2023_bracket`), not to a single named law.

Because of property 1, the decision table for the paper is cheap: for each case, `v_P` under each law = the posterior of
`(P0 − Pf)/(τ L²)` times that law's `D(T)`.

## 6. Validation (what proves the new code right)
1. **Port fidelity** (existing): Octave forward ≤ 1e-14; replay of 1,000 saved MATLAB fits within control tolerances.
2. **Exact properties**: `test_scaling_invariance`, `test_kd_linearity`, `test_kd_marginal_matches_quadrature`.
3. **Recovery on synthetic data**: profiles generated with known `(v_P, K, P0, Pf, T)` and noise at the case's σ; across ≥ 200
   synthetic data sets the 68 % and 95 % intervals contain the truth at their nominal rates (± binomial error) — `test_coverage`.
4. **Agreement with 2023**: for each case, the posterior median `v_P` (2023 law, 2023 priors) vs the median of the saved 2023 MC fits;
   differences reported, with the expected size from the fminsearch tolerance and the D-prior rescaling. Not forced to agree.
5. **Grid convergence** as in 4.3.
