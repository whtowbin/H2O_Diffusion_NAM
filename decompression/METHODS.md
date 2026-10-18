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
Because `G` does not depend on the data, it is tabulated once per case and degassing path, storing the whole profile on the `ξ`
grid, over:
- `log10 τ`: 400 points from the explicit solver's floor (≈ −3.1 to −3.5, where a step count < 10 would be needed) to **2.5**, not a
  range chosen from the likelihood. The priors reach log τ ≈ 3.6; above the table `ℓ` is held at its edge value. This is an
  **approximation**: for `τ ≫ 1` the crystal tracks its rim and `G → H(Pf)`, but slowly, because the last degassing near `Pf` is steep.
  Measured (2026-10-17, max |G − H(Pf)| as a fraction of `H(P0) − H(Pf)`): at log τ 2.5, 0.7 % (olivine), 1.1 % (cpx interior 2),
  3.4 % (opx exterior); G(2.5) vs G(3.0) differ by 0.4–2.2 %. It only matters for posterior mass in the slow-ascent tail
  (log τ > 2.5); none of the ten cases' posteriors reach it, and the §4.6 flags report any that would. If one does, extend the table
  (cost grows ∝ τ). (Corrected twice on 2026-10-17: the first version asked for "the full range implied by the priors", infeasible
  with an explicit solver; a draft of this correction claimed the profile is converged at 2.5 "to rounding", which the measurement
  above disproves.)
- `P0`, `Pf`: table **nodes at the ends of each prior range**, default 5 P0 nodes and 2 Pf nodes, or 6 Pf nodes when the Pf prior
  spans more than 10 bar (the interior cpx and grain-boundary cases, 200–400 bar). The posterior interpolates the profile linearly onto
  15 P0 × 6 Pf integration cells. (Measured: treating 5 × 2 nodes as discrete prior points failed the K coverage test, 95 % interval
  0.88, and 2 Pf nodes moved log v_P by up to 0.07 dex for a 400-bar Pf prior.)
- space: a grid **2× finer** than the 2023 one (4× moved log v_P quantiles by ≤ 0.0017 dex in three cases and costs 3–5× more; the
  2023 grids themselves carry 0.2–0.8 % spatial error vs a fine reference).

From the table, the K-marginal log-likelihood `ℓ(log τ, P0, Pf)` (4.2) is computed for a data set in ~0.1 s. The posterior over the
physical parameters is

    p(log v_P, log D, P0, Pf | h) ∝ exp ℓ(log D + log(P0 − Pf) − 2 log L − log v_P, P0, Pf) · p(log v_P) p(log D) p(P0) p(Pf),

evaluated on a grid of `L` (11 nodes when `L` varies) × `log v_P` (350 cells, 0.02 dex) × `log D` (21 nodes from the D prior) ×
the P0 × Pf cells, with `ℓ` interpolated in `log τ`. This is exact up to grid resolution and has no MCMC noise. Temperature enters
only inside the `log D` prior, and each named diffusion law gets its own posterior on the same table. (Section 2, consequence 4 —
fitting `τ` and multiplying afterwards — is the flat-prior special case; the code does not use it.)
**Cost (measured 2026-10-17, 12 cores, machine shared, so upper bounds):** table build 8–90 s per case; posterior with 40k draws
≈ 0.1 s. A 2023-style 100-fit MC is 3–30 s per case on one core. The table is therefore still slower for a *single* fit and pays off
when reused — new priors, σ, diffusion laws, temperatures, L priors, or more profiles on the same geometry.
*Corrected 2026-10-17: an earlier version called the table "cheap" without measuring it (first build 12–725 s).*
**Checked:** `test_grid_convergence` (log10 v_P quantiles change < 0.01 dex when the τ and v grids are doubled and numx 2× → 4×;
measured 0.0091, dominated by the 0.02-dex v cells — refine the v grid before tightening this); `test_g_table_matches_forward`
(table entries equal direct forward runs); `test_coverage` — calibrated under the clamp above: about 25 % of prior-drawn truths
have log τ > 2.5 and are generated at the table edge.

### 4.4 Decompression rate, ascent speed, and the half-width
`v_P` comes directly from the posterior in 4.3. Ascent speed `u = v_P / (ρ g)` with `v_P` in Pa/s, `ρ` the density of the
overlying column (default 2800 ± 100 kg/m³, a documented input) and `g = 9.81 m/s²`.
`L` is a posterior dimension, not a post-hoc factor (section 2, consequence 3), and it needs no extra diffusion runs: the table
stores the whole dimensionless profile `G(ξ; τ, P0, Pf)`, so a value of `L` only changes where the data are read from it —
`ξ_i = |d_i|/L` for centre-referenced distances, `ξ_i = 1 − d_rim,i/L` for rim-referenced ones — and the conversion
`v_P = D (P0 − Pf)/(τ L²)`. Values of `L` that put a measured point outside the crystal have zero likelihood: centre-referenced
`|d_i| > L`; rim-referenced `d_rim,i > 2L`. A rim-referenced point with `L < d_rim,i ≤ 2L` lies past the centre in the other half of
the slab and is read at `|ξ_i| = |1 − d_rim,i/L|` (valid for the symmetric slab; it assumes the traverse crossed the centre — reject
such `L` instead if it did not). Prior: the
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
such cases as one-sided bounds (95 % limit) and flags them, instead of a median. With priors on the physical parameters (4.1), all ten
2023 cases are bounded (none flagged). *Corrected 2026-10-17: an earlier version said `opx_wht1_interior` and `cpx_wht1_interior1`
were unbounded; that came from the old flat-in-τ prior. Their 2.5× and 12.5× gaps to the 2023 medians come instead from saved runs
that do not match the preserved scripts (Kd, P0; open question Q26) and from the 2023 fminsearch starting range.*

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

## 7. Which inputs need a new G table

The table `G(ξ; τ, P0, Pf)` is the solution of the dimensionless problem in section 2. Anything that only rescales that solution, or
only changes where and how it is read, reuses the table; anything that changes the *shape* of the problem the PDE solves needs a new
one. Building a table costs seconds to minutes (5 × 2 P0 × Pf grid; olivine ≈ 1 s on 12 cores); after that each combination of
everything in 7.1 costs ≈ 0.01 s.

### 7.1 Reuse the table (no new diffusion runs)

| input | why | where it enters |
|---|---|---|
| `v_P`, `D`, `T`, diffusion law | only through `τ = D (P0 − Pf)/(v_P L²)` (property 1) | the `τ ↔ v_P` conversion and the `log D` prior |
| half-width `L` | changes where the data are read, `ξ_i(L)`, and the `τ ↔ v_P` conversion | section 4.4 |
| constant partition coefficient `K` (value or prior) | profile ∝ `K` (property 2) | closed-form marginal, section 4.2 |
| degassing path multiplied by a constant | indistinguishable from a different `K` | — |
| `P0`, `Pf` inside the tabulated range | already table axes | widen the range once if a new prior needs it |
| SIMS σ, model-error σ_m, priors, new profiles, other grains | only the likelihood changes | — |
| mineral phase | `G` is dimensionless and phase-free; phases differ only in `D` and `K` | profiles sharing a degassing path, pressure history and P0/Pf ranges share one table |

### 7.2 Need a new table

| input | why | tables |
|---|---|---|
| shape of the degassing path `H(P)` (solubility model, starting H2O/CO2, open/closed/half-open, degassing T) | the rim boundary follows a different curve, not a rescaled one | one per path |
| shape of the pressure history (stalling, two-stage or accelerating ascent) | write `P(t) = P0 − (P0 − Pf) π(s)`; the linear ascent is `π(s) = s`. The rim value at reduced time `s` is `H(P0 − (P0 − Pf) π(s))`, a different function of `s` for each `π` | one per `π`; a stall depth/duration can be a table axis |
| `K` that varies along the path, `K(P) = K0 f(P)` (e.g. olivine/melt partitioning falling with melt H2O; Peterson et al.) | the rim sees `K0 f(P) H(P)`: an effective path `f·H` | one per shape `f`; `K0` still integrates out in closed form |
| initial profile not in equilibrium with the melt at `P0` (mantle value, pre-existing zoning) | breaks property 2 (initial value no longer ∝ `K`) | two tables combined by superposition (7.3) |
| `D` changing during ascent (cooling, or `T` rising/falling with depth) | with reduced time `θ = ∫ D dt / L²` the PDE is again `∂C/∂θ = ∂²C/∂ξ²`, but the rim value as a function of `θ` depends on the shape `φ` of `D(t)/D_ref` | one per `φ` |
| geometry (sphere, 3-D, anisotropic 3-D), concentration-dependent `D` | a different PDE (the latter non-linear: properties 1–2 no longer hold) | a new solver |
| numerics (`numx`, `τ` range) | the table is a discretisation | rebuild; convergence test 4.3 |

### 7.3 Superposition for a non-equilibrium initial profile
Let the crystal start at a uniform value `c0` (e.g. a mantle concentration) that need not equal `K H(P0)`. The problem is linear, so
its solution is the sum of two problems:

    A(ξ, s): ∂A/∂s = τ ∂²A/∂ξ²,  A(ξ, 0) = 0,  ∂A/∂ξ(0, s) = 0,  A(1, s) = H(P0 − (P0 − Pf) π(s))   (rim-driven part)
    B(ξ, s): ∂B/∂s = τ ∂²B/∂ξ²,  B(ξ, 0) = 1,  ∂B/∂ξ(0, s) = 0,  B(1, s) = 0                           (decay of the initial value)

    C(ξ, 1) = K · A(ξ, 1; τ, P0, Pf) + c0 · B(ξ, 1; τ).

Proof: both sides satisfy the same linear PDE, the same no-flux centre, the rim value `K H` (since `B = 0` there) and the initial
value `c0` (since `A = 0` there); the solution is unique. ∎ `B` depends on `τ` only (no path), so it is a single 1-D table shared by
everything. The equilibrium start is the special case `c0 = K H(P0)`, giving `G = A + H(P0) B` (a check: it must equal the existing
table to rounding). With Gaussian priors on `K` and `c0`, the likelihood is linear-Gaussian in `(K, c0)` and both integrate out in
closed form — section 4.2 with the scalar `a` replaced by the 2 × 2 matrix `M = Gᵀ S⁻¹ G + Σ_prior⁻¹`, `b = Gᵀ S⁻¹ h + Σ_prior⁻¹ μ`,
and `log p(h | τ, P0, Pf) = −½ [hᵀ S⁻¹ h + μᵀ Σ_prior⁻¹ μ − bᵀ M⁻¹ b] − ½ log det(M Σ_prior) − Σ log(√(2π) s_i)`, where `G` has the
two columns `A(ξ_i)` and `B(ξ_i)`.

## 8. Inferring pyroxene diffusivity from the olivine rate (the paper's approach)

The 2023 chapter fixes Stage II decompression from the olivine phenocryst (≈ 0.2 MPa/s, Figs 11–12) and solves for the opx and cpx
H diffusivities that reproduce the pyroxene profiles at that rate; experimental laws (Demers-Roberge et al. 2021, Ferriss et al. 2016)
are comparisons, not inputs, because pyroxene composition (Al, Cr) is expected to lower D. Property 1 makes this exact: a pyroxene
profile constrains `τ_px = D_px (P0 − Pf)/(v_P L_px²)`, so a known `v_P` determines `D_px`.

**Joint posterior.** Let `q(v_P, P0, Pf)` be the olivine posterior (section 4) for the shared ascent path. For a pyroxene profile, put a
flat prior on `log D_px` (wide range, e.g. −14 to −8) and use `q` as the prior on `(v_P, P0, Pf)`:

    p(log D_px | h_px, h_ol) ∝ ∫ exp ℓ_px(log D_px + log(P0 − Pf) − 2 log L_px − log v_P, P0, Pf) · q(v_P, P0, Pf | h_ol) dv_P dP0 dPf.

With samples from `q` and the pyroxene table this costs ≈ 0.1 s per profile. Assumptions, to state in the paper: the olivine phenocryst
and the xenolith pyroxenes experienced the same Stage II path (P0, Pf, history) at the same T; the pyroxene K prior and L as in
section 4. If the paths differ (e.g. stalling), `q` must be the olivine posterior *for that path*.

**Quick check (2026-10-17, ratio of posterior medians, 2023 brackets; to be replaced by the joint posterior).** With `v_P` ≈ 1.7 bar/s
(olivine, 1.5–1.9): implied log D (1200 °C, m²/s) ≈ −11.4 (opx WHT1 exterior), −11.0 (opx exterior, higher K), −12.0 (opx WHT1
interior), −11.4 (cpx WHT1 exterior). The deposit text gives cpx ≈ 25× below olivine (≈ −11.4) and the lowest opx −11.15, so the
new fit reproduces the chapter. Relative to experiments: cpx sits at Ferriss et al.'s mantle-cpx estimate (~10⁻¹¹) and between the
anchors in section 5; opx is 1–2 log units below Demers-Roberge et al. 2021 (−9.95 to −10.21) — "low end, attributed to composition".
