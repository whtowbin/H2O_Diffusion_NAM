# VESIcal degassing paths as the melt-H2O boundary: numbers

VESIcal 1.2.12 (Python 3.13, ThermoEngine absent, so MagmaSat cannot run and `degassing_path` raises for it).
Numbers come from `tests/test_degassing.py` (`DEGASSING_NUMBERS_JSON=... uv run pytest tests/test_degassing.py`).
MI compositions and Sullivan Ranch values are read at test time from the Ch3 repo and are not stored here.

## Layout of the existing boundary file
`VesiCal1200_VaporBubble_H2O_only.txt` = the VESIcal export (descending P, 202 rows, last row 1 bar) plus an
appended anchor row `0 0` (203 rows, 8 s.f.). `boundary_arrays` reproduces it (rtol 1e-6) from the
`GCB_Closed_Degassing_1200C_CO2 Highest Bubble_Corrected.xlsx` export.

## Which composition the Feb 2023 exports started from
- Dixon/Shishkina/IaconoMarziano files: first row is the *starting pressure given by the user* (5000, 5000, 10000 bar),
  fluid-free, H2O_liq 3.36 and CO2_liq 0.3296 wt% exactly; 200 steps down to 100 bar. Called as
  `pressure=<number>, final_pressure=100, steps=200`.
- No single T3 GCB inclusion has that (H2O, CO2) pair. H2O 3.36 is the largest GCB H2O (inclusion ol23); CO2 3296.89
  ppm is the largest GCB CO2 (inclusion "Corrected Ol 11"), truncated to 0.3296 wt%.
- The Dixon model in VESIcal sees SiO2 only (±1 % on any other oxide changes the path by < 1e-4 wt%). Fitting SiO2
  to the Dixon export gives a best-fit that matches `mean(SIO2normANHYD)` over the 14 GCB inclusions to 4 s.f.
  (fit SiO2 differs from that mean by < 1e-3). Raw SiO2 of every single GCB inclusion is worse: best single inclusion
  max abs 1.97e-2 wt% H2O vs 2.5e-4 for the mean composition.
- Identified start: mean of the 14 GCB `*normANHYD` oxide columns (SIO2normANHYD ... P2O5normANHYD), H2O = max GCB H2O,
  CO2 = max GCB CO2 truncated to 4 dp (wt%). Same composition is used for all three models.
- MagmaSat files (unprefixed): cannot be regenerated. "Highest_Measured_Inclusion" starts at 3810.9 bar, H2O 3.359282,
  CO2 0.327198 (not the same CO2 as the input 0.3296, so MagmaSat's saturation solve is not simply the input).
  The "CO2 Highest Bubble_Corrected" file (6756.1 bar, 3.35925, 0.847607 wt% CO2): no T3 column matches 0.8476 wt%.
  The largest GCB `CO2vbg` is 8528.83 ppm (ratio 0.9938 to the file's value; the measured-case ratio is 0.9927), so
  that column is the likely source, but this is unverified without MagmaSat.

## T1: regenerate Feb 2023 paths (H2O_liq vs pressure, regenerated path interpolated onto the saved pressures)
| model | max abs (wt%) | max rel | control: same call twice | control: round input to 2 dp | control: oxides +0.5 % | control: oxides ±2 % random (4 draws), misfit to saved |
|---|---|---|---|---|---|---|
| Dixon | 2.5e-4 | 8.0e-5 | bit-identical | 1.4e-4 | 1.2e-2 | 1.5e-3 .. 4.0e-2 |
| ShishkinaIdealMixing | 2.85e-2 | 1.10e-2 | bit-identical | 2.4e-4 | 1.2e-4 | 2.58e-2 .. 3.08e-2 |
| IaconoMarziano | 2.25e-1 | 9.6e-2 | bit-identical | 3.3e-4 | 2.2e-3 | 2.17e-1 .. 2.44e-1 |

(CO2_liq max abs: Dixon 2.2e-4, Shishkina 5.2e-3, IaconoMarziano 1.3e-2.)

Reading the numbers:
- Dixon: agreement is at the level of composition rounding (2 dp rounding gives 1.4e-4, same size as the difference).
  The Dixon model is unchanged between 2023 and 1.2.12 as far as this test can see.
- Shishkina and IaconoMarziano: the differences are insensitive to the composition (±2 % jitter of all oxides moves the
  misfit by < 5e-3 and < 3e-2 respectively; Fe-speciation variants and raw-vs-normalised oxides give 0.028-0.031 and
  0.21-0.30). Differences start at the onset of degassing and grow smoothly toward low pressure (Shishkina: 1e-4 at
  4015 bar, 2.4e-2 at 1060 bar; IaconoMarziano: onset of H2O loss differs, 0.11 wt% at 1045 bar). The composition would
  need to differ by far more than the Dixon-pinned SiO2 allows, so a model change between the 2023 VESIcal and 1.2.12
  is the more likely cause. Not proven: the 2023 VESIcal version is not installed here.

## T2: Dixon path through the forward model
Dixon path: identified composition, 1200 C, start 7000 bar, 200 steps to 1 bar, `boundary_arrays` (with the 0,0 anchor).
`nc.forward` and `nc.misfit` run on it with finite results; the second call is served from the cache and is identical.

Refit of 3 saved rows (rows 0, 40, 80; Pf = 2.5, fminsearch from the saved x0, tolx = tolfun = 1; MagmaSat txt vs Dixon):

| row | saved dP/dt | MagmaSat txt dP/dt (fval) | Dixon dP/dt (fval) | Dixon / MagmaSat | melt H2O at P0: MagmaSat / Dixon |
|---|---|---|---|---|---|
| 0 | 1.5345 | 1.5447 (0.0789) | 0.4184 (0.4195) | 0.271 | 3.324 / 3.360 |
| 40 | 1.7520 | 1.7329 (0.1712) | 0.5568 (0.5156) | 0.321 | 3.316 / 3.360 |
| 80 | 1.7214 | 1.7664 (0.1146) | 0.6178 (0.5409) | 0.350 | 3.335 / 3.360 |

The Dixon path is a different scenario from the saved one (starts undersaturated at 3.36 wt% H2O / 0.33 wt% CO2, not at the
6756 bar, 0.85 wt% CO2 bubble-corrected start, and is flat in H2O until saturation); the ratios are the effect of swapping the whole boundary path, not of the
solubility model alone.

## Follow-up: VESIcal 1.2.4 (the current release in Feb 2023) vs 1.2.12
Isolated env outside the repo (`uv run --no-project --python 3.11 --with VESIcal==1.2.4 --with 'numpy<2' --with 'pandas<2.2'`),
same identified composition and call as T1 (1200 C, start pressure and step count taken from each export).

| model | 1.2.4 vs export: max abs / max rel | 1.2.12 vs export | 1.2.4 vs 1.2.12 (pressure grid and H2O_liq) |
|---|---|---|---|
| Dixon | 2.539e-4 / 8.04e-5 | 2.539e-4 / 8.04e-5 | identical (0.0) |
| ShishkinaIdealMixing | 2.847e-2 / 1.101e-2 | 2.847e-2 / 1.101e-2 | identical (0.0) |
| IaconoMarziano | 2.250e-1 / 9.58e-2 | 2.250e-1 / 9.58e-2 | identical (0.0) |

Result: 1.2.4 does not reproduce the Shishkina or IaconoMarziano exports; the two versions give bit-identical
paths for all three models. This retracts the T1 suggestion that a model change between 2023 and 1.2.12 explains the
mismatch: there is no change to name between 1.2.4 and 1.2.12 for these inputs. The cause lies in the inputs of the
2023 runs (composition beyond SiO2, which only Dixon is sensitive to, or model options), not in the package version.
Temperature is not it: Shishkina is independent of T (1000-1400 C identical), IaconoMarziano moves only 0.222-0.229.
Untested: versions before 1.2.4, and composition/normalisation variants beyond those in T1.

## Note added 2026-10-17 (main session)
The 2023 MiMiC+degassing workbook ("GCB Degassing Models" sheet) stores only the model outputs, not the input compositions, so the
Shishkina/IaconoMarziano start compositions cannot be recovered from it. Its IaconoMarziano column is headed "IaconoMarziano 1200 C
Pressure doesn't seem to be correct" (Henry, 2023): that export was already considered suspect. Neither export fed the deposited
fits (those used the MagmaSat path), so the mismatch does not affect the replayed results.
