# H2O diffusion in nominally anhydrous minerals

Finite-difference models of hydrogen diffusion in olivine and mantle clinopyroxene, using diffusivities from Ferriss et al. (2016, 2018). The code builds a diffusion matrix, steps profiles through time, and scores model profiles against measured ones by the sum of squared residuals.

```bash
uv run python "Diffusion Modeling.py"
```

For H2O profiles measured in two olivines (3Ol2 and 12Ol3, entered in the script), it models diffusion at 1100 °C, finds the diffusion time that best fits each profile, and plots the model against the data.

## Status

This is a working research repository, not a finished package. Some files are exploratory analyses or earlier versions kept for reference, and the layout will change as I clean it up. Coupling the model to a decompression path is not finished. This code will be used to update the hydrogen diffusion fits from chapter 3 of my dissertation for an upcoming paper.

## License

The code is under the MIT License (see LICENSE).
