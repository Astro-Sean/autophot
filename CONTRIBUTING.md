# Contributing to AutoPhOT

## Development setup

```bash
conda env create -f environment.yml
conda activate autophot
# or: pip install -e ".[all,test]"
```

## Layout

All importable code lives in the `autophot/` package:

- `autophot/pipeline/` - per-image stages (prepare, wcs, background,
  cosmic, fwhm, mask, telescope, main orchestrator)
- `autophot/photometry/` - aperture, PSF, differential, host-galaxy fits
- `autophot/calibration/` - catalogs, zeropoint, color/filter transforms
- `autophot/subtraction/` - template alignment and difference imaging
- `autophot/analysis/` - lightcurves, limits, plotting, QA
- `autophot/services/` - TNS, additional targets, survey downloads
- `autophot/core/` - shared helpers (functions.py, plotting_utils.py)
- `autophot/data/` - bundled YAML/config (use `functions.data_path()`)
- `autophot/gaia_curves/` - Gaia XP synthetic photometry catalogs

Put new modules inside `autophot/` and use relative imports
(`from ..core.functions import ...`). The old flat-layout top-level
module names were removed; imports must use `autophot.<sub>.<module>`
paths.

## Tests

```bash
pytest                        # full suite (slow; needs network for some)
pytest -m "not slow and not mcmc and not injection"   # fast subset
```

Scientific outputs must stay bit-for-bit compatible unless the change
is intentional; see AGENTS.md for the photometric invariants.
