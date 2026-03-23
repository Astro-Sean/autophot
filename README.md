AutoPHoT
========

Automated photometry pipeline for transients and variable sources.

AutoPHoT processes FITS images, solves/updates WCS, measures aperture and PSF
photometry, calibrates zeropoints, and can run template subtraction (SFFT,
HOTPANTS, or ZOGY).

Project links:
- GitHub: https://github.com/Astro-Sean/autophot
- Paper: https://ui.adsabs.harvard.edu/abs/2022A%26A...667A..62B

## Install

Recommended in a fresh conda environment:

```bash
conda create -n autophot python=3.11 -y
conda activate autophot
conda install astro-sean::autophot
```

Quick check:

```bash
python -c "from autophot import AutomatedPhotometry; print('AutoPHoT import OK')"
autophot-main -h
```

## External tools

Common optional tools used by the pipeline:

```bash
conda install -c conda-forge astrometry-net
conda install -c conda-forge astromatic-source-extractor
conda install -c conda-forge astromatic-scamp
conda install -c conda-forge astromatic-swarp
```

For `template_subtraction.method: hotpants`, install HOTPANTS separately.

## Minimal usage

```python
from autophot import AutomatedPhotometry

cfg = AutomatedPhotometry.load()
cfg["fits_dir"] = "/path/to/images"
cfg["outdir_name"] = "REDUCED"
cfg["target_name"] = "SN2024abc"
cfg["target_ra"] = 150.1234
cfg["target_dec"] = 2.3456
cfg["catalog"]["use_catalog"] = "gaia"

# WCS mode:
# TPV (default): solve-field + SCAMP
# SIP: solve-field only
cfg["wcs"]["projection_type"] = "TPV"
cfg["wcs"]["redo_wcs"] = True

# Optional subtraction
cfg["template_subtraction"]["do_subtraction"] = True
cfg["template_subtraction"]["method"] = "sfft"
cfg["template_subtraction"]["alignment_method"] = "reproject"

AutomatedPhotometry.run_photometry(default_input=cfg)
```

## Template subtraction

If you use local templates, place them under:

```text
<fits_dir>/templates/<filter>_template/*.fits
```

For common optical filters (`g`, `r`, `i`, `z`, `u`) use:

```text
<fits_dir>/templates/gp_template/*.fits
<fits_dir>/templates/rp_template/*.fits
...
```

## Key config options

- `wcs.projection_type`: `TPV` (default) or `SIP`
- `wcs.redo_wcs`: force fresh WCS solve
- `template_subtraction.method`: `sfft` (default), `hotpants`, `zogy`
- `template_subtraction.alignment_method`: `reproject`, `swarp`, `astroalign`
- `catalog.use_catalog`: `gaia`, `pan_starrs`, `sdss`, `custom`, etc.

See `databases/default_input.yml` for full configuration.

## Citation

If you use AutoPHoT in a publication, please cite:

- https://ui.adsabs.harvard.edu/abs/2022A%26A...667A..62B/abstract
