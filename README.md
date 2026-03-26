![AutoPhOT logo](logo.png)

[![Anaconda Version](https://anaconda.org/astro-sean/autophot/badges/version.svg)](https://anaconda.org/astro-sean/autophot)
[![Latest Release Date](https://anaconda.org/astro-sean/autophot/badges/latest_release_date.svg)](https://anaconda.org/astro-sean/autophot)
[![Latest Release Relative Date](https://anaconda.org/astro-sean/autophot/badges/latest_release_relative_date.svg)](https://anaconda.org/astro-sean/autophot)
[![License](https://anaconda.org/astro-sean/autophot/badges/license.svg)](https://anaconda.org/astro-sean/autophot)
[![Downloads](https://anaconda.org/astro-sean/autophot/badges/downloads.svg)](https://anaconda.org/astro-sean/autophot)

# AutoPhOT

The Automated Photometry of Transients (AutoPhOT) pipeline, built on Photutils and Astropy, provides a comprehensive photometric solution for transients and variable sources, offering aperture/PSF photometry, catalogue calibration, WCS solving, and optional template subtraction.


Project links:
- Conda: [https://anaconda.org/channels/astro-sean/packages/autophot/overview](https://anaconda.org/channels/astro-sean/packages/autophot/overview)
- Paper: [https://ui.adsabs.harvard.edu/abs/2022A%26A...667A..62B](https://ui.adsabs.harvard.edu/abs/2022A%26A...667A..62B)


> [!NOTE]
> I am the sole developer and maintainer of AutoPhOT and also a full-time researcher at MPE.
> Please open issues on GitHub and I will do my best to resolve them.

## Installation (Conda)

```bash
conda create -n autophot python=3.11 -y
conda activate autophot
conda install astro-sean::autophot
```

Verify installation:

```bash
python -c "from autophot import AutomatedPhotometry; print('AutoPhOT import OK')"
autophot-main -h
```

## Optional External Tools

### Astrometry.net (`solve-field`)

```bash
conda install -c conda-forge astrometry-net
solve-field --help
```

### Astromatic tools (SExtractor/SCAMP/SWarp)

```bash
conda install -c conda-forge astromatic-source-extractor astromatic-scamp astromatic-swarp
```

### HOTPANTS

```bash
conda install -c conda-forge cfitsio make gcc
git clone https://github.com/acbecker/hotpants
cd hotpants
make
```

## Driver Script Example (Sanitized)

The script below follows your requested workflow style while avoiding sensitive paths/tokens.

> [!NOTE]
> FITS images require the TELESCOP and INSTRUME header keywords (otherwise they will be ignored), as well as a keyword giving the image bandpass (e.g., FILTER).


```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run automated photometry with AutoPhOT.
Override default_input and run pipeline; optionally plot lightcurve and tables.
"""

import os
from autophot import AutomatedPhotometry, prepare_template_directory


def main() -> int:
    autophot_input = AutomatedPhotometry.load()
    autophot_input["nCPU"] = 1

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    autophot_input["outdir_name"] = "REDUCED"
    autophot_input["wdir"] = "/path/to/autophot_db"
    autophot_input["fits_dir"] = "/path/to/images"

    # Optional: do not re-run files already processed.
    # autophot_input["restart"] = False

    # ------------------------------------------------------------------
    # Target
    # ------------------------------------------------------------------
    autophot_input["target_name"] = "SNXXXXabc"
    autophot_input["target_ra"] = 123.456789
    autophot_input["target_dec"] = -12.345678

    # ------------------------------------------------------------------
    # Catalog
    # ------------------------------------------------------------------
    autophot_input["catalog"]["use_catalog"] = {
        "griz": "refcat",
        "u": "gaia",
        "UBVRI": "apass",
        # "default": "gaia",
    }

    # Optional: Gaia + custom throughput curves ("gaia_custom")
    # - Set one or more mapping entries to "gaia_custom"
    # - Provide `catalog.transition_curve_map` with matching band keys (e.g. g/r/i)
    # AutoPhOT will build/reuse a Gaia curve-map catalog and route those bands
    # through the `custom` backend automatically.
    #
    # autophot_input["catalog"]["use_catalog"] = {
    #     "gri": "gaia_custom",
    #     "zJHK": "refcat",
    #     "u": "gaia",
    #     "UBVRI": "apass",
    # }
    # autophot_input["catalog"]["transition_curve_map"] = {
    #     "g": "/path/to/throughputs/g.dat",
    #     "r": "/path/to/throughputs/r.dat",
    #     "i": "/path/to/throughputs/i.dat",
    # }

    # Optional credentials from environment (do not hard-code secrets):
    # TNS credentials are only needed if you want TNS lookups from target_name.
    # export MASTCASJOBS_WSID="..."
    # export MASTCASJOBS_PWD="..."
    # export TNS_BOT_ID="..."
    # export TNS_BOT_NAME="..."
    # export TNS_BOT_API="..."
    if os.getenv("MASTCASJOBS_WSID"):
        autophot_input["catalog"]["MASTcasjobs_wsid"] = os.getenv("MASTCASJOBS_WSID")
    if os.getenv("MASTCASJOBS_PWD"):
        autophot_input["catalog"]["MASTcasjobs_pwd"] = os.getenv("MASTCASJOBS_PWD")
    if os.getenv("TNS_BOT_ID"):
        autophot_input["wcs"]["TNS_BOT_ID"] = os.getenv("TNS_BOT_ID")
    if os.getenv("TNS_BOT_NAME"):
        autophot_input["wcs"]["TNS_BOT_NAME"] = os.getenv("TNS_BOT_NAME")
    if os.getenv("TNS_BOT_API"):
        autophot_input["wcs"]["TNS_BOT_API"] = os.getenv("TNS_BOT_API")

    # ------------------------------------------------------------------
    # Preprocessing / photometry / WCS
    # ------------------------------------------------------------------
    autophot_input["cosmic_rays"]["remove_cmrays"] = False
    autophot_input["preprocessing"]["trim_image"] = 5
    autophot_input["photometry"]["perform_emcee_fitting_s2n"] = 10
    autophot_input["wcs"]["redo_wcs"] = True

    # ------------------------------------------------------------------
    # Template subtraction
    # ------------------------------------------------------------------
    autophot_input["template_subtraction"]["do_subtraction"] = True
    autophot_input["template_subtraction"]["alignment_method"] = "reproject"
    autophot_input["template_subtraction"]["method"] = "sfft"
    autophot_input["template_subtraction"]["kernel_order"] = 1

    # Create template folder structure and ask before continuing.
    prepare_template_directory(
        fits_dir=autophot_input["fits_dir"],
        include_legacy_p_folders=False,  # create only *_template by default
        confirm_before_continue=True,
    )

    loc = AutomatedPhotometry.run_photometry(
        default_input=autophot_input,
        do_photometry=True,
    )

    # Optional post-run products.
    from lightcurve import plot_lightcurve, check_detection_plots, generate_photometry_table

    detections_loc = plot_lightcurve(
        loc,
        snr_limit=3,
        method="PSF",
        format="png",
        offset=1,
        show=True,
        plot_color=False,
        color_match_days=0.5,
    )
    check_detection_plots(detections_loc, method="PSF")
    generate_photometry_table(loc, snr_limit=3, method="PSF", reference_epoch=0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

## Preparing Template-Subtracted Photometry

1. Set subtraction options:
   - `autophot_input["template_subtraction"]["do_subtraction"] = True`
   - `autophot_input["template_subtraction"]["alignment_method"] = "reproject"`
   - `autophot_input["template_subtraction"]["method"] = "sfft"` (or `hotpants`, `zogy`)
2. Create template directories:
   - Call `prepare_template_directory(...)`.
  - AutoPhOT prints where folders were created and asks if you want to continue.
3. Place template FITS files:
   - Put one usable template per filter in `fits_dir/templates/<filter>_template/`.
4. Run photometry.

Notes:
- Modern folder names are preferred: `r_template`, `g_template`, etc.
- Legacy `rp_template` naming is still supported by the pipeline if needed.
- Keep paths and credentials sanitized in public scripts.

## Citation

If you use AutoPhOT in your research, please cite:
- ADS: [https://ui.adsabs.harvard.edu/abs/2022A%26A...667A..62B](https://ui.adsabs.harvard.edu/abs/2022A%26A...667A..62B)

```bibtex
@ARTICLE{2022A&A...667A..62B,
       author = {{Brennan}, S.~J. and {Fraser}, M.},
       title = "{The Automated Photometry of Transients pipeline (AutoPhOT)}",
      journal = {\aap},
     keywords = {techniques: photometric, techniques: image processing, methods: data analysis, Astrophysics - Instrumentation and Methods for Astrophysics, Astrophysics - High Energy Astrophysical Phenomena},
         year = 2022,
        month = nov,
       volume = {667},
          eid = {A62},
        pages = {A62},
          doi = {10.1051/0004-6361/202243067},
archivePrefix = {arXiv},
       eprint = {2201.02635},
 primaryClass = {astro-ph.IM},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2022A&A...667A..62B},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```