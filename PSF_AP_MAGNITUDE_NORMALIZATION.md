# PSF vs Aperture Magnitude Normalization

## Investigation summary

AP and PSF magnitudes use the **same** flux convention and formula; the systematic offset is due to **what** is measured, not a normalization bug.

### Shared convention (consistent)

- **Units**: Both fluxes are in **electrons per second** (e/s).
  - **Aperture** (`aperture.py`): `flux_AP = aperture_sum * (1/exposure_time)` where `aperture_sum` is electron counts in the aperture (image is converted to electrons via `image_e = image * gain` before photometry).
  - **PSF** (`psf.py`): `flux_PSF = flux_fit_e / exposure_time` where `flux_fit_e` is the fitted total flux in electrons (NDData and fit are in electrons).
- **Exposure time**: Both use `input_yaml['exposure_time']` (aperture via `self.input_yaml.get('exposure_time', 30.0)`, PSF via `self.input_yaml.get('exposure_time', 30.0)`).
- **Magnitude formula**: Both use `inst_mag = -2.5 * log10(flux)` with that flux in e/s (aperture via `mag(flux_ap)` in `functions.mag`, PSF via `-2.5 * np.log10(updated['flux_PSF'])`).
- **Zeropoint**: `zeropoint._compute_delta_mag()` uses the same formula for both (`inst_mag = -2.5 * np.log10(flux)`), so ZP is on the same scale.

So there is **no** inconsistency in how the two are “normalised” (same units, same exposure time, same mag definition).

### Why they differ: aperture vs total flux

- **Aperture magnitude**: Uses only the light **inside the circular aperture** (typically radius ≈ 2× FWHM). Flux outside the aperture is not included.
- **PSF magnitude**: Uses the **total** flux from the PSF fit (integrated over the model). So for the same star, `flux_PSF` > `flux_AP`, hence **PSF mag is brighter** (smaller number) than AP mag.

So a “subtle” systematic offset (AP fainter than PSF) is **expected** and is not a normalisation error.

### Optional: put AP on the same scale as PSF (total flux)

To make AP and PSF directly comparable (both representing “total” flux), you can apply an **aperture correction** so that aperture magnitudes are corrected to an “infinite” aperture (total flux):

- **Existing code**: `aperture.compute_aperture_correction()` uses a curve-of-growth to compute the correction in mag (amount to subtract from aperture mag to get total mag). In `main.py` the call is currently **commented out** (around line 1683).
- **Options**:
  1. **Enable and apply** `compute_aperture_correction()` in the pipeline, then subtract the correction from the stored aperture magnitudes (or add it to the stored `flux_AP` in linear space), so that AP mag represents total flux like PSF.
  2. **Document only**: Keep current behaviour and document that `flux_AP` / AP mag are “aperture flux” and `flux_PSF` / PSF mag are “total flux”, so a small AP−PSF offset is expected.

### Code references

| Item            | Aperture | PSF |
|-----------------|----------|-----|
| Flux in         | e/s      | e/s |
| Flux from       | `aperture_sum * inv_exposure_time` (`aperture.py` ~195, 216) | `flux_fit_e / exposure_time` (`psf.py` 1828) |
| Exposure time   | `input_yaml.get('exposure_time', 30.0)` (`aperture.py` 524) | `input_yaml.get('exposure_time', 30.0)` (`psf.py` 1567) |
| Inst mag        | `mag(flux_ap)` → -2.5*log10(flux) (`aperture.py` 198, `functions.mag`) | `-2.5 * np.log10(flux_PSF)` (`psf.py` 1838) |
| Aperture corr.   | Available in `compute_aperture_correction()`; not applied in main | N/A (already total) |
