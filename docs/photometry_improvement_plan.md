# Photometry Robustness & Precision Improvement Plan

Research-driven upgrade plan for AutoPhOT (see docs/photometry_audit_design_note.md
context). Ordered by effort/dependency; each phase lands independently.

## Phase A - Small wiring changes

### A1. SFFT kernel regularization (Hu et al. 2024)

sfft 1.7.3 `BSpline_Packet.BSP` supports `REGULARIZE_KERNEL`, `XY_REGULARIZE`,
`WEIGHT_REGULARIZE`, `LAMBDA_REGULARIZE` - a Laplacian penalty on the matching
kernel that suppresses delta-basis overfitting on sparse priors.

NOTE: in sfft 1.7.3 the B-Spline path is Cupy/GPU-only. On the Numpy backend
`SingleSFFTConfigure.SSC` returns None and `ElementalSFFTSubtract.ESS` raises
UnboundLocalError - currently caught silently as "B-Spline kernel refinement
failed". This change (a) skips BSP cleanly with an accurate warning on Numpy,
(b) plumbs the regularization options so GPU deployments can use them.

- utils/run_sfft.py: new CLI args `-regularize_kernel` (auto/true/false),
  `-regularize_lambda` (1e-6), `-regularize_sparse_threshold` (30);
  Numpy-backend guard on the BSP block; XY_REGULARIZE = matched source coords.
- templates.py `_build_sfft_cmd`: forward `sfft_regularize_kernel`,
  `sfft_regularize_lambda`, `sfft_regularize_sparse_threshold`.
- autophot/data/default_input.yml: document keys; fix stale claim that BSP works on
  Numpy.

### A2. Wire in empirical uncertainty calibration

`utils/uncertainty_calibration.py` was dead code (never called). Injection/
recovery trials in `limits.py::Limits.get_injected_limit` already produce
(true_flux, measured_flux, measured_err) per trial - exactly what
`calibrate_uncertainties` needs.

- limits.py: collect per-trial samples in `run_trials_at_mag` (PSF/EMCEE
  methods: true = injected F_amp; AP method: true = F_amp * aperture fraction
  from input_yaml["aperture_correction"]). After the search, run
  `calibrate_uncertainties`, stash `self.last_uncal = {"result": res,
  "method": recovery_method}`, expose via details dict.
- utils/uncertainty_calibration.py: add `scale_error_columns(df, factor,
  method, filt)` - scales flux_*_err, inst_*_err, recomputes calibrated
  mag_err via quadrature delta sqrt(mag_err^2 + (f^2-1)*inst_err^2), rescales
  SNR columns. Testable as a pure function.
- main.py: after the limiting-magnitude block (~line 10070, before the output
  dict literal at ~10090), apply the factor to TargetPosition when it passes
  gates (n >= uncal_min_sources, factor within uncal_factor_clip). All
  downstream consumers (SNR, detection flag, output CSV) then see calibrated
  errors consistently.
- Config under `limiting_magnitude`: `uncal_from_injections: True`,
  `uncal_min_sources: 30`, `uncal_factor_clip: [0.5, 2.0]`, `uncal_apply: True`.
- Output columns: `uncal_factor`, `uncal_n_sources`, `uncal_z_std`,
  `uncal_applied`.

## Phase B - Spatially-varying zeropoint

After the color-term fit in `zeropoint.py::fit_zeropoint`, fit a tilted plane
r = a*X + b*Y + c to the inlier residuals in pixel coords, gated like
`fit_color_term` (n_inliers >= 15, coeffs >= 3 sigma, corrected MAD <
uncorrected MAD, |delta at target| <= zp_spatial_max_corr). Store
`spatial_coeffs` in fit_params; main.py replaces scalar `_zp` with
`_zp_at(x, y)` at every `inst + zp` site (target ~9706, additional targets
~10867, limiting mags ~10165/10971, Calib catalog write). New output columns
`zp_spatial_corr_{psf,ap}`, `zp_spatial_applied`.

Config: `zeropoint.fit_spatial_residuals: False`, `zp_spatial_max_corr: 0.10`,
`zp_spatial_min_inliers: 15`.

## Phase C - Cross-epoch ensemble calibration post-step

New `ensemble_calibration.py` cloned from `color_correction.py` structure
(standalone CLI + postprocessing hook in autophot.py next to
`_run_transient_color_correction`). Glob Calib_*.csv, build star x epoch
matrix of calibrated mags per band+telescope+instrument (star_id via rounded
ra/dec, as in lightcurve.py ~2590), solve per-epoch offsets d_e minimizing
sum (m_se - mbar_s - d_e)^2/sig^2 with sum(d)=0, iterate with MAD clipping to
veto variables. Optional SysRem modes (default 0). Gates: >=5 epochs, >=10
stars, |d| <= 0.2 mag. Writes LightCurve_Output_ensemblecal.csv with
ens_delta/ens_delta_err/ens_applied columns.

Config: `postprocessing.ensemble_calibration: False`,
`ensemble_cal_min_stars: 10`, `ensemble_cal_max_abs_delta: 0.2`,
`ensemble_cal_sysrem_modes: 0`.

## Phase D - Spatially-varying PSF (GriddedPSFModel)

Partition psf_source_pool into detector cells (2x2 or 3x3 via
`psf_spatial_grid: "auto"`, min `psf_grid_min_stars_per_cell: 8`), build ePSF
per cell with the existing builder, assemble with `photutils.psf.grid_from_epsfs`
into `GriddedPSFModel`. Constraints: N_psf not in {2,3} (partial grids filled
by nearest-cell copy or rejected), identical stamp shapes, all-finite.

Adaptations needed:
- `convolve_epsf_with_kernel` (psf.py:888): convolve each cell ePSF with the
  SFFT kernel realized at that cell's position (Realize_MatchingKernel pattern
  already at main.py:8052).
- `epsf_model.data` accesses (psf.py ~5085/5207/5385/5478/5568): gridded data
  is 3D - run diagnostics per-cell or on nearest-cell model; write cube +
  grid_xypos to PSF FITS.
- Fitters probe `evaluate` compatibility already (MCMC fast path) - verify.
- `psf_model` output column records e.g. `gridded-epsf-3x3`.

Land in order: (i) ImagePSF/GriddedPSFModel abstraction refactor, (ii) per-cell
build, (iii) kernel-per-cell convolution, (iv) validation on synthetic
position-varying-PSF image. Default `psf_spatial_grid: "off"` initially.

## Phase E - Robustness extras (independent)

- Neighbor pre-subtraction before ePSF build in crowded fields
  (`psf_neighbor_presubtract: False`, IterativePSFPhotometry pre-pass).
- Brighter-fatter flag: quadratic trend of bright-end ZP residuals ->
  WARNING > 0.02 mag. Diagnostic only.
- DCR warning when airmass > 2.
- TRANSLIENT-style sub-pixel dipole statistic on the difference stamp at the
  target -> `dipole_stat` output column (flag, not reject).

## Phase F - Scene modeling photometry (outline)

Forward-model host (frozen from template) + PSF*flux_epoch + sky jointly over
all epochs in pixel space - the DES-SN5YR / ZTF-DR2 / Roman approach.
Entry point: `scene_model.py` post-step loading reduced cutouts, host fit once
on stacked template, per-epoch fluxes via existing MCMCFitter with shared
position. Revisit after B-D land; spatial PSF (D) is a prerequisite.

## Status

- [x] Plan written
- [ ] A1 SFFT regularization plumbing + Numpy BSP guard
- [ ] A2 Uncertainty calibration wiring
- [ ] B Spatial zeropoint
- [ ] C Ensemble calibration
- [ ] D Gridded PSF
- [ ] E Robustness extras
- [ ] F Scene modeling (not scheduled)
