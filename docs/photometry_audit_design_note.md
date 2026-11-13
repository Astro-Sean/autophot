# AutoPhOT Photometry Pipeline Audit & Implementation Plan

**Date:** 2026-08-21
**Scope:** Photometric-measurement pipeline robustness, reproducibility,
statistical defensibility, and scientific transparency.
**Repository state:** `master` @ `163b7f9`, pyproject v1.1.11

---

## 1. Current Behavior

### 1.1 End-to-End Workflow (main.py:run_photometry, ~10,900 lines)

| Stage | Module(s) | Key Lines | Notes |
|-------|-----------|-----------|-------|
| Image loading & header | `functions.get_image_and_header` | 997-1096 | Validates TELESCOP/INSTRUME; extracts gain/saturate/read_noise/airmass |
| WCS solving | `wcs.WCSSolver` | 2193-2287 | astrometry.net + Gaia DR3; fallback to existing WCS |
| Preprocessing (NaN trim, crop) | `main._trim_nan_boundaries` | 1707-1956 | Target-preserving NaN trim; optional arcmin crop |
| Source detection & FWHM | `utils.run_sex.SExtractorWrapper`, `fwhm.Find_FWHM` | 1988-2090 | SExtractor with pythonic fallback; FWHM uncertainty computed |
| Cosmic ray removal | `cosmic` | 2141-2190 | astroscrappy / LACosmic |
| Background estimation | `background.BackgroundSubtractor` | 2039-2057, 2458-2483 | Background2D (BiweightLocation + MADStdRMS); 3-pass source masking |
| Sampling regime | `functions.classify_sampling_regime` | 2072-2090 | 4-tier: undersampled/critical/well_sampled/oversampled |
| Template alignment | `templates.Templates.align` | 2634-2822 | Cascade: spalipy -> SWarp -> reproject -> AstroAlign |
| Calibration (catalog, ZP) | `catalog.Catalog`, `zeropoint.Zeropoint` | 2866-4430 | Aperture + PSF photometry on catalog stars; color terms |
| PSF construction | `psf.PSF.build` | 4070-4193 | ePSF via photutils EPSFBuilder; adaptive oversampling |
| PSF photometry (sequence) | `psf.PSF.fit` | 4219-4231 | Tiered SNR: LSQ / Poisson / MCMC |
| Template subtraction | `templates.Templates.subtract` | 4546-6887 | Cascade: ZOGY -> SFFT -> HOTPANTS; quality assessment |
| Target photometry | `aperture.Aperture`, `psf.PSF.fit` | 8197-8657 | AP + PSF; inverted-image retry for negative detections |
| Calibrated magnitudes | `main` | 9088-9377 | Quadrature error propagation; systematic floors |
| Limiting magnitude (injection) | `limits.Limits` | 9535-9838 | PSF-based injection; bracket-and-bisect completeness |
| Output serialization | `main` | 9840-10927 | CSV output with ~40+ fields; provenance metadata |

### 1.2 PSF Construction (psf.py:PSF.build, lines 2371-3743)

**Star selection:** Adaptive cuts with safety nets:
- Saturation (peak < 0.9 * saturate, auto-relaxed for small pools)
- SExtractor FLAGS (<= 1, rejects blended)
- Elongation (<= 1.5, rejects trailed)
- FWHM consistency ([0.5x, 2.5x] global FWHM + sigma-clip)
- Sharpness (<= 1.0, rejects extended)
- Profile concentration (FLUX_RADIUS/FWHM <= 0.9, rejects galaxies and extended defects)
- Neighbour isolation (cKDTree, 3x FWHM)
- Pixel-level contamination (6 tests: asymmetry, edge, radial, secondary peak, streak, mask fraction)
- FFT power-spectrum outlier rejection
- Spatial uniformity (grid-based O(N log N) selection)

**ePSF build:** photutils EPSFBuilder with:
- Adaptive oversampling (2x-4x based on star count)
- Adaptive sigma clipping (3.0-5.0 based on N stars)
- Quartic or Gaussian smoothing kernel
- Moffat initial PSF
- Adaptive iteration retry (5 -> 10 iterations)

**Diagnostics:** Encircled energy check, edge background check, oversampled PSF plot, star cutout grid.

**Returns:** `(epsf, df)` - ImagePSF model and selected star DataFrame.

### 1.3 PSF Fitting (psf.py:PSF.fit, lines 4106-5871)

**Tiered SNR approach:**
- Bright (SNR >= 8): LSQ (TRFLSQFitter) or Poisson likelihood
- Faint (4 <= SNR < 8): LSQ with retry
- Very faint (SNR < 4): MCMC (emcee) if `perform_emcee_fitting_s2n > 0`

**Fitters:**
- `TRFLSQFitter` (default, trust-region reflective least-squares)
- `PoissonLikelihoodFitter` (Fermilab TM-2543-AE, Newton-Raphson)
- `MCMCFitter` (emcee, StretchMove + DEMove, adaptive convergence)

**Free parameters:** x, y, flux. PSF shape fixed from ePSF.
**Background:** LocalBackground (MedianBackground) with FWHM-scaled annulus.
**Neighbours:** SourceGrouper (2-3x FWHM); simultaneous fitting of grouped sources.
**Bounds:** x/y within +- xy_bounds; flux >= 1e-6.
**Uncertainty:** LSQ curvature-based; MCMC chain std; chi2 scaling (capped 10x).
**Quality:** photutils flags (decoded), cfit, qfit, reduced_chi2.
**Failure handling:** LSQ retry (larger box, tighter bounds); inverted-image retry for negative flux.

### 1.4 MCMC (psf.py:MCMCFitter, lines 489-1107)

- 32 walkers (default), 5000 steps or adaptive until tau < 50
- Mixed moves: 70% StretchMove + 30% DEMove
- Parameter-aware jitter (absolute for position, relative for flux)
- Rejection-sampling boundary guard (no reflection artifacts)
- Gaussian likelihood with per-pixel variance (Poisson + sky + read noise)
- Prior: uniform within bounds, flux >= 0, shape params > 0
- Convergence: autocorrelation time, acceptance fraction check (0.15-0.8)
- Burn-in: 10x tau (or 30% fallback); thinning: max(10, tau/2)
- Posterior: median + 16/84 percentiles; covariance from chain
- Effective sample size warning if < 100

### 1.5 Injection/Recovery (limits.py:Limits, lines 608-4358)

- PSF-based injection using same ePSF as pipeline
- Correct gain/exposure normalization
- Site selection: Fibonacci spiral + quiet-site or representative mode
- Filtering: target exclusion, edge clearance, aperture validity, annulus support, pixel statistics
- Recovery: AP / PSF / EMCEE (matches transient photometry method)
- Detection: S/N gate + positive flux + finite error
- Search: bracket-and-bisect for completeness target (default 50%)
- Optional logistic MCMC completeness fit
- Deterministic seeds supported via `rng_seed`

### 1.6 Output

- CSV format: `Output_{base}.csv`, `Calib_{base}.csv`
- ~40+ fields including photometry, quality, calibration, provenance
- Quality flags: `is_detection`, `diff_quality_class`, `flags_normal/inverted`
- Provenance: filename, telescope, FWHM, ZP, alignment RMS, subtraction quality

### 1.7 Test Suite

**No test suite exists.** Confirmed by:
- `find` returns zero test files
- `pyproject.toml` has no test configuration
- `agents.md` documents: "No unit tests (entire codebase has no test suite)"

---

## 2. Current Assumptions

1. **Single global PSF:** One ePSF per image; no spatial variation model.
2. **Gaussian noise:** LSQ and MCMC use Gaussian per-pixel likelihood. Poisson fitter available but not default.
3. **Background-subtracted data:** Photometry operates on background-subtracted images; local background is a small residual correction.
4. **ADU units:** Image remains in ADU; gain applied internally for noise model.
5. **FWHM is representative:** Global median FWHM used for all sizing decisions.
6. **PSF stars are point sources:** Selection assumes stars are unresolved; galaxies/cosmic rays rejected.
7. **MCMC flux >= 0:** Prior rejects negative flux (problematic for difference images with negative sources).
8. **Uncorrelated noise:** Per-pixel variance model; no correlated-noise treatment in fitting (only in subtraction quality assessment).
9. **Injection after preprocessing:** Sources injected into background-subtracted image; does not test detection or PSF construction end-to-end.

---

## 3. Known Weaknesses

### 3.1 Critical (affects scientific validity)

| ID | Weakness | Impact | Location |
|----|----------|--------|----------|
| W1 | **No PSF held-out validation** | PSF quality unverified; model mismatch undetected | psf.py:build |
| W2 | **No injection catalog saved** | Cannot audit/reproduce injection results; no uncertainty calibration | limits.py |
| W3 | **No uncertainty calibration check** | Reported errors may be underestimated/overestimated | limits.py |
| W4 | **No difference-image injection** | Subtraction quality not validated by injection/recovery | limits.py |
| W5 | **MCMC prior hard-rejects negative flux** | Cannot sample negative-flux posteriors for difference images | psf.py:575 |
| W6 | **No MCMC convergence flags in output** | Non-convergence not propagated to quality flags | psf.py, main.py |
| W7 | **No test suite** | No regression protection; refactoring risk | entire repo |

### 3.2 Significant (affects robustness)

| ID | Weakness | Impact | Location |
|----|----------|--------|----------|
| W8 | **No spatial PSF variation** | PSF mismatch across field | psf.py:build |
| W9 | **No planar background option** | Structured backgrounds not modeled in PSF fit | psf.py:fit |
| W10 | **No split-chain MCMC diagnostic** | Non-convergence may be missed | psf.py:MCMCFitter |
| W11 | **Fragmented quality flags** | No unified quality assessment | main.py |
| W12 | **No PSF-star residual RMS metric** | PSF quality not quantified | psf.py:build |
| W13 | **beta_aperture clips negative flux** | Inconsistent with beta_psf for difference images | functions.py:871 |
| W14 | **No data validation module** | Missing metadata not systematically checked | main.py |
| W15 | **Injection not stratified by flux** | Recovery tested at single flux per magnitude | limits.py |

### 3.3 Minor (engineering / convenience)

| ID | Weakness | Impact |
|----|----------|--------|
| W16 | No `--validate-config`, `--dry-run`, `--resume` CLI | Operational friction |
| W17 | No spatial FWHM variation measurement | Cannot detect PSF variation |
| W18 | qfit-based error scaling removed | May under-estimate errors in some cases |
| W19 | Bootstrap flux estimation can be unreliable | NaN-heavy patches |
| W20 | No centralized quality score | Users must interpret multiple flags |

---

## 4. Proposed Changes

### Phase 2: Validation and Diagnostics (no scientific behavior change)

1. **Create test infrastructure:** `tests/` directory, `conftest.py`, pytest configuration in `pyproject.toml`.
2. **Synthetic data generators:** Functions to create known-truth images (Gaussian/Moffat PSF, Poisson+Gaussian noise, background, masks).
3. **Data validation module (`utils/data_validation.py`):** Systematic checks for image dimensionality, dtype, finite values, exposure time, gain, read noise, saturation, WCS, mask consistency. Returns structured validation report with quality flags. Does NOT change pipeline behavior; wraps existing checks and adds missing ones.
4. **Quality flag system (`utils/quality_flags.py`):** Composable bitmask flags for all failure modes. Maps existing flags to unified system. Does NOT replace existing flags; provides additional structured field.

### Phase 3: PSF Construction and Validation

5. **PSF held-out validation (`utils/psf_validation.py`):** Leave-one-out or k-fold cross-validation. For each held-out star: fit flux with fixed PSF, measure residual RMS, flux bias, centroid bias, reduced chi2, residual spatial correlations. Generate validation report and plots.
6. **PSF residual diagnostics:** Per-star residual images, radial profile comparison, residual vs. position/brightness plots.
7. **PSF quality metrics in output:** Add `psf_validation_rms`, `psf_validation_bias`, `psf_n_stars`, `psf_oversampling` to output dict.
8. **PSF-star diagnostics recording:** Save per-star SNR, crowding score, centroid stability, FWHM, rejection reasons.

### Phase 4: Deterministic Fitting and Uncertainty

9. **Empirical uncertainty calibration:** Compare reported flux errors to empirical scatter from injection/recovery. Apply calibration factor if systematic discrepancy found.
10. **Fitting-region sensitivity test:** Optional diagnostic that refits with +-20% fit region and reports flux change.
11. **Planar background option:** Add configurable planar background mode for structured backgrounds.
12. **beta_aperture symmetry fix:** Use abs(flux) for difference-image consistency (matching beta_psf fix).

### Phase 5: MCMC Improvements

13. **Optional negative-flux prior:** Configurable `mcmc_allow_negative_flux` for difference-image photometry.
14. **Split-chain diagnostic:** Gelman-Rubin R-hat from split chains; flag if R-hat > 1.1.
15. **MCMC convergence flags in output:** Add `mcmc_converged`, `mcmc_n_eff`, `mcmc_rhat`, `mcmc_acc_frac` to output.
16. **Explicit fallback recording:** If MCMC falls back to LSQ, record reason in output.
17. **Posterior predictive check:** Generate residual image from posterior median; compare to data.

### Phase 6: Injection/Recovery

18. **Save injection catalog:** Write per-trial CSV with position, true flux, recovered flux, error, detection flag, quality flags, seed.
19. **Uncertainty calibration metric:** Compute (F_meas - F_true) / sigma_F distribution; check zero mean, unit scale.
20. **Flux stratification:** Inject sources at multiple fluxes per magnitude bin.
21. **Difference-image injection:** Implement injection/recovery on difference images.
22. **Injection before PSF construction (optional):** End-to-end pipeline test mode.

### Phase 7: Regression Tests

23. **Synthetic truth-recovery tests:** Known-truth images with quantitative acceptance criteria.
24. **PSF validation tests:** Held-out star residuals consistent with noise.
25. **MCMC convergence tests:** Known posterior shape; verify convergence diagnostics.
26. **Injection reproducibility tests:** Fixed seed produces identical results.
27. **Failure mode tests:** Saturated, masked, edge, NaN, non-convergence cases.
28. **Serialization round-trip tests:** Output CSV loads correctly with all fields.

### Phase 8: Documentation

29. **PSF validation report:** Template and example with quantitative metrics.
30. **Injection/recovery validation report:** Template with completeness, bias, uncertainty calibration.
31. **Quality flag documentation:** All flags defined with meaning and recommended actions.
32. **API documentation:** Updated docstrings and user guide.
33. **Changelog:** All compatibility changes documented.

---

## 5. Expected Compatibility Risks

| Change | Risk | Mitigation |
|--------|------|------------|
| New test infrastructure | None - additive | Separate `tests/` directory |
| Data validation module | None - additive, wraps existing checks | Returns report; does not block |
| Quality flag system | None - additive field | Existing flags preserved; new `quality_flags` field added |
| PSF validation | Minimal - runs after build | Does not change PSF model; adds diagnostics |
| Empirical uncertainty calibration | Low - configurable, default off | Only applied when `uncertainty_calibration=True` |
| Planar background option | Low - default remains constant | Configurable via `psf_background_order` |
| beta_aperture symmetry fix | Low - changes detection confidence for negative fluxes | Only affects difference images; matches beta_psf behavior |
| Negative-flux MCMC prior | Low - configurable, default off | Only enabled when `mcmc_allow_negative_flux=True` |
| MCMC convergence flags | None - additive output fields | New fields in output dict |
| Injection catalog saving | None - additive output file | New CSV alongside existing outputs |
| Difference-image injection | None - new capability | Does not change existing science-image injection |

**Key principle:** All changes are additive or configurable with defaults matching current behavior. No existing output fields are removed or renamed. No existing API signatures are broken.

---

## 6. Test Infrastructure Plan

### 6.1 Directory Structure

```
tests/
  conftest.py              # Shared fixtures, synthetic data generators
  synthetic/
    __init__.py
    images.py              # Synthetic image generators
    psf_models.py          # Known-truth PSF models
    catalogs.py            # Known-truth source catalogs
  test_data_validation.py  # Data validation module tests
  test_psf_validation.py   # PSF held-out validation tests
  test_psf_fit.py          # PSF fitting tests (LSQ, Poisson, MCMC)
  test_mcmc.py             # MCMC convergence and posterior tests
  test_injection.py        # Injection/recovery tests
  test_uncertainty.py      # Uncertainty calibration tests
  test_quality_flags.py    # Quality flag system tests
  test_serialization.py    # Output round-trip tests
  test_regression.py       # Regression tests with synthetic data
```

### 6.2 Synthetic Data Generators

- `make_gaussian_image()`: 2D Gaussian PSF, Poisson+Gaussian noise, flat background
- `make_moffat_image()`: Moffat PSF with configurable beta
- `make crowded_field()`: Multiple overlapping sources
- `make_masked_image()`: With bad pixels, saturation, edge effects
- `make_difference_image()`: With negative sources, correlated noise

### 6.3 Quantitative Acceptance Criteria

- High-S/N isolated source: flux bias < 1%
- PSF held-out residuals: consistent with noise (reduced chi2 ~ 1)
- Uncertainty coverage: 68% of truths within 1-sigma, 95% within 2-sigma
- MCMC: R-hat < 1.1, n_eff > 100 for converged chains
- Injection reproducibility: identical results with fixed seed
- All degraded cases: appropriate quality flags set

---

## 7. Scientific Caution

Per the project's scientific caution guidelines:

- Do not manufacture precision. If the PSF, background, noise model, calibration, or posterior is inadequate, report a degraded measurement with transparent uncertainty and quality flag.
- Prefer a scientifically honest failure or warning over a numerically precise but biased result.
- All tolerance values reflect the instrument, image scale, PSF sampling, and intended science. No arbitrary tolerances chosen merely to make tests pass.
- Bias corrections preserve uncorrected measurements and document the calibration procedure.
