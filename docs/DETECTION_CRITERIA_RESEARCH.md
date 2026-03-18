# Research: More Effective Ways to Determine If a Source Is Detected

## Current Implementation (AutoPhOT)

Detection is decided in **lightcurve.py** (and in table-generation helpers there) using one of two modes:

1. **SNR-based** (`use_SNR_limit=True`, default):  
   **Detected** if:
   - Magnitude and error are finite  
   - **SNR >= snr_limit** (default 3), where SNR is either `SNR_PSF`, `SNR_AP`, or flux/flux_err  
   - **beta > beta_limit** (default 0.5) when a `beta` column exists  

2. **Limit-based** (`use_SNR_limit=False`):  
   **Detected** if magnitude is finite and **mag < limiting magnitude (lmag)** (and beta when present).

**Beta** (from `functions.beta_psf` / `beta_aperture`) is the probability that the true flux is above an n-sigma detection threshold, given the measured flux and its uncertainty (Gaussian assumption). It is already a probabilistic detection criterion.

So the pipeline combines:
- A **hard SNR cut** (e.g. SNR >= 3)
- An optional **probabilistic cut** (beta > 0.5)

---

## Literature and Best Practices

### 1. Sigma/SNR thresholds

- **3–5 sigma** above background is the usual range in photometry; 3σ gives higher completeness, 5σ fewer false positives (e.g. ~1 spurious per 10^6 pixels at 3σ).
- There is **no universal “correct” sigma**; it depends on noise properties and the trade-off between false positives and missed detections.
- For **transients**, optimal detection is often formulated as a **statistic** (e.g. matched filter or proper image subtraction) with a known distribution under the null, rather than a single fixed sigma.

### 2. Probabilistic / Bayesian detection

- **Beta-style metrics**: The probability that flux is above a threshold (as in your `beta_psf` / `beta_aperture`) is a sound way to combine flux and error into one detection number. It is already more informative than SNR alone because it uses the full uncertainty.
- **Bayesian model comparison**: Comparing “source present” vs “noise only” via marginal likelihoods (or odds) can improve sensitivity over simple thresholding (e.g. reported gains for low-S/N flares).
- **Caution**: Converting Bayes factors to “sigma” is often done incorrectly and can overstate confidence; better to work with probabilities or odds directly.

### 3. Consistency with limiting magnitude

- Using **mag < limiting magnitude** as the sole criterion can be inconsistent if the limit is defined at a different confidence (e.g. 3σ or a given beta) than the detection rule. Aligning both to the same formalism (e.g. same n_sigma or same beta) is recommended.

---

## Recommendations for a More Effective Detection Criterion

### 1. Prefer beta (or equivalent probability) as the primary criterion

- **Why**: Beta encodes both flux and uncertainty; two measurements with the same SNR but different errors get different betas. A single threshold on beta is therefore a more consistent “detection probability” rule than a single SNR cut.
- **Suggestion**: Treat **beta > beta_limit** as the main detection rule when `beta` is available (PSF or aperture), and use **SNR >= snr_limit** only as a fallback when beta is missing (with a clear comment in code/doc).

### 2. Unify with the limiting-magnitude definition

- Define the **limiting magnitude** using the same n_sigma (and, if possible, same beta threshold) used for “detected”. Then:
  - Either use **beta > beta_limit** for both “is detected?” and “is above limit?” (when beta exists),  
  - Or use **SNR >= snr_limit** for detection and ensure the limit is quoted at that same SNR (e.g. 3σ limit when snr_limit=3).

### 3. Optional: Single helper for “is_detected”

- Centralize the logic in one function, e.g. `is_detected(row, method, snr_limit, beta_limit, use_snr_primary=False)`, that:
  - Prefers beta when present and uses `beta > beta_limit`;
  - Falls back to SNR or mag vs limit when beta is absent;
  - Ensures finite mag/error and consistent handling of PSF vs AP columns.  
- This reduces duplication between `plot_lightcurve`, colour pairing, and table generation, and makes it easier to switch to a beta-primary policy everywhere.

### 4. Optional: Configurable policy

- Allow a small set of policies in config or function args, e.g.:
  - **"snr"**: current behaviour (SNR >= snr_limit and optionally beta > beta_limit).
  - **"beta"**: primary criterion is beta > beta_limit; SNR only as sanity check or fallback.
  - **"limit"**: mag < lmag (and optionally beta), for compatibility with limit-based workflows.  
- Default could be **"beta"** when the photometry table includes `beta`, and **"snr"** otherwise.

### 5. Do not convert beta/Bayes factors to sigma for reporting

- Keep reporting **beta** (and SNR) as-is. Avoid converting beta or Bayes factors to “equivalent sigma” for detection decisions; that conversion is often misleading.

---

## Summary

- The current combination of **SNR threshold** and **beta** is already reasonable; the main improvement is to **treat beta as the primary criterion** when available and align detection with how the limiting magnitude is defined.
- Adding a **single “is_detected” helper** and an optional **policy** (snr vs beta vs limit) would make behaviour clearer and easier to tune without changing the underlying photometry.
