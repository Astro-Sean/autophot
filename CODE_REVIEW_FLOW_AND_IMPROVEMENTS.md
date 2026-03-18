# Code Review: Flow, Speed, Performance & Accuracy

Review of the autophot_object pipeline with concrete improvement suggestions.

---

## 1. Pipeline flow (high level)

1. **Setup** – Parse args, load YAML, set work dir, skip if output exists (`restart=False`).
2. **Load image/header** – `get_image(fpath)`, `get_header(fpath)` (first of many).
3. **Metadata** – Telescope/instrument/filter, gain, read noise, saturation, MJD, pixel scale (WCS or telescope.yml).
4. **Preprocessing** – Optional trim, recrop; each step reloads image/header after writing.
5. **SExtractor (initial)** – FWHM + source detection; on failure, fallback to `Find_FWHM.measure_image`.
6. **Background (1st)** – `bg_remover.remove(..., plot=False)` for cosmic-ray step.
7. **Cosmic rays** – Optional removal; image/header updated in memory.
8. **WCS** – Solve/refine; optionally write to FITS.
9. **Background (2nd)** – `bg_remover.remove(..., plot=True, galaxies=...)`; then background subtraction applied and file written.
10. **Reload** – `get_image`/`get_header` again after template block.
11. **Background (3rd)** – `bg_remover.remove(..., plot=False, galaxies=...)`; weight map written.
12. **Catalog** – Build or download, clean, border mask.
13. **SExtractor (2nd block)** – Run twice in a row: first run (get FWHM), second run with `use_FWHM=ImageFWHM`.
14. **Aperture/PSF** – Photometry on catalog sources; zeropoint (color term + estimate_zeropoint).
15. **Template subtraction** – If enabled: align, subtract, then target/limits on difference.
16. **Target & limits** – Measure target, injection/recovery; write outputs.

---

## 2. Speed & performance

### 2.1 Repeated FITS I/O (high impact)

**Issue:** `get_image(fpath)` and `get_header(fpath)` are called many times in `main.py` (on the order of 15+ times for the science file). Each call opens the FITS file separately. After steps that modify the file (trim, recrop, cosmic, background subtract, WCS write), a reload is needed, but several reloads are redundant because nothing was written in between.

**Examples of redundant reloads:**

- After recrop (lines 855–856): reload is correct (file was just written). Then the same `image`/`header` are used for SExtractor and first background.
- Lines 1366–1367: reload after template block; then background is run again. If no template block ran, this is still a fresh reload before catalog/SExtractor.
- Lines 2035–2036, 2468, 2473: further reloads in template/target sections.

**Recommendation:**

- Keep a single “current” `image` and `header` in the main loop and pass them into functions that only read.
- Reload only after a step that does `fits.writeto(fpath, ...)` (trim, recrop, cosmic, background subtract, WCS update).
- Where a step both reads and writes (e.g. background subtract), do: update in-memory `image`/`header` from the step’s result, then one `fits.writeto` and optionally one reload for the next block, instead of multiple separate reads.

This reduces disk I/O and parsing work, especially on slow or network filesystems.

---

### 2.2 Duplicate SExtractor runs (medium impact)

**Issue:** In the “Run source detection on final calibrated image” block (lines 1440–1456), SExtractor is run twice in succession:

```python
ImageFWHM, FWHMSources, scale = SExtractorWrapper(...).run(fpath, ...)  # 1st
ImageFWHM, FWHMSources, scale = SExtractorWrapper(...).run(fpath, ..., use_FWHM=ImageFWHM, ...)  # 2nd
```

The first run estimates FWHM; the second uses that FWHM for detection. Both runs read the same image from disk and do full detection.

**Recommendation:**

- If the wrapper supports it: one run that returns both FWHM and source list (e.g. estimate FWHM from a first pass, then run detection with that FWHM inside the same process/subprocess).
- If two runs are required by the current design: add a short comment explaining why (e.g. “First run: estimate FWHM; second: detect with fixed FWHM for stability”). Consider caching the first run’s FWHM in a small temp file or in-memory so that in debugging/re-run scenarios you can optionally skip the first run when a recent FWHM is already available (optional, lower priority).

---

### 2.3 Background removal called three times (medium impact)

**Issue:** On the main science path, `bg_remover.remove()` is called three times:

1. **~890:** `remove(image, plot=False, fwhm=ImageFWHM)` – no galaxies, no plot.  
   Used for cosmic-ray step (background surface/rms).
2. **~1087:** `remove(image, header=header, plot=True, fwhm=ImageFWHM, galaxies=variable_sources, ...)` – with galaxies and plotting.  
   Result used for subtraction and writing the calibrated image.
3. **~1370:** After reload, `remove(image, header=header, plot=False, galaxies=variable_sources, ...)` – same as (2) but no plot.  
   Result used for weight map and downstream catalog/SExtractor.

So the same image is background-estimated twice with the same options (with galaxies) apart from `plot` (2nd and 3rd). The 3rd call is after a reload; if the file written after (2) is the background-subtracted image, then (3) is run on the subtracted image (correct for weight map). If the file was not updated between (2) and the reload at 1366, then (3) is redundant with (2).

**Recommendation:**

- Clarify in comments or in the code: “Weight map and catalog use background estimated from the *current* (possibly background-subtracted) image.”
- If the image on disk at 1366 is already background-subtracted: keep one `remove` for that image (for weight/catalog). If it is not, then the two “with galaxies” calls could be unified: run once with `plot=True` and reuse the same result for both writing the calibrated image and the weight map, and avoid the 3rd full background run (unless the weight map must be from a different image).

---

### 2.4 Redundant path computation (low impact)

**Issue:** The same output-path logic appears twice (lines 194–201 and 211–218): `wdir`, `newDir`, `baseDir`, `workLoc`, `new_output_dir` (and then `cur_dir`, `output_csv_path`, etc.) are recomputed from `input_yaml`.

**Recommendation:** Compute `new_output_dir`, `cur_dir`, `output_csv_path`, and `calibration_file` once (e.g. right after setting `input_yaml["base"]`) and reuse. Reduces duplication and the chance of one branch getting out of sync.

---

### 2.5 Imports inside `run_photometry` (low impact)

**Issue:** Heavy imports (numpy, pandas, astropy, photutils, scipy, etc.) are inside `run_photometry()` (lines 71–130). So every call to `run_photometry()` pays the cost of importing.

**Recommendation:** Move as many imports as possible to the top of `main.py`. Keep inside the function only imports that are optional or that must run after the `os.environ` thread limits (e.g. if a library sets OpenMP at import time). This speeds up repeated invocations (e.g. tests or batch runs that call `run_photometry` multiple times).

---

## 3. Accuracy & correctness

### 3.1 Zeropoint and color term

- **Color term:** Recent robustness improvements (adaptive RANSAC threshold, multi-seed RANSAC, ODR fallback, slope sanity) are in place in `zeropoint.fit_color_term`; no further change suggested for this review.
- **Zeropoint inlier selection:** The RANSAC-based inlier selection in `_robust_RANSAC_fit` could be hardened in the same way as the color term (e.g. multi-seed, min inlier fraction, sigma-clip fallback) if you observe unstable ZPs in practice.

### 3.2 WCS and pixel scale

- Pixel scale is taken from WCS when valid, else from telescope.yml; special cases (e.g. MPI-2.2) are handled. Ensure any new instruments either have WCS or a telescope.yml `pixel_scale` so limits and aperture sizes are correct.

### 3.3 Exception handling

- Several `except Exception` blocks only log and continue. For critical steps (e.g. no WCS, no catalog), the pipeline already raises or skips explicitly. Consider narrowing `except Exception` to specific exceptions where a fallback is intended (e.g. `KeyError`, `OSError`) so programming errors are not silently swallowed.

---

## 4. Structural / maintainability

### 4.1 Single large function

- `run_photometry()` is a very long function (thousands of lines). Breaking it into phases (e.g. `_setup_paths_and_logging()`, `_load_and_validate_image()`, `_preprocess_image()`, `_run_detection_and_background()`, `_solve_wcs()`, `_build_catalog()`, `_run_photometry_and_zeropoint()`, `_template_subtract()`, `_measure_target_and_limits()`) would improve readability and testing. Each phase could take `input_yaml`, `image`, `header`, and return updated versions plus any new objects (catalog, ZP, etc.).

### 4.2 Pass image/header explicitly

- Once “current” image and header are maintained as above, pass them explicitly into helper functions and submodules (catalog, aperture, zeropoint, etc.) instead of having them call `get_image(fpath)` / `get_header(fpath)` internally where possible. That keeps I/O in one place and makes the data flow obvious.

---

## 5. Summary of recommended changes (priority)

| Priority | Change | Benefit |
|----------|--------|---------|
| High | Reduce redundant `get_image`/`get_header`: keep in-memory `image`/`header`, reload only after writes | Fewer FITS reads; faster and more predictable I/O |
| Medium | Run SExtractor once if possible, or document and optionally cache FWHM | Fewer subprocess/disk reads |
| Medium | Clarify background-removal sequence; avoid duplicate “with galaxies” run if not needed | Less CPU and clearer behavior |
| Low | Compute output paths once; reuse variables | Less duplication, fewer bugs |
| Low | Move imports to module top where safe | Faster repeated runs |
| Structural | Split `run_photometry()` into phased helpers; pass image/header explicitly | Easier to test and maintain |

If you want, the next step can be a concrete patch for one of these (e.g. “single image/header reload policy” or “single SExtractor run with clear FWHM flow”) in `main.py`.
