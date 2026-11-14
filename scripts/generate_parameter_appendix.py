#!/usr/bin/env python3
"""Generate the LaTeX parameter-reference appendix from default_input.yml.

Parses ``autophot/data/default_input.yml`` textually (comments carry the
``# type --- description`` annotations that PyYAML discards) and emits one
``longtable`` per top-level config section, ready for ``\\input`` in the
methods document.

Usage:
    python scripts/generate_parameter_appendix.py \
        > docs/autophot_parameters.tex
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
YML = REPO / "autophot" / "data" / "default_input.yml"

# Ordered presentation of the YAML sections (matches the file's own order).
SECTION_TITLES = {
    "default_input": "General and run-control parameters",
    "preprocessing": "Pre-processing",
    "photometry": "Photometry",
    "wcs": "World coordinate system",
    "catalog": "Reference catalogues",
    "zeropoint": "Photometric zero-point",
    "cosmic_rays": "Cosmic rays and defects",
    "source_detection": "Source detection and FWHM",
    "limiting_magnitude": "Limiting magnitudes",
    "host_model": "Host-galaxy modelling",
    "alignment": "Image reprojection",
    "template_subtraction": "Template subtraction",
    "background": "Background estimation",
    "postprocessing": "Post-processing",
}

# Thematic sub-grouping applied inside the (large) ``photometry`` section.
# First matching prefix wins; unmatched keys fall into "Other".
PHOTOMETRY_GROUPS = [
    ("detection", "Detection and forced-photometry decision"),
    ("force_photometry", "Detection and forced-photometry decision"),
    ("fitting_xy_bounds", "Centroid window of the PSF fit"),
    ("psf_group_separation", "Centroid window of the PSF fit"),
    ("centroid_max_wander", "Centroid window of the PSF fit"),
    ("aperture_", "Aperture photometry"),
    ("apply_aperture", "Aperture photometry"),
    ("plot_aperture", "Aperture photometry"),
    ("find_optimum", "Aperture photometry"),
    ("optimum_radius", "Aperture photometry"),
    ("annulus_", "Aperture photometry"),
    ("min_aperture", "Aperture photometry"),
    ("crowded_optimum", "Aperture photometry"),
    ("crowded_field", "Aperture photometry"),
    ("empirical_noise_factor", "Aperture photometry"),
    ("enforce_nonnegative", "Aperture photometry"),
    ("systematic_error_floor", "Aperture photometry"),
    ("apply_colorterms", "Instrumental colour terms"),
    ("color_term", "Instrumental colour terms"),
    ("lpi_", "Local background infill (LPI)"),
    ("psf_contam_", "ePSF star vetoes: cutout contamination"),
    ("psf_shape_", "ePSF star vetoes: ensemble shape outliers"),
    ("psf_fft_", "ePSF star vetoes: Fourier-domain outliers"),
    ("psf_norm_radius", "ePSF star vetoes: normalisation mask"),
    ("psf_grid", "Spatially varying (gridded) ePSF"),
    ("psf_spatial", "Spatially varying (gridded) ePSF"),
    ("psf_analytic_", "Analytic composite-Moffat PSF"),
    ("psf_use_analytic", "Analytic composite-Moffat PSF"),
    ("psf_init_moffat", "Analytic composite-Moffat PSF"),
    ("psf_fit_", "PSF fit and fitters"),
    ("psf_diff_negative", "PSF fit and fitters"),
    ("psf_target_fit", "PSF fit and fitters"),
    ("psf_localbkg", "PSF fit and fitters"),
    ("psf_fwhm_guard", "PSF fit and fitters"),
    ("psf_fit_positive", "PSF fit and fitters"),
    ("perform_emcee", "PSF fit and fitters"),
    ("check_inverted", "PSF fit and fitters"),
    ("use_poisson", "PSF fit and fitters"),
    ("poisson_", "PSF fit and fitters"),
    ("emcee_", "PSF fit and fitters"),
    ("psf_", "ePSF construction"),
    ("fwhm_", "Field FWHM estimation"),
    ("sextractor_reject", "ePSF construction"),
    ("oversample_psf", "ePSF construction"),
    ("linearity_", "ePSF construction"),
    ("undersampled_fwhm", "Sampling regime"),
    ("critical_fwhm", "Sampling regime"),
    ("oversampled_fwhm", "Sampling regime"),
    ("wcs_offset_min_snr", "Diagnostics"),
    ("target_defect", "Diagnostics"),
    ("aperture_n_jobs", "Parallelism"),
    ("photutils_n_threads", "Parallelism"),
    ("redo_sources", "Other"),
]

GROUP_ORDER = [
    "Detection and forced-photometry decision",
    "Centroid window of the PSF fit",
    "Aperture photometry",
    "Instrumental colour terms",
    "Local background infill (LPI)",
    "Sampling regime",
    "Field FWHM estimation",
    "ePSF construction",
    "ePSF star vetoes: cutout contamination",
    "ePSF star vetoes: ensemble shape outliers",
    "ePSF star vetoes: Fourier-domain outliers",
    "ePSF star vetoes: normalisation mask",
    "Analytic composite-Moffat PSF",
    "Spatially varying (gridded) ePSF",
    "PSF fit and fitters",
    "Diagnostics",
    "Parallelism",
    "Other",
]

# Per-table introductions: a short scope paragraph plus the minimal
# ``autophot_input[...][...] = ...`` edits a user is most likely to
# need.  Shown values are illustrative; the table itself carries the
# real defaults.
SECTION_INTROS = {
    "default_input": (
        (
            r"Run definition and control: input and output locations, the "
            r"target specification, filter selection, image-level "
            r"parallelism, and console verbosity. The tail of the table "
            r"holds runtime placeholders that are filled internally per "
            r"image; they exist in the schema so every \code{.get()} has a "
            r"declared type (Section~\ref{sec:config}) -- leave them alone. "
            r"A typical run edits only this table plus whichever stage "
            r"tables it is tuning."
        ),
        [
            'autophot_input["fits_dir"] = "/data/SN2026abc/images"',
            'autophot_input["wdir"] = "/data/SN2026abc/work"',
            'autophot_input["target_name"] = "SN2026abc"',
            'autophot_input["selected_filters"] = ["g", "r"]',
            'autophot_input["nCPU"] = 8',
            'autophot_input["rng_seed"] = 42',
        ],
    ),
    "preprocessing": (
        (
            r"Frame cleanup applied before any measurement: trimming edge "
            r"rows and columns, and dropping duplicate exposures so the "
            r"same data are not reduced twice."
        ),
        [
            'autophot_input["preprocessing"]["trim_image"] = 5  # px off every edge',
            'autophot_input["preprocessing"]["deduplicate_input_files"] = True',
        ],
    ),
    "photometry": (
        (
            r"The largest section: it decides which sources get measured, "
            r"how the aperture radius and local sky are chosen, how the "
            r"empirical PSF is built and vetted, which fitter measures the "
            r"target, and what counts as a detection at all. The sub-tables "
            r"below follow that order; most runs touch only the detection "
            r"and aperture groups."
        ),
        [],
    ),
    "wcs": (
        (
            r"Verify-first astrometry: the existing WCS is validated "
            r"against Gaia and a full astrometry.net solve -- guided by "
            r"header hints, tweak orders, and CPU budgets -- runs only when "
            r"verification fails or \code{redo\_wcs} is set."
        ),
        [
            'autophot_input["wcs"]["redo_wcs"] = False  # trust the header WCS',
            'autophot_input["wcs"]["verify_input_wcs"] = True',
            'autophot_input["wcs"]["projection_type"] = "SIP"  # or "TPV"',
        ],
    ),
    "catalog": (
        (
            r"Which survey supplies the calibrator sequence and how the "
            r"field is queried: backend selection (a single survey, a "
            r"per-band mapping, or a user CSV), query geometry, retries, "
            r"and published band transformations for bands the survey does "
            r"not natively carry."
        ),
        [
            'autophot_input["catalog"]["use_catalog"] = "panstarrs"',
            'autophot_input["catalog"]["allow_filter_transformations"] = True',
            (
                'autophot_input["catalog"]["include_IR_sequence_data"]'
                " = True  # add 2MASS JHK"
            ),
        ],
    ),
    "zeropoint": (
        (
            r"The calibrator vetoes and the fit that turns them into a "
            r"zero point: magnitude range, saturation and linearity gates, "
            r"blend and defect rejection, the fit method, and the "
            r"colour/spatial correction terms."
        ),
        [
            'autophot_input["zeropoint"]["fit_method"] = "odr"  # mcmc | odr | ransac',
            'autophot_input["zeropoint"]["faint_mag_limit"] = 21.0',
            'autophot_input["zeropoint"]["min_source_no"] = 5',
        ],
    ),
    "cosmic_rays": (
        (
            r"Cosmic-ray rejection and defect masking: the LA.Cosmic-style "
            r"clip or the astroscrappy backend, mask dilation, and the "
            r"guards that stop a cosmic ray being carved out of the target "
            r"itself."
        ),
        [
            'autophot_input["cosmic_rays"]["remove_cmrays"] = True',
            (
                'autophot_input["cosmic_rays"]["cr_backend"] = "lacosmic"'
                "  # ccdproc instead of astroscrappy"
            ),
            'autophot_input["cosmic_rays"]["cr_sigclip"] = 4.5',
        ],
    ),
    "source_detection": (
        (
            r"How the source list is found and measured: the finder and "
            r"its threshold, the FWHM estimation set, and the shape cuts "
            r"(elongation, $\chi^2$, segment size) that separate stars "
            r"from artifacts."
        ),
        [
            'autophot_input["source_detection"]["dao_nsigma"] = 4.0',
            'autophot_input["source_detection"]["elongation_max"] = 1.6',
            'autophot_input["source_detection"]["min_segment_npixels"] = 3',
        ],
    ),
    "limiting_magnitude": (
        (
            r"Injection/recovery limiting magnitudes: how many artificial "
            r"sources are drawn and where they are placed, the S/N "
            r"threshold(s) each limit is reported against, and the "
            r"completeness-fit machinery that turns recoveries into "
            r"$m_{50}$."
        ),
        [
            'autophot_input["limiting_magnitude"]["snr_thresholds"] = [3.0, 5.0]',
            'autophot_input["limiting_magnitude"]["inject_final_n_sites"] = 30',
            (
                'autophot_input["limiting_magnitude"]["completeness_solver"]'
                ' = "bisect"  # or the emcee logistic fit'
            ),
        ],
    ),
    "host_model": (
        (
            r"Optional parametric host-galaxy subtraction: an external "
            r"GALFIT call fits S\'ersic (and optional point-source) "
            r"components around the target before photometry."
        ),
        [
            'autophot_input["host_model"]["enabled"] = True',
            'autophot_input["host_model"]["galfit_path"] = "/usr/local/bin/galfit"',
            'autophot_input["host_model"]["n_sersic"] = 1',
        ],
    ),
    "alignment": (
        (
            r"Reprojection settings used whenever one frame is resampled "
            r"onto another (template to science, diagnostic checks): "
            r"method, interpolation order, and flux conservation."
        ),
        [
            (
                'autophot_input["alignment"]["reproject_method"]'
                ' = "adaptive"  # exact | adaptive | interp'
            ),
            'autophot_input["alignment"]["reproject_interp_order"] = "bicubic"',
            'autophot_input["alignment"]["reproject_adaptive_conserve_flux"] = True',
        ],
    ),
    "template_subtraction": (
        (
            r"Reference-template discovery, the alignment cascade, flux "
            r"matching, and the three subtraction backends with their "
            r"vetoes and quality gates. Disabled by default; enabling the "
            r"stage is one flag, the rest is tuning."
        ),
        [
            'autophot_input["template_subtraction"]["do_subtraction"] = True',
            (
                'autophot_input["template_subtraction"]["method"] = "sfft"'
                "  # sfft | hotpants | zogy | auto"
            ),
            'autophot_input["template_subtraction"]["alignment_method"] = "reproject"',
            'autophot_input["template_subtraction"]["forceconv"] = "REF"',
        ],
    ),
    "background": (
        (
            r"The two-pass sky estimate: the iterative source mask, the "
            r"mesh-binning statistics, the clipped interpolation back to "
            r"the grid, the optional MaxiMask defect classifier, and the "
            r"target-local \code{local\_*} fit."
        ),
        [
            'autophot_input["background"]["use_maximask"] = True  # needs tensorflow',
            'autophot_input["background"]["mesh_scale_min"] = 4.0  # mesh size in FWHM',
            'autophot_input["background"]["mesh_scale_max"] = 10.0',
        ],
    ),
    "postprocessing": (
        (
            r"What happens after the per-image pipeline: light-curve "
            r"colour-term correction, RGB composite images, the "
            r"Lomb--Scargle periodogram, and instrument-alias handling in "
            r"the aggregated table."
        ),
        [
            'autophot_input["postprocessing"]["color_image"] = True',
            'autophot_input["postprocessing"]["periodogram"] = True',
            'autophot_input["postprocessing"]["transient_color_correction"] = True',
        ],
    ),
}

# Same treatment for the photometry sub-tables.
GROUP_INTROS = {
    "Detection and forced-photometry decision": (
        (
            r"Whether the measured target flux counts as a detection: the "
            r"S/N threshold, fit-quality and sign-consistency vetoes, the "
            r"empirical false-alarm annulus test, the matched-filter and "
            r"aperture-fallback rescue channels, and the SNU upper limit "
            r"reported for non-detections."
        ),
        [
            'autophot_input["photometry"]["detection_limit"] = 5.0',
            (
                'autophot_input["photometry"]["force_photometry"] = True'
                "  # pin to catalogue coords"
            ),
            (
                'autophot_input["limiting_magnitude"]["snr_thresholds"]'
                " = [3.0, 5.0]  # SNU = 2nd entry"
            ),
        ],
    ),
    "Centroid window of the PSF fit": (
        (
            r"How far a fitted centroid may move from its seed position, "
            r"and when neighbouring sources are solved as one blended "
            r"group."
        ),
        [
            'autophot_input["photometry"]["fitting_xy_bounds"] = 4.0  # px',
            'autophot_input["photometry"]["psf_group_separation_fwhm"] = 2.5',
            'autophot_input["photometry"]["centroid_max_wander_fwhm"] = 0.5',
        ],
    ),
    "Aperture photometry": (
        (
            r"The radius (a fixed FWHM multiple, explicit pixels, or the "
            r"per-image curve-of-growth optimum), the sky annulus, and the "
            r"noise bookkeeping -- including the empirical factor that "
            r"rescales the propagated variance against measured "
            r"sky-aperture scatter."
        ),
        [
            'autophot_input["photometry"]["aperture_size"] = 1.7  # radius in FWHM',
            'autophot_input["photometry"]["find_optimum_radius"] = True',
            'autophot_input["photometry"]["annulus_min_pixels"] = 20',
            'autophot_input["photometry"]["empirical_noise_factor"] = True',
        ],
    ),
    "Instrumental colour terms": (
        (
            r"An optional colour term between the instrumental and "
            r"catalogue magnitudes, fitted as a polynomial or a piecewise "
            r"line."
        ),
        [
            'autophot_input["photometry"]["apply_colorterms"] = True',
            'autophot_input["photometry"]["color_term_poly_order"] = 2  # 1 = linear',
        ],
    ),
    "Local background infill (LPI)": (
        (
            r"(In development) An LPI-style local background infill "
            r"behind the target, for structured backgrounds that defeat "
            r"the global mesh."
        ),
        [
            'autophot_input["photometry"]["lpi_background_for_target"] = True',
            'autophot_input["photometry"]["lpi_n_samples"] = 250',
        ],
    ),
    "Sampling regime": (
        (
            r"The FWHM thresholds that classify each frame as "
            r"undersampled, critically sampled, or oversampled; "
            r"detection, aperture, and PSF behaviour adapt to the regime."
        ),
        [
            'autophot_input["photometry"]["undersampled_fwhm_threshold"] = 2.5  # px',
            'autophot_input["photometry"]["oversampled_fwhm_threshold"] = 5.0  # px',
        ],
    ),
    "Field FWHM estimation": (
        (
            r"Robustness guards on the per-image seeing measurement "
            r"itself: clipping iterations, how far the fitted FWHM may "
            r"inflate past the robust estimate, and which sources may "
            r"vote."
        ),
        [
            'autophot_input["photometry"]["fwhm_inflation_max_factor"] = 1.5',
            'autophot_input["photometry"]["fwhm_sigma_clip_maxiters"] = 15',
        ],
    ),
    "ePSF construction": (
        (
            r"Everything about building the empirical PSF: the candidate "
            r"pool and its vetoes, cutout size, recentering, the "
            r"oversampling-factor pick, smoothing kernels, the iterative "
            r"build loop, and the linearity/saturation screens applied to "
            r"the pool."
        ),
        [
            (
                'autophot_input["photometry"]["oversample_psf"] = True'
                "  # force adaptive oversampling"
            ),
            'autophot_input["photometry"]["sextractor_reject_constant_regions"] = True',
        ],
    ),
    "ePSF star vetoes: cutout contamination": (
        (
            r"Pixel-level vetoes on the candidate cutouts: quadrant "
            r"asymmetry, edge and border excess, radial reversals, "
            r"secondary peaks, and streak tests that catch what "
            r"catalogue-level isolation misses."
        ),
        [
            'autophot_input["photometry"]["psf_cutout_contamination_check"] = True',
            'autophot_input["photometry"]["psf_contam_asymmetry_sigma"] = 3.0',
        ],
    ),
    "ePSF star vetoes: ensemble shape outliers": (
        (
            r"Rejects cutouts whose flux-normalised shape disagrees with "
            r"the ensemble median stack -- miscentred, contaminated, or "
            r"deviant morphologies that would stop the build converging."
        ),
        [
            'autophot_input["photometry"]["psf_shape_outlier_rejection"] = True',
            'autophot_input["photometry"]["psf_shape_outlier_nsigma"] = 4.0',
        ],
    ),
    "ePSF star vetoes: Fourier-domain outliers": (
        (
            r"Fourier-domain outlier rejection on the cutouts; it catches "
            r"artifacts (hot pixels, cosmic-ray clusters) whose near-flat "
            r"azimuthal power spectrum the image-domain tests miss."
        ),
        [
            'autophot_input["photometry"]["psf_fft_rejection"] = True',
            'autophot_input["photometry"]["psf_fft_n_sigma"] = 4.0',
        ],
    ),
    "ePSF star vetoes: normalisation mask": (
        (
            r"How many masked or non-finite pixels may sit inside the "
            r"flux-normalisation circle of a candidate before it is "
            r"rejected."
        ),
        [
            'autophot_input["photometry"]["psf_norm_radius_mask_frac_max"] = 0.02',
        ],
    ),
    "Analytic composite-Moffat PSF": (
        (
            r"The fitted composite-Moffat model that seeds the iterative "
            r"ePSF build, serves as the residual diagnostic, and is "
            r"delivered as a fallback when too few vetted stars survive."
        ),
        [
            'autophot_input["photometry"]["psf_use_analytic_init"] = True',
            'autophot_input["photometry"]["psf_analytic_components"] = 3',
            'autophot_input["photometry"]["psf_analytic_fallback"] = True',
        ],
    ),
    "Spatially varying (gridded) ePSF": (
        (
            r"An optional spatially varying model: the field is split "
            r"into detector grid cells, each running the full "
            r"vet-and-build pipeline, and the per-cell ePSFs are combined "
            r"into a \code{GriddedPSFModel}."
        ),
        [
            (
                'autophot_input["photometry"]["psf_spatial_grid"] = "auto"'
                "  # 3x3 then 2x2, else single ePSF"
            ),
            'autophot_input["photometry"]["psf_grid_min_stars_per_cell"] = 25',
        ],
    ),
    "PSF fit and fitters": (
        (
            r"The fit itself: box sizes and shape priors, the "
            r"non-negativity bound, and which of the three fitters runs "
            r"-- least-squares, Poisson likelihood, or MCMC below a "
            r"chosen S/N."
        ),
        [
            'autophot_input["photometry"]["use_poisson_likelihood_fitter"] = True',
            (
                'autophot_input["photometry"]["perform_emcee_fitting_s2n"]'
                " = 10  # MCMC below this S/N"
            ),
            (
                'autophot_input["photometry"]["check_inverted_image"] = True'
                "  # fading dips on diffs"
            ),
        ],
    ),
    "Diagnostics": (
        (
            r"Warning thresholds only -- they flag suspicious conditions "
            r"(astrometric offset, target on a defect) without changing "
            r"any measurement."
        ),
        [
            'autophot_input["photometry"]["wcs_offset_min_snr"] = 5.0',
            'autophot_input["photometry"]["target_defect_warning_fwhm"] = 3.0',
        ],
    ),
    "Parallelism": (
        (
            r"Per-image thread count for photutils calls; multiply by "
            r"\code{nCPU} (Table~\ref{tab:default_input}) and keep the "
            r"product under the core count."
        ),
        [
            'autophot_input["photometry"]["photutils_n_threads"] = 4',
        ],
    ),
    "Other": (
        (r"Keys that fit nowhere else."),
        [
            (
                'autophot_input["photometry"]["redo_sources"] = True'
                "  # rebuild the source list"
            ),
        ],
    ),
}

KEY_RE = re.compile(r"^(?P<indent>\s*)(?P<key>[A-Za-z0-9_]+)\s*:(?P<rest>.*)$")

# Keys with no consumer in the autophot package, verified with
# scripts/audit_config_usage.py plus manual checks against dynamic key
# construction and external-tool pass-through.  Dotted section.key paths.
# Currently empty: the 2026-11 audits removed the twelve dead/no-op keys
# found (see _CONFIG_REMOVED_KEYS in autophot/core/functions.py).
UNREFERENCED_KEYS: set[str] = set()


def esc(text: str) -> str:
    """Escape a plain-text fragment for LaTeX."""
    if text is None:
        return ""
    t = str(text).strip()
    t = t.replace("\\", "\\textbackslash{}")
    for a, b in (
        ("&", "\\&"),
        ("%", "\\%"),
        ("#", "\\#"),
        ("_", "\\_"),
        ("$", "\\$"),
        ("{", "\\{"),
        ("}", "\\}"),
        ("~", "\\textasciitilde{}"),
        ("^", "\\textasciicircum{}"),
        ("<", "\\textless{}"),
        (">", "\\textgreater{}"),
        ("|", "\\textbar{}"),
    ):
        t = t.replace(a, b)
    # YAML comments mark inline code with backticks; render it in tt.
    t = re.sub(r"`([^`]+)`", r"\\texttt{\1}", t)
    return t


def breakable(tt_text: str) -> str:
    """Allow LaTeX to break a \\texttt{...} span after underscores."""
    return r"\texttt{" + tt_text.replace(r"\_", r"\_{\allowbreak}") + "}"


def break_tt_spans(text: str) -> str:
    """Insert breakpoints inside every \\texttt{...} in text.

    Long literal values (env vars, key lists, dicts, paths) have to wrap
    inside the narrow table columns; breaks are allowed after underscores,
    commas, slashes, dots, and colons.
    """

    def _fix(m):
        inner = m.group(1).replace(r"\_", r"\_{\allowbreak}")
        for ch in (",", "/", ".", ":"):
            inner = inner.replace(ch, ch + r"\allowbreak{}")
        return r"\texttt{" + inner + "}"

    return re.sub(r"\\texttt\{([^}]*)\}", _fix, text)


def break_plain(text: str) -> str:
    """Insert breakpoints in plain (non-tt) text segments.

    Descriptions quote long identifiers in roman text (env vars, paths,
    host names); breaks are allowed after escaped underscores, slashes,
    and letter-followed dots.
    """
    parts = re.split(r"(\\texttt\{[^}]*\})", text)
    for i, part in enumerate(parts):
        if part.startswith(r"\texttt{"):
            continue
        part = part.replace(r"\_", r"\_{\allowbreak}")
        part = part.replace("/", r"/\allowbreak{}")
        part = re.sub(r"\.(?=[A-Za-z])", r".\\allowbreak ", part)
        parts[i] = part
    return "".join(parts)


def parse_yml(path: Path):
    """Return {section: [(key, default, type, desc, runtime)]} preserving order."""
    sections: dict[str, list] = {}
    stack: list[tuple[int, str]] = []  # (indent, key) of open mappings

    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        m = KEY_RE.match(raw)
        if not m:
            continue
        indent = len(m.group("indent").replace("\t", "    "))
        key = m.group("key")
        rest = m.group("rest") or ""

        # Pop closed mappings.
        while stack and indent <= stack[-1][0]:
            stack.pop()

        has_value = bool(rest.strip()) and not rest.strip().startswith("#")
        is_mapping = not rest.strip()  # "key:" alone opens a section

        if is_mapping:
            stack.append((indent, key))
            if key not in SECTION_TITLES:
                # nested mapping, e.g. wcs.crowded -> keep under parent title
                parent = stack[-2][1] if len(stack) > 1 else "default_input"
                sections.setdefault(parent, [])
                sections[parent].append(("__HEADER__", key, "", "", False))
            else:
                sections.setdefault(key, [])
            continue

        # Scalar/list entry.
        value_part, _, comment = rest.partition("#")
        value = value_part.strip() or "null"
        dtype, desc = "", comment.strip()
        if "---" in desc:
            dtype, _, desc = desc.partition("---")
            dtype, desc = dtype.strip(), desc.strip()
        runtime = desc.lower().startswith("runtime")
        # Attach to the nearest ancestor that is a documented section;
        # nested mappings (wcs.crowded etc.) stay inside their parent table.
        parent = "default_input"
        for _, skey in reversed(stack):
            if skey in SECTION_TITLES:
                parent = skey
                break
        sections.setdefault(parent, []).append((key, value, dtype, desc, runtime))

    return sections


def phot_group(key: str) -> str:
    for prefix, group in PHOTOMETRY_GROUPS:
        if key.startswith(prefix):
            return group
    return "Other"


def fmt_default(v: str) -> str:
    if v in ("null", "None", ""):
        return r"\emph{null}"
    return break_tt_spans(r"\texttt{" + esc(v) + "}")


def emit_table(title: str, rows, path_prefix: str, section: str, fid):
    """Emit one longtable; rows may contain __HEADER__ pseudo-rows."""
    label = re.sub(r"[^A-Za-z0-9_-]+", "-", path_prefix).strip("-")
    fid.write(
        "{\\footnotesize\n\\begin{longtable}{@{}p{5.6cm}p{2.2cm}p{7.4cm}@{}}\n"
        "\\caption{%s --- configuration reference.} \\label{tab:%s}\\\\\n\\toprule\n"
        "\\textbf{Key} & \\textbf{Default (type)} & \\textbf{Description} \\\\\n"
        "\\midrule\n"
        "\\endfirsthead\n\\multicolumn{3}{l}{\\footnotesize\\emph{continued}}\\\\\n"
        "\\toprule\n"
        "\\textbf{Key} & \\textbf{Default (type)} & \\textbf{Description} \\\\\n"
        "\\midrule\n\\endhead\n"
        "\\bottomrule\n"
        "\\endlastfoot\n" % (esc(title), label)
    )
    for key, value, dtype, desc, runtime in rows:
        if key == "__HEADER__":
            fid.write(
                "\\multicolumn{3}{l}{\\textbf{sub-section \\texttt{%s}}}\\\\\n"
                "\\midrule\n" % esc(value)
            )
            continue
        dv = fmt_default(value)
        if dtype:
            # Long union/list type annotations must wrap in the narrow
            # Default column.
            t = esc(dtype).replace(r"\textbar{}", r"\textbar{}\allowbreak{}")
            t = t.replace("[", r"[\allowbreak{}")
            dv += " {\\scriptsize(%s)}" % t
        d = break_plain(break_tt_spans(esc(desc)))
        marks = []
        if runtime:
            marks.append("runtime")
        # "Deprecated" at the head of a description is the deprecation
        # marker; the same word later in the text just refers to old keys.
        if "deprecat" in desc[:15].lower():
            marks.append("deprecated")
        path = key if section == "default_input" else "%s.%s" % (section, key)
        if path in UNREFERENCED_KEYS:
            marks.append("unreferenced")
        if marks:
            d = "\\emph{(%s)} %s" % (", ".join(marks), d)
        fid.write("%s & %s & %s \\\\\n" % (breakable(esc(key)), dv, d))
    fid.write("\\end{longtable}}\n\n")


def main() -> int:
    sections = parse_yml(YML)

    out = sys.stdout
    out.write(
        "% Auto-generated by scripts/generate_parameter_appendix.py\n"
        "% Source: autophot/data/default_input.yml\n"
        "% Do not edit by hand; regenerate after changing default_input.yml.\n\n"
    )

    order = [s for s in SECTION_TITLES if s in sections]
    order += [s for s in sections if s not in SECTION_TITLES]
    for sec in order:
        rows = sections[sec]
        title = SECTION_TITLES.get(sec, sec)
        paragraph, code_lines = SECTION_INTROS.get(sec, ("", []))
        if paragraph:
            out.write("\\subsection{%s}\n\n%s\n" % (title, paragraph))
            if code_lines:
                out.write(
                    "\n\\begin{lstlisting}[language=Python]\n%s\n"
                    "\\end{lstlisting}\n\n" % "\n".join(code_lines)
                )
        if sec == "photometry":
            # Split the large photometry block into thematic sub-tables.
            grouped: dict[str, list] = {}
            for r in rows:
                grouped.setdefault(phot_group(r[0]), []).append(r)
            for gname in GROUP_ORDER:
                grows = grouped.get(gname)
                if not grows:
                    continue
                gpara, gcode = GROUP_INTROS.get(gname, ("", []))
                if gpara:
                    out.write("\\subsubsection{%s}\n\n%s\n" % (gname, gpara))
                    if gcode:
                        out.write(
                            "\n\\begin{lstlisting}[language=Python]\n"
                            "%s\n\\end{lstlisting}\n\n" % "\n".join(gcode)
                        )
                emit_table(
                    "%s: %s" % (title, gname),
                    grows,
                    f"photometry-{gname}",
                    sec,
                    out,
                )
            leftovers = [k for k in grouped if k not in GROUP_ORDER]
            for k in leftovers:
                emit_table(f"{title}: {k}", grouped[k], f"photometry-{k}", sec, out)
        else:
            emit_table(title, rows, sec, sec, out)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
