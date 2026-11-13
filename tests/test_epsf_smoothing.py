"""Tests for ePSF smoothing-kernel sizing and adaptive oversampling.

Covers:
* ``get_smoothing_kernel`` -- nearest-odd auto-sizing, the FWHM-relative
  footprint cap (which also bounds explicit ``kernel_size``), and the
  ``none`` opt-out.
* ``_select_adaptive_oversample`` -- for undersampled data the fine grid is
  the binding constraint: synthetic pixel-integrated builds show a 2x grid
  biases the ePSF ~4% broad regardless of kernel, while thin sampling is
  handled by the kernel + phase-class interpolation, so 4x is preferred at
  every usable star count.
* An end-to-end undersampled ePSF build against the known
  pixel-integrated truth.
"""

from __future__ import annotations

import numpy as np
import pytest

from autophot.photometry.psf import (
    _epsf_phase_diagnostics,
    _select_adaptive_oversample,
    clean_epsf_stamp,
    get_smoothing_kernel,
    measure_epsf_fwhm_native,
    pixel_integrate_oversampled,
)


class TestSmoothingKernelSizing:
    def test_canonical_quartic_at_4x(self):
        """scale=1.0 at osamp=4 reproduces the A&K 5x5 kernel."""
        k = get_smoothing_kernel(3.0, oversample=4, size_scale_px=1.0)
        assert k.shape == (5, 5)
        assert np.isclose(k.sum(), 1.0)

    def test_fwhm_cap_bounds_boosted_footprint(self):
        """A boosted 2.5 px request on a 2 px PSF respects the FWHM cap:
        ~1xFWHM -> 7x7 (1.75 px), the default 1.25xFWHM -> 9x9 (2.25 px)."""
        k = get_smoothing_kernel(
            2.0, oversample=4, size_scale_px=2.5, max_fwhm_frac=1.0
        )
        assert k.shape == (7, 7)
        k = get_smoothing_kernel(2.0, oversample=4, size_scale_px=2.5)
        assert k.shape == (9, 9)

    def test_cap_floor_keeps_usable_kernel(self):
        """At FWHM=1.5 px the cap floor (~1.75 px) still permits the
        empirically optimal 7x7 kernel at osamp=4."""
        k = get_smoothing_kernel(1.5, oversample=4, size_scale_px=2.5)
        assert k.shape == (7, 7)

    def test_explicit_size_respects_cap(self):
        """An explicit kernel wider than the cap is truncated (a >=3x3
        kernel at osamp=1 spans several FWHM and broadens the ePSF
        catastrophically)."""
        k = get_smoothing_kernel(2.0, oversample=1, kernel_size=5)
        assert k.shape == (1, 1)

    def test_none_disables_smoothing(self):
        assert get_smoothing_kernel(2.0, oversample=4, kind="none") is None

    def test_osamp2_minimal_kernel(self):
        k = get_smoothing_kernel(2.0, oversample=2, size_scale_px=1.0)
        assert k.shape == (3, 3)

    def test_cap_disabled(self):
        k = get_smoothing_kernel(
            2.0, oversample=4, size_scale_px=2.5, max_fwhm_frac=0.0
        )
        assert k.shape == (9, 9)

    def test_auto_sizes_from_fwhm(self):
        """'auto' mirrors photutils 3.1: footprint 0.7xFWHM in
        oversampled cells (4.6 px at osamp=2 -> 6.4 -> 7x7)."""
        k = get_smoothing_kernel(4.6, oversample=2, kind="auto")
        assert k.shape == (7, 7)
        assert np.isclose(k.sum(), 1.0)

    def test_auto_skips_heavily_undersampled(self):
        """'auto' disables smoothing when the 0.7xFWHM kernel spans
        fewer than 5 oversampled grid points (raw width <5, not the
        odd-rounded size)."""
        assert get_smoothing_kernel(1.7, oversample=4, kind="auto") is None
        assert get_smoothing_kernel(6.0, oversample=1, kind="auto") is None

    def test_auto_explicit_size(self):
        k = get_smoothing_kernel(4.6, oversample=2, kind="auto", kernel_size=9)
        assert k.shape == (9, 9)


class TestAdaptiveOversample:
    def test_prefers_4x_when_well_sampled(self):
        # ~3 samples/gridpoint required: n >= 48 gives 4x (>=3 samples/cell)
        for n in (48, 64, 128, 256):
            assert _select_adaptive_oversample(n, {})[0] == 4

    def test_thin_sampling_descends(self):
        # n=44 -> floor(sqrt(44/3)) = 3 (4.9 samples/cell)
        assert _select_adaptive_oversample(44, {})[0] == 3
        # n=20 -> floor(sqrt(20/3)) = 2 (5 samples/cell)
        assert _select_adaptive_oversample(20, {})[0] == 2
        # n=32 -> floor(sqrt(32/3)) = 3
        assert _select_adaptive_oversample(32, {})[0] == 3

    def test_starved_fields_get_1x(self):
        # starved pools reach the robust 1x regime (every star constrains
        # every grid cell) rather than a noise-dominated high-factor build
        for n in (5, 6, 7, 11):
            assert _select_adaptive_oversample(n, {})[0] == 1

    def test_below_hard_min_disables(self):
        assert _select_adaptive_oversample(4, {})[0] == 1
        assert _select_adaptive_oversample(0, {})[0] == 1

    def test_samples_threshold_respected(self):
        cfg = {"psf_oversample_min_samples_per_pixel": 2.0}
        # n=8 -> floor(sqrt(4)) = 2
        assert _select_adaptive_oversample(8, cfg)[0] == 2
        # n=32 -> floor(sqrt(16)) = 4
        assert _select_adaptive_oversample(32, cfg)[0] == 4

    def test_min_floor_config(self):
        # an explicit floor still applies when set
        cfg = {"psf_oversample_min_undersampled": 2}
        assert _select_adaptive_oversample(8, cfg)[0] == 2

    def test_understaffed_flag(self):
        # Below the hard star minimum the pool cannot support even a 1x
        # build; the flag must surface that to the caller.
        assert _select_adaptive_oversample(4, {}) == (1, True)
        # A forced 2x floor on a thin pool (10 stars -> 2.5 samples/cell
        # < 3) is flagged too.
        cfg = {"psf_oversample_min_undersampled": 2}
        assert _select_adaptive_oversample(10, cfg) == (2, True)
        # Adequately staffed: no flag.
        assert _select_adaptive_oversample(48, {}) == (4, False)


class TestUndersampledEpsfFidelity:
    """End-to-end: the pipeline kernel choice on an undersampled ePSF build
    recovers the true effective (pixel-integrated) FWHM."""

    BETA = 3.0
    FWHM_CONT = 1.8

    @classmethod
    def _truth_fwhm(cls):
        from scipy.signal import fftconvolve

        gamma = cls.FWHM_CONT / (
            2.0 * np.sqrt(2.0 ** (1.0 / cls.BETA) - 1.0)
        )
        xg = np.linspace(-8, 8, 513)
        xx, yy = np.meshgrid(xg, xg)
        cont = (1.0 + (np.hypot(xx, yy) / gamma) ** 2) ** (-cls.BETA)
        taps = np.ones(32) / 32
        eff = fftconvolve(cont, np.outer(taps, taps), mode="same")
        r = np.hypot(xx, yy)
        rb = np.arange(0, 4, 0.02) + 0.01
        rp = np.array(
            [eff[(r >= ri - 0.01) & (r < ri + 0.01)].mean() for ri in rb]
        )
        i = np.where(rp <= eff.max() / 2)[0][0]
        f = (rp[i - 1] - eff.max() / 2) / (rp[i - 1] - rp[i])
        return 2.0 * (rb[i - 1] + f * 0.02)

    def _make_stars(self, n=16, cutout=13, seed=7):
        from photutils.psf import EPSFStars, EPSFStar

        rng = np.random.default_rng(seed)
        sub = 8
        gamma = self.FWHM_CONT / (
            2.0 * np.sqrt(2.0 ** (1.0 / self.BETA) - 1.0)
        )
        c = (cutout - 1) / 2.0
        yy, xx = (
            np.mgrid[0 : cutout * sub, 0 : cutout * sub].astype(float) / sub
            - 0.5 / sub
        )
        stars = []
        for _ in range(n):
            dx, dy = rng.uniform(-0.5, 0.5, 2)
            prof = (1.0 + (np.hypot(xx - c - dx, yy - c - dy) / gamma) ** 2) ** (
                -self.BETA
            )
            pix = prof.reshape(cutout, sub, cutout, sub).mean(axis=(1, 3))
            data = rng.uniform(3e4, 3e5) * pix + 200.0
            data = rng.poisson(data) + rng.normal(0, 5.0, data.shape)
            stars.append(EPSFStar(np.ascontiguousarray(data)))
        return EPSFStars(stars)

    @pytest.mark.parametrize(
        "n_stars,expected_osamp", [(16, 2), (64, 4)]
    )
    def test_build_recovers_effective_fwhm(self, n_stars, expected_osamp):
        pytest.importorskip("photutils")
        from photutils.psf import EPSFBuilder
        from photutils.centroids import centroid_2dg

        stars = self._make_stars(n=n_stars)
        truth = self._truth_fwhm()  # ~2.0 px effective
        # Pipeline choice: the selector picks the highest factor that
        # still leaves ~3 residual samples per grid cell (n=16 -> 2x,
        # n=64 -> 4x).
        osamp = _select_adaptive_oversample(len(stars), {})[0]
        assert osamp == expected_osamp
        kernel = get_smoothing_kernel(
            truth, oversample=osamp, size_scale_px=2.5
        )
        epsf = EPSFBuilder(
            oversampling=osamp,
            smoothing_kernel=kernel,
            recentering_func=centroid_2dg,
            recentering_boxsize=(5, 5),
            maxiters=10,
            fit_shape=5,
            progress_bar=False,
        ).build_epsf(stars).epsf
        fwhm_meas = measure_epsf_fwhm_native(epsf.data, osamp)
        assert np.isfinite(fwhm_meas)
        assert abs(fwhm_meas / truth - 1.0) < 0.08


class TestStarWeights:
    """Uniform weightmaps must still produce per-star weight *arrays* --
    photutils' ``_fit_star`` slices ``weights[large_slc]`` -- and the
    assignment must not wipe the contamination-mask zeros."""

    def _extract(self, weightmap):
        import logging
        import pandas as pd
        from astropy.nddata import NDData

        from autophot.photometry.psf import PSF

        rng = np.random.default_rng(3)
        img = rng.normal(100.0, 5.0, (60, 60))
        yy, xx = np.mgrid[0:60, 0:60]
        for cx, cy in [(20.4, 20.7), (40.6, 20.2), (30.5, 40.4), (45.1, 45.9)]:
            img += 2e4 * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / 1.4**2)
        # The norm-radius mask gate is exercised separately in
        # TestNormRadiusMaskGate; disable it here so the weightmap
        # mechanics are tested in isolation (synthetic noise triggers
        # keep-floor annulus masking that would drop the stars).
        p = PSF(
            {
                "photometry": {
                    "undersampled_fwhm_threshold": 2.5,
                    "psf_norm_radius_mask_frac_max": 1.0,
                }
            },
            img,
        )
        df = pd.DataFrame(
            {"x_pix": [20.4, 40.6, 30.5, 45.1], "y_pix": [20.7, 20.2, 40.4, 45.9]}
        )
        log = logging.getLogger("test")
        log.setLevel(logging.ERROR)
        stars, _ = p.robust_extract_stars(
            NDData(img), df, (15, 15), 2.0, 7, log, weightmap=weightmap
        )
        return stars

    def test_uniform_weightmap_gives_array_weights(self):
        pytest.importorskip("photutils")
        weightmap = np.full((60, 60), 0.02)
        stars = self._extract(weightmap)
        assert len(stars) == 4
        for st in stars:
            assert isinstance(st.weights, np.ndarray)
            assert st.weights.shape == st.data.shape
            # Masked pixels must stay zero-weight after assignment.
            assert (st.weights[np.asarray(st.mask, bool)] == 0.0).all()
            assert (st.weights[~np.asarray(st.mask, bool)] > 0.0).all()

class TestNormRadiusMaskGate:
    """Stars with masked/non-finite pixels inside the flux-normalisation
    circle (radius = aperture_radius) are rejected: the ePSF build
    normalises each star within that circle, so masked pixels there
    silently shrink the normalised area and bias the flux scale."""

    def _extract(self, mask=None, phot_extra=None):
        import logging
        import pandas as pd
        from astropy.nddata import NDData

        from autophot.photometry.psf import PSF

        rng = np.random.default_rng(3)
        img = rng.normal(100.0, 5.0, (60, 60))
        yy, xx = np.mgrid[0:60, 0:60]
        for cx, cy in [(20.4, 20.7), (40.6, 20.2), (30.5, 40.4), (45.1, 45.9)]:
            img += 2e4 * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / 1.4**2)
        # Contamination check off: its keep-floor masking on synthetic
        # noise would add masked annulus pixels inside the r=3 circle and
        # confound the gate under test.
        phot = {
            "undersampled_fwhm_threshold": 2.5,
            "aperture_radius": 3.0,
            "psf_cutout_contamination_check": False,
        }
        phot.update(phot_extra or {})
        p = PSF({"photometry": phot}, img)
        df = pd.DataFrame(
            {"x_pix": [20.4, 40.6, 30.5, 45.1], "y_pix": [20.7, 20.2, 40.4, 45.9]}
        )
        log = logging.getLogger("test")
        log.setLevel(logging.ERROR)
        return p.robust_extract_stars(
            NDData(img, mask=mask), df, (15, 15), 2.0, 7, log
        )[0]

    def test_masked_pixel_inside_norm_radius_rejected(self):
        pytest.importorskip("photutils")
        mask = np.zeros((60, 60), dtype=bool)
        # star 2 is at (x=40.6, y=20.2): (y=20, x=42) is ~1.5 px from the
        # centre, inside the r=3 normalisation circle.
        mask[20, 42] = True
        # star 4 at (x=45.1, y=45.9): (y=45, x=52) is ~7 px out - inside
        # the cutout but outside the normalisation circle.
        mask[45, 52] = True
        # Rescue disabled: this test exercises the strict gate itself;
        # the keep-floor rescue is covered separately below.
        stars = self._extract(
            mask=mask,
            phot_extra={"psf_norm_radius_mask_frac_rescue_max": 0.0},
        )
        assert len(stars) == 3

    def test_masked_pixel_outside_norm_radius_kept(self):
        pytest.importorskip("photutils")
        mask = np.zeros((60, 60), dtype=bool)
        mask[45, 52] = True  # ~7 px from star 4 - outside r=3
        stars = self._extract(mask=mask)
        assert len(stars) == 4

    def test_tolerance_allows_isolated_bad_pixel(self):
        pytest.importorskip("photutils")
        mask = np.zeros((60, 60), dtype=bool)
        mask[20, 42] = True  # 1 of ~29 pixels in the r=3 circle (~3%)
        stars = self._extract(
            mask=mask,
            phot_extra={"psf_norm_radius_mask_frac_max": 0.05},
        )
        assert len(stars) == 4

    def test_keep_floor_rescue_keeps_least_bad(self):
        """Sparse field: strict rejection of every cutout masked inside the
        normalisation circle would starve the build (fewer than
        psf_min_candidates survive).  The rescue keeps the least-bad
        cutouts under the rescue cap; heavily masked ones stay rejected."""
        pytest.importorskip("photutils")
        mask = np.zeros((60, 60), dtype=bool)
        # star 2 (x=40.6, y=20.2): 2 px inside the r=3 circle (~7%) -
        # under the 25% rescue cap.
        mask[20, 42] = True
        mask[19, 41] = True
        # star 4 (x=45.1, y=45.9): dense block inside the circle - over
        # the rescue cap.
        mask[43:50, 42:49] = True
        stars = self._extract(mask=mask)
        assert len(stars) == 3

    def test_rescue_cap_zero_disables_rescue(self):
        pytest.importorskip("photutils")
        mask = np.zeros((60, 60), dtype=bool)
        mask[20, 42] = True  # inside r=3 of star 2
        stars = self._extract(
            mask=mask,
            phot_extra={"psf_norm_radius_mask_frac_rescue_max": 0.0},
        )
        assert len(stars) == 3


class TestCoreEllipticity:
    """The second-moment core-ellipticity test must not fire on
    noise-dominated cores: signed background-subtracted pixels acted as
    negative moment weights, driving the moment sum through zero and
    producing ellipticity ~1e10 flags on junk cutouts.  Weights are now
    clipped non-negative and the test is gated on core peak SNR."""

    def _check(self, cutouts, phot_extra=None):
        import logging

        from astropy.table import Table
        from photutils.psf import EPSFStar, EPSFStars

        from autophot.photometry.psf import PSF

        # Disable every other flag so only the ellipticity test can fire.
        phot = {
            "psf_cutout_contamination_check": True,
            "psf_contam_asymmetry_frac": 1e9,
            "psf_contam_edge_excess_sigma": 1e9,
            "psf_contam_radial_reversal_frac": 1e9,
            "psf_contam_border_excess_sigma": 1e9,
            "psf_contam_secondary_peak_sigma": 1e9,
            "psf_contam_streak_sigma": 1e9,
            "psf_contam_annulus_peak_sigma": 1e9,
            "psf_contam_flux_excess_sigma": 1e9,
            "psf_contam_border_max_sigma": 1e9,
            "psf_contam_core_ellipticity_max": 0.15,
            "psf_contam_core_ellipticity_snr_min": 8.0,
            "psf_build_min_keep_frac_after_cut": 0.0,
            "psf_min_candidates": 4,
        }
        phot.update(phot_extra or {})
        p = PSF({"photometry": phot}, np.zeros((2, 2)))
        stars = EPSFStars(
            [EPSFStar(np.ascontiguousarray(c)) for c in cutouts]
        )
        tbl = Table({"x": np.zeros(len(cutouts)), "y": np.zeros(len(cutouts))})
        log = logging.getLogger("test")
        log.setLevel(logging.ERROR)
        out, _ = p._check_cutout_contamination(
            stars, tbl, cutouts[0].shape, 3.0, log
        )
        return out

    def test_noise_core_never_flagged(self):
        pytest.importorskip("photutils")
        rng = np.random.default_rng(11)
        cutouts = [rng.normal(100.0, 5.0, (25, 25)) for _ in range(5)]
        assert len(self._check(cutouts)) == 5

    def test_elongated_core_flagged(self):
        pytest.importorskip("photutils")
        rng = np.random.default_rng(5)
        yy, xx = np.mgrid[0:25, 0:25]
        c = 12.0
        round_star = 100.0 + 5e3 * np.exp(
            -((xx - c) ** 2 + (yy - c) ** 2) / (2 * 2.0**2)
        )
        # sigma_x=4, sigma_y=1 -> ellipticity ~0.8 > 0.15
        blend = 100.0 + 5e3 * np.exp(
            -((xx - c) ** 2 / (2 * 4.0**2) + (yy - c) ** 2 / (2 * 1.0))
        )
        cutouts = [round_star + rng.normal(0, 2, (25, 25)) for _ in range(4)]
        cutouts.append(blend + rng.normal(0, 2, (25, 25)))
        assert len(self._check(cutouts)) == 4


class TestHotPixelCluster:
    """A hot-pixel cluster concentrates nearly all core flux into 1-2
    pixels (peak fraction -> 1); a genuine PSF spreads the same flux
    across the seeing disk.  The annulus tests and the core-ellipticity
    test cannot see a symmetric central blob, so the dedicated
    peak-fraction test rejects it."""

    FWHM = 3.0
    CUT = 25

    def _cutouts(self, seed=7):
        rng = np.random.default_rng(seed)
        c = (self.CUT - 1) / 2.0
        yy, xx = np.mgrid[0 : self.CUT, 0 : self.CUT]
        sig = self.FWHM / 2.354820045
        prof = np.exp(
            -((xx - c) ** 2 + (yy - c) ** 2) / (2.0 * sig**2)
        )
        out = []
        for _ in range(6):
            out.append(
                100.0 + 5e3 * prof + rng.normal(0.0, 0.5, prof.shape)
            )
        # Hot-pixel cluster: a few very bright pixels on sky, no PSF
        # wings.  Peak fraction within r<=1.5 px is ~0.8 vs ~0.2 for the
        # Gaussian stars.
        hot = 100.0 + rng.normal(0.0, 0.5, prof.shape)
        hot[int(c), int(c)] += 5e4
        hot[int(c), int(c) + 1] += 1e4
        out.append(hot)
        return out

    def _check(self, cutouts, phot_extra=None):
        import logging

        from astropy.table import Table
        from photutils.psf import EPSFStar, EPSFStars

        from autophot.photometry.psf import PSF

        # Disable every other gate so only the hot-pixel test can fire.
        phot = {
            "psf_cutout_contamination_check": True,
            "psf_contam_asymmetry_sigma": 1e9,
            "psf_contam_edge_excess_sigma": 1e9,
            "psf_contam_radial_reversal_sigma": 1e9,
            "psf_contam_border_excess_sigma": 1e9,
            "psf_contam_border_max_sigma": 1e9,
            "psf_contam_secondary_peak_sigma": 1e9,
            "psf_contam_streak_sigma": 1e9,
            "psf_contam_annulus_peak_sigma": 1e9,
            "psf_contam_flux_excess_sigma": 1e9,
            "psf_contam_core_ellipticity_max": 0.0,
            "psf_contam_hotpix_sigma": 5.0,
            "psf_contam_hotpix_max_peak_frac": 0.65,
            # A hot-pixel cluster is also a spike on the growth curve;
            # keep the CoG test off so these tests isolate the hotpix gate.
            "psf_contam_cog_check": False,
            "psf_build_min_keep_frac_after_cut": 0.0,
            "psf_min_candidates": 2,
        }
        phot.update(phot_extra or {})
        p = PSF({"photometry": phot}, np.zeros((2, 2)))
        stars = EPSFStars(
            [EPSFStar(np.ascontiguousarray(c_)) for c_ in cutouts]
        )
        for i, st in enumerate(stars):
            st.id_label = i
        tbl = Table(
            {"x": np.full(len(cutouts), 12.0), "y": np.full(len(cutouts), 12.0)}
        )
        log = logging.getLogger("test")
        log.setLevel(logging.ERROR)
        out, _ = p._check_cutout_contamination(
            stars, tbl, cutouts[0].shape, self.FWHM, log
        )
        return [int(getattr(s, "id_label", -1)) for s in out]

    def test_hot_pixel_cluster_rejected(self):
        pytest.importorskip("photutils")
        kept = self._check(self._cutouts())
        assert set(kept) == {0, 1, 2, 3, 4, 5}

    def test_disabled_by_sigma_gate(self):
        pytest.importorskip("photutils")
        kept = self._check(
            self._cutouts(), {"psf_contam_hotpix_sigma": 0.0}
        )
        assert set(kept) == {0, 1, 2, 3, 4, 5, 6}

    def test_clean_psf_cores_not_flagged(self):
        pytest.importorskip("photutils")
        # A pool of only clean stars keeps every member.
        kept = self._check(self._cutouts()[:6])
        assert len(kept) == 6


class TestCurveOfGrowth:
    """Enclosed flux normalised by the plateau removes the brightness
    scale, so genuine stars share one growth curve while defect spikes
    saturate early and diffuse/wing-contaminated sources rise late.  A
    cutout is rejected when its curve is an outlier against the
    Moffat-consistent ensemble or outside the plausible Moffat EE
    envelope entirely - the second path still works when the pool is
    dominated by non-stellar cutouts."""

    FWHM = 3.4
    CUT = 41

    def _sig(self):
        return self.FWHM / 2.354820045

    def _stars(self, rng, n=6):
        c = (self.CUT - 1) / 2.0
        yy, xx = np.mgrid[0 : self.CUT, 0 : self.CUT]
        sig = self._sig()
        out = []
        for _ in range(n):
            amp = rng.uniform(500.0, 5000.0)
            out.append(
                amp * np.exp(
                    -((xx - c) ** 2 + (yy - c) ** 2) / (2.0 * sig**2)
                )
                + rng.normal(0.0, 3.0, (self.CUT, self.CUT))
            )
        return out, yy, xx, c

    def _check(self, cutouts, phot_extra=None):
        import logging

        from astropy.table import Table
        from photutils.psf import EPSFStar, EPSFStars

        from autophot.photometry.psf import PSF

        # Disable every other gate so only the CoG test can fire.
        phot = {
            "psf_cutout_contamination_check": True,
            "psf_contam_asymmetry_sigma": 1e9,
            "psf_contam_edge_excess_sigma": 1e9,
            "psf_contam_radial_reversal_sigma": 1e9,
            "psf_contam_border_excess_sigma": 1e9,
            "psf_contam_border_max_sigma": 1e9,
            "psf_contam_secondary_peak_sigma": 1e9,
            "psf_contam_streak_sigma": 1e9,
            "psf_contam_annulus_peak_sigma": 1e9,
            "psf_contam_flux_excess_sigma": 1e9,
            "psf_contam_core_ellipticity_max": 0.0,
            "psf_contam_hotpix_sigma": 0.0,
            "psf_build_min_keep_frac_after_cut": 0.0,
            "psf_min_candidates": 2,
        }
        phot.update(phot_extra or {})
        p = PSF({"photometry": phot}, np.zeros((2, 2)))
        stars = EPSFStars(
            [EPSFStar(np.ascontiguousarray(c_)) for c_ in cutouts]
        )
        for i, st in enumerate(stars):
            st.id_label = i
        tbl = Table(
            {"x": np.full(len(cutouts), 20.0), "y": np.full(len(cutouts), 20.0)}
        )
        log = logging.getLogger("test")
        log.setLevel(logging.ERROR)
        out, _ = p._check_cutout_contamination(
            stars, tbl, cutouts[0].shape, self.FWHM, log
        )
        return [int(getattr(s, "id_label", -1)) for s in out]

    def test_spike_and_diffuse_rejected(self):
        pytest.importorskip("photutils")
        rng = np.random.default_rng(7)
        stars, yy, xx, c = self._stars(rng)
        sig = self._sig()
        spike = 2000.0 * np.exp(
            -((xx - c) ** 2 + (yy - c) ** 2) / (2.0 * 0.6**2)
        ) + rng.normal(0.0, 3.0, (self.CUT, self.CUT))
        diffuse = 300.0 * np.exp(
            -((xx - c) ** 2 + (yy - c) ** 2) / (2.0 * 64.0)
        ) + rng.normal(0.0, 3.0, (self.CUT, self.CUT))
        kept = self._check(stars + [spike, diffuse])
        assert set(kept) == set(range(6))

    def test_junk_dominated_pool_keeps_stars(self):
        """When most cutouts are spikes the ensemble median converges on
        the junk curve; the Moffat envelope must still flag the spikes
        and never the real stars."""
        pytest.importorskip("photutils")
        rng = np.random.default_rng(3)
        stars, yy, xx, c = self._stars(rng, n=3)
        spikes = [
            2000.0 * np.exp(
                -((xx - c) ** 2 + (yy - c) ** 2) / (2.0 * s**2)
            )
            + rng.normal(0.0, 3.0, (self.CUT, self.CUT))
            for s in (0.4, 0.5, 0.6, 0.7, 0.8)
        ]
        kept = self._check(stars + spikes)
        assert set(kept) == {0, 1, 2}

    def test_disabled_by_config(self):
        pytest.importorskip("photutils")
        rng = np.random.default_rng(7)
        stars, yy, xx, c = self._stars(rng)
        spike = 2000.0 * np.exp(
            -((xx - c) ** 2 + (yy - c) ** 2) / (2.0 * 0.6**2)
        ) + rng.normal(0.0, 3.0, (self.CUT, self.CUT))
        kept = self._check(
            stars + [spike], {"psf_contam_cog_check": False}
        )
        assert len(kept) == 7

    def test_low_snr_gated_not_flagged(self):
        """Below psf_contam_cog_snr_min the plateau estimate is
        noise-dominated; the test must not fire on faint stars."""
        pytest.importorskip("photutils")
        c = (self.CUT - 1) / 2.0
        yy, xx = np.mgrid[0 : self.CUT, 0 : self.CUT]
        sig = self._sig()
        rng = np.random.default_rng(11)
        faint = [
            30.0 * np.exp(
                -((xx - c) ** 2 + (yy - c) ** 2) / (2.0 * sig**2)
            )
            + rng.normal(0.0, 3.0, (self.CUT, self.CUT))
            for _ in range(5)
        ]
        kept = self._check(faint)
        assert len(kept) == 5


class TestAnalyticModelNormalization:
    """photutils ePSF convention: model data sums to prod(oversampling) so
    that ``evaluate(flux=F)`` yields a source of total flux F on the native
    grid.  A unit-sum oversampled stamp silently inflates fitted fluxes by
    ~osamp^2 (~3 mag at osamp=4) - the analytic Moffat fallback/injection
    models must follow the same convention as a built ePSF."""

    def _fit_flux(self, model, star_data, fit_shape=5):
        import warnings as _warnings

        from photutils.psf import EPSFFitter, EPSFStar, EPSFStars

        star = EPSFStar(np.ascontiguousarray(star_data))
        # EPSFFitter is deprecated in photutils >= 3.x but still functional;
        # the test only needs a fitted flux, not the builder pipeline.
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            fitted = EPSFFitter(fit_boxsize=fit_shape)(
                model, EPSFStars([star])
            )[0][0]
        return float(fitted.flux)

    def _moffat_star(self, flux, fwhm=2.0, size=15, dx=0.0, dy=0.0):
        beta = 4.765
        gamma = fwhm / (2.0 * np.sqrt(2.0 ** (1.0 / beta) - 1.0))
        c = (size - 1) / 2.0
        # Pixel-integrated so the star is an effective PSF like real data.
        sub = 8
        gy = np.linspace(0, size - 1, size * sub)
        gx = np.linspace(0, size - 1, size * sub)
        gxx, gyy = np.meshgrid(gx, gy)
        prof = (1.0 + ((gxx - c - dx) ** 2 + (gyy - c - dy) ** 2) / gamma**2) ** (
            -beta
        )
        pix = prof.reshape(size, sub, size, sub).mean(axis=(1, 3))
        return flux * pix / pix.sum()

    def test_injection_model_sum_matches_convention(self):
        pytest.importorskip("photutils")
        from autophot.analysis.limits import _analytic_psf_for_injection

        for osamp in (1, 2, 4):
            m = _analytic_psf_for_injection(2.0, oversampling=osamp)
            assert np.isclose(
                m.data.sum(), float(osamp) ** 2, rtol=1e-6
            )

    def test_fitted_flux_recovers_source_flux(self):
        pytest.importorskip("photutils")
        from autophot.analysis.limits import _analytic_psf_for_injection

        f_true = 5e4
        star = self._moffat_star(f_true, fwhm=2.0)
        model = _analytic_psf_for_injection(2.0, oversampling=4)
        f_fit = self._fit_flux(model, star)
        # Same Moffat shape on both sides; recovery should be tight.
        assert abs(f_fit / f_true - 1.0) < 0.15

    def test_injection_render_scales_like_empirical(self):
        pytest.importorskip("photutils")
        from autophot.analysis.limits import _analytic_psf_for_injection, _render_epsf_on_cutout

        model = _analytic_psf_for_injection(2.0, oversampling=4)
        stamp = _render_epsf_on_cutout(model, 21, 21, 10.0, 10.0, 1e4, 4)
        total = float(np.nansum(stamp))
        # ePSF data sums to osamp^2 by the photutils convention, but the
        # render divides that back out so the stamp carries true detector
        # flux -- matching the osamp<=1 evaluate() path and the model's
        # own flux parameter.
        assert np.isclose(total / 1e4, 1.0, rtol=0.05)
        # The same model rendered natively (osamp=1) must land on the same
        # detector flux scale.
        m1 = _analytic_psf_for_injection(2.0, oversampling=1)
        native = _render_epsf_on_cutout(m1, 21, 21, 10.0, 10.0, 1e4, 1)
        assert np.isclose(float(np.nansum(native)), total, rtol=0.1)

class TestCleanEpsfStamp:
    """Post-build stamp cleanup: wing speckle suppression at osamp>=2 and
    edge taper, while the core, flux sum, and coarse ringing are preserved."""

    def _stamp(self, osamp=4, fwhm=2.0, noise=0.02, edge_artifact=True, seed=0):
        rng = np.random.default_rng(seed)
        beta = 4.765
        gamma = fwhm / (2.0 * np.sqrt(2.0 ** (1.0 / beta) - 1.0))
        n_native = 27
        size = osamp * n_native
        c = (size - 1) / 2.0
        yy, xx = np.mgrid[0:size, 0:size]
        d = (1.0 + ((xx - c) ** 2 + (yy - c) ** 2) / gamma**2) ** (-beta)
        d *= float(osamp) ** 2 / d.sum()
        # gridpoint speckle in the wings
        r = np.hypot(xx - c, yy - c) / osamp
        d = d + rng.normal(0, noise, d.shape) * (r > 3.0)
        if edge_artifact:
            # Kept faint so the leaked flux stays a small fraction of the
            # stamp total, matching real edge artifacts (~1% of peak).
            d[:, -2:] += 0.01 * np.nanmax(d)
        return d, r

    def test_sum_and_core_preserved(self):
        d, r = self._stamp(osamp=4, edge_artifact=False)
        c = clean_epsf_stamp(d, 4, 2.0)
        assert np.isclose(c.sum(), d.sum(), rtol=1e-9)
        # Core shape is preserved: the blend mask is 1 inside
        # wing_start_fwhm*FWHM, so core pixels only share the global
        # renorm scale factor, which cancels in peak-normalised shape.
        core = r < 1.5 * 2.0
        assert np.allclose(c[core] / c.max(), d[core] / d.max(), atol=1e-6)

    def test_wing_speckle_reduced_at_osamp_ge2(self):
        d, r = self._stamp(osamp=4)
        c = clean_epsf_stamp(d, 4, 2.0)
        mad = lambda a: 1.4826 * np.median(np.abs(a - np.median(a)))

        def speckle(a):
            # Detrend the radial profile so the metric isolates
            # pixel-to-pixel gridpoint noise, not the blend ramp.
            rb = np.floor(r).astype(int)
            out = np.zeros_like(a)
            for k in np.unique(rb):
                m = rb == k
                out[m] = a[m] - np.median(a[m])
            return out

        wing = (r > 7.0) & (r < 12.0)
        assert mad(speckle(c)[wing]) < 0.4 * mad(speckle(d)[wing])

    def test_edge_artifact_removed(self):
        d, r = self._stamp(osamp=4)
        c = clean_epsf_stamp(d, 4, 2.0, edge_px=1.5)
        # Outermost ~1 native px should be (near) zero
        assert np.abs(c[:, -4:]).max() < 0.02 * np.nanmax(d)

    def test_corner_blob_clipped(self):
        # A leaked neighbour blob is azimuthally localised; the wing
        # reference is azimuthally coherent.  A ~10%-of-peak blob at the
        # stamp corner should be removed while coherent wing structure
        # (built into the reference) survives.
        d, r = self._stamp(osamp=4, edge_artifact=False)
        blob = np.zeros_like(d)
        yy, xx = np.indices(d.shape)
        blob[(yy - 12) ** 2 + (xx - 14) ** 2 < 36] = 0.10 * d.max()
        d2 = d + blob
        c = clean_epsf_stamp(d2, 4, 2.0)
        peak = d2.max()
        # Residual at the blob location vs azimuthal reference elsewhere
        zone = (yy - 12) ** 2 + (xx - 14) ** 2 < 36
        assert np.abs(c[zone]).max() < 0.03 * peak
        # Sum convention preserved
        assert np.isclose(c.sum(), d2.sum(), rtol=1e-9)

    def test_osamp1_wings_untouched(self):
        # At 1x, wing ringing is indistinguishable from signal -- only the
        # edge taper applies.
        d, r = self._stamp(osamp=1, edge_artifact=False)
        c = clean_epsf_stamp(d, 1, 2.0, edge_px=0.5)
        wing = (r > 4.0) & (r < 8.0) & (r < 11.0)
        interior = np.ones_like(d, bool)
        interior[:2] = interior[-2:] = interior[:, :2] = interior[:, -2:] = False
        m = wing & interior
        # Untouched pixels share only the global renorm factor: the
        # border row/col is zeroed, the rest scaled by d.sum/tapered.sum.
        tapered = d.copy()
        tapered[0] = tapered[-1] = tapered[:, 0] = tapered[:, -1] = 0.0
        scale = d.sum() / tapered.sum()
        assert np.allclose(c[m], d[m] * scale, rtol=0, atol=1e-12)


    def test_far_wing_flattened_to_radial_profile(self):
        # Beyond flatten_fwhm the wings should approach the azimuthal
        # median profile: azimuthal mottle is suppressed while the radial
        # falloff is preserved.
        d, r = self._stamp(osamp=4, noise=0.05, edge_artifact=False)
        c = clean_epsf_stamp(d, 4, 2.0)
        far = (r > 3.5 * 2.0 + 2.0) & (r < 10.0)
        # azimuthal mottle: MAD of deviations from the per-radius median
        def az_mad(a):
            out = []
            rb = np.floor(r[far] * 2) / 2
            v = a[far]
            for k in np.unique(rb):
                vals = v[rb == k]
                out.extend(vals - np.median(vals))
            return 1.4826 * np.median(np.abs(np.asarray(out)))
        assert az_mad(c) < 0.3 * az_mad(d)
        # radial profile tracks the true wing: at r>~9 px on a FWHM=2
        # stamp the true Moffat is ~1e-4, so the noisy input median is
        # itself noise -- the flattened output should sit closer to the
        # noise-free profile than the noisy input does.
        d0, _ = self._stamp(osamp=4, noise=0.0, edge_artifact=False)
        rb = np.floor(r[far] * 2) / 2
        for k in np.unique(rb):
            m = np.zeros_like(d, bool)
            m[far] = rb == k
            med_true = np.median(d0[m])
            med_out = np.median(c[m])
            assert abs(med_out - med_true) <= abs(
                np.median(d[m]) - med_true
            ) + 0.3 * abs(med_true) + 1e-3
        assert np.isclose(c.sum(), d.sum(), rtol=1e-9)

    def test_flatten_disabled(self):
        d, r = self._stamp(osamp=4, noise=0.05, edge_artifact=False)
        c = clean_epsf_stamp(d, 4, 2.0, flatten_fwhm=0.0)
        far = r > 9.0
        # with flattening off the far wing still carries residual mottle
        v = c[far]
        assert 1.4826 * np.median(np.abs(v - np.median(v))) > 0



class TestEpsfResidualMad:
    """The residual-accuracy gate must rank the model that truly
    represents the stars above a wrong model."""

    FWHM_CONT = 2.0
    BETA = 4.765

    def _stars(self, n=10, cutout=15, seed=11):
        from photutils.psf import EPSFStars, EPSFStar

        rng = np.random.default_rng(seed)
        sub = 8
        gamma = self.FWHM_CONT / (
            2.0 * np.sqrt(2.0 ** (1.0 / self.BETA) - 1.0)
        )
        c = (cutout - 1) / 2.0
        yy, xx = (
            np.mgrid[0 : cutout * sub, 0 : cutout * sub].astype(float) / sub
            - 0.5 / sub
        )
        stars = []
        for _ in range(n):
            dx, dy = rng.uniform(-0.5, 0.5, 2)
            prof = (
                1.0 + (np.hypot(xx - c - dx, yy - c - dy) / gamma) ** 2
            ) ** (-self.BETA)
            pix = prof.reshape(cutout, sub, cutout, sub).mean(axis=(1, 3))
            data = rng.uniform(3e4, 3e5) * pix + 200.0
            data = rng.poisson(data) + rng.normal(0, 5.0, data.shape)
            stars.append(EPSFStar(np.ascontiguousarray(data)))
        return EPSFStars(stars)

    def _moffat(self, fwhm, osamp, cutout=15):
        from astropy.convolution import Moffat2DKernel
        from photutils.psf import ImagePSF

        gamma = (osamp * fwhm) / (
            2.0 * np.sqrt(2.0 ** (1.0 / self.BETA) - 1.0)
        )
        kern = Moffat2DKernel(
            gamma=gamma, alpha=self.BETA,
            x_size=osamp * cutout, y_size=osamp * cutout,
        )
        data = pixel_integrate_oversampled(kern.array, osamp) * osamp**2
        return ImagePSF(
            data=data,
            x_0=(osamp * cutout - 1) / 2.0,
            y_0=(osamp * cutout - 1) / 2.0,
            oversampling=osamp,
        )

    def test_true_model_beats_speckled_model(self):
        # The gate exists to catch starved-build pathology: a stamp at the
        # correct FWHM whose grid cells are noise-drawn (speckled wings).
        # A few-% speckle on the stamp lands in every star residual.
        pytest.importorskip("photutils")
        from autophot.photometry.psf import _epsf_residual_mad

        stars = self._stars()
        good = self._moffat(self.FWHM_CONT * 1.05, 1)
        rng = np.random.default_rng(3)
        bad_data = np.asarray(good.data, float).copy()
        bad_data += rng.normal(0, 0.05 * bad_data.max(), bad_data.shape)
        from photutils.psf import ImagePSF

        bad = ImagePSF(
            data=bad_data,
            x_0=good.x_0,
            y_0=good.y_0,
            oversampling=good.oversampling,
        )
        m_good = _epsf_residual_mad(stars, good, 4.5)
        m_bad = _epsf_residual_mad(stars, bad, 4.5)
        assert np.isfinite(m_good) and np.isfinite(m_bad)
        assert m_bad > 1.5 * m_good

    def test_degenerate_inputs_give_nan(self):
        pytest.importorskip("photutils")
        from autophot.photometry.psf import _epsf_residual_mad

        assert not np.isfinite(_epsf_residual_mad(None, None, 4.5))

class TestAnalyticFallbackPolicy:
    """An analytic Moffat may substitute for a failed ePSF build ONLY on
    starved fields and ONLY as a flagged model: psf_model_kind records
    "analytic-moffat" so the substitution is never silent, and the
    fallback is opt-out via psf_analytic_fallback (default True).  A
    healthy-field run must still keep the empirical model (or revert to
    aperture-only).  The analytic composite also seeds the ePSF build via
    psf_use_analytic_init."""

    def test_analytic_fallback_is_opt_out(self):
        import yaml
        from pathlib import Path

        root = Path(__file__).resolve().parent.parent
        cfg = yaml.safe_load(
            (root / "autophot" / "data" / "default_input.yml").read_text()
        )
        phot = cfg["default_input"]["photometry"]
        # Shipped default must enable the sparse-field fallback; setting it
        # False restores the strict empirical-or-aperture-only policy.
        assert phot.get("psf_analytic_fallback") is True
        assert phot.get("psf_build_min_stars", 2) >= 1

    def test_psf_model_kind_defaults_none(self):
        pytest.importorskip("photutils")
        from autophot.photometry.psf import PSF

        p = PSF(input_yaml={}, image=np.zeros((5, 5)))
        assert p.psf_model_kind == "none"

    def test_analytic_moffat_psf_wellformed(self):
        pytest.importorskip("photutils")
        from autophot.photometry.psf import _analytic_moffat_psf, _epsf_usable, measure_epsf_fwhm_native

        m = _analytic_moffat_psf(fwhm=3.0, oversampling=2, cutout_n=25, beta=4.765)
        assert _epsf_usable(m)
        assert int(np.atleast_1d(m.oversampling)[0]) == 2
        # photutils flux convention: stamp sums to prod(oversampling).
        assert np.isclose(float(np.nansum(m.data)), 4.0, rtol=1e-3)
        fwhm_meas = measure_epsf_fwhm_native(np.asarray(m.data, float), 2)
        assert np.isfinite(fwhm_meas)
        # Effective (pixel-integrated) FWHM is at least the continuous one.
        assert 2.5 < fwhm_meas < 4.0

    def test_analytic_moffat_psf_guards_bad_inputs(self):
        pytest.importorskip("photutils")
        from autophot.photometry.psf import _analytic_moffat_psf

        # beta below the physical floor is clamped, not propagated.
        m = _analytic_moffat_psf(fwhm=3.0, oversampling=0, cutout_n=25, beta=0.5)
        assert int(np.atleast_1d(m.oversampling)[0]) == 1
        assert np.isfinite(np.asarray(m.data, float)).all()

    def test_flat_noise_plate_is_not_usable(self):
        """A non-converged ePSF is a flat noise plate: finite data, positive
        range and sum, but no half-max crossing so no measurable FWHM.  It
        must fail _epsf_usable so the build reports no model (aperture-only
        downstream) instead of keeping a degenerate model (SLT z-band
        regression)."""
        pytest.importorskip("photutils")
        from photutils.psf import ImagePSF
        from autophot.photometry.psf import _epsf_usable

        rng = np.random.default_rng(0)
        n = 119
        # Uniform floor ~1/n^2 with small noise spikes: looks "usable" to
        # the finite/range checks but has no stellar core.
        data = np.full((n, n), 1.0 / (n * n)) + rng.normal(
            0.0, 3e-5, (n, n)
        )
        data = np.abs(data)
        m = ImagePSF(
            data=data,
            x_0=(n - 1) / 2.0,
            y_0=(n - 1) / 2.0,
            oversampling=1,
        )
        assert np.isfinite(data).all()
        assert np.ptp(data) > 0 and np.nansum(data) > 0
        assert not _epsf_usable(m)

    def test_offcenter_spike_model_is_not_usable(self):
        """A model whose only peak is a noise spike far from the stamp
        centre has no measurable core either."""
        pytest.importorskip("photutils")
        from photutils.psf import ImagePSF
        from autophot.photometry.psf import _epsf_usable

        n = 119
        data = np.full((n, n), 1.0 / (n * n))
        data[5, 5] = 1e-3  # corner spike: never anchors a core at centre
        m = ImagePSF(
            data=data,
            x_0=(n - 1) / 2.0,
            y_0=(n - 1) / 2.0,
            oversampling=1,
        )
        assert not _epsf_usable(m)


def _moffat_profile(r, comps):
    # Components are (amp, gamma, beta) or (amp, gamma, beta, ox, oy);
    # an offset component's radial profile about the origin is
    # evaluated along the offset direction.
    out = np.zeros_like(r, dtype=float)
    for c in comps:
        a, g, b = c[0], c[1], c[2]
        if len(c) > 4:
            r_c = np.abs(r - np.hypot(c[3], c[4]))
        else:
            r_c = r
        out += a * (1.0 + (r_c / g) ** 2) ** (-b)
    return out


def _make_epsf_star(data, flux, center=(20.0, 20.0)):
    pytest.importorskip("photutils")
    from photutils.psf.epsf_stars import EPSFStar

    star = EPSFStar(np.asarray(data, float))
    star.flux = flux
    star.cutout_center = np.asarray(center, float)
    return star


class TestMoffatComposite:
    """The analytic PSF is a sum of Moffat components: a single Moffat
    cannot represent a real PSF's narrow core plus extended wings."""

    TRUE_COMPS = (
        (0.05, 2.5, 4.0),
        (0.008, 6.0, 2.2),
        (0.0012, 14.0, 1.6),
    )

    def _synthetic_stars(self, n=6, noise=0.0004, seed=0):
        rng = np.random.default_rng(seed)
        yy, xx = np.indices((41, 41))
        r = np.hypot(xx - 20.0, yy - 20.0)
        prof = _moffat_profile(r, self.TRUE_COMPS)
        stars = []
        for i in range(n):
            flux = 4000.0 * (1.0 + 0.4 * i)
            data = prof * flux + rng.normal(0.0, noise * flux, r.shape)
            stars.append(_make_epsf_star(data, flux))
        return stars

    def test_shipped_default_component_count(self):
        import yaml
        from pathlib import Path

        root = Path(__file__).resolve().parent.parent
        cfg = yaml.safe_load(
            (root / "autophot" / "data" / "default_input.yml").read_text()
        )
        n = cfg["default_input"]["photometry"]["psf_analytic_components"]
        assert 1 <= int(n) <= 4

    def test_default_composite_wellformed(self):
        from autophot.photometry.psf import _default_moffat_composite

        comps = _default_moffat_composite(8.0, 4.765)
        assert len(comps) == 3
        for amp, gamma, beta in comps:
            assert np.isfinite(amp) and amp > 0
            assert np.isfinite(gamma) and gamma > 0
            assert np.isfinite(beta) and beta > 1.0
        # Core must dominate the peak.
        assert comps[0][0] > comps[1][0] > comps[2][0]

    def test_composite_stamp_flux_convention(self):
        pytest.importorskip("photutils")
        from autophot.photometry.psf import _composite_moffat_psf, _default_moffat_composite, _epsf_usable

        comps = _default_moffat_composite(8.0, 4.765)
        for osamp, expect in ((1, 1.0), (2, 4.0), (4, 16.0)):
            m = _composite_moffat_psf(comps, osamp, 51)
            assert _epsf_usable(m)
            assert np.isclose(float(np.nansum(m.data)), expect, rtol=1e-3)
            assert np.isfinite(np.asarray(m.data, float)).all()
        assert m._autophot_kind == "analytic-composite"

    def test_composite_stamp_fwhm_native_units(self):
        # Component gammas are native px: the stamp FWHM must track the
        # requested core FWHM regardless of oversampling.
        pytest.importorskip("photutils")
        from autophot.photometry.psf import (
            _composite_moffat_psf,
            _default_moffat_composite,
            measure_epsf_fwhm_native,
        )

        comps = _default_moffat_composite(6.0, 4.765)
        for osamp in (1, 4):
            m = _composite_moffat_psf(comps, osamp, 51)
            f = measure_epsf_fwhm_native(np.asarray(m.data, float), osamp)
            assert 4.0 < f < 10.0

    def test_fit_recovers_profile(self):
        pytest.importorskip("photutils")
        from autophot.photometry.psf import _fit_moffat_composite

        fit = _fit_moffat_composite(self._synthetic_stars(), 7.0)
        assert fit is not None and 1 <= len(fit) <= 3
        yy, xx = np.indices((41, 41))
        r = np.hypot(xx - 20.0, yy - 20.0)
        truth = _moffat_profile(r, self.TRUE_COMPS)
        pred = _moffat_profile(r, fit)
        core = r < 8.0
        # Sum-of-Moffats parameterisations are degenerate; the fitted
        # profile -- not the component values -- is what must match.
        assert np.allclose(pred[core], truth[core], rtol=0.3)

    def test_fit_handles_junk_and_empty(self):
        pytest.importorskip("photutils")
        from autophot.photometry.psf import _fit_moffat_composite

        assert _fit_moffat_composite([], 7.0) is None
        # A star with only a few valid pixels (<9) is skipped by the
        # fitter; with nothing else to fit the result is None.  Six
        # valid pixels keep EPSFStar's interpolation-based flux
        # estimator alive while staying under the fit floor.
        bad = np.full((9, 9), np.nan)
        bad[3:6, 3:5] = 1.0
        stars = [_make_epsf_star(bad, 1000.0, center=(4.0, 4.0))]
        assert _fit_moffat_composite(stars, 7.0) is None

    def test_noise_clump_rejected_by_snr_gate(self):
        """A cutout that is pure sky noise must not enter the joint
        fit: its positive aperture sum would pass the flux floor, but
        it constrains nothing and drags the shared profile."""
        pytest.importorskip("photutils")
        from autophot.photometry.psf import _composite_star_profile, _fit_moffat_composite

        rng = np.random.default_rng(3)
        noise = rng.normal(0.0, 2.0, (61, 61))
        junk = _make_epsf_star(noise + 0.5, 800.0, center=(30.0, 30.0))
        assert _composite_star_profile(junk, 7.0, None, snr_min=5.0) is None
        stars = self._synthetic_stars(n=4) + [junk]
        fit = _fit_moffat_composite(stars, 7.0)
        assert fit is not None  # real stars still fit; junk is skipped

    def test_stamp_without_stars_uses_canonical(self):
        pytest.importorskip("photutils")
        from autophot.photometry.psf import _analytic_psf_stamp, _default_moffat_composite

        m, comps = _analytic_psf_stamp(8.0, 2, 51, 4.765, stars=None)
        assert len(comps) == 3
        canon = _default_moffat_composite(8.0, 4.765)
        assert comps == canon

    def test_stamp_with_stars_uses_fit(self):
        pytest.importorskip("photutils")
        from autophot.photometry.psf import _analytic_psf_stamp, _epsf_usable

        m, comps = _analytic_psf_stamp(
            7.0, 2, 51, 4.765, stars=self._synthetic_stars()
        )
        assert _epsf_usable(m)
        assert np.isclose(float(np.nansum(m.data)), 4.0, rtol=1e-3)


class TestMoffatJointFit:
    """The analytic composite is fitted jointly over all stars: shared
    component shapes, per-star position and amplitude only.  Component
    centre offsets give the model its asymmetry freedom."""

    @staticmethod
    def _stars(n=6, seed=9, wing_offset=(1.5, 0.3), centroid_err=0.7):
        from types import SimpleNamespace

        def moffat(r, a, g, b):
            return a * (1.0 + (r / g) ** 2) ** (-b)

        rng = np.random.default_rng(seed)
        stars, shifts = [], []
        for _ in range(n):
            yy, xx = np.indices((41, 41))
            dx = rng.uniform(-centroid_err, centroid_err)
            dy = rng.uniform(-centroid_err, centroid_err)
            shifts.append((dx, dy))
            # The star is really at 20+shift but the recorded centroid
            # sits at 20 - the joint fit must absorb the difference.
            r1 = np.hypot(xx - 20.0 - dx, yy - 20.0 - dy)
            r2 = np.hypot(
                xx - 20.0 - dx - wing_offset[0],
                yy - 20.0 - dy - wing_offset[1],
            )
            prof = moffat(r1, 0.06, 2.2, 4.5) + moffat(r2, 0.006, 7.0, 1.9)
            data = 9000.0 * prof + 50.0 + rng.normal(0, 0.6, prof.shape)
            stars.append(
                SimpleNamespace(
                    data=data,
                    cutout_center=np.array([20.0, 20.0]),
                    mask=None,
                    weights=np.ones_like(data),
                    flux=9000.0,
                )
            )
        return stars, shifts

    def test_per_star_positions_recovered(self):
        pytest.importorskip("photutils")
        from autophot.photometry.psf import _fit_moffat_composite_full

        stars, shifts = self._stars()
        comps, sfits = _fit_moffat_composite_full(stars, 5.0)
        assert comps is not None and len(sfits) == len(stars)
        # Gauge freedom: component offsets and per-star dx share a
        # translation, so compare effective positions dx_i + ox_core.
        oc = np.array(comps[0][3:5]) if len(comps[0]) > 4 else np.zeros(2)
        for (dx, dy, _a), (tx, ty) in zip(sfits, shifts):
            assert np.hypot(dx + oc[0] - tx, dy + oc[1] - ty) < 0.15

    def test_component_offsets_capture_asymmetry(self):
        pytest.importorskip("photutils")
        from autophot.photometry.psf import _fit_moffat_composite_full

        stars, _ = self._stars()
        comps, _ = _fit_moffat_composite_full(stars, 5.0)
        assert comps is not None
        # The displaced wing must show up as a centre-offset component.
        off = [
            np.hypot(c[3], c[4])
            for c in comps
            if len(c) > 4 and np.hypot(c[3], c[4]) > 0.3
        ]
        assert off, f"no offset component fitted: {comps}"

    def test_concentric_variant_keeps_3_tuples(self):
        pytest.importorskip("photutils")
        from autophot.photometry.psf import _fit_moffat_composite_full

        stars, _ = self._stars()
        comps, sfits = _fit_moffat_composite_full(
            stars, 5.0, component_offsets=False
        )
        assert comps is not None
        assert all(len(c) == 3 for c in comps)
        assert all(a > 0 for a, _g, _b in comps)
        assert sfits

    def test_stamp_valid_for_fitting(self):
        # The delivered object must be a usable photutils ImagePSF:
        # finite, positive-normalised, and carrying the composite
        # provenance the later PSF-fitting path consumes.
        pytest.importorskip("photutils")
        from photutils.psf import ImagePSF
        from autophot.photometry.psf import _composite_moffat_psf, _epsf_usable

        stars, _ = self._stars()
        from autophot.photometry.psf import _fit_moffat_composite_full

        comps, sfits = _fit_moffat_composite_full(stars, 5.0)
        stamp = _composite_moffat_psf(comps, 1, 41)
        assert isinstance(stamp, ImagePSF)
        assert _epsf_usable(stamp)
        assert np.isfinite(np.asarray(stamp.data, float)).all()
        assert np.isclose(float(np.nansum(stamp.data)), 1.0, rtol=1e-3)
        assert stamp._autophot_components == [
            tuple(float(v) for v in c) for c in comps
        ]

    def test_amplitudes_stay_positive(self):
        pytest.importorskip("photutils")
        from autophot.photometry.psf import _fit_moffat_composite_full

        stars, _ = self._stars()
        comps, sfits = _fit_moffat_composite_full(stars, 5.0)
        assert all(c[0] > 0 for c in comps)
        assert all(s[2] > 0 for s in sfits)


class TestSpatialGridConfig:
    """psf_spatial_grid: opt-in spatially-varying PSF (GriddedPSFModel).

    Parser/config tests only; the build-time gating and fallback
    behaviour is covered in tests/test_spatial_psf.py.
    """

    def test_shipped_default_is_off(self):
        import yaml
        from pathlib import Path

        root = Path(__file__).resolve().parent.parent
        cfg = yaml.safe_load(
            (root / "autophot" / "data" / "default_input.yml").read_text()
        )
        phot = cfg["default_input"]["photometry"]
        assert phot["psf_spatial_grid"] is False
        assert int(phot["psf_grid_min_stars_per_cell"]) >= 1
        assert phot["psf_spatial_fill_missing"] in ("nearest", "fail")

    def test_parse_disabled_values(self):
        from autophot.photometry.psf import _parse_spatial_grid

        for v in (False, None, "false", "off", "none", "0", "", 0, 1):
            assert _parse_spatial_grid(v) is None, v

    def test_parse_auto(self):
        from autophot.photometry.psf import _parse_spatial_grid

        assert _parse_spatial_grid(True) == "auto"
        assert _parse_spatial_grid("auto") == "auto"
        assert _parse_spatial_grid(" AUTO ") == "auto"

    def test_parse_int_grid(self):
        from autophot.photometry.psf import _parse_spatial_grid

        assert _parse_spatial_grid(2) == 2
        assert _parse_spatial_grid("3") == 3
        # Out-of-range ints clamp to the supported 2-3 grid sizes.
        assert _parse_spatial_grid(4) == 3
        assert _parse_spatial_grid(9) == 3

    def test_parse_garbage_disables(self):
        from autophot.photometry.psf import _parse_spatial_grid

        for v in ("bogus", "2x2", object()):
            assert _parse_spatial_grid(v) is None, v


class TestInverseSigmaWeights:
    """The weightmap handed to EPSFBuilder must be 1/sigma, not
    1/sigma**2: the astropy fit objective is weights*(model - data), so
    a variance map squares the inverse noise a second time and the
    bright core dominates while the wings go unconstrained."""

    STARS = [(20.4, 20.7), (40.6, 20.2), (30.5, 40.4), (45.1, 45.9)]

    def _image(self):
        rng = np.random.default_rng(3)
        img = rng.normal(100.0, 5.0, (60, 60))
        yy, xx = np.mgrid[0:60, 0:60]
        for cx, cy in self.STARS:
            img += 2e4 * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / 1.4**2)
        return img, xx, yy

    def _extract(self, nddata):
        import logging

        import pandas as pd

        from autophot.photometry.psf import PSF

        # Isolate the weight mechanics: disable the contamination gate
        # and the norm-radius mask gate, both exercised separately.
        p = PSF(
            {
                "photometry": {
                    "undersampled_fwhm_threshold": 2.5,
                    "psf_cutout_contamination_check": False,
                    "psf_norm_radius_mask_frac_max": 1.0,
                }
            },
            np.asarray(nddata.data),
        )
        df = pd.DataFrame(
            {
                "x_pix": [s[0] for s in self.STARS],
                "y_pix": [s[1] for s in self.STARS],
            }
        )
        log = logging.getLogger("test")
        log.setLevel(logging.ERROR)
        return p.robust_extract_stars(nddata, df, (15, 15), 2.0, 7, log)[0]

    def test_weights_are_inverse_sigma(self):
        pytest.importorskip("photutils")
        from astropy.nddata import NDData, StdDevUncertainty

        img, xx, yy = self._image()
        # A mild radial ramp keeps the map non-uniform so the per-star
        # cutout path (not the uniform fast path) is exercised.
        sigma = 8.0 + 0.05 * np.hypot(xx - 30.0, yy - 30.0)
        stars = self._extract(
            NDData(img, uncertainty=StdDevUncertainty(sigma))
        )
        assert len(stars) == 4
        for st in stars:
            ox, oy = st.origin
            ny, nx = st.data.shape
            expected = 1.0 / sigma[oy : oy + ny, ox : ox + nx]
            np.testing.assert_allclose(
                np.asarray(st.weights, float), expected, rtol=1e-6
            )

    def test_masked_and_nan_pixels_zero_weight(self):
        pytest.importorskip("photutils")
        from astropy.nddata import NDData, StdDevUncertainty

        img, xx, yy = self._image()
        mask = np.zeros((60, 60), dtype=bool)
        # ~5 px off star 2's centre: inside the 15x15 cutout, outside the
        # normalisation circle.
        mask[20, 46] = True
        img[45, 50] = np.nan  # star 4's cutout, also ~5 px off-centre
        # Snapshot before extraction: the pipeline zeroes non-finite
        # cutout pixels in place and photutils cutouts alias the source
        # image, so img has no NaN left by the time stars come back.
        bad_img = mask | ~np.isfinite(img)
        stars = self._extract(
            NDData(
                img,
                uncertainty=StdDevUncertainty(np.full((60, 60), 8.0)),
                mask=mask,
            )
        )
        assert len(stars) == 4
        for st in stars:
            ox, oy = st.origin
            ny, nx = st.data.shape
            bad = bad_img[oy : oy + ny, ox : ox + nx]
            w = np.asarray(st.weights, float)
            assert (w[bad] == 0.0).all()
            assert (w[~bad] > 0.0).all()


def _psf_build_yaml(tmp_path, phot_extra=None):
    """Minimal input_yaml that reaches the ePSF build in PSF.build()."""
    phot = {
        "aperture_radius": 5.0,
        "psf_cutout_contamination_check": False,
        "psf_fft_rejection": False,
        "psf_shape_outlier_rejection": False,
    }
    phot.update(phot_extra or {})
    return {
        "fpath": "test.fits",
        "write_dir": str(tmp_path),
        "fwhm": 4.0,
        "scale": 12.0,
        "gain": 1.0,
        "photometry": phot,
    }


def _star_field(asym_wing=True, seed=1):
    """Nine isolated stars on a 170x170 field.

    With ``asym_wing`` each PSF carries a one-sided exponential wing, so
    the isophotal COM the cutout validator measures sits measurably off
    the extraction centroid -- a ``cutout_center`` overwrite moves the
    centres this test asserts on.
    """
    import pandas as pd

    rng = np.random.default_rng(seed)
    img = rng.normal(100.0, 5.0, (170, 170))
    yy, xx = np.mgrid[0:170, 0:170]
    sig = 4.0 / 2.354820045
    pos = [(x, y) for x in (30.0, 80.0, 130.0) for y in (30.0, 80.0, 130.0)]
    for cx, cy in pos:
        core = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sig**2))
        wing = (
            np.where(xx > cx, np.exp(-(xx - cx) / 3.0), 0.0)
            if asym_wing
            else 0.0
        )
        img += 3e3 * (core + 0.35 * wing)
    df = pd.DataFrame(
        {
            "x_pix": [p[0] for p in pos],
            "y_pix": [p[1] for p in pos],
            "fwhm": np.full(9, 4.0),
            "flux_radius": np.full(9, 2.4),
            "peak_flux": np.full(9, 3e3),
            "flux_AP": np.full(9, 6e4),
            "a": np.full(9, 1.7),
            "b": np.full(9, 1.7),
            "ELONGATION": np.full(9, 1.2),
            "flags": np.zeros(9),
            "SNR": np.full(9, 40.0),
        }
    )
    return img, df


def _run_build_with_spies(tmp_path, monkeypatch, phot_extra=None,
                          asym_wing=True):
    """Run PSF.build() while capturing the star table and the exact
    (stars, kwargs) handed to every EPSFBuilder.build_epsf call."""
    import functools

    import autophot.photometry.psf as psf_mod
    from photutils.psf import EPSFBuilder
    from autophot.photometry.psf import PSF

    captured = {"tbl": None, "calls": []}
    orig_extract = psf_mod.extract_stars

    @functools.wraps(orig_extract)
    def extract_spy(data, catalog, **kwargs):
        if captured["tbl"] is None and "x" in catalog.colnames:
            captured["tbl"] = catalog.copy()
        return orig_extract(data, catalog, **kwargs)

    monkeypatch.setattr(psf_mod, "extract_stars", extract_spy)

    orig_build = EPSFBuilder.build_epsf

    @functools.wraps(orig_build)
    def build_spy(self, stars, **kwargs):
        flat = getattr(stars, "all_stars", stars)
        # Snapshot at entry: the fit mutates cutout_center on the same
        # star objects, so recording the objects themselves would read
        # the post-fit centres.
        captured["calls"].append(
            (
                [
                    (
                        int(getattr(s, "id_label", -1)),
                        tuple(np.asarray(s.cutout_center, float)),
                        tuple(np.asarray(s.origin, float)),
                    )
                    for s in flat
                ],
                dict(kwargs),
            )
        )
        return orig_build(self, stars, **kwargs)

    monkeypatch.setattr(EPSFBuilder, "build_epsf", build_spy)

    img, df = _star_field(asym_wing=asym_wing)
    p = PSF(_psf_build_yaml(tmp_path, phot_extra), img)
    p.build(psfSources=df, plot=False)
    return captured


class TestEpsfBuildContract:
    """End-to-end build() checks on the star ensemble and the seed the
    empirical build hands to EPSFBuilder."""

    def test_cutout_center_matches_extraction_table(
        self, tmp_path, monkeypatch
    ):
        """The cutout validator must not replace the extraction
        centroid: the two-pass centroid_sources result carries the
        subpixel phase the ePSF grid needs, and overwriting it with the
        crude 10%-of-peak COM injected star-to-star phase jitter."""
        pytest.importorskip("photutils")
        captured = _run_build_with_spies(tmp_path, monkeypatch)
        tbl = captured["tbl"]
        assert tbl is not None, "extract_stars was never called"
        assert captured["calls"], "EPSFBuilder.build_epsf never ran"
        checked = 0
        for star_snaps, _kwargs in captured["calls"]:
            for id_label, center, origin in star_snaps:
                i = id_label - 1
                exp_x = float(tbl["x"][i]) - origin[0]
                exp_y = float(tbl["y"][i]) - origin[1]
                np.testing.assert_allclose(
                    center, (exp_x, exp_y), atol=1e-6
                )
                checked += 1
        assert checked > 0

    def test_analytic_init_seeds_build_by_default(
        self, tmp_path, monkeypatch
    ):
        pytest.importorskip("photutils")
        import autophot.photometry.psf as psf_mod

        captured = _run_build_with_spies(tmp_path, monkeypatch)
        assert captured["calls"], "EPSFBuilder.build_epsf never ran"
        assert captured["calls"][0][1].get(psf_mod._EPSF_KWARG) is not None

    def test_analytic_init_disabled_builds_from_scratch(
        self, tmp_path, monkeypatch
    ):
        pytest.importorskip("photutils")
        import autophot.photometry.psf as psf_mod

        captured = _run_build_with_spies(
            tmp_path, monkeypatch, {"psf_use_analytic_init": False}
        )
        assert captured["calls"], "EPSFBuilder.build_epsf never ran"
        assert captured["calls"][0][1].get(psf_mod._EPSF_KWARG) is None

    def test_wing_cleanup_off_by_default(self, tmp_path, monkeypatch):
        """clean_epsf_stamp modifies the converged empirical model
        (edge taper + wing smoothing + radial flattening), so it must
        stay opt-in."""
        pytest.importorskip("photutils")
        import autophot.photometry.psf as psf_mod
        from autophot.photometry.psf import PSF

        calls = {"n": 0}
        real = psf_mod.clean_epsf_stamp

        def spy(data, *args, **kwargs):
            calls["n"] += 1
            return real(data, *args, **kwargs)

        monkeypatch.setattr(psf_mod, "clean_epsf_stamp", spy)
        img, df = _star_field(asym_wing=False)
        p = PSF(_psf_build_yaml(tmp_path), img)
        p.build(psfSources=df, plot=False)
        assert calls["n"] == 0

    def test_wing_cleanup_opt_in(self, tmp_path, monkeypatch):
        pytest.importorskip("photutils")
        import autophot.photometry.psf as psf_mod
        from autophot.photometry.psf import PSF

        calls = {"n": 0}
        real = psf_mod.clean_epsf_stamp

        def spy(data, *args, **kwargs):
            calls["n"] += 1
            return real(data, *args, **kwargs)

        monkeypatch.setattr(psf_mod, "clean_epsf_stamp", spy)
        img, df = _star_field(asym_wing=False)
        p = PSF(
            _psf_build_yaml(tmp_path, {"psf_epsf_wing_smooth": True}), img
        )
        p.build(psfSources=df, plot=False)
        # Requires a usable empirical ePSF; a failed build has no model
        # to clean.
        if p.psf_model_kind != "none" and p.psf_converged is not None:
            assert calls["n"] > 0


class TestNonconvergedFwhmDiscard:
    """Policy for a non-converged ePSF whose measured FWHM misses the
    [psf_epsf_fwhm_min_frac, psf_epsf_fwhm_max_frac]x band.

    A GROSS miss (outside the discard_* band) is a defect stack and is
    discarded unconditionally.  A MARGINAL miss is ambiguous -- thin
    clean pools drift a few tenths past the warn edge without settling
    (GROND_g_TDP8_cor_ana_5_MJD61233_29029: 4 usable stars, measured
    0.83x, non-converged at 0.032 px) -- so the residual gate arbitrates:
    the model is kept when it fits its own stars no worse than
    psf_epsf_residual_ratio_max x the analytic seed, and discarded when
    it loses or the comparison cannot run.
    """

    # maxiters=1 forces non-convergence; the relaxed settle bound and the
    # unsettled-star prune are disabled so the verdict stays deterministic.
    _NONCONV = {
        "psf_maxiters_initial": 1,
        "psf_maxiters_max": 1,
        "psf_center_accuracy_relaxed_px": 0.0,
        "psf_prune_unsettled_retry": False,
    }

    def _run(self, tmp_path, monkeypatch, meas_fwhm, mad_seq=None,
             phot_extra=None):
        pytest.importorskip("photutils")
        import itertools

        import autophot.photometry.psf as psf_mod
        from autophot.photometry.psf import PSF

        phot = dict(self._NONCONV)
        phot.update(phot_extra or {})
        monkeypatch.setattr(
            psf_mod,
            "measure_epsf_fwhm_native",
            lambda *a, **k: meas_fwhm,
        )
        calls = {"mad": []}
        if mad_seq is not None:
            it = iter(mad_seq)
            monkeypatch.setattr(
                psf_mod, "_epsf_residual_mad", lambda *a, **k: next(it)
            )
            calls["mad"] = mad_seq
        img, df = _star_field(asym_wing=False)
        p = PSF(_psf_build_yaml(tmp_path, phot), img)
        model, _ = p.build(psfSources=df, plot=False)
        return p, model

    def test_marginal_miss_rescued_when_empirical_wins(
        self, tmp_path, monkeypatch
    ):
        """measured 0.80x image FWHM sits between discard_min (0.6) and
        the warn floor (0.85); the empirical model fits its own stars at
        parity with the analytic seed, so it is kept."""
        import itertools

        import autophot.photometry.psf as psf_mod

        real = psf_mod._epsf_residual_mad
        seen = []

        def spy(stars, model, fit_rad):
            v = real(stars, model, fit_rad)
            seen.append(v)
            return v

        monkeypatch.setattr(psf_mod, "_epsf_residual_mad", spy)
        p, model = self._run(
            tmp_path, monkeypatch, meas_fwhm=3.2  # 0.80 x fwhm=4.0
        )
        assert p.psf_converged is False
        assert model is not None
        assert len(seen) == 2  # empirical first, analytic second
        assert seen[0] <= 1.3 * seen[1]

    def test_gross_miss_discards_unconditionally(
        self, tmp_path, monkeypatch
    ):
        """measured 0.50x is inside defect-stack territory; the residual
        gate must not even run."""
        import itertools

        import autophot.photometry.psf as psf_mod

        it = iter([0.01, 1.0])
        monkeypatch.setattr(
            psf_mod, "_epsf_residual_mad", lambda *a, **k: next(it)
        )
        p, model = self._run(
            tmp_path, monkeypatch, meas_fwhm=2.0  # 0.50 x fwhm=4.0
        )
        assert p.psf_converged is False
        assert model is None

    def test_marginal_miss_discards_when_empirical_loses(
        self, tmp_path, monkeypatch
    ):
        """measured 0.80x but the empirical model fits its own stars
        2x worse than the analytic seed (> residual_ratio_max) --
        that is a degraded build, not a thin pool."""
        p, model = self._run(
            tmp_path,
            monkeypatch,
            meas_fwhm=3.2,
            mad_seq=[1.0, 0.5],  # empirical, analytic
        )
        assert p.psf_converged is False
        assert model is None

    def test_marginal_miss_discards_when_gate_disabled(
        self, tmp_path, monkeypatch
    ):
        """With psf_epsf_residual_gate off there is no evidence to
        arbitrate with, so the conservative discard stands."""
        import itertools

        import autophot.photometry.psf as psf_mod

        it = iter([0.01, 1.0])
        monkeypatch.setattr(
            psf_mod, "_epsf_residual_mad", lambda *a, **k: next(it)
        )
        p, model = self._run(
            tmp_path,
            monkeypatch,
            meas_fwhm=3.2,
            phot_extra={"psf_epsf_residual_gate": False},
        )
        assert p.psf_converged is False
        assert model is None


class TestContaminationSigmaKeys:
    """psf_contam_asymmetry_frac / psf_contam_radial_reversal_frac were
    documented as flux fractions but applied in annulus-sigma units at
    0.3 -- a sub-sigma quadrant or ring excursion flagged legitimate
    stars and pushed the pool into the masked keep-floor.  The *_sigma
    names at 3.0 are the physically meaningful thresholds; the old names
    are still honoured."""

    FWHM = 3.0
    CUT = 25

    def _cutouts(self, seed=5):
        """4 clean stars + 1 mild-asymmetry + 1 mild ring-reversal +
        1 strong contamination.

        Mild raises are ~1.5 sigma of the annulus MAD: they trip the
        legacy 0.3 ``*_frac`` thresholds but stay under the 3.0-sigma
        keys.  7 stars keep the flagged count above the keep-floor
        (min_candidates is floored at 4), so rejection is actually
        applied rather than converted to annulus masking.
        """
        rng = np.random.default_rng(seed)
        c = (self.CUT - 1) / 2.0
        yy, xx = np.mgrid[0 : self.CUT, 0 : self.CUT]
        sig = self.FWHM / 2.354820045
        prof = np.exp(
            -((xx - c) ** 2 + (yy - c) ** 2) / (2.0 * sig**2)
        )
        ann = np.hypot(xx - c, yy - c)
        # Annulus runs from ceil(2*FWHM) = 6 px to the cutout centre
        # distance; ring edges are linspace(6, 12, 5), so the second
        # ring covers r in [7.5, 9).
        ann_mask = (ann >= 6) & (ann < c)
        quad = ann_mask & (xx > c) & (yy < c)
        ring2 = (ann >= 7.5) & (ann < 9.0)
        out = []
        for i in range(7):
            # Low noise keeps clean stars under the 0.3-sigma legacy
            # threshold; the raises scale against it via ann_std.
            data = 5e3 * prof + rng.normal(0.0, 0.5, prof.shape)
            if i == 4:
                data[quad] += 0.7  # ~1.4-sigma one-sided annulus raise
            elif i == 5:
                data[ring2] += 0.7  # ~1.4-sigma ring-to-ring increase
            elif i == 6:
                data[quad] += 5.0  # ~10-sigma, flagged at any threshold
            out.append(data)
        return out

    def _check(self, cutouts, phot_extra=None):
        import logging

        from astropy.table import Table
        from photutils.psf import EPSFStar, EPSFStars

        from autophot.photometry.psf import PSF

        phot = {
            "psf_cutout_contamination_check": True,
            # Only the asymmetry / ring-reversal gates are under test;
            # every other gate is disabled so nothing else can flag.
            "psf_contam_edge_excess_sigma": 1e9,
            "psf_contam_border_excess_sigma": 1e9,
            "psf_contam_border_max_sigma": 1e9,
            "psf_contam_secondary_peak_sigma": 1e9,
            "psf_contam_streak_sigma": 1e9,
            "psf_contam_annulus_peak_sigma": 1e9,
            "psf_contam_flux_excess_sigma": 1e9,
            "psf_contam_core_ellipticity_max": 0.0,
            "psf_build_min_keep_frac_after_cut": 0.0,
            "psf_min_candidates": 2,
        }
        phot.update(phot_extra or {})
        p = PSF({"photometry": phot}, np.zeros((2, 2)))
        stars = EPSFStars(
            [EPSFStar(np.ascontiguousarray(c_)) for c_ in cutouts]
        )
        for i, st in enumerate(stars):
            st.id_label = i
        tbl = Table(
            {"x": np.full(len(cutouts), 12.0), "y": np.full(len(cutouts), 12.0)}
        )
        log = logging.getLogger("test")
        log.setLevel(logging.ERROR)
        out, _ = p._check_cutout_contamination(
            stars, tbl, cutouts[0].shape, self.FWHM, log
        )
        return [int(getattr(s, "id_label", -1)) for s in out]

    def test_sigma_default_flags_only_strong_contamination(self):
        pytest.importorskip("photutils")
        kept = self._check(self._cutouts())
        # The ~1.4-sigma bumps are real but mild asymmetries; only the
        # 10-sigma quadrant raise is unambiguous contamination.
        assert set(kept) == {0, 1, 2, 3, 4, 5}

    def test_legacy_frac_key_still_honored(self):
        pytest.importorskip("photutils")
        kept = self._check(
            self._cutouts(), {"psf_contam_asymmetry_frac": 0.3}
        )
        # The 1.4-sigma asymmetry trips the old sub-sigma threshold;
        # the ring-raised star stays (azimuthally symmetric) and the
        # strong contam still drops.
        assert 4 not in kept
        assert 6 not in kept
        assert 5 in kept

    def test_new_key_wins_over_legacy(self):
        pytest.importorskip("photutils")
        kept = self._check(
            self._cutouts(),
            {
                "psf_contam_asymmetry_sigma": 3.0,
                "psf_contam_asymmetry_frac": 0.3,
            },
        )
        assert set(kept) == {0, 1, 2, 3, 4, 5}

    def test_legacy_reversal_key_still_honored(self):
        pytest.importorskip("photutils")
        kept = self._check(
            self._cutouts(), {"psf_contam_radial_reversal_frac": 0.3}
        )
        # The reversal key only changes the ring test: the ring-raised
        # star drops at 0.3 sigma while the asymmetric star survives
        # under the 3.0-sigma asymmetry default.
        assert 5 not in kept
        assert 6 not in kept
        assert 4 in kept


class TestShippedPsfDefaults:
    """The shipped config must carry the diagnosis-verified defaults."""

    def _phot(self):
        import yaml
        from pathlib import Path

        root = Path(__file__).resolve().parent.parent
        cfg = yaml.safe_load(
            (root / "autophot" / "data" / "default_input.yml").read_text()
        )
        return cfg["default_input"]["photometry"]

    def test_wing_smoothing_default_off(self):
        assert self._phot()["psf_epsf_wing_smooth"] is False

    def test_contamination_sigma_keys(self):
        phot = self._phot()
        assert phot["psf_contam_asymmetry_sigma"] == 3.0
        assert phot["psf_contam_radial_reversal_sigma"] == 3.0

    def test_analytic_init_toggle_documented(self):
        assert "psf_use_analytic_init" in self._phot()

    def test_ee_cutout_sizing_keys(self):
        phot = self._phot()
        assert phot["psf_cutout_ee_fraction"] == 0.9995
        assert phot["psf_cutout_ee_margin_fwhm"] == 1.0
        assert phot["psf_cutout_size_scale_fwhm"] == 10.0


class TestMatchedFilterDetectionProb:
    """Likelihood-ratio detection probability (PSF + sky vs sky only)."""

    @staticmethod
    def _psf_stamp(sigma_px=1.5, size=15):
        yy, xx = np.mgrid[0:size, 0:size]
        c = (size - 1) / 2.0
        stamp = np.exp(
            -((xx - c) ** 2 + (yy - c) ** 2) / (2.0 * sigma_px**2)
        )
        stamp /= stamp.sum()
        return stamp

    def _model(self):
        from photutils.psf import ImagePSF

        stamp = self._psf_stamp()
        c = (stamp.shape[0] - 1) / 2.0
        return ImagePSF(
            data=stamp, x_0=c, y_0=c, oversampling=1, fill_value=0.0
        )

    def _field(self, flux=0.0, pos=(25.4, 25.2), seed=7, noise_sigma=1.0,
               shape_sigma=None, size=51):
        """Image with a point source (or alternative shape) plus noise."""
        from astropy.nddata import NDData, StdDevUncertainty

        model = self._model()
        rng = np.random.default_rng(seed)
        yy, xx = np.mgrid[0:size, 0:size]
        if shape_sigma is None:
            # Photometrically injected point source at a subpixel offset.
            src = flux * model.evaluate(
                xx, yy, flux=1.0, x_0=pos[0], y_0=pos[1]
            )
        else:
            c = (size - 1) / 2.0
            src = flux * np.exp(
                -((xx - c) ** 2 + (yy - c) ** 2) / (2.0 * shape_sigma**2)
            )
        img = rng.normal(0.0, noise_sigma, (size, size)) + src
        return (
            NDData(data=img, uncertainty=StdDevUncertainty(
                np.full_like(img, noise_sigma)
            )),
            model,
        )

    def test_strong_source_high_prob(self):
        """A z ~ 6 point source must report prob_detect near 1."""
        from autophot.photometry.psf import matched_filter_detection_prob

        nd, model = self._field(flux=60.0)
        prob, z, dchi2 = matched_filter_detection_prob(
            nd, model, 25.4, 25.2, half_box=5
        )
        assert prob > 0.99
        assert z > 5.0
        assert np.isclose(dchi2, z * z)

    def test_noise_only_is_undecided(self):
        """No source -> prob_detect ~0.5, not a confident rejection."""
        from autophot.photometry.psf import matched_filter_detection_prob

        probs = []
        for seed in range(20):
            nd, model = self._field(flux=0.0, seed=seed)
            p, _, _ = matched_filter_detection_prob(
                nd, model, 25.4, 25.2, half_box=5
            )
            probs.append(p)
        probs = np.asarray(probs)
        # Under the null, Phi(z) is ~Uniform(0,1): mean ~0.5, no
        # systematic pile-up at the extremes.
        assert 0.35 < probs.mean() < 0.65
        assert np.count_nonzero(probs > 0.99) <= 2

    def test_negative_dip_low_prob(self):
        """PSF-shaped negative flux (oversubtraction) -> prob ~0."""
        from autophot.photometry.psf import matched_filter_detection_prob

        nd, model = self._field(flux=-60.0)
        prob, z, _ = matched_filter_detection_prob(
            nd, model, 25.4, 25.2, half_box=5
        )
        assert prob < 0.01
        assert z < -5.0

    def test_inverted_evaluates_sign_flipped(self):
        """inverted=True treats a negative source as the detection."""
        from autophot.photometry.psf import matched_filter_detection_prob

        nd, model = self._field(flux=-60.0)
        prob, z, _ = matched_filter_detection_prob(
            nd, model, 25.4, 25.2, half_box=5, inverted=True
        )
        assert prob > 0.99
        assert z > 5.0

    def test_shape_mismatch_suppressed(self):
        """Templates that fit better than the PSF must suppress
        prob_detect: a single-pixel spike (cosmic-ray-like) carries a
        matched z that naively claims a detection, but the narrow-Gaussian
        alternative explains the box better."""
        import math

        from astropy.nddata import NDData, StdDevUncertainty
        from autophot.photometry.psf import matched_filter_detection_prob

        model = self._model()
        rng = np.random.default_rng(3)
        img = rng.normal(0.0, 1.0, (51, 51))
        img[25, 25] += 15.0  # sub-PSF spike: delta-shape, not PSF-shape
        nd = NDData(
            data=img, uncertainty=StdDevUncertainty(np.ones((51, 51)))
        )
        prob, z, _ = matched_filter_detection_prob(
            nd, model, 25.4, 25.2, half_box=5
        )
        naive = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
        assert z > 4.0 and naive > 0.99  # naive matched filter "detects"
        assert prob < 0.8  # shape comparison vetoes it

    def test_extended_source_suppressed(self):
        """A resolved blob is not a point-source detection."""
        from autophot.photometry.psf import matched_filter_detection_prob

        nd, model = self._field(flux=30.0, shape_sigma=4.0, seed=3)
        prob_blob, _, _ = matched_filter_detection_prob(
            nd, model, 25.0, 25.0, half_box=6
        )
        assert prob_blob < 0.7

    def test_empirical_fap_strong_source(self):
        """Blank-site FAP is small when the target statistic is strong."""
        from autophot.photometry.psf import (
            empirical_false_alarm_prob,
            matched_filter_detection_prob,
        )

        nd, model = self._field(flux=200.0, pos=(50.0, 50.0), seed=0, size=101)
        prob_row, _, _ = matched_filter_detection_prob(
            nd, model, 50.0, 50.0, half_box=5
        )
        assert prob_row > 0.99
        p_false, n_sites = empirical_false_alarm_prob(
            nd,
            model,
            50.0,
            50.0,
            prob_row,
            5,
            n_trials=40,
            r_in=8.0,
            r_out=40.0,
            min_sites=20,
            rng=np.random.default_rng(1),
        )
        assert n_sites >= 20
        assert p_false < 0.10

    def test_empirical_fap_uses_snr_when_probability_saturates(self, monkeypatch):
        from astropy.nddata import NDData
        import autophot.photometry.psf as psf_mod

        nd = NDData(data=np.zeros((101, 101)))

        def fake_statistic(ndimage, epsf_model, x, y, half_box, inverted=False):
            return 1.0, 9.0, 81.0

        monkeypatch.setattr(
            psf_mod, "matched_filter_detection_prob", fake_statistic
        )
        p_false, n_sites = psf_mod.empirical_false_alarm_prob(
            nd,
            None,
            50.0,
            50.0,
            1.0,
            5,
            n_trials=10,
            r_in=8.0,
            r_out=30.0,
            min_sites=10,
            rng=np.random.default_rng(11),
            z_row=10.0,
        )
        assert n_sites == 10
        assert p_false == pytest.approx(1.0 / 11.0)

    def test_empirical_fap_noise_field(self):
        """On pure noise the row statistic sits mid-null: FAP ~ 0.5."""
        from astropy.nddata import NDData, StdDevUncertainty
        from autophot.photometry.psf import (
            empirical_false_alarm_prob,
            matched_filter_detection_prob,
        )

        model = self._model()
        rng = np.random.default_rng(2)
        nd = NDData(
            data=rng.normal(0.0, 1.0, (101, 101)),
            uncertainty=StdDevUncertainty(np.ones((101, 101))),
        )
        prob_row, _, _ = matched_filter_detection_prob(
            nd, model, 50.0, 50.0, half_box=5
        )
        p_false, n_sites = empirical_false_alarm_prob(
            nd,
            model,
            50.0,
            50.0,
            prob_row,
            5,
            n_trials=40,
            r_in=8.0,
            r_out=40.0,
            min_sites=20,
            rng=np.random.default_rng(3),
        )
        assert n_sites >= 20
        # Any position in a pure-noise null: FAP is in (0, 1) and the
        # estimate must not crash toward the rails.
        assert 0.1 < p_false < 0.95

    def test_empirical_fap_insufficient_sites(self):
        """Too few valid sites -> NaN, not a spurious small p-value."""
        from astropy.nddata import NDData, StdDevUncertainty
        from autophot.photometry.psf import empirical_false_alarm_prob

        model = self._model()
        rng = np.random.default_rng(4)
        nd = NDData(
            data=rng.normal(0.0, 1.0, (101, 101)),
            uncertainty=StdDevUncertainty(np.ones((101, 101))),
        )
        # Position too close to the edge: the annulus cannot fit.
        p_false, n_sites = empirical_false_alarm_prob(
            nd,
            model,
            6.0,
            6.0,
            0.5,
            5,
            n_trials=40,
            r_in=8.0,
            r_out=40.0,
            min_sites=20,
            rng=np.random.default_rng(5),
        )
        assert not np.isfinite(p_false)
        assert n_sites == 0

    def test_empirical_fap_exclusion(self):
        """Excluded positions are never sampled as blank sites."""
        from astropy.nddata import NDData, StdDevUncertainty
        from autophot.photometry.psf import empirical_false_alarm_prob

        model = self._model()
        rng = np.random.default_rng(6)
        nd = NDData(
            data=rng.normal(0.0, 1.0, (101, 101)),
            uncertainty=StdDevUncertainty(np.ones((101, 101))),
        )
        gy, gx = np.mgrid[0:101, 0:101]
        every_pixel = np.column_stack([gx.ravel(), gy.ravel()])
        p_false, n_sites = empirical_false_alarm_prob(
            nd,
            model,
            50.0,
            50.0,
            0.5,
            5,
            n_trials=20,
            r_in=8.0,
            r_out=40.0,
            min_sites=5,
            rng=np.random.default_rng(7),
            exclude_xy=every_pixel,
            exclude_radius=1.0,
        )
        assert not np.isfinite(p_false)
        assert n_sites == 0

    def test_dipole_suppressed(self):
        """A +PSF/-PSF dipole (misregistration signature) must be
        suppressed below the naive matched-filter probability -- the
        dipole alternative model explains the box better."""
        import math

        from astropy.nddata import NDData, StdDevUncertainty
        from autophot.photometry.psf import matched_filter_detection_prob

        model = self._model()
        gy, gx = np.mgrid[0:51, 0:51]
        rng = np.random.default_rng(3)
        img = rng.normal(0.0, 1.0, (51, 51))
        img += 45.0 * model.evaluate(
            gx, gy, flux=1.0, x_0=26.2, y_0=25.2
        ) - 40.0 * model.evaluate(
            gx, gy, flux=1.0, x_0=24.6, y_0=25.2
        )
        nd = NDData(
            data=img, uncertainty=StdDevUncertainty(np.ones((51, 51)))
        )
        prob, z, _ = matched_filter_detection_prob(
            nd, model, 25.4, 25.2, half_box=5
        )
        naive = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
        assert prob < naive
        assert prob < 0.75

    def test_nonfinite_pixels_excluded(self):
        """NaN data / inf sigma in the box are ignored, not fatal."""
        from autophot.photometry.psf import matched_filter_detection_prob

        nd, model = self._field(flux=60.0)
        img = np.asarray(nd.data, dtype=float).copy()
        img[25, 25] = np.nan
        sig = np.asarray(nd.uncertainty.array, dtype=float).copy()
        sig[24, 24] = np.inf
        from astropy.nddata import NDData, StdDevUncertainty

        nd2 = NDData(data=img, uncertainty=StdDevUncertainty(sig))
        prob, z, _ = matched_filter_detection_prob(
            nd2, model, 25.4, 25.2, half_box=5
        )
        assert np.isfinite(prob) and prob > 0.9

    def test_empty_or_all_nan_returns_nan(self):
        from astropy.nddata import NDData
        from autophot.photometry.psf import matched_filter_detection_prob

        nd = NDData(data=np.full((21, 21), np.nan))
        prob, z, d = matched_filter_detection_prob(
            nd, self._model(), 10.0, 10.0, half_box=5
        )
        assert not np.isfinite(prob) and not np.isfinite(z)


class TestEeCutoutSizing:
    """Encircled-energy-driven PSF star cutout sizing in build().

    The fixed 10xFWHM multiplier is replaced by the radius enclosing
    ``psf_cutout_ee_fraction`` of the analytic init PSF core profile
    plus a FWHM-scaled margin.  Compact profiles shrink the stamp;
    broad-wing profiles (low ``psf_init_moffat_beta``) grow it past
    the formula result.  The formula is the fallback, and the
    fit-region/arcsecond/12-px floors always apply.
    """

    def _capture_sizes(self, tmp_path, monkeypatch, phot_extra=None,
                       yaml_extra=None):
        """Run build() and record every ``size`` kwarg extract_stars saw."""
        pytest.importorskip("photutils")
        import autophot.photometry.psf as psf_mod
        from autophot.photometry.psf import PSF

        captured = []
        orig = psf_mod.extract_stars

        def spy(data, catalog, **kwargs):
            if "size" in kwargs:
                captured.append(kwargs["size"])
            return orig(data, catalog, **kwargs)

        monkeypatch.setattr(psf_mod, "extract_stars", spy)
        img, df = _star_field(asym_wing=False)
        cfg = _psf_build_yaml(tmp_path, phot_extra)
        cfg.update(yaml_extra or {})
        PSF(cfg, img).build(psfSources=df, plot=False)
        assert captured, "extract_stars was never called"
        return captured

    def test_ee_radius_on_moffat_core(self):
        """The EE radius of the analytic init core matches the Moffat
        closed form (within the integer-ring binning tolerance)."""
        pytest.importorskip("photutils")
        from autophot.photometry.psf import (
            _composite_moffat_psf,
            _default_moffat_composite,
            _epsf_ee_radius_native,
        )

        fwhm, beta = 3.0, 4.765
        comps = _default_moffat_composite(fwhm, beta)[0:1]
        stamp = _composite_moffat_psf(comps, 4, 121)
        r = _epsf_ee_radius_native(
            np.asarray(stamp.data, dtype=float), 4, 0.9995
        )
        gamma = fwhm / (2.0 * np.sqrt(2.0 ** (1.0 / beta) - 1.0))
        r_true = gamma * np.sqrt(
            (1.0 - 0.9995) ** (1.0 / (1.0 - beta)) - 1.0
        )
        assert np.isfinite(r)
        assert abs(r - r_true) < 2.0

    def test_ee_radius_grows_with_lower_beta(self):
        pytest.importorskip("photutils")
        from autophot.photometry.psf import (
            _composite_moffat_psf,
            _default_moffat_composite,
            _epsf_ee_radius_native,
        )

        radii = []
        for beta in (4.765, 2.0):
            comps = _default_moffat_composite(3.0, beta)[0:1]
            stamp = _composite_moffat_psf(comps, 4, 241)
            radii.append(
                _epsf_ee_radius_native(
                    np.asarray(stamp.data, dtype=float), 4, 0.999
                )
            )
        assert radii[1] > radii[0]

    def test_compact_profile_shrinks_below_formula(
        self, tmp_path, monkeypatch
    ):
        """Default beta=4.765 at FWHM=4: EE sizing lands below the
        10xFWHM formula (41 px) while still covering the fit floor."""
        sizes = self._capture_sizes(tmp_path, monkeypatch)
        n = sizes[0][0]
        assert n < 41
        assert n % 2 == 1
        # The 2*fit_boxsize+5 floor (39 px for the default fit box)
        # still binds, so the stamp is not shrunk into the fit region.
        assert n >= 39

    def test_broad_wing_grows_past_formula(self, tmp_path, monkeypatch):
        """beta=2.0 wings need more than 10xFWHM; the EE radius
        saturates the probe and the cutout grows beyond the formula."""
        sizes = self._capture_sizes(
            tmp_path, monkeypatch, {"psf_init_moffat_beta": 2.0}
        )
        assert sizes[0][0] > 41

    def test_ee_disabled_uses_formula(self, tmp_path, monkeypatch):
        sizes = self._capture_sizes(
            tmp_path, monkeypatch, {"psf_cutout_ee_fraction": 0.0}
        )
        assert sizes[0][0] == 41

    def test_ee_failure_falls_back_to_formula(self, tmp_path, monkeypatch):
        pytest.importorskip("photutils")
        import autophot.photometry.psf as psf_mod

        def _boom(*args, **kwargs):
            raise RuntimeError("forced EE sizing failure")

        monkeypatch.setattr(psf_mod, "_composite_moffat_psf", _boom)
        sizes = self._capture_sizes(tmp_path, monkeypatch)
        assert sizes[0][0] == 41

    def test_formula_scale_override_still_beaten_by_ee(
        self, tmp_path, monkeypatch
    ):
        """A large psf_cutout_size_scale_fwhm raises only the formula
        fallback -- the EE result is still used when it is smaller."""
        sizes = self._capture_sizes(
            tmp_path, monkeypatch, {"psf_cutout_size_scale_fwhm": 20.0}
        )
        assert sizes[0][0] < 81

    def test_margin_grows_cutout(self, tmp_path, monkeypatch):
        sizes = self._capture_sizes(
            tmp_path, monkeypatch, {"psf_cutout_ee_margin_fwhm": 8.0}
        )
        assert sizes[0][0] > 41

    def test_fixed_cutout_override_wins(self, tmp_path, monkeypatch):
        """``_psf_fixed_cutout`` (the spatial-grid shared shape) must
        override both EE and formula sizing."""
        sizes = self._capture_sizes(
            tmp_path, monkeypatch, {"_psf_fixed_cutout": 55}
        )
        assert all(s == (55, 55) for s in sizes)

    def test_fit_box_floor_applies(self, tmp_path, monkeypatch):
        """A large fit box forces the cutout to 2*fit_boxsize+5 even
        when the EE radius alone would be smaller."""
        sizes = self._capture_sizes(
            tmp_path, monkeypatch, {"psf_fit_boxsize_scale_fwhm": 8.0}
        )
        assert sizes[0][0] >= 71

    def test_arcsec_floor_applies(self, tmp_path, monkeypatch):
        """With a pixel scale, psf_cutout_min_arcsec floors both the
        formula and the EE result."""
        sizes = self._capture_sizes(
            tmp_path, monkeypatch, yaml_extra={"pixel_scale": 0.1}
        )
        assert sizes[0][0] == 91


class TestPhaseDiagnostics:
    """``_epsf_phase_diagnostics`` -- the post-build subpixel-phase
    coverage/uniformity check from the photutils ePSF guidelines."""

    class _Star:
        def __init__(self, x, y):
            self.center = np.array([x, y], float)

    def test_uniform_phases_pass(self):
        rng = np.random.default_rng(7)
        stars = [
            self._Star(10.0 + rng.uniform(), 20.0 + rng.uniform())
            for _ in range(40)
        ]
        d = _epsf_phase_diagnostics(stars, 2)
        assert d["n_stars"] == 40
        assert d["samples_per_cell"] == pytest.approx(10.0)
        assert d["coverage"] == pytest.approx(1.0)
        assert d["uniform_p"] > 0.01

    def test_clustered_phases_flagged(self):
        rng = np.random.default_rng(3)
        stars = [
            self._Star(10.0 + 0.2 * rng.uniform(), 20.0 + 0.2 * rng.uniform())
            for _ in range(40)
        ]
        d = _epsf_phase_diagnostics(stars, 2)
        assert d["coverage"] == pytest.approx(0.25)
        assert d["uniform_p"] < 1e-6

    def test_tiny_pool_skips_test(self):
        stars = [self._Star(10.3, 20.7), self._Star(11.1, 21.2)]
        d = _epsf_phase_diagnostics(stars, 4)
        assert d["n_stars"] == 2
        assert not np.isfinite(d["uniform_p"])
        assert d["coverage"] == pytest.approx(2.0 / 16.0)

    def test_osamp1_single_cell(self):
        rng = np.random.default_rng(5)
        stars = [
            self._Star(10.0 + rng.uniform(), 20.0 + rng.uniform())
            for _ in range(12)
        ]
        d = _epsf_phase_diagnostics(stars, 1)
        assert d["coverage"] == pytest.approx(1.0)
        # one cell -> no degrees of freedom for the chi-square test
        assert not np.isfinite(d["uniform_p"])

    def test_missing_and_bad_centres(self):
        stars = [
            self._Star(np.nan, 5.0),
            self._Star(10.2, 20.4),
        ]
        d = _epsf_phase_diagnostics(stars, 2)
        assert d["n_stars"] == 1
        d_none = _epsf_phase_diagnostics(None, 2)
        assert d_none["n_stars"] == 0

    def test_psf_defaults(self):
        from autophot.photometry.psf import PSF

        p = PSF(input_yaml={"photometry": {}}, image=np.zeros((5, 5)))
        assert p.psf_grid_underresolved is False
        assert p.psf_phase_diag is None
        assert p.psf_phase_nonuniform is False

    def test_build_populates_phase_diag(self, tmp_path, monkeypatch):
        """A completed empirical build records the subpixel-phase
        diagnostics of the fitted star centres."""
        pytest.importorskip("photutils")
        from autophot.photometry.psf import PSF

        img, df = _star_field(asym_wing=False)
        p = PSF(_psf_build_yaml(tmp_path), img)
        epsf, _ = p.build(psfSources=df, plot=False)
        if epsf is None or p.psf_model_kind == "none":
            pytest.skip("no empirical model built on the synthetic field")
        assert p.psf_phase_diag is not None
        assert p.psf_phase_diag["n_stars"] > 0
        assert p.psf_phase_diag["samples_per_cell"] > 0
        assert np.isfinite(p.psf_phase_diag["coverage"])
