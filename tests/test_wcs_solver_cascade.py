"""Tests for the solve-field cascade hardening in wcs.py.

Covers the argv flag-splice helper, the invocation-scoped artifact glob,
the cpulimit clamp, timeout partial-WCS cleanup, and the backup/restore
tweak-order descent - all with a mocked solve-field run (no subprocess,
no index files, no network).
"""

from __future__ import annotations

import subprocess

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from autophot.pipeline.wcs import (
    WCSSolver,
    _set_flag_value,
    _solve_artifact_glob,
    _wcs_cfg_bool,
)


# ---------------------------------------------------------------------------
# _set_flag_value
# ---------------------------------------------------------------------------


def test_set_flag_value_replaces_only_owned_pair():
    """A colliding value for another flag must survive the splice."""
    args = [
        "solve-field",
        "--downsample",
        "3",
        "--tweak-order",
        "4",
        "--nsigma",
        "4",
        "--objs",
        "4",
        "input.fits",
    ]
    out = _set_flag_value(args, "--nsigma", "3")
    # --downsample 3 and --objs 4 keep their values; --tweak-order 4 keeps
    # its colliding "4"; the --nsigma pair moves to the end with "3".
    assert out[out.index("--downsample") + 1] == "3"
    assert out[out.index("--tweak-order") + 1] == "4"
    assert out[out.index("--objs") + 1] == "4"
    assert out.count("--nsigma") == 1
    assert out[out.index("--nsigma") + 1] == "3"
    # Every flag still has exactly one value arg after it.
    for flag in ("--downsample", "--tweak-order", "--objs"):
        i = out.index(flag)
        assert not out[i + 1].startswith("--"), f"{flag} left dangling"


def test_set_flag_value_appends_when_absent():
    args = ["solve-field", "in.fits"]
    out = _set_flag_value(args, "--cpulimit", 45)
    assert out[-2:] == ["--cpulimit", "45"]


# ---------------------------------------------------------------------------
# _solve_artifact_glob
# ---------------------------------------------------------------------------


def test_solve_artifact_glob_scoped_to_invocation(tmp_path):
    """Concurrent solves in the same dir keep each other's artifacts."""
    tag_this = "img_a_111"
    tag_other = "img_b_222"
    for name in (
        f"astrometry_{tag_this}.match",
        f"astrometry_{tag_other}.match",
        f"{tag_this}.xyls",
        f"{tag_other}.xyls",
    ):
        (tmp_path / name).write_text("x")
    hits = _solve_artifact_glob(str(tmp_path), tag_this, "img_a", ".match")
    assert hits == [str(tmp_path / f"astrometry_{tag_this}.match")]
    hits = _solve_artifact_glob(str(tmp_path), tag_this, "img_a", ".xyls")
    assert hits == [str(tmp_path / f"{tag_this}.xyls")]


def test_solve_artifact_glob_axy_covers_input_stem(tmp_path):
    """solve-field names .axy after the input file stem, not the tag."""
    (tmp_path / "img_a.axy").write_text("x")  # image-input axy
    (tmp_path / "img_b.axy").write_text("x")  # other worker's axy
    (tmp_path / "img_a_111.axy").write_text("x")  # xylist-input axy
    hits = _solve_artifact_glob(str(tmp_path), "img_a_111", "img_a", ".axy")
    assert sorted(hits) == [
        str(tmp_path / "img_a.axy"),
        str(tmp_path / "img_a_111.axy"),
    ]


# ---------------------------------------------------------------------------
# _wcs_cfg_bool
# ---------------------------------------------------------------------------


def test_wcs_cfg_bool_values():
    assert _wcs_cfg_bool({}, "k", False) is False
    assert _wcs_cfg_bool({"k": None}, "k", True) is True
    assert _wcs_cfg_bool({"k": "true"}, "k") is True
    assert _wcs_cfg_bool({"k": "0"}, "k", True) is False
    assert _wcs_cfg_bool({"k": 1}, "k") is True


# ---------------------------------------------------------------------------
# Integration harness: plate_solve with a mocked _run_solve_field
# ---------------------------------------------------------------------------


def _tan_wcs_header(crval=(150.0, 30.0), npix=200, scale_arcsec=1.0):
    w = WCS(naxis=2)
    w.wcs.crpix = [npix / 2.0, npix / 2.0]
    w.wcs.crval = [float(crval[0]), float(crval[1])]
    s = scale_arcsec / 3600.0
    w.wcs.cd = np.array([[-s, 0.0], [0.0, s]])
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    # NAXIS1/2 are left for fits.writeto to generate in the right place.
    return w.to_header()


def _fake_solved_wcs_file(path, crval=(150.0, 30.0), sip_order=None):
    """Write a minimal solve-field-style .wcs.fits file."""
    hdr = _tan_wcs_header(crval=crval)
    if sip_order:
        hdr["A_ORDER"] = sip_order
        # All-zero coefficients -> _has_nonzero_sip is False.
        for i in range(sip_order + 1):
            for j in range(sip_order + 1 - i):
                if i + j >= 2:
                    hdr[f"A_{i}_{j}"] = 0.0
                    hdr[f"B_{i}_{j}"] = 0.0
        hdr["AP_ORDER"] = 0
        hdr["BP_ORDER"] = 0
    fits.writeto(path, np.zeros((2, 2)), hdr, overwrite=True)


def _make_solver(tmp_path, wcs_cfg):
    img = np.random.default_rng(0).normal(100.0, 5.0, (200, 200))
    hdr = _tan_wcs_header()
    fpath = tmp_path / "img_a.fits"
    fits.writeto(fpath, img, hdr)
    default_input = {
        "pixel_scale": 1.0,
        "target_ra": 150.0,
        "target_dec": 30.0,
        "wcs": {
            "redo_wcs": True,
            "projection_type": "SIP",
            "solver": "astrometry",
            "search_radius": 0.5,
            "cpulimit": 45,
            "late_cpulimit": 120,
            "blind_cpulimit": 300,
            "solve_field_total_budget": 1800,
            "downsample": 2,
            "solve_field_use_xylist": False,
            "solve_field_tweak_orders": [4],
            "compare_input_vs_solved_wcs": False,
            "refit_wcs_from_points": False,
            **wcs_cfg,
        },
    }
    solver = WCSSolver(
        fpath=str(fpath), image=img, header=hdr, default_input=default_input
    )
    return solver


def test_plate_solve_args_no_newfits_and_cpulimit_floor(tmp_path, monkeypatch):
    """Mocked solve captures argv: no --new-fits, --cpulimit >= 1."""
    captured = []

    def fake_run(self, args, wcs_file, timeout_sec, logpath):
        captured.append(list(args))
        _fake_solved_wcs_file(wcs_file)
        return True

    monkeypatch.setattr(WCSSolver, "_run_solve_field", fake_run)
    solver = _make_solver(
        tmp_path, {"cpulimit": 0.4, "solve_field_tweak_orders": [0]}
    )
    out = solver.plate_solve(solvefield_exe="/bin/true")
    assert isinstance(out, fits.Header)
    first = captured[0]
    assert "--new-fits" not in first
    assert "--wcs" in first
    assert int(first[first.index("--cpulimit") + 1]) >= 1


def test_plate_solve_no_fits2fits_flag(tmp_path, monkeypatch):
    captured = []

    def fake_run(self, args, wcs_file, timeout_sec, logpath):
        captured.append(list(args))
        _fake_solved_wcs_file(wcs_file)
        return True

    monkeypatch.setattr(WCSSolver, "_run_solve_field", fake_run)
    solver = _make_solver(
        tmp_path,
        {"solve_field_no_fits2fits": True, "solve_field_tweak_orders": [0]},
    )
    solver.plate_solve(solvefield_exe="/bin/true")
    assert "--no-fits2fits" in captured[0]


def test_zero_sip_descent_restores_backup_and_nsigma_args_clean(
    tmp_path, monkeypatch
):
    """Order-4 zero-SIP solve: lower orders fail, backup is restored
    without a re-solve, and the nsigma retry argv keeps every flag paired.
    """
    captured = []
    calls = {"n": 0}

    def fake_run(self, args, wcs_file, timeout_sec, logpath):
        calls["n"] += 1
        captured.append(list(args))
        # Only the first call (order 4) solves, and it has zero SIP.
        if calls["n"] == 1:
            _fake_solved_wcs_file(wcs_file, sip_order=4)
            return True
        # The nsigma retry also "solves" (still zero SIP).
        if "--nsigma" in args and args[args.index("--nsigma") + 1] == "3":
            _fake_solved_wcs_file(wcs_file, sip_order=4)
            return True
        return False

    monkeypatch.setattr(WCSSolver, "_run_solve_field", fake_run)
    solver = _make_solver(
        tmp_path,
        {
            "solve_field_tweak_orders": [4],
            "solve_field_nsigma": 4,  # collides with --tweak-order 4 value
        },
    )
    out = solver.plate_solve(solvefield_exe="/bin/true")
    assert isinstance(out, fits.Header)
    # Old code re-solved the original order to restore the deleted file:
    # order-4 (1) + orders 3,2,1,0 (4) + re-solve (1) + nsigma (1) = 7.
    # Backup restore drops the re-solve -> 6 calls.
    assert calls["n"] == 6
    nsigma_call = [
        a
        for a in captured
        if "--nsigma" in a and a[a.index("--nsigma") + 1] == "3"
    ][0]
    # The tweak-order pair survives the splice; nsigma lands as a pair.
    i = nsigma_call.index("--tweak-order")
    assert nsigma_call[i + 1] == "4"
    assert nsigma_call.count("--nsigma") == 1


def test_plate_solve_rejects_unparseable_wcs_file(tmp_path, monkeypatch):
    """A wcs file with no WCS keywords is rejected, not adopted silently."""

    def fake_run(self, args, wcs_file, timeout_sec, logpath):
        # Header with no WCS keywords at all.
        fits.writeto(wcs_file, np.zeros((2, 2)), fits.Header(), overwrite=True)
        return True

    monkeypatch.setattr(WCSSolver, "_run_solve_field", fake_run)
    solver = _make_solver(tmp_path, {"solve_field_tweak_orders": [0]})
    out = solver.plate_solve(solvefield_exe="/bin/true")
    # Rejected solve falls back to the (valid) input header WCS.
    assert out is solver.header
    assert solver.solved_new_wcs is False
    # The rejected wcs file must not be left behind.
    assert not list(tmp_path.glob("*.wcs.fits*"))


# ---------------------------------------------------------------------------
# _run_solve_field timeout: partial wcs file is dropped
# ---------------------------------------------------------------------------


class _FakeProc:
    """Popen stub: first wait() times out, bare wait() reaps fine."""

    def __init__(self, args, **kwargs):
        self.args = args
        self.pid = 99999
        self._wcs_file = args[args.index("--wcs") + 1]

    def wait(self, timeout=None):
        if timeout is not None:
            # Leave a truncated WCS file behind like a SIGKILLed write.
            with open(self._wcs_file, "wb") as fh:
                fh.write(b"partial")
            raise subprocess.TimeoutExpired(self.args, timeout)
        return 0

    def kill(self):
        pass


def test_run_solve_field_timeout_drops_partial_wcs(tmp_path, monkeypatch):
    import autophot.pipeline.wcs as wcs_mod

    monkeypatch.setattr(wcs_mod.subprocess, "Popen", _FakeProc)
    solver = _make_solver(tmp_path, {})
    solver._solvefield_exe = "solve-field"
    wcs_file = tmp_path / "out.wcs.fits"
    ok = solver._run_solve_field(
        ["solve-field", "--wcs", str(wcs_file), "in.fits"],
        str(wcs_file),
        timeout_sec=5,
        logpath=str(tmp_path / "sf.log"),
    )
    assert ok is False
    assert not wcs_file.exists()
