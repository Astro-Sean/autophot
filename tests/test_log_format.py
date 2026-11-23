"""Unit tests for autophot.core.log_format helpers."""

import logging

import pytest

from autophot.core.log_format import (
    _RUN,
    LOG_WIDTH,
    SUB_WIDTH,
    WarningTagger,
    banner,
    begin_stage,
    dedup,
    dedup_reset,
    dedup_suppressed,
    end_stage,
    reset_run_state,
    run_header,
    scrub_secrets,
    subheader,
    table,
    wrap_message,
    write_debug_dump,
)


class TestBanner:
    def test_width_80(self):
        out = banner("Source detection and FWHM").splitlines()
        assert len(out) == 3
        assert out[0] == "=" * LOG_WIDTH
        assert out[2] == "=" * LOG_WIDTH
        assert out[1] == "Source detection and FWHM"

    def test_subheader_uses_dash(self):
        out = subheader("PSF model").splitlines()
        assert out[0] == "-" * SUB_WIDTH
        assert out[1] == "PSF model"

    def test_long_title_truncated(self):
        out = banner("x" * 100).splitlines()
        assert len(out[1]) <= LOG_WIDTH
        assert out[1].endswith("...")

    def test_whitespace_collapsed(self):
        assert banner("  a\n\n  b ").splitlines()[1] == "a b"


class TestTable:
    def test_alignment_and_separator(self):
        out = table(
            ("star", "snr", "name"),
            [("1", "52.5", "alpha"), ("22", "7.1", "beta-long")],
        ).splitlines()
        assert out[0].startswith("  star")
        assert set(out[1].strip()) <= {"-", " "} and "-" in out[1]
        # numeric column right-aligned: '52.5' and '7.1' share right edge
        snr_col = out[0].index("snr")
        assert out[2][snr_col : snr_col + 3] == " 52" or "52.5" in out[2]
        # right edge alignment check: both snr values end on same column
        end52 = out[2].index("52.5") + 4
        end71 = out[3].index("7.1") + 3
        assert end52 == end71

    def test_empty_rows(self):
        out = table(("a", "b"), [])
        assert "(none)" in out
        assert "a" in out.splitlines()[0]

    def test_very_wide_values_wrapped(self):
        wide = "word " * 40
        out = table(("id", "note"), [("1", wide)], max_width=40)
        for line in out.splitlines():
            assert len(line) <= 42
        # information preserved: every word fragment still present
        assert out.count("word") == 40

    def test_rows_padded_and_truncated(self):
        out = table(("a", "b"), [("1",), ("2", "3", "ignored")])
        assert "ignored" not in out
        assert len(out.splitlines()) == 4  # header, sep, 2 rows

    def test_blank_line_framing_left_to_caller(self):
        # Table itself has no leading/trailing blank lines; callers log
        # it inside a record that carries its own spacing.
        out = table(("a",), [("1",)])
        assert not out.startswith("\n")
        assert not out.endswith("\n")


class TestWrap:
    def test_hanging_indent_aligns_under_body(self):
        out = wrap_message("alpha " * 40, level="WARNING", width=60)
        lines = out.splitlines()
        assert len(lines) > 1
        indent = len("WARNING - ")
        for cont in lines[1:]:
            assert cont.startswith(" " * indent)
            assert len(cont) <= 60

    def test_single_line_unchanged(self):
        assert wrap_message("short", level="INFO") == "short"


class TestDedup:
    def setup_method(self):
        dedup_reset()

    def test_first_true_repeat_false(self):
        assert dedup("k1") is True
        assert dedup("k1") is False
        assert dedup("k2") is True
        assert dedup_suppressed() == 1

    def test_namespaces_isolated(self):
        assert dedup("k", "a") is True
        assert dedup("k", "b") is True
        assert dedup("k", "a") is False
        assert dedup_suppressed("a") == 1
        assert dedup_suppressed("b") == 0


class TestRunState:
    def setup_method(self):
        reset_run_state()

    def test_stage_warn_flips_status(self, caplog):
        begin_stage("Test stage")
        _RUN.record_warning("PSF", "something odd")
        with caplog.at_level(25, logger="autophot.core.log_format"):
            end_stage("3 stars kept")
        assert "[WARN] 3 stars kept" in caplog.text

    def test_stage_ok_line(self, caplog):
        begin_stage("Calm stage")
        with caplog.at_level(25, logger="autophot.core.log_format"):
            end_stage("all good")
        assert "[OK] all good" in caplog.text

    def test_warnings_recap_merges_repeats(self):
        _RUN.record_warning("ZP", "dup message")
        _RUN.record_warning("ZP", "dup message")
        _RUN.record_warning("WCS", "other")
        from autophot.core.log_format import warnings_summary_lines

        lines = warnings_summary_lines()
        assert "WARNINGS (3)" in lines[1]
        assert any("(x2)" in ln for ln in lines)


class TestWarningTagger:
    def test_tags_and_registers_once(self):
        reset_run_state()
        tagger = WarningTagger()
        rec = logging.LogRecord(
            "autophot.photometry.psf",
            logging.WARNING,
            __file__,
            1,
            "ePSF degraded: %s",
            ("x4",),
            None,
        )
        tagger.filter(rec)
        assert rec.msg == "[PSF] ePSF degraded: x4"
        assert _RUN.warnings[0][0] == "PSF"
        # idempotent on second handler
        tagger.filter(rec)
        assert rec.msg == "[PSF] ePSF degraded: x4"
        assert len(_RUN.warnings) == 1

    def test_existing_tag_kept(self):
        reset_run_state()
        tagger = WarningTagger()
        rec = logging.LogRecord(
            "autophot.pipeline.main",
            logging.WARNING,
            __file__,
            1,
            "[CUSTOM] already tagged",
            None,
            None,
        )
        tagger.filter(rec)
        assert rec.msg.startswith("[CUSTOM]")

    def test_info_untouched(self):
        tagger = WarningTagger()
        rec = logging.LogRecord("x", logging.INFO, __file__, 1, "plain", None, None)
        tagger.filter(rec)
        assert rec.msg == "plain"


class TestHeader:
    def test_run_header_tokens(self):
        out = run_header(
            "1.5.6",
            "img.fits",
            "2026-01-01 00:00:00",
            [("nCPU", "8"), ("catalog", "ps1")],
        )
        for tok in ("v1.5.6", "img.fits", "2026-01-01", "nCPU", "ps1"):
            assert tok in out
        assert out.splitlines()[0] == "=" * LOG_WIDTH


class TestDump:
    def test_no_filehandler_returns_empty(self, tmp_path):
        # Root logger in pytest has no FileHandler -> safe no-op.
        for h in logging.getLogger("").handlers:
            if isinstance(h, logging.FileHandler):
                pytest.skip("file handler present in test session")
        assert write_debug_dump("t", "body") == ""


class TestSecretScrub:
    @pytest.mark.parametrize(
        "text",
        [
            "{'MASTcasjobs_pwd': 'fake_pwd_123', 'n': 1}",
            "MASTcasjobs_pwd=hunter2",
            "api_token: s3cr3t",
            'password="p@ss word"',
        ],
    )
    def test_values_masked(self, text):
        out = scrub_secrets(text)
        for tok in ("fake_pwd_123", "hunter2", "s3cr3t", "p@ss"):
            assert tok not in out

    def test_nonsecret_keys_untouched(self):
        out = scrub_secrets(
            "{'MASTcasjobs_wsid': 1234567890, 'catalog_radius': 0.25}"
        )
        assert "1234567890" in out and "0.25" in out

    def test_formatter_masks(self):
        from autophot.core.functions import PlainFormatter

        rec = logging.LogRecord(
            "x", logging.WARNING, "", 0, "cfg %s",
            ({"MASTcasjobs_pwd": "fake_pwd_123"},), None,
        )
        assert "fake_pwd_123" not in PlainFormatter().format(rec)

    def test_run_header_masks_dict_values(self):
        out = run_header(
            "1.5.6", "i.fits", "now",
            [("Catalog", {"use_catalog": "refcat", "MASTcasjobs_pwd": "x9"})],
        )
        assert "x9" not in out and "refcat" in out
