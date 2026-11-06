"""Tests for autophot._run_main_subprocess (image-level parallel driver).

The suppression flag must be explicit: ProcessPoolExecutor workers do not
reliably inherit module state (spawn/forkserver re-import the module), so
reading QUIET_MODE inside the worker would leak child output to the
terminal on non-fork platforms.
"""

import os
import sys

import pytest

from autophot import _run_main_subprocess


@pytest.fixture()
def fake_main(tmp_path):
    """A stand-in main.py that prints to both streams and exits 0."""
    script = tmp_path / "fake_main.py"
    script.write_text(
        "import sys\n"
        "print('child stdout noise')\n"
        "print('child stderr noise', file=sys.stderr)\n"
    )
    return str(script)


@pytest.fixture()
def input_yaml(tmp_path):
    p = tmp_path / "input.yaml"
    p.write_text("target_name: test\n")
    return str(p)


def test_missing_input_yaml_returns_rc2(tmp_path, fake_main):
    missing = str(tmp_path / "gone.yaml")
    fname, rc = _run_main_subprocess(
        sys.executable, fake_main, "img.fits", missing, False
    )
    assert fname == "img.fits"
    assert rc == 2


def test_return_code_passthrough(tmp_path, input_yaml):
    script = tmp_path / "fail_main.py"
    script.write_text("import sys\nsys.exit(3)\n")
    _, rc = _run_main_subprocess(
        sys.executable, str(script), "img.fits", input_yaml, False
    )
    assert rc == 3


def test_success(fake_main, input_yaml):
    fname, rc = _run_main_subprocess(
        sys.executable, fake_main, "img.fits", input_yaml, False
    )
    assert (fname, rc) == ("img.fits", 0)


def test_suppress_output_silences_child(capfd, fake_main, input_yaml):
    fname, rc = _run_main_subprocess(
        sys.executable,
        fake_main,
        "img.fits",
        input_yaml,
        False,
        suppress_output=True,
    )
    assert rc == 0
    captured = capfd.readouterr()
    assert "child stdout noise" not in captured.out
    assert "child stderr noise" not in captured.err


def test_unsuppressed_child_output_reaches_terminal(
    capfd, fake_main, input_yaml
):
    fname, rc = _run_main_subprocess(
        sys.executable,
        fake_main,
        "img.fits",
        input_yaml,
        False,
        suppress_output=False,
    )
    assert rc == 0
    captured = capfd.readouterr()
    assert "child stdout noise" in captured.out
    assert "child stderr noise" in captured.err


def test_suppression_ignores_quiet_mode(capfd, fake_main, input_yaml):
    """The flag, not module state, controls suppression - regression test
    for spawn-platform workers where QUIET_MODE does not propagate."""
    import autophot

    saved = autophot.QUIET_MODE
    try:
        autophot.QUIET_MODE = False
        _, rc = _run_main_subprocess(
            sys.executable,
            fake_main,
            "img.fits",
            input_yaml,
            False,
            suppress_output=True,
        )
        assert rc == 0
        captured = capfd.readouterr()
        assert "child stdout noise" not in captured.out
    finally:
        autophot.QUIET_MODE = saved


def test_bad_executable_returns_failure(input_yaml):
    _, rc = _run_main_subprocess(
        "/nonexistent/python", "whatever.py", "img.fits", input_yaml, False
    )
    assert rc == 1


def test_template_flag_appended(tmp_path, input_yaml, capfd):
    script = tmp_path / "echo_args.py"
    marker = tmp_path / "argv.txt"
    script.write_text(
        "import sys\n"
        f"open({str(marker)!r}, 'w').write(' '.join(sys.argv[1:]))\n"
    )
    _, rc = _run_main_subprocess(
        sys.executable, str(script), "img.fits", input_yaml, True
    )
    assert rc == 0
    argv = marker.read_text()
    assert "-temp" in argv
    assert "-f img.fits" in argv
