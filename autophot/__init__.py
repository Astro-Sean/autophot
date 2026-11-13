"""AutoPHOT - target-centric photometric pipeline for CCD/NIR imaging.

The public API is re-exported from :mod:`autophot.driver` so that
``from autophot import AutomatedPhotometry`` keeps working for existing
driver scripts. Exports resolve lazily: importing the package does not
pull in the driver (and its heavy dependency chain), and ``python -m
autophot.driver`` does not trip the runpy sys.modules warning.
"""

__all__ = [
    "AutomatedPhotometry",
    "find_variable_sources",
    "inspect_telescope",
    "list_parameters",
    "list_paramters",
    "main",
    "prepare_template_directory",
]

def _resolve_version() -> str:
    # The pyproject.toml beside the code wins over install metadata:
    # editable installs and PYTHONPATH runs keep __file__ in the source
    # tree, where the checkout's declared version is newer than any
    # install-time snapshot.
    try:
        import re
        from pathlib import Path

        pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
        m = re.search(
            r'^\s*version\s*=\s*"([^"]+)"', pyproject.read_text(), re.M
        )
        if m:
            return m.group(1)
    except Exception:
        pass
    try:
        from importlib.metadata import version as _pkg_version

        return _pkg_version("autophot")
    except Exception:
        return "0.0.0"


__version__ = _resolve_version()


def __getattr__(name: str):
    # Delegate to the driver, which applies typo-tolerant fuzzy matching
    # to its public symbols. import_module resolves the submodule directly;
    # ``from . import driver`` here would re-enter __getattr__ and recurse.
    import importlib

    _d = importlib.import_module(".driver", __name__)
    if name == "driver":
        return _d
    return getattr(_d, name)
