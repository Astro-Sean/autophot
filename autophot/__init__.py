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

try:
    from importlib.metadata import version as _pkg_version

    __version__ = _pkg_version("autophot")
except Exception:
    __version__ = "0.0.0"


def __getattr__(name: str):
    # Delegate to the driver, which applies typo-tolerant fuzzy matching
    # to its public symbols. import_module resolves the submodule directly;
    # ``from . import driver`` here would re-enter __getattr__ and recurse.
    import importlib

    _d = importlib.import_module(".driver", __name__)
    if name == "driver":
        return _d
    return getattr(_d, name)
