# Changelog

All notable changes to AutoPhOT are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Changed

- Repository restructured into a real ``autophot`` package. All pipeline
  modules now live under ``autophot/`` in domain subpackages
  (``pipeline``, ``photometry``, ``calibration``, ``subtraction``,
  ``analysis``, ``services``, ``core``, ``data``, ``gaia_curves``).
- ``databases/`` moved to ``autophot/data/`` and is resolved via
  ``importlib.resources`` (``functions.data_path``).
- ``check.py`` renamed to ``autophot.pipeline.telescope``;
  ``inspect_telescope.py`` merged into it.
- Installing the package no longer exposes ~25 generically named
  top-level modules (``functions``, ``check``, ``catalog``, ``main``,
  ...) in the global namespace.
- New unified ``autophot`` CLI with subcommands; the historical console
  scripts (``autophot-main``, ``autophot-driver``,
  ``autophot-inspect-telescope``, ``autophot-gaia-curves``) are unchanged.
- ``scripts/`` and ``docs/`` (minus the arXiv PDF) are now tracked.

### Removed

- Legacy flat-layout imports removed. ``import main``,
  ``from functions import ...``, ``from utils.run_sfft import ...``,
  ``import check``, ``import autophot_gaia_curves`` etc. no longer work.
  Use ``autophot.<sub>.<module>`` paths, e.g.
  ``from autophot.photometry.psf import PSF`` or
  ``from autophot.subtraction.run_sfft import run_sfft``.
- ``python main.py`` is replaced by ``python -m autophot.pipeline.main``
  (the driver already launches children this way).
