"""Unified ``autophot`` command-line entry point.

Dispatches to the pipeline stages. The historical console scripts
(``autophot-main``, ``autophot-driver``, ``autophot-inspect-telescope``,
``autophot-gaia-curves``) remain available unchanged.
"""

if __package__ in (None, ""):
    # Support "python <path>" execution: expose the repo root and
    # anchor the package so relative imports resolve.
    import os as _os
    import sys as _sys

    _sys.path.insert(
        0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    )
    __package__ = "autophot"

import argparse
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="autophot",
        description="AutoPHOT photometry pipeline.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("run", help="Per-image pipeline (autophot.pipeline.main)")
    sub.add_parser("driver", help="Full AutomatedPhotometry driver")
    sub.add_parser("batch", help="Batch driver over many images")
    sub.add_parser(
        "inspect-telescope", help="Build/update telescope.yml from FITS headers"
    )
    sub.add_parser(
        "gaia-curves", help="Build a custom catalog from Gaia XP curves"
    )

    args, rest = parser.parse_known_args(argv)

    if args.command == "run":
        # run_photometry parses sys.argv; drop the subcommand token.
        from .pipeline.main import run_photometry

        sys.argv = [sys.argv[0], *rest]
        return run_photometry()
    if args.command == "driver":
        from .driver import main as _main

        return _main(rest)
    if args.command == "batch":
        from .batch_main import main as _main

        return _main(rest)
    if args.command == "inspect-telescope":
        from .pipeline.telescope import main as _main

        return _main(rest)

    # gaia-curves parses sys.argv like run.
    from .gaia_curves.cli import main as _main

    sys.argv = [sys.argv[0], *rest]
    return _main()


if __name__ == "__main__":
    raise SystemExit(main())
