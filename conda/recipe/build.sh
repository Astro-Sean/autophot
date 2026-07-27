#!/usr/bin/env bash
set -euxo pipefail

# Install pip-only dependencies not available on conda-forge.
# conda-build runs pip with --no-index by default, blocking PyPI access.
# We explicitly set --index-url to allow downloading sfft (sdist only, no wheels).
${PYTHON} -m pip install --index-url https://pypi.org/simple sfft==1.7.3 sip_tpv==1.1 -vv

# Install the package itself
${PYTHON} -m pip install . -vv
