#!/usr/bin/env bash
set -euxo pipefail

# Install pip-only dependencies not available on conda-forge.
# conda-build sets PIP_NO_INDEX=1 and PIP_NO_DEPENDENCIES=1 as environment
# variables, which override --index-url on the CLI and block all PyPI access.
# We must unset them so pip can actually reach PyPI for sfft (sdist only).
unset PIP_NO_INDEX
unset PIP_NO_DEPENDENCIES

${PYTHON} -m pip install --no-build-isolation --index-url https://pypi.org/simple sfft==1.7.3 sip_tpv==1.1 -vv

# Install the package itself
${PYTHON} -m pip install . -vv
