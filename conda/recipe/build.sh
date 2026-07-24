#!/usr/bin/env bash
set -euxo pipefail

# Install pip-only dependencies not available on conda-forge
${PYTHON} -m pip install sfft==1.7.3 sip_tpv==1.1 -vv

# Install the package itself
${PYTHON} -m pip install . -vv
