#!/usr/bin/env bash
set -euxo pipefail

# Install the package itself.
# --no-deps is CRITICAL: without it, pip installs ALL dependencies from
# pyproject.toml into the build prefix, and conda-build packages them into
# the autophot .conda file.  This caused the package to balloon to 126MB
# with 13,000+ files from scipy, astropy, pandas, sklearn, etc.
# Dependencies are declared in meta.yaml run: section and installed by conda
# when the user runs `conda install autophot`.
#
# sfft and sip_tpv are NOT installed here.  They are pip-only packages with
# compiled Cython extensions (.so files) that are platform-specific.  Bundling
# them would break the noarch: python declaration.  Users install them
# separately after conda install:
#   pip install sfft==1.7.3 sip_tpv==1.1
${PYTHON} -m pip install --no-deps . -vv
