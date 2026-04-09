#!/bin/bash

# Autophot Installation Script - Fixes astroalign dependency issue
# This script addresses the conda installation problem with astroalign 2.6.1

set -e  # Exit on any error

echo "=== Autophot Installation Script ==="
echo "Fixing conda installation issues with astroalign dependency"
echo

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda is not installed or not in PATH"
    echo "Please install Anaconda or Miniconda first"
    exit 1
fi

echo "Step 1: Creating new conda environment..."
conda create -n autophot python=3.10 -y

echo "Step 2: Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate autophot

echo "Step 3: Configuring channels with correct priority..."
conda config --add channels conda-forge
conda config --add channels astro-sean
conda config --set channel_priority strict

echo "Step 4: Installing astroalign first..."
conda install astroalign>=2.6.0,<2.7.0 -y

echo "Step 5: Installing autophot..."
# Try conda first, fallback to pip if needed
if conda install astro-sean::autophot -y; then
    echo "SUCCESS: autophot installed via conda"
else
    echo "Conda installation failed, trying pip fallback..."
    pip install astroalign==2.6.1
    pip install git+https://github.com/Astro-Sean/autophot.git
fi

echo "Step 6: Verifying installation..."
python -c "import astroalign; print(f'astroalign version: {astroalign.__version__}')"
python -c "import main; print('autophot imported successfully')"

if command -v autophot-main &> /dev/null; then
    echo "SUCCESS: autophot command is available"
    autophot-main -h
else
    echo "WARNING: autophot-main command not found, but imports work"
fi

echo
echo "=== Installation Complete ==="
echo "To use autophot in the future, run:"
echo "conda activate autophot"
echo "autophot-main -h"
