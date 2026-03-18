#!/bin/bash
#SBATCH --job-name=autophot
#SBATCH --output=autophot_%j.out
#SBATCH --error=autophot_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --partition=normal
# -----------------------------------------------------------------------------
# Run AutoPhOT VIPER pipeline (template subtraction with SFFT).
#
# Usage: sbatch run_autophot_VIPER.sbatch
#   Or:   sbatch --partition=your_partition run_autophot_VIPER.sbatch
#   Or:   AUTOPHOT_DIR=/path/to/autophot CONDA_ENV=astro sbatch run_autophot_VIPER.sbatch
#
# Edit AUTOPHOT_DIR and CONDA_ENV below for your HPC; keep thread limits for SFFT.
# -----------------------------------------------------------------------------

# Where the autophot repo and run_autophot_VIPER.py live (edit if needed)
AUTOPHOT_DIR="${AUTOPHOT_DIR:-/ptmp/sbrennan/autophot}"
# Conda env with autophot deps (edit if needed)
CONDA_ENV="${CONDA_ENV:-astro}"

# Single thread/process to avoid libgomp and semaphore issues with SFFT
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export OPENMP_NUM_THREADS=1

# Activate conda (adjust to your cluster: module load conda, or source conda.sh)
if command -v conda &>/dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV" || true
elif [ -n "$CONDA_PREFIX" ]; then
    : # already in a conda env
fi

cd "$AUTOPHOT_DIR" || exit 1

# SFFT path uses nCPU=1 internally when method is sfft; --ncpu 1 keeps one image at a time
python run_autophot_VIPER.py --ncpu 1
