#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Image masking utilities for AutoPHOT.

Wraps the ``maximask`` external tool to generate pixel-quality masks
(cosmic rays, hot/dead columns, saturated pixels, etc.) and provides
helpers to combine selected mask layers into a single boolean array.

The mask layer codes follow the maximask convention:

    =====  ============================================
    Code   Layer
    =====  ============================================
    CR     Cosmic rays
    HCL    Hot columns / lines
    DCL    Dead columns / lines / clusters
    BG     Background artefacts
    DP     Dead pixels
    P      Persistence
    TRL    Trails
    FR     Fringe patterns
    NEB    Nebulosities
    SAT    Saturated pixels
    SP     Diffraction spikes
    OV     Overscanned pixels
    BBG    Bright background pixels
    =====  ============================================
"""

import logging
import os
import subprocess

import numpy as np
from astropy.io import fits

logger = logging.getLogger(__name__)


def append_id(filename):
    """Insert ``.mask`` before the file extension.

    Parameters
    ----------
    filename : str
        Original filename (e.g. ``image.fits``).

    Returns
    -------
    str
        Masked filename (e.g. ``image.mask.fits``).
    """
    return "{0}{2}.{1}".format(*filename.rsplit(".", 1) + [".mask"])


def is_program_installed(program_name):
    """Check whether *program_name* is on the system ``PATH``.

    Parameters
    ----------
    program_name : str
        Executable name to search for (e.g. ``"maximask"``).

    Returns
    -------
    bool
        ``True`` if the program is found, ``False`` otherwise.
    """
    try:
        result = subprocess.run(
            ["which", program_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return result.returncode == 0
    except Exception as exc:
        logger.error(
            "Error while checking for program '%s': %s",
            program_name,
            exc,
            exc_info=True,
        )
        return False


def run_maximask_with_file(filename):
    """Run ``maximask`` on *filename* and return the output mask path.

    Parameters
    ----------
    filename : str
        Path to the FITS image to mask.

    Returns
    -------
    str or None
        Path to the generated ``*.mask.fits`` file, or ``None`` if
        maximask is not installed or fails.
    """
    program_name = "maximask"
    if is_program_installed(program_name):
        try:
            # Run maximask with the given filename and suppress output
            subprocess.run(
                [program_name, filename],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            expected_fname = append_id(filename)

            if os.path.exists(expected_fname):
                logger.info(
                    "Image mask created using %s: %s", program_name, expected_fname
                )
                return expected_fname

            raise RuntimeError("maximask output file was not created.")

        except Exception as exc:
            logger.error(
                "An error occurred while running %s on '%s': %s",
                program_name,
                filename,
                exc,
                exc_info=True,
            )
    else:
        logger.warning(
            "%s is not installed; cannot generate image mask.", program_name
        )
        return None


def create_image_mask(filename, layers=None):
    """Create a combined integer mask from a maximask output file.

    Parameters
    ----------
    filename : str or None
        Path to the ``*.mask.fits`` file produced by :func:`run_maximask_with_file`.
        If ``None``, returns ``None`` immediately.
    layers : list of str, optional
        Maximask layer codes to include (default ``["CR"]`` — cosmic rays only).
        See the module docstring for the full list of codes.

    Returns
    -------
    numpy.ndarray or None
        2-D integer array (1 = masked, 0 = clean) with the same shape as
        the input image, or ``None`` if *filename* is ``None``.
    """
    if layers is None:
        layers = ["CR"]

    from functions import log_step

    if filename is None:
        return None

    fname = os.path.basename(filename)
    logger.info(log_step(f"Mask: {fname}"))

    maximask_masks = {
        "CR": [0, "Cosmic Rays"],
        "HCL": [1, "Hot Columns/Lines"],
        "DCL": [2, "Dead Columns/Lines/Clusters"],
        "BG": [3, "Background"],
        "DP": [4, "Dead Pixels"],
        "P": [5, "Persistence"],
        "TRL": [6, "TRaiLs"],
        "FR": [7, "FRinge patterns"],
        "NEB": [8, "NEBulosities"],
        "SAT": [9, "SATurated pixels"],
        "SP": [10, "diffraction SPikes"],
        "OV": [11, "OVerscanned pixels"],
        "BBG": [12, "Bright BackGround pixel"],
    }

    with fits.open(filename) as hdul:
        # Iterate through each HDU (Header Data Unit)
        imgs = hdul[0].data
        # Convert integer dtypes to float32 to preserve NaNs (chip gaps)
        if imgs.dtype.kind != 'f':
            imgs = imgs.astype(np.float32)

    selected_layers = [maximask_masks[i] for i in layers]

    mask = np.zeros(imgs[0].shape)

    for layer in selected_layers:
        logger.info("\t> Adding %s layer to mask", layer[1])
        mask += imgs[layer[0]]

    mask[mask > 1] = 1
    return mask.astype(int)
