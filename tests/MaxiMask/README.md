# MaxiMask end-to-end test

Exercises [MaxiMask](https://github.com/mpaillassa/MaxiMask)
(Paillassa, Bertin & Bouy 2020, arXiv:1907.08298) on a synthetic CCD
frame with known injected contaminants, and produces paper-style
diagnostic plots.

Requires the `astro` conda environment (`MaxiMask` + `tensorflow`).

## Contents

- `synthetic_image.py` - builds a 1000x1400 float32 frame (the network
  only processes HDUs larger than its 400 px tile size) with field
  stars plus injected contaminants matched to the MaxiMask classes:
  cosmic rays (CR), hot/dead columns (HCL/DCL), hot/dead pixels
  (HP/DP), a satellite trail (TRL), a saturated star with diffraction
  spikes and a bleed trail (SAT/SP), a smooth nebulosity blob (NEB),
  a fringing patch (FR) and an overscan strip (OV). Every feature is
  recorded in a boolean truth mask.
- `run_maximask.py` - writes the FITS inputs, runs the real
  TensorFlow inference (prior-modified probability maps, no
  thresholding so continuous maps can be plotted), applies the shipped
  per-class thresholds, and writes plots + a recall summary into
  `output/`.
- `test_maximask.py` - pytest wrapper (`slow` marker; run with
  `--run-slow`) asserting per-class recall, a low false-positive rate
  on clean sky, and the `background.use_maximask` integration.
- `maximask_on_fits.py` - runs MaxiMask on any real FITS image
  (in-memory inference; no `.mask.fits` side effect) and writes
  overlay/probmap/zoom plots to `output/real/<stem>/`.
- `output/` - generated artifacts: input/truth/mask FITS, PNG plots,
  `summary.txt`. FITS products are git-ignored.

## Pipeline integration

`background.use_maximask: true` in `databases/default_input.yml`
enables the stage inside `BackgroundSubtractor.remove()`: detections
from `background.maximask_classes` (default CR/HCL/DCL/HP/DP/P/TRL/
SAT/SP/OV; BG/BBG/FR/NEB excluded) are unioned into
`hardware_defects_mask` and seeded into the source mask. CPU-only by
default (`maximask_allow_gpu: false` hides GPUs from TensorFlow). If
the optional dependencies or inference fail, the stage warns and the
heuristic masks are used unchanged.

## Running

Standalone (regenerates only missing artifacts, `--force` reruns):

```bash
python tests/MaxiMask/run_maximask.py [--force]
```

Via pytest:

```bash
python -m pytest tests/MaxiMask/test_maximask.py --run-slow -v
```

## Plots

- `maximask_input.png` - the input frame with ground-truth contours
  per contaminant class.
- `maximask_overlay.png` - thresholded MaxiMask detections overlaid on
  the image.
- `maximask_probmaps.png` - all 14 class probability maps plus the
  input and a flagged-class count map.
- `maximask_zooms.png` - zoomed panels comparing detections (filled)
  against injected truth (white contours).

## Notes

Measured recall on this frame (seed 7): TRL 0.88, SAT 0.91, CR 0.83,
HCL 0.76, DCL 0.67, DP 0.82, HP 0.32, with a 0.4% defect-class false-
positive rate on clean sky. FR/NEB/OV/SP are injected and plotted but
not asserted: the synthetic versions are weak proxies for the real
signatures the network learned (the smooth nebulosity is absorbed by
MaxiMask's internal background normalisation or flagged as BBG, and
simplified diffraction spikes are only partially flagged).
