## Summary

<!-- What changed and why. Link the issue if there is one. -->

## Scientific correctness

<!-- For anything touching photometry, WCS, masking, calibration, or
subtraction: state the invariant preserved and how equivalence was
checked (reference run, tolerance). See agents.md section 2. -->

## Test plan

- [ ] `pytest -m "not slow and not mcmc and not injection"` passes
- [ ] New behavior covered by a test or a documented manual check
- [ ] No changes to `default_input.yml` semantics (or documented below)
