# Logging and printed messages

AutoPHOT uses a consistent layout and language for all log and print output.

## Layout

- **One message per call.** No multi-line messages unless a single logical sentence (e.g. border_msg).
- **Format:** Prefer `logging.info("Context: detail.")` or `logging.info("Context: %s", value)`. Use %-style for variables in logging so levels and filtering work correctly.
- **Scripts using print():** Use a single line per status. For CLI scripts (e.g. run_autophot_VIPER, batch_main), use clear prefixes only when helpful (e.g. "Error:", "Warning:").

## Language

- **Case:** Sentence case for messages (e.g. "Pixel scale from WCS: 0.4 arcsec/pixel").
- **Terms:** Use consistent terminology: "zeropoint", "FITS", "working directory", "template", "light curve".
- **Status:** Use past/imperative consistently: "Skipping ...", "Using ...", "Removed ...", "Wrote ...", "Updated ...", "Done."
- **Errors:** Start with "Error:" for user-facing errors; use `logging.error(...)` in pipeline code.
- **Warnings:** Start with "Warning:" when printing to stderr; use `logging.warning(...)` in pipeline code.

## Where messages go

- **Pipeline (main.py and modules):** Use the `logging` module (info, warning, error). Configuration is set in main.py (file + console).
- **CLI / helper scripts:** May use `print()` for direct user feedback. Use stderr for errors/warnings: `print("Error: ...", file=sys.stderr)`.
