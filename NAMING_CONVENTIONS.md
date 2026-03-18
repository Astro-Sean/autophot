# Variable and symbol naming

AutoPHOT follows PEP 8 and these conventions for clarity and consistency.

## Style

- **Variables and functions:** `snake_case` (e.g. `science_file`, `prepare_template`, `output_dir_suffix`).
- **Classes:** `CapitalizedWords` (e.g. `FitsInfo`, `BackgroundSubtractor`).
- **Constants:** `UPPER_SNAKE_CASE` or `UPPER` when truly constant (e.g. `INSTRUMENT_BLOCK_KEYS`).
- **Private or internal:** Leading underscore when not part of the public API (e.g. `_ncpu`, `_style`).

## Clarity

- Prefer **descriptive names** over short ones: `file_path` or `science_path` over `fpath`, `working_dir` over `wdir` when the scope is large; short names are fine in small scopes (e.g. loop variables `i`, `k`).
- **Paths:** Suffix or name by role: `*_path` for file paths, `*_dir` for directories, `*_suffix` for string suffixes (e.g. `output_dir_suffix` for `"_REDUCED"`).
- **Booleans:** Use `is_*`, `has_*`, or `*_flag` where it helps (e.g. `prepare_template` for a “prepare template” flag).
- **Config:** `input_yaml` is the loaded config dict; names that mirror YAML keys (e.g. `fits_dir`, `outdir_name`) are kept for consistency with config files.

## Names to prefer or avoid

| Prefer | Avoid | Reason |
|--------|--------|--------|
| `science_file` | `scienceFile` | PEP 8 snake_case |
| `prepare_template` | `prepareTemplate` | PEP 8 |
| `output_dir_suffix` | `newDir` | Describes content ("_REDUCED") |
| `fits_basename` | `baseDir` | Actually basename of fits_dir |
| `reduced_dir_name` | `workLoc` | e.g. "SN1987A_REDUCED" |
| `filename_with_ext` | `base_wext` | Clearer than “base with extension” |
| `file_extension` | `fname_ext` | Clearer |
| `file_path` / `science_path` | `fpath` | Clearer in large scope |
| `working_dir` | `wdir` | Clearer when used in many places |

YAML keys (e.g. `fits_dir`, `wdir`, `outdir_name`) are not changed so that config and code stay in sync.
