#!/usr/bin/env bash
set -euo pipefail

# Build and upload the AutoPHOT conda package to Anaconda.org.
#
# Usage examples:
#   ./scripts/build_and_upload_conda.sh
#   ./scripts/build_and_upload_conda.sh --label dev
#   ./scripts/build_and_upload_conda.sh --new-version 0.1.1
#
# This script also:
#   - Bumps the version in pyproject.toml and conda/recipe/meta.yaml
#   - Auto-regenerates requirements.txt from pyproject.toml dependencies
#   - Validates environment.yml and meta.yaml for dependency drift (warn-only)
#   - Tests the built package in a fresh conda env before uploading; the
#     test env is always removed afterwards, on success or failure
#   - Removes stale autophot/autophot-object installs and all build
#     artifacts (including the just-uploaded package) after a successful
#     upload so old code cannot shadow the new package
#
# Requirements:
#   - conda
#   - conda-build
#   - anaconda-client
#   - anaconda login already completed (or ANACONDA_API_TOKEN exported)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

RECIPE_PATH="conda/recipe"
CONDA_BUILD_ENV=""
CHANNEL_USERNAME="astro-sean"
LABEL=""
NEW_VERSION=""
AUTO_VERSION=0
TEST_PYTHON="3.12"

usage() {
  cat <<'EOF'
Build and upload AutoPHOT conda package.

Options:
  --channel-username NAME   Optional. Anaconda.org username/channel (default: astro-sean).
  --env NAME                Optional conda env used for build/upload tools.
  --recipe PATH             Optional recipe path (default: conda/recipe).
  --label NAME              Optional Anaconda label (e.g. dev, main).
  --new-version VERSION     Set an explicit new version (X.Y.Z) non-interactively.
  --update-version          Auto-increment the patch version (e.g. 1.0.1 -> 1.0.2) without prompting.
  --test-python VERSION     Python version for the test env (default: 3.12).
  -h, --help                Show this help.
EOF
}

log() {
  printf "[build-upload] %s\n" "$*"
}

die() {
  printf "[build-upload] ERROR: %s\n" "$*" >&2
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --channel-username)
      CHANNEL_USERNAME="${2:-}"
      shift 2
      ;;
    --env)
      CONDA_BUILD_ENV="${2:-}"
      shift 2
      ;;
    --recipe)
      RECIPE_PATH="${2:-}"
      shift 2
      ;;
    --label)
      LABEL="${2:-}"
      shift 2
      ;;
    --new-version)
      NEW_VERSION="${2:-}"
      shift 2
      ;;
    --update-version)
      AUTO_VERSION=1
      shift
      ;;
    --test-python)
      TEST_PYTHON="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "Unknown argument: $1"
      ;;
  esac
done

cd "${REPO_ROOT}"

[[ -d "${RECIPE_PATH}" ]] || die "Recipe path not found: ${RECIPE_PATH}"

if ! command -v conda >/dev/null 2>&1; then
  die "conda not found in PATH."
fi

run_in_env() {
  if [[ -n "${CONDA_BUILD_ENV}" ]]; then
    conda run --no-capture-output -n "${CONDA_BUILD_ENV}" "$@"
  else
    "$@"
  fi
}

validate_version() {
  local v="$1"
  [[ "${v}" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]
}

version_gt() {
  python - "$1" "$2" <<'PY'
import sys

def parse(v: str):
    parts = v.strip().split(".")
    if len(parts) != 3:
        raise ValueError("invalid version format")
    return tuple(int(p) for p in parts)

new_v = parse(sys.argv[1])
cur_v = parse(sys.argv[2])
print("1" if new_v > cur_v else "0")
PY
}

get_current_version() {
  python - <<'PY'
import re
from pathlib import Path
p = Path("pyproject.toml")
text = p.read_text(encoding="utf-8")
m = re.search(r'^version\s*=\s*"([^"]+)"\s*$', text, flags=re.M)
if not m:
    raise SystemExit("Could not find version in pyproject.toml")
print(m.group(1))
PY
}

increment_patch_version() {
  python - "$1" <<'PY'
import sys
v = sys.argv[1].strip().split(".")
if len(v) != 3:
    raise SystemExit("Invalid semantic version (expected X.Y.Z)")
major, minor, patch = map(int, v)
print(f"{major}.{minor}.{patch + 1}")
PY
}

set_version_everywhere() {
  local target="$1"
  python - "$target" <<'PY'
import re
import sys
from pathlib import Path

new_v = sys.argv[1]
pyproject = Path("pyproject.toml")
meta = Path("conda/recipe/meta.yaml")

py_text = pyproject.read_text(encoding="utf-8")
py_text_new, py_count = re.subn(
    r'(^version\s*=\s*")[^"]+("\s*$)',
    rf'\g<1>{new_v}\2',
    py_text,
    count=1,
    flags=re.M,
)
if py_count != 1:
    raise SystemExit("Failed to update version in pyproject.toml")
pyproject.write_text(py_text_new, encoding="utf-8")

meta_text = meta.read_text(encoding="utf-8")
meta_text_new, meta_count = re.subn(
    r'(\{%\s*set\s+version\s*=\s*")[^"]+("\s*%\})',
    rf'\g<1>{new_v}\2',
    meta_text,
    count=1,
)
if meta_count != 1:
    raise SystemExit("Failed to update version in conda/recipe/meta.yaml")
meta.write_text(meta_text_new, encoding="utf-8")
PY
}

# -----------------------------------------------------------------------------
# Remove stale installed copies and build artifacts so old code cannot shadow
# the new package.  The package was renamed (autophot-object -> autophot), so
# an env can carry BOTH installs: the old-name package leaves its flat .py
# files in site-packages where they import ahead of the new code.  Older
# autophot= conda installs are removed too; the build above supersedes them.
# Editable (pip -e) installs are left alone: they point at this repo.
# -----------------------------------------------------------------------------
remove_stale_autophot_code() {
  local prefix pkg_name pkg_version pkg_channel stale_dir pdir

  while IFS= read -r prefix; do
    [[ -n "${prefix}" && -d "${prefix}" ]] || continue

    while IFS=$'\t' read -r pkg_name pkg_version pkg_channel; do
      [[ -n "${pkg_name}" ]] || continue
      if [[ "${pkg_channel}" == "pypi" ]]; then
        [[ "${pkg_name}" != "autophot" || "${pkg_version}" == "${SELECTED_VERSION}" ]] && continue
        # A pip install only shadows the repo when it is not an editable
        # install of it.
        if conda run -p "${prefix}" pip list --editable 2>/dev/null | grep -qi '^autophot'; then
          continue
        fi
        log "Removing stale pip install ${pkg_name}=${pkg_version} from ${prefix}"
        conda run -p "${prefix}" pip uninstall -y autophot >/dev/null 2>&1 ||
          log "WARNING: could not uninstall autophot in ${prefix}; remove it manually."
        continue
      fi
      if [[ "${pkg_name}" == "autophot-object" ]] ||
         [[ "${pkg_name}" == "autophot" && "${pkg_version}" != "${SELECTED_VERSION}" ]]; then
        log "Removing stale conda package ${pkg_name}=${pkg_version} from ${prefix}"
        conda remove -y -p "${prefix}" "${pkg_name}" >/dev/null 2>&1 ||
          log "WARNING: could not remove ${pkg_name} from ${prefix}; remove it manually."
      fi
    done < <(conda list -p "${prefix}" --json 2>/dev/null | python -c '
import json, sys
try:
    pkgs = json.load(sys.stdin)
except Exception:
    pkgs = []
for p in pkgs:
    name = p.get("name", "")
    if name not in ("autophot", "autophot-object"):
        continue
    src = (p.get("channel") or "") + (p.get("platform") or "")
    chan = "pypi" if "pypi" in src else "conda"
    print(name + "\t" + p.get("version", "") + "\t" + chan)
')

    # Extractions in an env-local pkgs cache and artifacts in an env-local
    # conda-bld keep serving the old code to --use-local installs.  The
    # just-built artifact is removed too: it is on Anaconda.org already
    # and the local copy is a stale file.
    for stale_dir in "${prefix}/pkgs" "${prefix}/conda-bld"; do
      [[ -d "${stale_dir}" ]] || continue
      find "${stale_dir}" -maxdepth 2 \
        \( -name 'autophot-*' -o -name 'autophot-object-*' \) \
        -exec rm -rf {} + 2>/dev/null || true
    done
    # Re-index so the local channel stops listing the deleted artifacts.
    [[ -d "${prefix}/conda-bld" ]] &&
      run_in_env conda index "${prefix}/conda-bld" >/dev/null 2>&1 || true
  done < <(conda env list --json 2>/dev/null | python -c '
import json, sys
try:
    envs = json.load(sys.stdin).get("envs", [])
except Exception:
    envs = []
print("\n".join(envs))
')

  # pkgs caches that are not env prefixes (e.g. ~/.conda/pkgs) are not covered
  # by the env loop.
  while IFS= read -r pdir; do
    [[ -d "${pdir}" ]] || continue
    find "${pdir}" -maxdepth 2 \
      \( -name 'autophot-*' -o -name 'autophot-object-*' \) \
      -exec rm -rf {} + 2>/dev/null || true
  done < <(conda config --show pkgs_dirs --json 2>/dev/null | python -c '
import json, sys
try:
    print("\n".join(json.load(sys.stdin).get("pkgs_dirs", [])))
except Exception:
    pass
')
}

# -----------------------------------------------------------------------------
# Regenerate requirements.txt from pyproject.toml dependencies.
# requirements.txt is a pure mirror of [project.dependencies] — if you edit
# dependencies, do it in pyproject.toml and this function keeps requirements.txt
# in sync.  Preserves the header comment block and optional-deps comment section.
# -----------------------------------------------------------------------------
sync_requirements() {
  python - <<'PY'
import tomllib
from pathlib import Path

pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
deps = pyproject["project"]["dependencies"]

# Separate SFFT-support deps (matching the existing grouping in requirements.txt)
sfft_deps = {"fastremap", "llvmlite", "numba", "pyfftw", "sep"}
core = []
sfft = []
for d in deps:
    name = d.split(">")[0].split("<")[0].split("=")[0].split("!")[0].split("~")[0].split(";")[0].strip().lower()
    if name in sfft_deps:
        sfft.append(d)
    else:
        core.append(d)

# Preserve original order as much as possible (pyproject.toml order)
lines = [
    "# AutoPhOT requirements (mirrors pyproject.toml dependencies)",
    "# Install: pip install -r requirements.txt",
    "# Conda:   conda install -c conda-forge -c astro-sean autophot",
    "# AUTO-GENERATED by scripts/build_and_upload_conda.sh — do not edit manually.",
    "",
]
lines.extend(core)
lines.append("")
lines.append("# SFFT dependencies")
lines.extend(sfft)
lines.append("")
lines.append("# Optional (not required for core pipeline):")
lines.append("# aafitrans==0.2.1")
lines.append("# mastcasjobs==0.0.7")
lines.append("")

Path("requirements.txt").write_text("\n".join(lines), encoding="utf-8")
print("synced")
PY
}

# -----------------------------------------------------------------------------
# Validate that environment.yml and meta.yaml contain all packages listed in
# pyproject.toml [project.dependencies].  Warns on drift but does NOT fail the
# build — some files intentionally omit pip-only packages (sfft, sip_tpv in
# meta.yaml) or include extra conda-only packages (astromatic-*).
# -----------------------------------------------------------------------------
validate_dependency_files() {
  python - <<'PY'
import re
import sys
import tomllib
from pathlib import Path

pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
py_deps = {}
for d in pyproject["project"]["dependencies"]:
    name = re.split(r"[<>=!~;]", d.strip())[0].lower()
    py_deps[name] = d

def extract_pkg_name(spec):
    """Extract package name from a conda/pip spec line like 'numpy==2.0' or '- sfft==1.7.3'."""
    s = spec.strip()
    if s.startswith("- "):
        s = s[2:]
    s = s.split("#")[0].strip()
    if not s or s == "pip" or s == "pip:":
        return None
    return re.split(r"[<>=!~;]", s)[0].strip().lower()

# --- environment.yml ---
env_text = Path("environment.yml").read_text(encoding="utf-8")
env_pkgs = set()
for line in env_text.splitlines():
    s = line.split("#")[0].strip()
    if not s:
        continue
    # Skip structural lines
    if re.match(r'^(name|channels|dependencies):', s) or s.endswith(":"):
        continue
    if s.startswith("- "):
        pkg = extract_pkg_name(s)
        if pkg:
            env_pkgs.add(pkg)
    elif not s.startswith("-"):
        # Bare package line like "python==3.11.*"
        pkg = extract_pkg_name(s)
        if pkg:
            env_pkgs.add(pkg)

missing_env = sorted(set(py_deps) - env_pkgs)
if missing_env:
    print(f"WARN_ENV:{','.join(missing_env)}")

# --- meta.yaml run: section ---
meta_text = Path("conda/recipe/meta.yaml").read_text(encoding="utf-8")
meta_pkgs = set()
in_run = False
for line in meta_text.splitlines():
    s = line.strip()
    if s == "run:":
        in_run = True
        continue
    if s in ("test:", "about:"):
        in_run = False
        continue
    if in_run and s.startswith("- "):
        pkg = extract_pkg_name(s)
        if pkg:
            meta_pkgs.add(pkg)

# sfft and sip_tpv are intentionally pip-only in meta.yaml
intentionally_missing_meta = {"sfft", "sip_tpv"}
missing_meta = sorted((set(py_deps) - meta_pkgs) - intentionally_missing_meta)
if missing_meta:
    print(f"WARN_META:{','.join(missing_meta)}")
PY
}

log "Checking required tools..."
run_in_env conda-build --version >/dev/null
# anaconda-client CLI does not have --version; check availability via --help.
run_in_env anaconda --help >/dev/null 2>&1 || die "anaconda-client not available (needed for upload)."

# Verify that the libmamba solver plugin is actually installed in the
# build environment.  conda config may say solver=libmamba, but if the
# conda-libmamba-solver package is missing from the env, conda-build
# silently falls back to the classic solver, which hangs indefinitely
# on recipes with 20+ run dependencies.
log "Verifying libmamba solver availability..."
# Probe the interpreter that actually runs conda-build.  A stray
# ~/.local/bin/python ahead of the conda env on PATH would otherwise
# make this check test the wrong environment.
if [[ -n "${CONDA_BUILD_ENV}" ]]; then
  _solver_env_name="${CONDA_BUILD_ENV}"
  _solver_probe=(conda run --no-capture-output -n "${CONDA_BUILD_ENV}" python)
else
  _cb_bin="$(command -v conda-build || true)"
  _solver_probe=("$(dirname "${_cb_bin:-$(command -v python)}")/python")
  _cb_env_dir="$(dirname "$(dirname "${_cb_bin:-python}")")"
  if [[ "$(basename "$(dirname "${_cb_env_dir}")")" == "envs" ]]; then
    _solver_env_name="$(basename "${_cb_env_dir}")"
  else
    _solver_env_name="base"
  fi
fi
if ! "${_solver_probe[@]}" -c "import conda_libmamba_solver" 2>/dev/null; then
  log "WARNING: conda-libmamba-solver is not installed in the build environment."
  log "         The classic solver will hang on complex recipes (20+ deps)."
  log "         Install it with: conda install -n ${_solver_env_name} -c conda-forge conda-libmamba-solver"
  die "Install conda-libmamba-solver in the build environment and retry."
fi
log "libmamba solver confirmed available."

# Refresh the README citation count on every build.  The helper is
# warn-only (network or missing API token never blocks a build).
"${_solver_probe[@]}" "${SCRIPT_DIR}/update_readme_citations.py" || \
  log "WARNING: README citation-count update skipped."

# Regenerate the citation growth figure alongside the count.  Also
# warn-only: it needs matplotlib, which the build env may not have.
"${_solver_probe[@]}" "${SCRIPT_DIR}/plot_citation_growth.py" || \
  log "WARNING: citation figure update skipped."

# Clear stale repodata cache if the TTL is too short (default 1 second
# forces re-downloading 400+ MB repodata on every conda command).
_REPODATA_TTL="$(run_in_env python -c "import conda.base.context as c; print(c.context.local_repodata_ttl)" 2>/dev/null || echo "0")"
if [[ "${_REPODATA_TTL}" -lt 300 ]] 2>/dev/null; then
  log "Increasing repodata cache TTL (was ${_REPODATA_TTL}s) to 3600s to avoid re-downloading 400+ MB repodata..."
  run_in_env conda config --set local_repodata_ttl 3600 2>/dev/null || true
fi

# Warn if the git working tree is dirty — the version bump and requirements
# sync write to pyproject.toml, meta.yaml, and requirements.txt, which could
# mix with uncommitted changes.
if command -v git >/dev/null 2>&1 && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  if ! git diff --quiet -- pyproject.toml conda/recipe/meta.yaml requirements.txt 2>/dev/null; then
    log "WARNING: pyproject.toml, meta.yaml, or requirements.txt has uncommitted changes."
    log "         The version bump and requirements sync will modify these files. Commit or stash first."
  fi
fi

log "---"
CURRENT_VERSION="$(get_current_version)"

# Verify pyproject.toml and meta.yaml versions are in sync before proceeding.
META_VERSION="$(python - <<'PY'
import re
from pathlib import Path
m = re.search(r'\{%\s*set\s+version\s*=\s*"([^"]+)"\s*%\}', Path("conda/recipe/meta.yaml").read_text())
print(m.group(1) if m else "")
PY
)"
if [[ -n "${META_VERSION}" && "${META_VERSION}" != "${CURRENT_VERSION}" ]]; then
  die "Version mismatch: pyproject.toml=${CURRENT_VERSION}, meta.yaml=${META_VERSION}. Please sync before building."
fi
SUGGESTED_VERSION="$(increment_patch_version "${CURRENT_VERSION}")"
if [[ -n "${NEW_VERSION}" ]]; then
  SELECTED_VERSION="${NEW_VERSION}"
elif [[ "${AUTO_VERSION}" -eq 1 ]]; then
  SELECTED_VERSION="${SUGGESTED_VERSION}"
  log "Auto-incrementing version: ${CURRENT_VERSION} -> ${SELECTED_VERSION}"
else
  read -r -p "[build-upload] Current version is ${CURRENT_VERSION}. Suggested next version is ${SUGGESTED_VERSION}. Press Enter to accept or type a new version: " SELECTED_VERSION
  SELECTED_VERSION="${SELECTED_VERSION:-${SUGGESTED_VERSION}}"
fi
if ! validate_version "${SELECTED_VERSION}"; then
  die "Invalid version '${SELECTED_VERSION}'. Expected format: X.Y.Z"
fi
if [[ "${SELECTED_VERSION}" == "${CURRENT_VERSION}" ]]; then
  die "Version must change before upload. Current version is ${CURRENT_VERSION}."
fi
if [[ "$(version_gt "${SELECTED_VERSION}" "${CURRENT_VERSION}")" != "1" ]]; then
  die "New version (${SELECTED_VERSION}) must be greater than current version (${CURRENT_VERSION})."
fi
log "Updating version: ${CURRENT_VERSION} -> ${SELECTED_VERSION}"

# -----------------------------------------------------------------------------
# Back up files that will be modified (version bump + requirements sync) so
# they can be restored if the build, test, or upload fails.  This prevents the
# repo from being left with a bumped version but no uploaded package.
# -----------------------------------------------------------------------------
BACKUP_DIR="$(mktemp -d)"
VERSION_BUMPED=0
BUILD_LOG=""
TEST_ENV_NAME=""
TEST_WORKDIR=""
cleanup() {
  local ec=$?
  if [[ -n "${TEST_ENV_NAME}" ]]; then
    conda env remove -y -n "${TEST_ENV_NAME}" >/dev/null 2>&1 || true
  fi
  rm -f "${BUILD_LOG}"
  rm -rf "${TEST_WORKDIR}"
  if [[ "${ec}" -ne 0 ]]; then
    # A failed run leaves partial autophot builds in conda-bld; they are
    # stale files on the next run, so purge them here too.
    for _bld in "${CONDA_PREFIX:-}/conda-bld" \
                "$(conda info --base 2>/dev/null)/conda-bld"; do
      [[ -d "${_bld}" ]] || continue
      find "${_bld}" -maxdepth 2 \
        \( -name 'autophot-*' -o -name 'autophot-object-*' \) \
        -exec rm -rf {} + 2>/dev/null || true
      # Re-index so the local channel drops the deleted entries; a stale
      # index makes the next --use-local install fetch a missing file.
      run_in_env conda index "${_bld}" >/dev/null 2>&1 || true
    done
  fi
  # Restore version files on failure so the repo is left clean.
  if [[ "${ec}" -ne 0 && "${VERSION_BUMPED}" -eq 1 ]]; then
    log "Rolling back version changes (build/test/upload failed)..."
    cp -f "${BACKUP_DIR}/pyproject.toml" pyproject.toml 2>/dev/null || true
    cp -f "${BACKUP_DIR}/meta.yaml" conda/recipe/meta.yaml 2>/dev/null || true
    cp -f "${BACKUP_DIR}/requirements.txt" requirements.txt 2>/dev/null || true
    log "Restored pyproject.toml, meta.yaml, requirements.txt to ${CURRENT_VERSION}."
  fi
  rm -rf "${BACKUP_DIR}"
  exit "${ec}"
}
trap cleanup EXIT

cp pyproject.toml "${BACKUP_DIR}/pyproject.toml"
cp conda/recipe/meta.yaml "${BACKUP_DIR}/meta.yaml"
cp requirements.txt "${BACKUP_DIR}/requirements.txt"

set_version_everywhere "${SELECTED_VERSION}"
VERSION_BUMPED=1

# -----------------------------------------------------------------------------
# Sync requirements.txt from pyproject.toml and validate environment.yml /
# meta.yaml for dependency drift.  requirements.txt is auto-generated; the
# others are warn-only because they contain pinned versions and conda-specific
# packages that cannot be fully auto-derived.
# -----------------------------------------------------------------------------
log "Syncing requirements.txt from pyproject.toml..."
SYNC_RESULT="$(sync_requirements)"
if [[ "${SYNC_RESULT}" == "synced" ]]; then
  log "requirements.txt regenerated from pyproject.toml dependencies."
else
  log "WARNING: requirements.txt sync may have failed."
fi

log "Validating dependency files for drift..."
VALIDATION_OUTPUT="$(validate_dependency_files)"
if [[ -z "${VALIDATION_OUTPUT}" ]]; then
  log "All dependency files are in sync with pyproject.toml."
else
  while IFS= read -r line; do
    case "${line}" in
      WARN_ENV:*)
        missing="${line#WARN_ENV:}"
        log "WARNING: environment.yml is missing these pyproject.toml dependencies: ${missing}"
        log "         Add them to environment.yml to keep conda env in sync."
        ;;
      WARN_META:*)
        missing="${line#WARN_META:}"
        log "WARNING: meta.yaml run: section is missing these pyproject.toml dependencies: ${missing}"
        log "         Add them to conda/recipe/meta.yaml run: section (or document as pip-only)."
        ;;
      *)
        log "WARNING: ${line}"
        ;;
    esac
  done <<< "${VALIDATION_OUTPUT}"
fi

# -----------------------------------------------------------------------------
# Update the local editable install so the local environment picks up the
# new version immediately.  Without this, autophot.egg-info retains the
# old version and `from importlib.metadata import version` returns stale data.
# -----------------------------------------------------------------------------
log "Refreshing local editable install (pip install -e .)..."
# Remove stale egg-info that can shadow the new dist-info metadata.
rm -rf autophot.egg-info
PIP_LOG="$(mktemp)"
# Use the interpreter that runs conda-build (${_solver_probe}), not bare
# `python` -- a system Python ahead of the conda env on PATH fails with
# PEP 668 'externally-managed-environment'.
if "${_solver_probe[@]}" -m pip install -e . --no-deps --no-build-isolation >"${PIP_LOG}" 2>&1; then
  log "Local install updated to ${SELECTED_VERSION}"
else
  log "WARNING: Local pip install -e . failed; local environment may show stale version."
  log "         pip output:"
  sed 's/^/  /' "${PIP_LOG}" >&2
fi
rm -f "${PIP_LOG}"

if [[ -z "${ANACONDA_API_TOKEN:-}" ]]; then
  log "ANACONDA_API_TOKEN not set; using 'anaconda login' if needed."
fi

# -----------------------------------------------------------------------------
# Build
# -----------------------------------------------------------------------------
# Fail fast if 'defaults' is in the effective channel list.  conda channel
# lists merge across every .condarc on the search path -- CLI flags and
# env vars cannot remove an entry, only the file that declares it can.
# conda-build queries every configured channel while collecting package
# records during metadata finalization, so a 'defaults' entry pointing at
# an unreachable host (repo.anaconda.com) kills the build deep inside
# conda-libmamba-solver.
if run_in_env conda config --show channels 2>/dev/null | grep -qE '^\s*-\s*defaults\s*$'; then
  log "WARNING: 'defaults' is in the effective conda channels."
  log "         conda-build will query repo.anaconda.com/pkgs/main and fail if it is unreachable."
  log "         Remove it from whichever .condarc declares it (see: conda config --show-sources)."
fi

log "---"
log "Building recipe: ${RECIPE_PATH} (may take several minutes)..."
BUILD_LOG="$(mktemp)"
# --no-anaconda-upload: prevent conda-build from auto-uploading before our
#   post-build test.  ~/.condarc may have anaconda_upload: true, which would
#   push the package live before we've verified it works.
# --no-test: skip conda-build's internal test phase.  It solves a second
#   environment with ALL 26+ run dependencies, which takes a very long time.
#   Our script runs its own post-build test in a fresh conda env below, so
#   the internal test is redundant.
# --override-channels -c conda-forge: the recipe only uses conda-forge.
#   Without --override-channels, conda-build also queries the 'defaults'
#   channels from .condarc (repo.anaconda.com/pkgs/main), which fails
#   hard when that host is unreachable or DNS-filtered.
run_in_env conda-build --override-channels -c conda-forge --no-anaconda-upload --no-test "${RECIPE_PATH}" 2>&1 | tee "${BUILD_LOG}" || die "conda-build failed (see output above)."

# With --no-test, there is no "TEST END:" line.  Find the built package by
# searching the build log for any .conda/.tar.bz2 path, then fall back to
# scanning the conda-bld/noarch directory for the most recent matching file.
PACKAGE_PATH="$(grep -oE '/[^ ]*autophot[^ ]*\.(conda|tar\.bz2)' "${BUILD_LOG}" | tail -1)"
if [[ -z "${PACKAGE_PATH}" || ! -f "${PACKAGE_PATH}" ]]; then
  # Fall back: find the most recently built package for this version.
  # Use CONDA_PREFIX (always set when conda is activated) instead of
  # conda info --base, which can be slow and may output warnings.
  CONDA_BLDDIR="${CONDA_PREFIX}/conda-bld/noarch"
  if [[ ! -d "${CONDA_BLDDIR}" ]]; then
    CONDA_BLDDIR="$(conda info --base 2>/dev/null)/conda-bld/noarch"
  fi
  log "Searching for package in: ${CONDA_BLDDIR}"
  PACKAGE_PATH="$(ls -t "${CONDA_BLDDIR}"/autophot-"${SELECTED_VERSION}"-*.conda 2>/dev/null | head -1)"
  if [[ -z "${PACKAGE_PATH}" ]]; then
    PACKAGE_PATH="$(ls -t "${CONDA_BLDDIR}"/autophot-"${SELECTED_VERSION}"-*.tar.bz2 2>/dev/null | head -1)"
  fi
fi
[[ -n "${PACKAGE_PATH}" && -f "${PACKAGE_PATH}" ]] || die "Could not determine built package path from build output or conda-bld directory."

log "Package ready: ${PACKAGE_PATH}"

# -----------------------------------------------------------------------------
# Test the package in a fresh conda environment before upload
# -----------------------------------------------------------------------------
TEST_ENV_NAME="autophot_test_$$"

# Clean up any stale test environments from previous failed runs.
# The PID-based naming means failed runs leave orphaned envs that waste disk.
log "Cleaning up stale autophot_test_* environments..."
for _stale_env in $(conda env list 2>/dev/null | grep -oE 'autophot_test_[0-9]+' | sort -u); do
  if [[ "${_stale_env}" != "${TEST_ENV_NAME}" ]]; then
    log "  Removing stale env: ${_stale_env}"
    conda env remove -y -n "${_stale_env}" >/dev/null 2>&1 || true
  fi
done

log "---"
log "Creating test conda environment: ${TEST_ENV_NAME} (python=${TEST_PYTHON}, cython)"
# Include cython in the initial env creation so it's present before autophot
# is installed.  Installing cython separately after autophot triggers a
# re-solve that finds the local autophot package "inconsistent".  The test
# env Python must satisfy the package pin (python>=3.12,<3.13) or the
# install re-solve fails outright.
conda create -y -n "${TEST_ENV_NAME}" --override-channels -c conda-forge "python=${TEST_PYTHON}" cython 2>&1 || die "Failed to create test environment"

# Resolve the env prefix and call its binaries directly.  `conda run` is
# unsafe here: when it cannot find the executable inside the env it falls
# back to PATH resolution, which can silently run a bare system python
# (e.g. ~/.local/bin/python -> /usr/bin/python3, no numpy) and make the
# install test fail on the repo file instead of the real package.
TEST_ENV_PREFIX="$(conda env list --json 2>/dev/null | python -c '
import json, os, sys
name = sys.argv[1]
try:
    envs = json.load(sys.stdin).get("envs", [])
except Exception:
    envs = []
for p in envs:
    if os.path.basename(p.rstrip("/")) == name:
        print(p)
        break
' "${TEST_ENV_NAME}")"
[[ -x "${TEST_ENV_PREFIX}/bin/python" ]] || \
  die "Test env has no python binary at ${TEST_ENV_PREFIX}/bin/python"

# Install the local package by name from the local conda-bld channel.
# Using --use-local (equivalent to -c local) lets conda find the locally-
# built package and properly resolve its run dependencies (numpy, scipy,
# astropy, etc.).  Installing by file path (conda install /path/to.conda)
# does NOT install dependencies in conda 23.9.0, leaving the test env
# without numpy and other required packages.
log "Installing autophot package in test environment..."
if ! conda install -y -n "${TEST_ENV_NAME}" --override-channels --use-local -c conda-forge "autophot=${SELECTED_VERSION}" 2>&1; then
  die "Failed to install autophot in test environment"
fi

# sfft and sip_tpv are pip-only (not on conda-forge); install them for the test.
# sfft builds from sdist (Cython extension), so it needs cython + numpy + gcc.
# cython was installed in the conda create step above; numpy/scipy/etc. come
# from autophot's conda dependencies.
# Use --no-deps on the pip install so pip doesn't replace conda's numpy with
# a different version that breaks binary compatibility.
log "Installing pip-only deps (sfft, sip_tpv) in test environment..."
# PYTHONNOUSERSITE isolates the test env from ~/.local site-packages;
# without it the env's Python 3.12 sees user packages first on sys.path,
# pip skips installs it can already see there, and stale user copies
# shadow the env's own packages at test time.
if ! PYTHONNOUSERSITE=1 "${TEST_ENV_PREFIX}/bin/pip" install --no-deps sfft==1.7.3 sip_tpv==1.1 2>&1; then
  log "WARNING: Failed to install sfft/sip_tpv via pip in test environment"
  log "         The import test below may fail if these are required at import time."
fi

log "Testing autophot installation..."
TEST_OUTPUT="$(mktemp)"
# Mirror the README install check: import test first, then the CLI.  Run
# from a scratch dir -- `python -c` puts the cwd on sys.path, and the
# repo's flat autophot.py would shadow the installed package here.
TEST_WORKDIR="$(mktemp -d)"
if (cd "${TEST_WORKDIR}" && \
    PYTHONNOUSERSITE=1 "${TEST_ENV_PREFIX}/bin/python" -c "from autophot import AutomatedPhotometry; print('OK')" && \
    PYTHONNOUSERSITE=1 "${TEST_ENV_PREFIX}/bin/autophot-main" -h) >"${TEST_OUTPUT}" 2>&1; then
  log "Test passed: import check and autophot-main -h succeeded"
else
  log "ERROR: Package test failed. Output:"
  sed 's/^/  /' "${TEST_OUTPUT}" >&2
  log "Env contents for autophot/numpy:"
  conda list -n "${TEST_ENV_NAME}" 2>/dev/null | grep -E 'autophot|numpy' | sed 's/^/  /' >&2 || true
  rm -f "${TEST_OUTPUT}"
  die "Package test failed; aborting upload"
fi
rm -f "${TEST_OUTPUT}"
rm -rf "${TEST_WORKDIR}"
TEST_WORKDIR=""

log "Cleaning up test environment..."
conda env remove -y -n "${TEST_ENV_NAME}" >/dev/null 2>&1 || true
TEST_ENV_NAME=""
log "Test environment removed successfully"

UPLOAD_ARGS=("${PACKAGE_PATH}" "--user" "${CHANNEL_USERNAME}")
if [[ -n "${LABEL}" ]]; then
  UPLOAD_ARGS+=("--label" "${LABEL}")
fi

log "---"
log "Uploading to Anaconda.org channel '${CHANNEL_USERNAME}'..."
run_in_env anaconda upload --force "${UPLOAD_ARGS[@]}"

# Upload succeeded — keep the version bump.  Clear the rollback flag so the
# cleanup trap does not restore the old files.
VERSION_BUMPED=0

# -----------------------------------------------------------------------------
# Remove stale installed copies and old artifacts only AFTER the new package
# is uploaded: if the build or upload fails, existing installs stay untouched.
# -----------------------------------------------------------------------------
log "---"
log "Removing stale autophot installs and old build artifacts..."
remove_stale_autophot_code

# -----------------------------------------------------------------------------
# Commit and push the release changes: README citation refresh plus the
# version bump and requirements sync done above.  Warn-only: a git failure
# must not fail a build that already uploaded successfully.
# -----------------------------------------------------------------------------
if command -v git >/dev/null 2>&1 && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git add README.md figures/citations_over_time.png pyproject.toml conda/recipe/meta.yaml requirements.txt
  if git diff --cached --quiet; then
    log "No staged changes; nothing to commit."
  else
    git commit -m "Updated to version ${SELECTED_VERSION} & updated citation count" || \
      log "WARNING: git commit failed; release changes left uncommitted."
    if git rev-parse --abbrev-ref --symbolic-full-name '@{u}' >/dev/null 2>&1; then
      git push || log "WARNING: git push failed; push manually."
    else
      log "WARNING: current branch has no upstream; skipping git push."
    fi
  fi
else
  log "WARNING: not inside a git work tree; skipping release commit."
fi

log "---"
log "Done. Install with: conda install -c conda-forge -c ${CHANNEL_USERNAME}${LABEL:+/label/${LABEL}} autophot"
