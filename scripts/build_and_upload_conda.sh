#!/usr/bin/env bash
set -euo pipefail

# Build and upload the AutoPHOT conda package to Anaconda.org.
#
# Usage examples:
#   ./scripts/build_and_upload_conda.sh
#   ./scripts/build_and_upload_conda.sh --label dev
#   ./scripts/build_and_upload_conda.sh --new-version 0.1.1
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
SKIP_BUILD=0
PACKAGE_PATH=""
NEW_VERSION=""

usage() {
  cat <<'EOF'
Build and upload AutoPHOT conda package.

Options:
  --channel-username NAME   Optional. Anaconda.org username/channel (default: astro-sean).
  --env NAME                Optional conda env used for build/upload tools.
  --recipe PATH             Optional recipe path (default: conda/recipe).
  --label NAME              Optional Anaconda label (e.g. dev, main).
  --new-version VERSION     Set an explicit new version (X.Y.Z) non-interactively.
  --skip-build              Skip build and upload existing package path (not allowed with mandatory version bump).
  --package-path PATH       Path to built .conda/.tar.bz2 package (required with --skip-build).
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
    --skip-build)
      SKIP_BUILD=1
      shift
      ;;
    --package-path)
      PACKAGE_PATH="${2:-}"
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

if [[ "${SKIP_BUILD}" -eq 1 && -z "${PACKAGE_PATH}" ]]; then
  die "--package-path is required when --skip-build is used."
fi
if [[ "${SKIP_BUILD}" -eq 1 ]]; then
  die "--skip-build is disabled for this workflow because each upload must bump version."
fi

run_in_env() {
  if [[ -n "${CONDA_BUILD_ENV}" ]]; then
    conda run -n "${CONDA_BUILD_ENV}" "$@"
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

log "Checking required tools..."
run_in_env conda-build --version >/dev/null
# anaconda-client CLI does not have --version; check if command is available instead
run_in_env anaconda --help >/dev/null

log "---"
CURRENT_VERSION="$(get_current_version)"
SUGGESTED_VERSION="$(increment_patch_version "${CURRENT_VERSION}")"
if [[ -n "${NEW_VERSION}" ]]; then
  SELECTED_VERSION="${NEW_VERSION}"
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
set_version_everywhere "${SELECTED_VERSION}"

if [[ -z "${ANACONDA_API_TOKEN:-}" ]]; then
  log "ANACONDA_API_TOKEN not set; using 'anaconda login' if needed."
fi

if [[ "${SKIP_BUILD}" -eq 0 ]]; then
  log "---"
  log "Building recipe: ${RECIPE_PATH} (may take several minutes)..."
  BUILD_LOG="$(mktemp)"
  trap 'rm -f "${BUILD_LOG}"' EXIT
  run_in_env conda-build "${RECIPE_PATH}" 2>&1 | tee "${BUILD_LOG}"
  PACKAGE_PATH="$(grep -o 'TEST END: .*\.conda' "${BUILD_LOG}" | tail -1 | sed 's/^TEST END: //')"
  [[ -n "${PACKAGE_PATH}" && -f "${PACKAGE_PATH}" ]] || die "Could not determine built package path from build output."
fi

[[ -n "${PACKAGE_PATH}" ]] || die "Failed to resolve package path."
[[ -f "${PACKAGE_PATH}" ]] || die "Built package file not found: ${PACKAGE_PATH}"

log "Package ready: ${PACKAGE_PATH}"

# -----------------------------------------------------------------------------
# Test the package in a fresh conda environment before upload
# -----------------------------------------------------------------------------
TEST_ENV_NAME="autophot_test_$$"
log "---"
log "Creating test conda environment: ${TEST_ENV_NAME}"
conda create -y -n "${TEST_ENV_NAME}" python=3.11 -c conda-forge >/dev/null 2>&1 || die "Failed to create test environment"

log "Installing autophot package in test environment..."
conda install -y -n "${TEST_ENV_NAME}" -c conda-forge -c "${CHANNEL_USERNAME}" --override-channels autophot || {
  log "WARNING: Failed to install from channel, trying local package..."
  conda install -y -n "${TEST_ENV_NAME}" "${PACKAGE_PATH}" || {
    conda env remove -y -n "${TEST_ENV_NAME}" >/dev/null 2>&1
    die "Failed to install autophot in test environment"
  }
}

log "Testing autophot installation..."
if conda run -n "${TEST_ENV_NAME}" autophot-main -h >/dev/null 2>&1; then
  log "Test passed: autophot-main -h succeeded"
else
  log "Test failed: autophot-main -h failed"
  conda env remove -y -n "${TEST_ENV_NAME}" >/dev/null 2>&1
  die "Package test failed; aborting upload"
fi

log "Cleaning up test environment..."
conda env remove -y -n "${TEST_ENV_NAME}" >/dev/null 2>&1
log "Test environment removed successfully"

UPLOAD_ARGS=("${PACKAGE_PATH}" "--user" "${CHANNEL_USERNAME}")
if [[ -n "${LABEL}" ]]; then
  UPLOAD_ARGS+=("--label" "${LABEL}")
fi

log "---"
log "Uploading to Anaconda.org channel '${CHANNEL_USERNAME}'..."
run_in_env anaconda upload --force "${UPLOAD_ARGS[@]}"

log "---"
log "Done. Install with: conda install -c ${CHANNEL_USERNAME}${LABEL:+/label/${LABEL}} autophot"
