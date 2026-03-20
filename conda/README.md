# Conda package (maintainers)

## Build

From the repository root (or adjust paths):

```bash
conda build conda/recipe -c conda-forge
```

## Upload conflicts (HTTP 409)

Anaconda.org stores each file by **name + version + build string** (e.g. `autophot-0.1.8-py_1.conda`). If that exact file already exists on your channel, `anaconda upload` fails with:

`Distribution already exists ... Conflict ... 409`

**The local build still succeeds** — only the upload step is rejected. Users do not get a new package until a **new** artifact is uploaded.

**Fix (pick one):**

1. **Recommended:** Bump `build: number:` in `recipe/meta.yaml` (e.g. `1` → `2`), rebuild, then upload. The new file will be `autophot-0.1.8-py_2.conda`.
2. **Replace in place:** `anaconda upload --force /path/to/autophot-0.1.8-py_1.conda` (overwrites the existing file on the channel).
3. **Remove then upload:** `anaconda remove <user>/autophot/0.1.8/noarch/autophot-0.1.8-py_1.conda`, then upload again.

## Wrapper scripts

If a post-build script prints `[build-upload]`, ensure it checks the exit code of `anaconda upload` (e.g. `set -e` in bash, or `if ! anaconda upload ...; then exit 1; fi`) so a 409 does not look like a full success.
