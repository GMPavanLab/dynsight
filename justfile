# Works with both uv and conda:
# - If a project-local uv virtualenv (./.venv) exists, its tools are
#   used automatically (no activation needed).
# - Otherwise the active environment is used (conda, system, ...).
dot_venv_bin := justfile_directory() / ".venv/bin"
venv_bin := justfile_directory() / "venv/bin"
src_dir := justfile_directory() / "src"

export PATH := if path_exists(dot_venv_bin) == "true" {
  dot_venv_bin + ":" + env("PATH")
} else if path_exists(venv_bin) == "true" {
  venv_bin + ":" + env("PATH")
} else {
  env("PATH")
}

# Import the package from ./src regardless of the editable install.
# This keeps the checks working even when the .pth file of the install
# is unreadable to Python, which happens on macOS when a synced folder
# (iCloud Desktop/Documents) sets the "hidden" flag on it: Python >=
# 3.11 silently skips hidden .pth files.
export PYTHONPATH := if env_var_or_default("PYTHONPATH", "") == "" {
  src_dir
} else {
  src_dir + ":" + env_var_or_default("PYTHONPATH", "")
}

# List all commands.
default:
  @just --list

# Build docs.
docs:
  rm -rf docs/build docs/source/_autosummary
  make -C docs html
  echo Docs are in $PWD/docs/build/html/index.html

# Do a dev install (uv venv, conda or plain pip - autodetected).
dev:
  #!/usr/bin/env bash
  set -euo pipefail
  # An existing project venv wins over the active environment, and
  # ./.venv wins over ./venv (same order as the PATH setting above).
  target=""
  for candidate in .venv venv; do
    if [ -d "$candidate" ]; then target="$candidate"; break; fi
  done
  if [ -n "$target" ] && command -v uv >/dev/null 2>&1; then
    echo "Installing into ./$target with uv"
    uv pip install --python "$target/bin/python" -e '.[dev]'
  elif [ -n "${CONDA_PREFIX:-}" ]; then
    echo "Installing into conda env '${CONDA_DEFAULT_ENV:-}' with pip"
    pip install -e '.[dev]'
  elif command -v uv >/dev/null 2>&1; then
    echo "Creating ./.venv with uv"
    uv venv
    uv pip install -e '.[dev]'
  else
    pip install -e '.[dev]'
  fi
  # On macOS the .pth file of the editable install can carry the
  # "hidden" flag, which makes Python >= 3.11 skip it, so that the
  # package fails to import (including from the label_tool command).
  # Some setups keep re-applying the flag to dot-directories such as
  # ./.venv; a venv named ./venv avoids it. The recipes above do not
  # depend on the .pth anyway: they import the package from ./src.
  if [ "$(uname)" = "Darwin" ] && [ -n "$target" ]; then
    chflags nohidden "$target"/lib/python*/site-packages/*.pth 2>/dev/null || true
  fi

# Run code checks.
check:
  #!/usr/bin/env bash

  # bash 3.2 (the macOS default) does not run the ERR trap when a
  # subshell fails, so failures are collected explicitly: without this,
  # `just check` reported success even when a step failed.
  error=0
  failed=()

  run() {
    echo
    ( set -x; "$@" ) || { error=1; failed+=("$1"); }
  }

  run ruff check .
  run ruff format --check .
  run mypy .
  run pytest --cov=src --cov-report term-missing
  run make -C docs doctest

  echo
  if [ $error -ne 0 ]; then
    echo "FAILED: ${failed[*]}"
  else
    echo "All checks passed."
  fi
  exit $error

# Auto-fix code issues.
fix:
  ruff format .
  ruff check --fix .

# Build a release.
build:
  python -m build
