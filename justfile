# Works with both uv and conda:
# - If a project-local uv virtualenv (./.venv) exists, its tools are
#   used automatically (no activation needed).
# - Otherwise the active environment is used (conda, system, ...).
venv_bin := justfile_directory() / ".venv/bin"

export PATH := if path_exists(venv_bin) == "true" {
  venv_bin + ":" + env("PATH")
} else {
  env("PATH")
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
  if [ -d .venv ] && command -v uv >/dev/null 2>&1; then
    echo "Installing into ./.venv with uv"
    uv pip install -e '.[dev]'
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
  # macOS can end up with the "hidden" flag on .pth files, which makes
  # Python >= 3.11 silently skip them (ModuleNotFoundError on import).
  if [ "$(uname)" = "Darwin" ] && [ -d .venv ]; then
    chflags nohidden .venv/lib/python*/site-packages/*.pth 2>/dev/null || true
  fi

# Run code checks.
check:
  #!/usr/bin/env bash

  error=0
  trap error=1 ERR

  echo
  (set -x; ruff check . )

  echo
  ( set -x; ruff format --check . )

  echo
  ( set -x; mypy . )

  echo
  ( set -x; pytest --cov=src --cov-report term-missing )

  echo
  ( set -x; make -C docs doctest )

  test $error = 0

# Auto-fix code issues.
fix:
  ruff format .
  ruff check --fix .

# Build a release.
build:
  python -m build
