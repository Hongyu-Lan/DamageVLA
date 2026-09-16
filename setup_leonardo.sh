#!/usr/bin/env bash
# One-time dependency setup on a Leonardo login node. No GPU is required.
set -euo pipefail
cd "$(dirname "$0")"

# shellcheck disable=SC1091
source ./env_leonardo.sh

MODULE_PYTHON="$(python -c 'import sys; print(sys._base_executable)')"
UV_VENV="$PWD/.tools/uv"
if [ ! -x "$UV_VENV/bin/uv" ]; then
  echo "==== [1/4] installing uv in $UV_VENV ===="
  "$MODULE_PYTHON" -m venv "$UV_VENV"
  "$UV_VENV/bin/python" -m pip install --upgrade pip
  "$UV_VENV/bin/python" -m pip install uv
else
  echo "==== [1/4] reusing $UV_VENV/bin/uv ===="
fi
UV="$UV_VENV/bin/uv"

echo "==== [2/4] creating/syncing the Python 3.11 project environment ===="
GIT_LFS_SKIP_SMUDGE=1 "$UV" sync --frozen --python "$MODULE_PYTHON"

echo "==== [3/4] validating imports and locked versions ===="
"$UV" run --frozen python - <<'PY'
import importlib.metadata as metadata
import sys

import openpi

print("python", sys.version.split()[0])
for package in ("jax", "jaxlib", "flax", "torch", "transformers", "lerobot", "openpi"):
    print(package, metadata.version(package))
PY

echo "==== [4/4] validating the uploaded raw dataset ===="
"$UV" run --frozen python examples/force/validate_draftvla_data.py

echo
echo "Leonardo setup is ready."
echo "Next: bash submit_leonardo.sh smoke"
