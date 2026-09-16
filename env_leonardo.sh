#!/usr/bin/env bash
# Source this file to reproduce the Leonardo software and cache environment.

_DRAFTVLA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! type module >/dev/null 2>&1; then
  echo "ERROR: Leonardo environment-modules is unavailable. Start a login shell first." >&2
  return 1 2>/dev/null || exit 1
fi

module purge
# JAX/PyTorch CUDA runtimes are locked and installed from PyPI. Do not load Leonardo's CUDA module:
# its CUDA 12.2 path would enter LD_LIBRARY_PATH and can override the pip CUDA 12.6/cuDNN 9 wheels.
module load python/3.11.7

# Keep large caches and the project environment on the shared project filesystem, not the small
# user home quota. These paths are shared between login and compute nodes.
export UV_CACHE_DIR="${UV_CACHE_DIR:-$_DRAFTVLA_ROOT/.cache/uv}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
export UV_PYTHON_DOWNLOADS="${UV_PYTHON_DOWNLOADS:-never}"
export HF_HOME="${HF_HOME:-$_DRAFTVLA_ROOT/.cache/huggingface}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-$HF_HOME/lerobot}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$_DRAFTVLA_ROOT/.cache/openpi}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$_DRAFTVLA_ROOT/.cache/jax}"
export TMPDIR="${TMPDIR:-$_DRAFTVLA_ROOT/.cache/tmp}"

mkdir -p "$UV_CACHE_DIR" "$HF_LEROBOT_HOME" "$OPENPI_DATA_HOME" "$JAX_COMPILATION_CACHE_DIR" "$TMPDIR"

if [ -x "$_DRAFTVLA_ROOT/.tools/uv/bin/uv" ]; then
  export PATH="$_DRAFTVLA_ROOT/.tools/uv/bin:$PATH"
fi
if [ -d "$_DRAFTVLA_ROOT/.venv" ]; then
  export VIRTUAL_ENV="$_DRAFTVLA_ROOT/.venv"
  export PATH="$_DRAFTVLA_ROOT/.venv/bin:$PATH"
fi

export DRAFTVLA_ROOT="$_DRAFTVLA_ROOT"
unset _DRAFTVLA_ROOT
