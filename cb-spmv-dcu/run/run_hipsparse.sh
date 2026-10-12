#!/usr/bin/env bash
#
# Compile the hipSPARSE baseline (benchmark/hipSparse-bsr/) and run it serially
# over every matrix in data/. Matrices are referenced by a relative path but
# converted to an absolute path at run time, because the program requires one.
#
set -euo pipefail

# Locate this script (run/) and the repo root, so it works from any CWD.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DTK_ROOT="${DTK_ROOT:-/opt/dtk-25.04.4}"
GPU_ARCH="${GPU_ARCH:-gfx936}"

if [[ ! -x "$DTK_ROOT/bin/hipcc" ]]; then
    echo "[Error] hipcc not found at $DTK_ROOT/bin/hipcc" >&2
    exit 1
fi

# The DTK environment supplies the Hygon HIP runtime and libraries.
# shellcheck source=/dev/null
source "$DTK_ROOT/env.sh"

# 1) Enter the build directory and compile.
cd "$ROOT_DIR/benchmark/hipSparse-bsr"
make DTK_ROOT="$DTK_ROOT" GPU_ARCH="$GPU_ARCH"

# 2) Serially run over every matrix in data/ (relative path -> absolute path).
EXE=./hipsparse_spmv
DATA_REL=../../data

shopt -s nullglob
matrices=("$DATA_REL"/*.mtx)
shopt -u nullglob

if [ ${#matrices[@]} -eq 0 ]; then
    echo "[Error] No .mtx files found in $DATA_REL" >&2
    exit 1
fi

for mtx in "${matrices[@]}"; do
    "$EXE" "$(realpath "$mtx")" || echo "[Warn] $EXE failed on $mtx" >&2
done
