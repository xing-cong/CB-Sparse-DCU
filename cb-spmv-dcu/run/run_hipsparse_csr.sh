#!/usr/bin/env bash
#
# Compile the hipSPARSE CSR baseline and run it serially over data/.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DTK_ROOT="${DTK_ROOT:-/opt/dtk-25.04.4}"
GPU_ARCH="${GPU_ARCH:-gfx936}"

if [[ ! -x "$DTK_ROOT/bin/hipcc" ]]; then
    echo "[Error] hipcc not found at $DTK_ROOT/bin/hipcc" >&2
    exit 1
fi

# shellcheck source=/dev/null
source "$DTK_ROOT/env.sh"

cd "$ROOT_DIR/benchmark/hipSparse-csr"
make DTK_ROOT="$DTK_ROOT" GPU_ARCH="$GPU_ARCH" \
    HIPCC="$DTK_ROOT/bin/hipcc"

EXE=./hipsparse_csr_spmv
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
