# Code Guide for CB-Sparse-DCU

Each method in CB-Sparse-DCU is maintained in its own directory. Select the operation you want to evaluate and follow the instructions in the corresponding directory.

| Method | Status | Source and instructions |
| --- | --- | --- |
| CB-SpMV-DCU | Available | [`cb-spmv-dcu/`](cb-spmv-dcu/) · [`README`](cb-spmv-dcu/README.md) |
| CB-SpMM-DCU | Coming soon | — |
| CB-SpGEMM-DCU | Coming soon | — |

## CB-SpMV-DCU

The [`cb-spmv-dcu/`](cb-spmv-dcu/) directory contains the HIP implementation of CB-SpMV-DCU, example matrices, runnable scripts, and a hipSPARSE BSR baseline. See the [CB-SpMV-DCU README](cb-spmv-dcu/README.md) for the complete environment, usage documentation, and example output.

### Quick start

From the root of this repository, run:

```bash
cd cb-spmv-dcu

# Build CB-SpMV-DCU and run all example matrices in data/
./run/run_cbspmv_dcu.sh

# Build the hipSPARSE BSR baseline and run the same matrices
./run/run_hipsparse.sh
```

The scripts compile the corresponding implementation and run the included Matrix Market (`.mtx`) matrices serially.

### Run a custom matrix

CB-SpMV-DCU accepts an absolute path to a Matrix Market file:

```bash
cd cb-spmv-dcu/src
make
./cb-spmv-dcu /absolute/path/to/matrix.mtx
```

To run the hipSPARSE BSR baseline on the same input:

```bash
cd cb-spmv-dcu/benchmark/hipSparse-bsr
make
./hipsparse_spmv /absolute/path/to/matrix.mtx
```

Both implementations target Hygon `gfx936` DCUs by default and require a 64-lane wavefront. For a different DCU target, set `GPU_ARCH` when compiling both source directories.

## CB-SpMM-DCU

The source code and execution guide will be released after the corresponding work is available.

## CB-SpGEMM-DCU

The source code and execution guide will be released after the corresponding work is available.
