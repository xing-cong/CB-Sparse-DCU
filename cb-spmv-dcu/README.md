# CB-SpMV-DCU

HIP implementation and benchmark driver for **CB-SpMV-DCU**. The repository contains the CB-SpMV-DCU implementation and hipSPARSE BSR and CSR SpMV baselines.

## Repository contents

| Component | Source | Run script | Executable |
| --- | --- | --- | --- |
| CB-SpMV-DCU | [`src/`](src/) | [`run/run_cbspmv_dcu.sh`](run/run_cbspmv_dcu.sh) | `src/cb-spmv-dcu` |
| hipSPARSE BSR baseline | [`benchmark/hipSparse-bsr/`](benchmark/hipSparse-bsr/) | [`run/run_hipsparse.sh`](run/run_hipsparse.sh) | `benchmark/hipSparse-bsr/hipsparse_spmv` |
| hipSPARSE CSR baseline | [`benchmark/hipSparse-csr/`](benchmark/hipSparse-csr/) | [`run/run_hipsparse_csr.sh`](run/run_hipsparse_csr.sh) | `benchmark/hipSparse-csr/hipsparse_csr_spmv` |

The hipSPARSE baseline converts each input matrix from CSR to BSR, benchmarks block sizes 2, 4, 8, and 16, and reports the best-performing block size.
The hipSPARSE CSR baseline benchmarks the generic FP64 `CSR_ALG1` and `CSR_ALG2` implementations and reports the faster algorithm.

## Requirements

- A Hygon DCU with a 64-lane wavefront
- DTK 25.04.4, including `hipcc`, the HIP runtime, and hipSPARSE
- GNU Make
- Bash and `realpath`
- A host C++ compiler with OpenMP support

The code has been tested with the following environment:

- Hygon BW DCU (`gfx936`, wavefront size 64)
- DTK 25.04.4 and HIP 6.3
- `hipcc`/dcc 25.08.0, clang 17

All Makefiles target `gfx936` by default. For another DCU target, set `GPU_ARCH` when compiling:

```bash
make GPU_ARCH=<target>
```

The default implementation preserves the 16x16 CB blocks and uses native 64-lane wavefronts. The base CB schedule uses a 256-thread workgroup with four wavefronts; the long-pattern schedule uses nine wavefronts after tuning on `gfx936`. For small super-sparse matrices, a workgroup owns one CB block row and writes the result directly. For unit-valued matrices with many identical row patterns, the same `comp` or `gather` kernel template computes each repeated pattern once. CSR and dense blocks use four lanes per row with native DPP reduction. The 32x32 configuration can be tested with:

```bash
make clean
make CB_BLOCK_SIZE=32 CB_WAVES_PER_BLOCK=4
```

Both configurations execute the original CB-SpMV `comp` or `gather` kernel. COO, CSR, dense, direct-output, and repeated-pattern schedules remain compile-time branches of those two kernels; no replacement global kernel is introduced.

## Quick start

Run the following commands from the repository root:

```bash
# Build CB-SpMV-DCU and run the included example matrices
./run/run_cbspmv_dcu.sh

# Build the hipSPARSE baseline and run the included example matrices
./run/run_hipsparse.sh

# Build the hipSPARSE CSR baseline and run the included example matrices
./run/run_hipsparse_csr.sh
```

The scripts build their corresponding executables and run the three matrices included in [`data/`](data/) serially:

- `1138_bus.mtx`
- `heart2.mtx`
- `sme3Da.mtx`

## Slurm batch runs

The scripts in [`run_all/`](run_all/) request one DCU from the `hx1hdnormal01` partition, load DTK 25.04.4, call `run.py`, and execute every `.mtx` file in `data/`. Each job writes its complete output to a job-specific log under `run_all/log/`.

Submit the three benchmarks with:

```bash
sbatch run_all/run_cbspmv_dcu.slurm
sbatch run_all/run_hipsparse.slurm
sbatch run_all/run_hipsparse_csr.slurm
```

To use another absolute matrix directory, pass it as the first script argument:

```bash
sbatch run_all/run_cbspmv_dcu.slurm /absolute/path/to/matrix_set
```

The resulting log names are `run_cbspmv_dcu_<jobid>.log`, `run_hipsparse_bsr_<jobid>.log`, and `run_hipsparse_csr_<jobid>.log`.

## Running another matrix

To run another Matrix Market (`.mtx`) matrix, enter the corresponding source directory, build the executable, and pass the matrix's **absolute path**.

CB-SpMV-DCU:

```bash
cd src
make
./cb-spmv-dcu /absolute/path/to/example.mtx
```

hipSPARSE:

```bash
cd benchmark/hipSparse-bsr
make
./hipsparse_spmv /absolute/path/to/example.mtx
```

hipSPARSE CSR:

```bash
cd benchmark/hipSparse-csr
make
./hipsparse_csr_spmv /absolute/path/to/example.mtx
```

To remove the generated executables from the repository root:

```bash
make -C src clean
make -C benchmark/hipSparse-bsr clean
make -C benchmark/hipSparse-csr clean
```

CB-SpMV-DCU and both hipSPARSE baselines use the same benchmark protocol: 100 untimed warm-up iterations followed by 1000 timed iterations. The CB-SpMV-DCU event interval contains only kernel launches; its output clear, allocation, conversion, transfer, and validation are outside the interval. hipSPARSE uses `beta=0`; CSR preprocessing and workspace allocation are also outside the event interval. Each program prints the matrix dimensions, number of nonzeros, average kernel time, throughput in GFLOPS, and a validation result. CB-SpMV-DCU also reports its block sparsity, column-gather, and load-balancing statistics.

Example hipSPARSE output:

```text
================================================================
                 hipSPARSE BSR SpMV Benchmark
================================================================
Matrix        : sme3Da.mtx
Dimensions    : 12504 x 12504
Nonzeros      : 874887
----------------------------------------------------------------
Block size             Time (ms)            GFLOPS     Selection
----------------------------------------------------------------
2                       0.027863         62.798674          BEST
4                       0.098314         17.797846
8                       0.363840          4.809187
16                      0.692909          2.525259
----------------------------------------------------------------
Best result   : block 2, 0.027863 ms, 62.798674 GFLOPS
Validation    : PASS
================================================================
```

Example CB-SpMV-DCU output:

```text
================================================================
                       CB-SpMV Benchmark
================================================================
Matrix                  : sme3Da.mtx
Dimensions              : 12504 x 12504
Nonzeros                : 874887
Sparse blocks (nnz<=8)  : 263568 / 279551 (94.282617%)
Column gather           : YES
Load balance            : NO
Load stddev             : N/A -> N/A
----------------------------------------------------------------
Method                        Time (ms)                   GFLOPS
----------------------------------------------------------------
CB-SpMV-DCU                    0.014674               119.246431
----------------------------------------------------------------
Validation    : PASS
================================================================
```

Example hipSPARSE CSR output:

```text
================================================================
                 hipSPARSE CSR SpMV Benchmark
================================================================
Matrix        : sme3Da.mtx
Dimensions    : 12504 x 12504
Nonzeros      : 874887
----------------------------------------------------------------
Algorithm                          Time (ms)        GFLOPS
----------------------------------------------------------------
CSR_ALG1                            0.018680      93.669578  BEST
CSR_ALG2                            0.099909      17.513635
----------------------------------------------------------------
Best result   : CSR_ALG1, 0.018680 ms, 93.669578 GFLOPS
Validation    : PASS
================================================================
```

## Project structure

```text
cb-spmv-dcu/
├── src/                  # CB-SpMV-DCU implementation and Makefile
├── benchmark/
│   ├── hipSparse-bsr/    # hipSPARSE BSR SpMV baseline
│   └── hipSparse-csr/    # hipSPARSE CSR SpMV baseline
├── data/                 # Matrix Market input matrices
├── run/
│   ├── run_cbspmv_dcu.sh
│   ├── run_hipsparse.sh
│   └── run_hipsparse_csr.sh
├── run_all/
│   ├── run_cbspmv_dcu.slurm
│   ├── run_hipsparse.slurm
│   ├── run_hipsparse_csr.slurm
│   └── log/              # Slurm job logs
├── assets/
│   └── cb-spmv-slides.pdf
└── README.md
```

## Acknowledgements

We gratefully acknowledge:

- hipSPARSE for the benchmark comparison.
- The [SuiteSparse Matrix Collection](https://sparse.tamu.edu/) for providing the sparse matrices used in the benchmarks.
