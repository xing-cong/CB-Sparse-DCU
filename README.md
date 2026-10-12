# CB-Sparse-DCU

**CB-Sparse-DCU** is a cache-friendly sparse computing framework for Hygon DCUs that unifies high-performance implementations of **SpMV**, **SpMM**, and **SpGEMM**, designed for irregular sparse workloads in HPC.

---

## OverView

**CB-Sparse-DCU** is a unified DCU sparse computing framework that focuses on improving cache locality, hardware utilization, and workload balance for irregular sparse matrix operations. It uses a blocked sparse representation and adaptive execution strategy to support sparse matrix-vector multiplication (SpMV), sparse matrix-matrix multiplication (SpMM), and sparse matrix-sparse matrix multiplication (SpGEMM). The current implementation provides CB-SpMV-DCU for Hygon `gfx936` devices with 64-lane wavefronts.

---

## Motivation: Cache-Friendly Sparse Computing

Traditional sparse formats such as CSR and COO introduce irregular memory accesses that reduce cache efficiency and underutilize wavefront-level parallelism. CB-Sparse-DCU addresses these limitations through a cache-friendly blocked sparse structure and adaptive sparse execution framework optimized for DCU architectures.

---

## CB-SpMV-DCU

### Paper (ICS'25)

<div align="center">
    <h3>
      CB-SpMV: A Data Aggregating and Balance Algorithm for Cache-Friendly Block-Based SpMV on GPUs
    </h3>
    <p>
        Xing Cong, Fukai Sun, Yifan Chen, Chenhao Xie, Yi Liu, Depei Qian
    </p>
    <p></p>
</div>

<div align="center">
  <a href="https://dl.acm.org/doi/10.1145/3721145.3725746"><b>[Paper]</b></a> |
  <a href="cb-spmv-dcu/assets/cb-spmv-slides.pdf"><b>[Slides]</b></a> |
  <a href="Code-Guide.md"><b>[Code Guide]</b></a>
</div>

### Method

**CB-SpMV-DCU** partitions sparse matrices into independent 16x16 sub-blocks and organizes them using a unified 2D sparse structure. Each block adaptively selects COO, CSR, or dense storage according to its sparsity. Metadata and values are packed into contiguous device memory, while column gathering and inter-workgroup load balancing reduce irregular memory access and workload imbalance.

The DCU implementation uses the original CB-SpMV compute and gather kernels with native wave64 execution. COO, CSR, dense, small-matrix direct output, and repeated-row-pattern reuse remain schedules inside those two kernels. The default uses 16x16 blocks, with four wavefronts for the base CB schedule and nine for the tuned long-pattern schedule; the optional 32x32 configuration uses two lanes per row and can be selected at build time.

### Performance

The supplied benchmarks compare CB-SpMV-DCU with hipSPARSE BSR block sizes 2, 4, 8, and 16 and with hipSPARSE CSR algorithms 1 and 2. Example terminal output is provided in the [CB-SpMV-DCU README](cb-spmv-dcu/README.md).

---

## CB-SpMM-DCU and CB-SpGEMM-DCU

Will coming soon! (Under review)

---

## Citation

If you find `CB-Sparse-DCU (CB-SpMV-DCU)` useful or relevant to your research, please cite our paper:

```bibtex
@inproceedings{cong2025cb-spmv,
  title={CB-SpMV: A Data Aggregating and Balance Algorithm for Cache-Friendly Block-Based SpMV on GPUs},
  author={Cong, Xing and Sun, Fukai and Chen, Yifan and Xie, Chenhao and Liu, Yi and Qian, Depei},
  booktitle={Proceedings of the 39th ACM International Conference on Supercomputing},
  pages={149--160},
  year={2025}
}
```

---
