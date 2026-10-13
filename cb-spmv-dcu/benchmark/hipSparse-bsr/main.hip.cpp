#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <hip/hip_runtime.h>
#include <hipsparse.h>

#include "io/mmio_highlevel.h"
#include "macros.h"
#include "utils.h"

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <absolute_matrix_path>\n";
        return EXIT_FAILURE;
    }

    const std::string matrix_path = argv[1];
    if (matrix_path.empty() || matrix_path.front() != '/') {
        Error("Matrix path must be absolute: " + matrix_path);
        return EXIT_FAILURE;
    }
    const std::size_t separator = matrix_path.find_last_of('/');
    const std::string matrix_name = matrix_path.substr(separator + 1);

    CHECK_HIP(hipSetDevice(0));
    hipDeviceProp_t device_properties{};
    CHECK_HIP(hipGetDeviceProperties(&device_properties, 0));
    if (device_properties.warpSize != 64) {
        Error("This benchmark expects a 64-lane DCU wavefront");
        return EXIT_FAILURE;
    }

    CsrMatrix matrix;
    std::string read_error;
    if (!read_csr_matrix(matrix_path, matrix, read_error)) {
        Error(read_error);
        return EXIT_FAILURE;
    }
    if (matrix.rows < BSR_BLOCK_SIZES[0]
        || matrix.columns < BSR_BLOCK_SIZES[0]) {
        Error("Matrix is too small for the requested BSR block sizes");
        return EXIT_FAILURE;
    }
    if (matrix.nonzeros() == 0) {
        Error("Matrix has no nonzero entries");
        return EXIT_FAILURE;
    }

    int* device_csr_row_offsets = nullptr;
    int* device_csr_column_indices = nullptr;
    double* device_csr_values = nullptr;
    CHECK_HIP(hipMalloc(reinterpret_cast<void**>(&device_csr_row_offsets),
                        matrix.row_offsets.size() * sizeof(int)));
    CHECK_HIP(hipMalloc(reinterpret_cast<void**>(&device_csr_column_indices),
                        matrix.column_indices.size() * sizeof(int)));
    CHECK_HIP(hipMalloc(reinterpret_cast<void**>(&device_csr_values),
                        matrix.values.size() * sizeof(double)));
    CHECK_HIP(hipMemcpy(device_csr_row_offsets, matrix.row_offsets.data(),
                        matrix.row_offsets.size() * sizeof(int), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(device_csr_column_indices, matrix.column_indices.data(),
                        matrix.column_indices.size() * sizeof(int), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(device_csr_values, matrix.values.data(),
                        matrix.values.size() * sizeof(double), hipMemcpyHostToDevice));

    hipsparseHandle_t handle = nullptr;
    hipsparseMatDescr_t csr_descriptor = nullptr;
    hipsparseMatDescr_t bsr_descriptor = nullptr;
    CHECK_HIPSPARSE(hipsparseCreate(&handle));
    CHECK_HIPSPARSE(hipsparseCreateMatDescr(&csr_descriptor));
    CHECK_HIPSPARSE(hipsparseSetMatType(
        csr_descriptor, HIPSPARSE_MATRIX_TYPE_GENERAL));
    CHECK_HIPSPARSE(hipsparseSetMatIndexBase(
        csr_descriptor, HIPSPARSE_INDEX_BASE_ZERO));
    CHECK_HIPSPARSE(hipsparseCreateMatDescr(&bsr_descriptor));
    CHECK_HIPSPARSE(hipsparseSetMatType(
        bsr_descriptor, HIPSPARSE_MATRIX_TYPE_GENERAL));
    CHECK_HIPSPARSE(hipsparseSetMatIndexBase(
        bsr_descriptor, HIPSPARSE_INDEX_BASE_ZERO));

    constexpr double alpha = 1.0;
    constexpr double beta = 0.0;
    const hipsparseDirection_t direction = HIPSPARSE_DIRECTION_ROW;
    std::vector<BSR_BENCHMARK_RESULT> benchmark_results;
    int best_block_size = 0;
    double best_time = 0.0;
    PROGRAM_METRICS best_metrics;
    std::vector<double> best_result(matrix.rows, 0.0);

    constexpr unsigned int random_seed = 42;
    std::mt19937 random_engine(random_seed);
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    std::vector<double> input_x(matrix.columns);
    for (double& value : input_x) value = distribution(random_engine);

    for (const int block_size : BSR_BLOCK_SIZES) {
        if (block_size > matrix.rows || block_size > matrix.columns) continue;

        const int block_rows = (matrix.rows + block_size - 1) / block_size;
        const int block_columns = (matrix.columns + block_size - 1) / block_size;
        const int padded_columns = block_columns * block_size;
        const int padded_rows = block_rows * block_size;
        std::vector<double> host_x(padded_columns, 0.0);
        std::copy(input_x.begin(), input_x.end(), host_x.begin());

        double* device_x = nullptr;
        double* device_y = nullptr;
        int* device_bsr_row_offsets = nullptr;
        int* device_bsr_column_indices = nullptr;
        double* device_bsr_values = nullptr;
        CHECK_HIP(hipMalloc(reinterpret_cast<void**>(&device_x),
                            host_x.size() * sizeof(double)));
        CHECK_HIP(hipMalloc(reinterpret_cast<void**>(&device_y),
                            static_cast<std::size_t>(padded_rows) * sizeof(double)));
        CHECK_HIP(hipMemcpy(device_x, host_x.data(), host_x.size() * sizeof(double),
                            hipMemcpyHostToDevice));
        CHECK_HIP(hipMalloc(reinterpret_cast<void**>(&device_bsr_row_offsets),
                            static_cast<std::size_t>(block_rows + 1) * sizeof(int)));

        int nonzero_blocks = 0;
        CHECK_HIPSPARSE(hipsparseXcsr2bsrNnz(
            handle, direction, matrix.rows, matrix.columns, csr_descriptor,
            device_csr_row_offsets, device_csr_column_indices, block_size,
            bsr_descriptor, device_bsr_row_offsets, &nonzero_blocks));
        CHECK_HIP(hipMalloc(reinterpret_cast<void**>(&device_bsr_column_indices),
                            static_cast<std::size_t>(nonzero_blocks) * sizeof(int)));
        CHECK_HIP(hipMalloc(reinterpret_cast<void**>(&device_bsr_values),
                            static_cast<std::size_t>(nonzero_blocks)
                                * block_size * block_size * sizeof(double)));
        CHECK_HIPSPARSE(hipsparseDcsr2bsr(
            handle, direction, matrix.rows, matrix.columns, csr_descriptor,
            device_csr_values, device_csr_row_offsets, device_csr_column_indices,
            block_size, bsr_descriptor, device_bsr_values,
            device_bsr_row_offsets, device_bsr_column_indices));

        for (int iteration = 0; iteration < WARMUP_ITERATIONS; ++iteration) {
            CHECK_HIPSPARSE(hipsparseDbsrmv(
                handle, direction, HIPSPARSE_OPERATION_NON_TRANSPOSE,
                block_rows, block_columns, nonzero_blocks, &alpha,
                bsr_descriptor, device_bsr_values, device_bsr_row_offsets,
                device_bsr_column_indices, block_size, device_x, &beta, device_y));
        }
        CHECK_HIP(hipDeviceSynchronize());

        hipEvent_t start = nullptr;
        hipEvent_t stop = nullptr;
        CHECK_HIP(hipEventCreate(&start));
        CHECK_HIP(hipEventCreate(&stop));
        CHECK_HIP(hipEventRecord(start));
        for (int iteration = 0; iteration < ITERATIONS; ++iteration) {
            CHECK_HIPSPARSE(hipsparseDbsrmv(
                handle, direction, HIPSPARSE_OPERATION_NON_TRANSPOSE,
                block_rows, block_columns, nonzero_blocks, &alpha,
                bsr_descriptor, device_bsr_values, device_bsr_row_offsets,
                device_bsr_column_indices, block_size, device_x, &beta, device_y));
        }
        CHECK_HIP(hipEventRecord(stop));
        CHECK_HIP(hipEventSynchronize(stop));

        float elapsed_time = 0.0F;
        CHECK_HIP(hipEventElapsedTime(&elapsed_time, start, stop));
        PROGRAM_METRICS metrics;
        metrics.comp_time = static_cast<double>(elapsed_time) / ITERATIONS;
        metrics.gflops =
            2.0 * matrix.nonzeros() * 1.0e-6 / metrics.comp_time;
        benchmark_results.push_back({block_size, metrics});

        if (best_block_size == 0 || metrics.comp_time < best_time) {
            best_block_size = block_size;
            best_time = metrics.comp_time;
            best_metrics = metrics;
            CHECK_HIP(hipMemcpy(best_result.data(), device_y,
                                best_result.size() * sizeof(double),
                                hipMemcpyDeviceToHost));
        }

        CHECK_HIP(hipEventDestroy(start));
        CHECK_HIP(hipEventDestroy(stop));
        CHECK_HIP(hipFree(device_bsr_values));
        CHECK_HIP(hipFree(device_bsr_row_offsets));
        CHECK_HIP(hipFree(device_bsr_column_indices));
        CHECK_HIP(hipFree(device_x));
        CHECK_HIP(hipFree(device_y));
    }

    CHECK_HIPSPARSE(hipsparseDestroyMatDescr(csr_descriptor));
    CHECK_HIPSPARSE(hipsparseDestroyMatDescr(bsr_descriptor));
    CHECK_HIPSPARSE(hipsparseDestroy(handle));
    CHECK_HIP(hipFree(device_csr_row_offsets));
    CHECK_HIP(hipFree(device_csr_column_indices));
    CHECK_HIP(hipFree(device_csr_values));

    if (best_block_size == 0) {
        Error("No valid BSR block size for this matrix");
        return EXIT_FAILURE;
    }
    const bool result_is_correct = check_results_consistency(
        matrix.rows, matrix.row_offsets, matrix.column_indices,
        matrix.values, input_x, best_result);
    print_benchmark_report(
        matrix_name, matrix.rows, matrix.columns, matrix.nonzeros(),
        benchmark_results, best_block_size, result_is_correct);
    writeCSV(
        "/public/home/congxing/CB-Sparse-DCU/cb-spmv-dcu/run_all/csv/hipsparse_bsr.csv",
        best_metrics, matrix_name, matrix.nonzeros());
    return result_is_correct ? EXIT_SUCCESS : EXIT_FAILURE;
}
