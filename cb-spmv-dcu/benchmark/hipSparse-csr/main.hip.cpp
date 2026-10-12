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
    if (matrix.nonzeros() == 0) {
        Error("Matrix has no nonzero entries");
        return EXIT_FAILURE;
    }

    constexpr unsigned int random_seed = 42;
    std::mt19937 random_engine(random_seed);
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    std::vector<double> host_x(matrix.columns);
    for (double& value : host_x) {
        value = distribution(random_engine);
    }
    std::vector<double> host_y(matrix.rows, 0.0);

    int* device_row_offsets = nullptr;
    int* device_column_indices = nullptr;
    double* device_values = nullptr;
    double* device_x = nullptr;
    double* device_y = nullptr;

    CHECK_HIP(hipMalloc(reinterpret_cast<void**>(&device_row_offsets),
                        matrix.row_offsets.size() * sizeof(int)));
    CHECK_HIP(hipMalloc(reinterpret_cast<void**>(&device_column_indices),
                        matrix.column_indices.size() * sizeof(int)));
    CHECK_HIP(hipMalloc(reinterpret_cast<void**>(&device_values),
                        matrix.values.size() * sizeof(double)));
    CHECK_HIP(hipMalloc(reinterpret_cast<void**>(&device_x),
                        host_x.size() * sizeof(double)));
    CHECK_HIP(hipMalloc(reinterpret_cast<void**>(&device_y),
                        host_y.size() * sizeof(double)));

    CHECK_HIP(hipMemcpy(device_row_offsets, matrix.row_offsets.data(),
                        matrix.row_offsets.size() * sizeof(int),
                        hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(device_column_indices, matrix.column_indices.data(),
                        matrix.column_indices.size() * sizeof(int),
                        hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(device_values, matrix.values.data(),
                        matrix.values.size() * sizeof(double),
                        hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(device_x, host_x.data(),
                        host_x.size() * sizeof(double),
                        hipMemcpyHostToDevice));
    CHECK_HIP(hipMemset(device_y, 0, host_y.size() * sizeof(double)));

    hipsparseHandle_t handle = nullptr;
    hipsparseSpMatDescr_t matrix_descriptor = nullptr;
    hipsparseDnVecDescr_t x_descriptor = nullptr;
    hipsparseDnVecDescr_t y_descriptor = nullptr;
    CHECK_HIPSPARSE(hipsparseCreate(&handle));
    CHECK_HIPSPARSE(hipsparseCreateCsr(
        &matrix_descriptor, matrix.rows, matrix.columns, matrix.nonzeros(),
        device_row_offsets, device_column_indices, device_values,
        HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_32I,
        HIPSPARSE_INDEX_BASE_ZERO, HIP_R_64F));
    CHECK_HIPSPARSE(hipsparseCreateDnVec(
        &x_descriptor, matrix.columns, device_x, HIP_R_64F));
    CHECK_HIPSPARSE(hipsparseCreateDnVec(
        &y_descriptor, matrix.rows, device_y, HIP_R_64F));

    constexpr double alpha = 1.0;
    constexpr double beta = 0.0;
    struct AlgorithmConfig {
        const char* name;
        hipsparseSpMVAlg_t algorithm;
    };
    constexpr AlgorithmConfig algorithms[] = {
        {"CSR_ALG1", HIPSPARSE_SPMV_CSR_ALG1},
        {"CSR_ALG2", HIPSPARSE_SPMV_CSR_ALG2},
    };
    std::vector<CSR_BENCHMARK_RESULT> benchmark_results;
    std::vector<double> best_result(matrix.rows, 0.0);
    std::string best_algorithm;
    double best_time = 0.0;

    for (const AlgorithmConfig& config : algorithms) {
        std::size_t workspace_size = 0;
        CHECK_HIPSPARSE(hipsparseSpMV_bufferSize(
            handle, HIPSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, matrix_descriptor, x_descriptor, &beta, y_descriptor,
            HIP_R_64F, config.algorithm, &workspace_size));

        void* workspace = nullptr;
        if (workspace_size > 0) {
            CHECK_HIP(hipMalloc(&workspace, workspace_size));
        }
        CHECK_HIPSPARSE(hipsparseSpMV_preprocess(
            handle, HIPSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, matrix_descriptor, x_descriptor, &beta, y_descriptor,
            HIP_R_64F, config.algorithm, workspace));
        CHECK_HIP(hipMemset(device_y, 0, host_y.size() * sizeof(double)));

        auto launch_spmv = [&]() {
            CHECK_HIPSPARSE(hipsparseSpMV(
                handle, HIPSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha, matrix_descriptor, x_descriptor, &beta, y_descriptor,
                HIP_R_64F, config.algorithm, workspace));
        };

        for (int iteration = 0; iteration < WARMUP_ITERATIONS; ++iteration) {
            launch_spmv();
        }
        CHECK_HIP(hipDeviceSynchronize());

        hipEvent_t start = nullptr;
        hipEvent_t stop = nullptr;
        CHECK_HIP(hipEventCreate(&start));
        CHECK_HIP(hipEventCreate(&stop));
        CHECK_HIP(hipEventRecord(start));
        for (int iteration = 0; iteration < ITERATIONS; ++iteration) {
            launch_spmv();
        }
        CHECK_HIP(hipEventRecord(stop));
        CHECK_HIP(hipEventSynchronize(stop));

        float elapsed_time = 0.0F;
        CHECK_HIP(hipEventElapsedTime(&elapsed_time, start, stop));
        PROGRAM_METRICS metrics;
        metrics.comp_time = static_cast<double>(elapsed_time) / ITERATIONS;
        metrics.gflops =
            2.0 * static_cast<double>(matrix.nonzeros()) * 1.0e-6
            / metrics.comp_time;
        benchmark_results.push_back({config.name, metrics});

        if (best_algorithm.empty() || metrics.comp_time < best_time) {
            best_algorithm = config.name;
            best_time = metrics.comp_time;
            CHECK_HIP(hipMemcpy(best_result.data(), device_y,
                                best_result.size() * sizeof(double),
                                hipMemcpyDeviceToHost));
        }

        CHECK_HIP(hipEventDestroy(start));
        CHECK_HIP(hipEventDestroy(stop));
        if (workspace != nullptr) {
            CHECK_HIP(hipFree(workspace));
        }
    }

    const bool result_is_correct = check_results_consistency(
        matrix.rows, matrix.row_offsets, matrix.column_indices,
        matrix.values, host_x, best_result);

    CHECK_HIPSPARSE(hipsparseDestroyDnVec(x_descriptor));
    CHECK_HIPSPARSE(hipsparseDestroyDnVec(y_descriptor));
    CHECK_HIPSPARSE(hipsparseDestroySpMat(matrix_descriptor));
    CHECK_HIPSPARSE(hipsparseDestroy(handle));
    CHECK_HIP(hipFree(device_row_offsets));
    CHECK_HIP(hipFree(device_column_indices));
    CHECK_HIP(hipFree(device_values));
    CHECK_HIP(hipFree(device_x));
    CHECK_HIP(hipFree(device_y));

    print_benchmark_report(
        matrix_name, matrix.rows, matrix.columns, matrix.nonzeros(),
        benchmark_results, best_algorithm, result_is_correct);
    return result_is_correct ? EXIT_SUCCESS : EXIT_FAILURE;
}
