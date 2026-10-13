#pragma once

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "format.h"
#include "macros.h"

inline void Error(const std::string& message) {
    std::cerr << COLOR_RED << "[Error]" << COLOR_RESET << ": "
              << message << '\n';
}

inline bool check_results_consistency(
    int rows,
    const std::vector<int>& row_offsets,
    const std::vector<int>& column_indices,
    const std::vector<double>& values,
    const std::vector<double>& x,
    const std::vector<double>& actual) {
    std::vector<double> expected(rows, 0.0);

    for (int row = 0; row < rows; ++row) {
        for (int index = row_offsets[row]; index < row_offsets[row + 1]; ++index) {
            expected[row] += values[index] * x[column_indices[index]];
        }
    }

    constexpr double absolute_tolerance = 1e-3;
    constexpr double relative_tolerance = 1e-4;
    for (int row = 0; row < rows; ++row) {
        if (!std::isfinite(expected[row]) || !std::isfinite(actual[row])) {
            Error("Non-finite result at index " + std::to_string(row));
            return false;
        }
        const double error = std::abs(expected[row] - actual[row]);
        const double tolerance =
            absolute_tolerance + relative_tolerance * std::abs(expected[row]);
        if (error > tolerance) {
            Error("Result differs at index " + std::to_string(row)
                  + ": expected " + std::to_string(expected[row])
                  + ", got " + std::to_string(actual[row])
                  + ", absolute error " + std::to_string(error)
                  + ", tolerance " + std::to_string(tolerance));
            return false;
        }
    }

    return true;
}

inline void print_benchmark_report(
    const std::string& matrix_name,
    int rows,
    int columns,
    int nonzeros,
    const std::vector<CSR_BENCHMARK_RESULT>& results,
    const std::string& best_algorithm,
    bool result_is_correct) {
    const std::string divider(64, '=');
    const std::string separator(64, '-');

    std::cout << '\n'
              << divider << '\n'
              << "                 hipSPARSE CSR SpMV Benchmark\n"
              << divider << '\n'
              << std::left
              << std::setw(14) << "Matrix" << ": " << matrix_name << '\n'
              << std::setw(14) << "Dimensions" << ": "
              << rows << " x " << columns << '\n'
              << std::setw(14) << "Nonzeros" << ": " << nonzeros << '\n'
              << separator << '\n'
              << std::left << std::setw(24) << "Algorithm"
              << std::right << std::setw(20) << "Time (ms)"
              << std::setw(14) << "GFLOPS"
              << std::setw(6) << "" << '\n'
              << separator << '\n';

    const CSR_BENCHMARK_RESULT* best_result = nullptr;
    for (const CSR_BENCHMARK_RESULT& result : results) {
        const bool is_best = result.algorithm == best_algorithm;
        if (is_best) {
            best_result = &result;
        }
        std::cout << std::left << std::setw(24) << result.algorithm
                  << std::right << std::fixed << std::setprecision(6)
                  << std::setw(20) << result.metrics.comp_time
                  << std::setw(14) << result.metrics.gflops
                  << std::setw(6) << (is_best ? "BEST" : "") << '\n';
    }

    std::cout << separator << '\n';
    if (best_result != nullptr) {
        std::cout << std::left << std::setw(14) << "Best result" << ": "
                  << best_result->algorithm << ", "
                  << best_result->metrics.comp_time << " ms, "
                  << best_result->metrics.gflops << " GFLOPS\n";
    }
    std::cout << std::left << std::setw(14) << "Validation" << ": "
              << (result_is_correct ? COLOR_GREEN : COLOR_RED)
              << (result_is_correct ? "PASS" : "FAIL")
              << COLOR_RESET << '\n'
              << divider << '\n';
}

inline bool writeCSV(
    const std::string& filename,
    const PROGRAM_METRICS& metrics,
    const std::string& matrix_name,
    int nonzeros) {
    std::ofstream file(filename, std::ios::app);
    if (!file) {
        Error("Failed to open " + filename);
        return false;
    }

    file << matrix_name << ','
         << nonzeros << ','
         << metrics.comp_time << ','
         << metrics.gflops << '\n';
    file.close();
    std::cout << "[Info]: Data written to " << filename << " successfully.\n";
    return true;
}
