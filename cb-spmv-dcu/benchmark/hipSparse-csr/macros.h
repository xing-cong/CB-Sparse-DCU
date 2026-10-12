#pragma once

#include <cstdlib>
#include <iostream>

#include <hip/hip_runtime.h>
#include <hipsparse.h>

constexpr int ITERATIONS = 1000;
constexpr int WARMUP_ITERATIONS = 100;

constexpr char COLOR_RESET[] = "\033[0m";
constexpr char COLOR_RED[] = "\033[31m";
constexpr char COLOR_GREEN[] = "\033[32m";

inline void check_hip(hipError_t status, const char* expression,
                      const char* file, int line) {
    if (status == hipSuccess) {
        return;
    }

    std::cerr << "HIP error at " << file << ':' << line
              << " for " << expression << ": "
              << hipGetErrorString(status) << '\n';
    std::exit(EXIT_FAILURE);
}

inline void check_hipsparse(hipsparseStatus_t status, const char* expression,
                            const char* file, int line) {
    if (status == HIPSPARSE_STATUS_SUCCESS) {
        return;
    }

    std::cerr << "hipSPARSE error at " << file << ':' << line
              << " for " << expression << ": "
              << static_cast<int>(status) << '\n';
    std::exit(EXIT_FAILURE);
}

#define CHECK_HIP(expression) \
    check_hip((expression), #expression, __FILE__, __LINE__)

#define CHECK_HIPSPARSE(expression) \
    check_hipsparse((expression), #expression, __FILE__, __LINE__)
