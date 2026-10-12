#pragma once

#include <string>

struct PROGRAM_METRICS {
    double comp_time = 0.0;
    double gflops = 0.0;
};

struct CSR_BENCHMARK_RESULT {
    std::string algorithm;
    PROGRAM_METRICS metrics;
};
