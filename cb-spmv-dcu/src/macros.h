#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <cstdint>
#include <unordered_map>
#include <assert.h>
#include <stdio.h>
#include <cstdlib>
#include <hip/hip_runtime.h>
#include <fstream>
#include <queue>
#include <algorithm>
#include <map>

#define CHECK_HIP(call)                                                     \
do                                                                          \
{                                                                           \
    const hipError_t error_code = call;                                     \
    if (error_code != hipSuccess)                                           \
    {                                                                       \
        printf("HIP Error:\n");                                             \
        printf("     - File: %s\n", __FILE__);                              \
        printf("     - Line: %d\n", __LINE__);                              \
        printf("     - Error code: %d\n", error_code);                      \
        printf("     - Error text: %s\n", hipGetErrorString(error_code));   \
        static_cast<void>(hipDeviceReset());                                \
        exit(1);                                                            \
    }                                                                       \
} while (0)

#define WARMUP_ITERATIONS 100
#define ITERATIONS 1000

typedef double ValType;

#ifndef CB_BLOCK_SIZE
#define CB_BLOCK_SIZE 16
#endif

#if CB_BLOCK_SIZE == 2
typedef uint32_t IndexTypeComp;
typedef uint8_t IndexTypeOrig;
typedef uint8_t DenseMaskType;
#define BLOCK_COORD_BITS 1
#elif CB_BLOCK_SIZE == 4
typedef uint32_t IndexTypeComp;
typedef uint8_t IndexTypeOrig;
typedef uint8_t DenseMaskType;
#define BLOCK_COORD_BITS 2
#elif CB_BLOCK_SIZE == 8
typedef uint32_t IndexTypeComp;
typedef uint8_t IndexTypeOrig;
typedef uint8_t DenseMaskType;
#define BLOCK_COORD_BITS 3
#elif CB_BLOCK_SIZE == 16
typedef uint32_t IndexTypeComp;
typedef uint8_t IndexTypeOrig;
typedef uint16_t DenseMaskType;
#define BLOCK_COORD_BITS 4
#elif CB_BLOCK_SIZE == 32
typedef uint32_t IndexTypeComp;
typedef uint16_t IndexTypeOrig;
typedef uint32_t DenseMaskType;
#define BLOCK_COORD_BITS 5
#elif CB_BLOCK_SIZE == 64
typedef uint32_t IndexTypeComp;
typedef uint16_t IndexTypeOrig;
typedef uint64_t DenseMaskType;
#define BLOCK_COORD_BITS 6
#else
#error "CB_BLOCK_SIZE must be 2, 4, 8, 16, 32, or 64"
#endif

#define BLOCK_SIZE CB_BLOCK_SIZE
#define WARP_SIZE 64
#define WAVE_SIZE 64
#ifndef CB_WAVES_PER_BLOCK
#define CB_WAVES_PER_BLOCK 8
#endif
#define WARP_PER_BLOCK CB_WAVES_PER_BLOCK
#define THREADS_PER_BLOCK (WARP_PER_BLOCK * WARP_SIZE)
#define LANES_PER_ROW (WAVE_SIZE / BLOCK_SIZE)
#define BLOCK_COLUMN_MASK ((IndexTypeComp(1) << BLOCK_COORD_BITS) - 1)
#ifndef CB_ROW_OWNED
#define CB_ROW_OWNED 0
#endif
#ifndef CB_FORCE_COO64
#define CB_FORCE_COO64 0
#endif
#ifndef CB_DENSE_SIXTEENTHS
#define CB_DENSE_SIXTEENTHS 15
#endif
#ifndef CB_PATTERN_WAVE
#define CB_PATTERN_WAVE 0
#endif

typedef uint8_t BlockType; 
#define COO 0
#define CSR 1
#define ELL 2
#define DENSE 3
#define BLOCK_TYPE_MASK 0x7f
#define UNIFORM_ROWS 0x80

#define RESET "\033[0m"
#define RED "\033[31m"
#define GREEN "\033[32m"
#define YELLOW "\033[33m"
#define BLUE "\033[34m"
#define MAGENTA "\033[35m"
#define CYAN "\033[36m"
#define WHITE "\033[37m"
