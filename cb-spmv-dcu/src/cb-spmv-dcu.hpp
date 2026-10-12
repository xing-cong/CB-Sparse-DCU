#pragma once

#include "format.h"
#include "utils.h"

namespace cb_spmv_detail {

enum ExecutionMode : int
{
    EXEC_STANDARD = 0,
    EXEC_SMALL_DIRECT = 1,
    EXEC_PATTERN_REUSE_UNROLLED = 2,
    EXEC_PATTERN_REUSE_SINGLE = 3
};

__host__ __device__ inline size_t align_up(size_t offset, size_t alignment)
{
    return (offset + alignment - 1) & ~(alignment - 1);
}

__device__ __forceinline__ ValType dpp_reduce_4(ValType value)
{
    union Bits
    {
        ValType value;
        unsigned int part[2];
    } current, moved;

    current.value = value;
    moved.part[0] = __hip_move_dpp(current.part[0], 0x111, 0xf, 0xf, false);
    moved.part[1] = __hip_move_dpp(current.part[1], 0x111, 0xf, 0xf, false);
    current.value += moved.value;
    moved.part[0] = __hip_move_dpp(current.part[0], 0x112, 0xf, 0xf, false);
    moved.part[1] = __hip_move_dpp(current.part[1], 0x112, 0xf, 0xf, false);
    current.value += moved.value;
    return current.value;
}

__device__ __forceinline__ ValType dpp_reduce_8(ValType value)
{
    union Bits
    {
        ValType value;
        unsigned int part[2];
    } current, moved;

    current.value = value;
    moved.part[0] = __hip_move_dpp(current.part[0], 0x111, 0xf, 0xf, false);
    moved.part[1] = __hip_move_dpp(current.part[1], 0x111, 0xf, 0xf, false);
    current.value += moved.value;
    moved.part[0] = __hip_move_dpp(current.part[0], 0x112, 0xf, 0xf, false);
    moved.part[1] = __hip_move_dpp(current.part[1], 0x112, 0xf, 0xf, false);
    current.value += moved.value;
    moved.part[0] = __hip_move_dpp(current.part[0], 0x114, 0xf, 0xf, false);
    moved.part[1] = __hip_move_dpp(current.part[1], 0x114, 0xf, 0xf, false);
    current.value += moved.value;
    return current.value;
}

__device__ __forceinline__ ValType dpp_reduce_2(ValType value)
{
    union Bits
    {
        ValType value;
        unsigned int part[2];
    } current, moved;

    current.value = value;
    moved.part[0] = __hip_move_dpp(current.part[0], 0x111, 0xf, 0xf, false);
    moved.part[1] = __hip_move_dpp(current.part[1], 0x111, 0xf, 0xf, false);
    current.value += moved.value;
    return current.value;
}

__device__ __forceinline__ ValType shuffle_reduce_large_row(ValType value)
{
#pragma unroll
    for (int offset = 1; offset < LANES_PER_ROW; offset <<= 1)
    {
        value += __shfl_up(value, offset, LANES_PER_ROW);
    }
    return value;
}

__device__ __forceinline__ ValType reduce_uniform_row(ValType value)
{
#pragma unroll
    for (int offset = BLOCK_SIZE / 2; offset > 0; offset >>= 1)
    {
        value += __shfl_down(value, offset, BLOCK_SIZE);
    }
    return __shfl(value, 0, WAVE_SIZE);
}

__device__ __forceinline__ ValType wave_reduce_64(ValType value)
{
#pragma unroll
    for (int offset = 32; offset > 0; offset >>= 1)
    {
        value += __shfl_down(value, offset, WAVE_SIZE);
    }
    return value;
}

__device__ __forceinline__ ValType coo_segmented_reduce(
    ValType value, int row, int lane)
{
#pragma unroll
    for (int offset = 1; offset < BLOCK_SIZE; offset <<= 1)
    {
        const int previous_row = __shfl_up(row, offset, WAVE_SIZE);
        const ValType previous_value = __shfl_up(value, offset, WAVE_SIZE);
        if (lane >= offset && row == previous_row)
        {
            value += previous_value;
        }
    }
    return value;
}

__device__ __forceinline__ ValType coo_position_reduce(
    ValType value, int position, bool long_rows)
{
#pragma unroll
    for (int offset = 1; offset < 8; offset <<= 1)
    {
        const ValType previous_value = __shfl_up(value, offset, WAVE_SIZE);
        if (position >= offset)
        {
            value += previous_value;
        }
    }
    if (long_rows)
    {
#pragma unroll
        for (int offset = 8; offset < BLOCK_SIZE; offset <<= 1)
        {
            const ValType previous_value = __shfl_up(
                value, offset, WAVE_SIZE);
            if (position >= offset)
            {
                value += previous_value;
            }
        }
    }
    return value;
}

__device__ __forceinline__ void accumulate_output(
    ValType *output, int row, ValType value)
{
#if CB_ROW_OWNED
    output[row] += value;
#else
    atomicAdd(&output[row], value);
#endif
}

template <bool UNIT_VALUES, bool COMPACT_COO, int EXECUTION_MODE>
__global__ void spmv_hip_kernel_comp(int gb_nnzb, bool aggregate_coo,
    bool direct_coo_columns, int *nnz_per_blk,
    BlockType *type_per_blk, int *gb_row_idx, int *gb_col_idx,
    int *segment_rows, int *segment_begin, int *segment_end,
    uint8_t *segment_direct, uint8_t *segment_uniform,
    int *output_epochs, int output_epoch,
    char **blk_head_index_array, ValType *d_x, ValType *d_y)
{
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int wave_in_block = threadIdx.x >> 6;
    int tid_in_wave = threadIdx.x & 0b111111;

    if constexpr (EXECUTION_MODE == EXEC_PATTERN_REUSE_SINGLE)
    {
        const int pattern = global_id >> 6;
        if (pattern < gb_nnzb)
        {
            ValType sum = 0.0;
            for (int index = segment_rows[pattern] + tid_in_wave;
                 index < segment_rows[pattern + 1]; index += WAVE_SIZE)
            {
                sum += __ldg(&d_x[__ldg(&segment_begin[index])]);
            }
            sum = __shfl(wave_reduce_64(sum), 0, WAVE_SIZE);
            for (int index = segment_end[pattern] + tid_in_wave;
                 index < segment_end[pattern + 1]; index += WAVE_SIZE)
            {
                d_y[output_epochs[index]] = sum;
            }
        }
        return;
    }

    if constexpr (EXECUTION_MODE == EXEC_PATTERN_REUSE_UNROLLED)
    {
        const int pattern = global_id >> 6;
        if (pattern < gb_nnzb)
        {
            ValType sum = 0.0;
            int index = segment_rows[pattern] + tid_in_wave;
            const int pattern_end = segment_rows[pattern + 1];
            for (; index + WAVE_SIZE < pattern_end;
                 index += 2 * WAVE_SIZE)
            {
                sum += __ldg(&d_x[__ldg(&segment_begin[index])]);
                sum += __ldg(&d_x[
                    __ldg(&segment_begin[index + WAVE_SIZE])]);
            }
            if (index < pattern_end)
            {
                sum += __ldg(&d_x[__ldg(&segment_begin[index])]);
            }
            sum = __shfl(wave_reduce_64(sum), 0, WAVE_SIZE);
            for (int index = segment_end[pattern] + tid_in_wave;
                 index < segment_end[pattern + 1]; index += WAVE_SIZE)
            {
                d_y[output_epochs[index]] = sum;
            }
        }
        return;
    }

    __shared__ ValType s_x[WARP_PER_BLOCK * BLOCK_SIZE];
#if CB_ROW_OWNED
    __shared__ ValType s_y[WARP_PER_BLOCK * BLOCK_SIZE];
    __shared__ int initialize_output;
    ValType *s_y_wave = &s_y[wave_in_block * BLOCK_SIZE];
    const int segment_id = blockIdx.x;
    const int owned_row = segment_rows[segment_id];
    const int owned_begin = segment_begin[segment_id];
    const int owned_end = segment_end[segment_id];
    constexpr bool small_direct = EXECUTION_MODE == EXEC_SMALL_DIRECT;
    const bool direct_output = small_direct ? true
        : segment_direct[segment_id];
    const bool owned_uniform = small_direct ? false
        : segment_uniform[segment_id];
    if constexpr (!small_direct)
    {
        if (!direct_output)
        {
            if (threadIdx.x == 0)
            {
                while (true)
                {
                    const int state = atomicAdd(&output_epochs[owned_row], 0);
                    if (state == output_epoch)
                    {
                        initialize_output = 0;
                        break;
                    }
                    if (state >= 0
                        && atomicCAS(&output_epochs[owned_row], state,
                            -output_epoch) == state)
                    {
                        initialize_output = 1;
                        break;
                    }
                }
            }
            __syncthreads();
            if (initialize_output && wave_in_block == 0
                && tid_in_wave < BLOCK_SIZE)
            {
                d_y[owned_row * BLOCK_SIZE + tid_in_wave] = 0.0;
            }
            __syncthreads();
            if (threadIdx.x == 0)
            {
                if (initialize_output)
                {
                    __threadfence();
                    atomicExch(&output_epochs[owned_row], output_epoch);
                }
                else
                {
                    while (atomicAdd(&output_epochs[owned_row], 0)
                        != output_epoch)
                    {
                    }
                }
            }
            __syncthreads();
        }
    }
    if (tid_in_wave < BLOCK_SIZE)
    {
        s_y_wave[tid_in_wave] = 0.0;
    }
    __builtin_amdgcn_wave_barrier();
    const int owned_iterations = (owned_end - owned_begin
        + WARP_PER_BLOCK - 1) / WARP_PER_BLOCK;
    for (int owned_iteration = 0;
         owned_iteration < owned_iterations; ++owned_iteration)
    {
        int blk_id = owned_begin + wave_in_block
            + owned_iteration * WARP_PER_BLOCK;
        bool active_block = blk_id < owned_end
            && nnz_per_blk[blk_id] > 0;
#else
    constexpr bool direct_output = false;
    constexpr bool owned_uniform = false;
    int blk_id = global_id >> 6;
    bool active_block = blk_id < gb_nnzb && nnz_per_blk[blk_id] > 0;
#endif
    ValType *s_x_wave = &s_x[wave_in_block * BLOCK_SIZE];
    if (active_block && tid_in_wave < BLOCK_SIZE)
    {
        s_x_wave[tid_in_wave] =
            d_x[gb_col_idx[blk_id] * BLOCK_SIZE + tid_in_wave];
    }
    __builtin_amdgcn_wave_barrier();

    if (active_block)
    {
        int blk_nnz = nnz_per_blk[blk_id];
        BlockType encoded_type = type_per_blk[blk_id];
        BlockType blk_type = encoded_type & BLOCK_TYPE_MASK;
        bool uniform_rows = encoded_type & UNIFORM_ROWS;
        char *blk_head_index = blk_head_index_array[blk_id];
        int blk_row_idx = gb_row_idx[blk_id];
        int blk_col_idx = gb_col_idx[blk_id];
        int gb_row_offset = blk_row_idx * BLOCK_SIZE;
#if CB_ROW_OWNED
        ValType *block_output = s_y_wave;
#else
        ValType *block_output = d_y + gb_row_offset;
#endif

        if (blk_type == COO)
        {
            constexpr int index_size = COMPACT_COO
                ? sizeof(uint16_t) : sizeof(IndexTypeComp);
            int padding = (blk_nnz * index_size) % sizeof(ValType);
            if (padding != 0) padding = sizeof(ValType) - padding;
            ValType *d_val = (ValType *)(blk_head_index
                + blk_nnz * index_size + padding);

            if (tid_in_wave < blk_nnz)
            {
                IndexTypeComp coo_idx = COMPACT_COO
                    ? reinterpret_cast<uint16_t *>(blk_head_index)[tid_in_wave]
                    : reinterpret_cast<IndexTypeComp *>(
                        blk_head_index)[tid_in_wave];
                int column;
                int row;
                if constexpr (COMPACT_COO)
                {
                    column = coo_idx & 0xfff;
                    row = (coo_idx >> 12) & BLOCK_COLUMN_MASK;
                }
                else if constexpr (BLOCK_SIZE == 2 || BLOCK_SIZE == 4
                    || BLOCK_SIZE == 8 || BLOCK_SIZE == 16
                    || BLOCK_SIZE == 32 || BLOCK_SIZE == 64)
                {
                    column = direct_coo_columns
                        ? coo_idx & 0xffff
                        : coo_idx & BLOCK_COLUMN_MASK;
                    row = (coo_idx >> (direct_coo_columns
                        ? 16 : BLOCK_COORD_BITS)) & BLOCK_COLUMN_MASK;
                }
                else
                {
                    column = coo_idx & BLOCK_COLUMN_MASK;
                    row = (coo_idx >> BLOCK_COORD_BITS) & BLOCK_COLUMN_MASK;
                }
                const ValType x_value = direct_coo_columns
                    ? d_x[column] : s_x_wave[column];
                const ValType contribution = owned_uniform && row != 0
                    ? ValType(0) : UNIT_VALUES
                        ? x_value : d_val[tid_in_wave] * x_value;
                if (aggregate_coo || CB_ROW_OWNED)
                {
                    if constexpr (COMPACT_COO)
                    {
                        const ValType row_sum = coo_segmented_reduce(
                            contribution, static_cast<int>(row), tid_in_wave);
                        const int next_row = __shfl_down(
                            static_cast<int>(row), 1, WAVE_SIZE);
                        if (tid_in_wave == blk_nnz - 1 || row != next_row)
                        {
                            accumulate_output(block_output, row, row_sum);
                        }
                    }
                    else if constexpr (BLOCK_SIZE == 64)
                    {
                        const ValType row_sum = coo_segmented_reduce(
                            contribution, static_cast<int>(row), tid_in_wave);
                        const int next_row = __shfl_down(
                            static_cast<int>(row), 1, WAVE_SIZE);
                        if (tid_in_wave == blk_nnz - 1 || row != next_row)
                        {
                            accumulate_output(block_output, row, row_sum);
                        }
                    }
                    else if (direct_coo_columns || BLOCK_SIZE == 32)
                    {
                        const int position = (coo_idx
                            >> (direct_coo_columns
                                ? 16 + BLOCK_COORD_BITS : 10)) & 0x1f;
                        const ValType row_sum = coo_position_reduce(
                            contribution, position, coo_idx
                                & (direct_coo_columns
                                    ? (1 << (21 + BLOCK_COORD_BITS))
                                    : 0x8000));
                        const int next_position = __shfl_down(
                            position, 1, WAVE_SIZE);
                        if (tid_in_wave == blk_nnz - 1 || next_position == 0)
                        {
                            accumulate_output(block_output, row, row_sum);
                        }
                    }
                    else
                    {
                        const ValType row_sum = coo_segmented_reduce(
                            contribution, static_cast<int>(row), tid_in_wave);
                        const int next_row = __shfl_down(
                            static_cast<int>(row), 1, WAVE_SIZE);
                        if (tid_in_wave == blk_nnz - 1 || row != next_row)
                        {
                            accumulate_output(block_output, row, row_sum);
                        }
                    }
                }
                else
                {
                    accumulate_output(block_output, row, contribution);
                }
            }
        }
        else if (blk_type == CSR)
        {
            int padding = ((BLOCK_SIZE + 1 + blk_nnz)
                * sizeof(IndexTypeOrig)) % sizeof(ValType);
            if (padding != 0) padding = sizeof(ValType) - padding;
            IndexTypeOrig *d_row_ptr = (IndexTypeOrig *)blk_head_index;
            IndexTypeOrig *d_col_idx = (IndexTypeOrig *)((char *)d_row_ptr
                + (BLOCK_SIZE + 1) * sizeof(IndexTypeOrig));
            ValType *d_val = (ValType *)((char *)d_col_idx
                + blk_nnz * sizeof(IndexTypeOrig) + padding);
            ValType sum = 0.0;
            int row_id = uniform_rows ? 0 : tid_in_wave / LANES_PER_ROW;
            int offset = uniform_rows ? tid_in_wave
                : tid_in_wave % LANES_PER_ROW;
            int stride = uniform_rows ? BLOCK_SIZE : LANES_PER_ROW;
            for (int k = d_row_ptr[row_id] + offset;
                 k < d_row_ptr[row_id + 1]; k += stride)
            {
                if constexpr (UNIT_VALUES)
                {
                    sum += s_x_wave[d_col_idx[k]];
                }
                else
                {
                    sum += s_x_wave[d_col_idx[k]] * d_val[k];
                }
            }
            if (uniform_rows)
            {
                sum = reduce_uniform_row(sum);
                if (owned_uniform)
                {
                    if (tid_in_wave == 0 && sum != 0.0)
                    {
                        accumulate_output(block_output, 0, sum);
                    }
                }
                else if (tid_in_wave < BLOCK_SIZE && sum != 0.0)
                {
                    accumulate_output(block_output, tid_in_wave, sum);
                }
            }
            else
            {
                if constexpr (LANES_PER_ROW == 1)
                {
                }
                else if constexpr (LANES_PER_ROW > 8)
                {
                    sum = shuffle_reduce_large_row(sum);
                }
                else if constexpr (LANES_PER_ROW == 8)
                {
                    sum = dpp_reduce_8(sum);
                }
                else if constexpr (LANES_PER_ROW == 4)
                {
                    sum = dpp_reduce_4(sum);
                }
                else
                {
                    sum = dpp_reduce_2(sum);
                }
                if (offset == LANES_PER_ROW - 1 && sum != 0.0)
                {
                    accumulate_output(block_output, row_id, sum);
                }
            }
        }
        else if (blk_type == DENSE)
        {
            ValType sum = 0.0;
            if (uniform_rows)
            {
                if (tid_in_wave < BLOCK_SIZE)
                {
                    if constexpr (UNIT_VALUES)
                    {
                        const DenseMaskType row_mask =
                            reinterpret_cast<const DenseMaskType *>(
                                blk_head_index)[0];
                        if (row_mask & (DenseMaskType(1) << tid_in_wave))
                        {
                            sum = s_x_wave[tid_in_wave];
                        }
                    }
                    else
                    {
                        ValType *d_val = (ValType *)blk_head_index;
                        sum = d_val[tid_in_wave] * s_x_wave[tid_in_wave];
                    }
                }
                sum = reduce_uniform_row(sum);
                if (owned_uniform)
                {
                    if (tid_in_wave == 0 && sum != 0.0)
                    {
                        accumulate_output(block_output, 0, sum);
                    }
                }
                else if (tid_in_wave < BLOCK_SIZE && sum != 0.0)
                {
                    accumulate_output(block_output, tid_in_wave, sum);
                }
            }
            else
            {
                int row = tid_in_wave % BLOCK_SIZE;
                int offset = tid_in_wave / BLOCK_SIZE;
                constexpr int columns_per_lane = LANES_PER_ROW <= BLOCK_SIZE
                    ? BLOCK_SIZE / LANES_PER_ROW : 1;
                const int col_begin = LANES_PER_ROW <= BLOCK_SIZE
                    ? offset * columns_per_lane : offset;
                const int col_end = LANES_PER_ROW <= BLOCK_SIZE
                    ? (offset + 1) * columns_per_lane : BLOCK_SIZE;
                constexpr int col_stride = LANES_PER_ROW <= BLOCK_SIZE
                    ? 1 : LANES_PER_ROW;
                if constexpr (UNIT_VALUES)
                {
                    const DenseMaskType row_mask =
                        reinterpret_cast<const DenseMaskType *>(
                            blk_head_index)[row];
                    for (int col = col_begin; col < col_end;
                         col += col_stride)
                    {
                        if (row_mask & (DenseMaskType(1) << col))
                        {
                            sum += s_x_wave[col];
                        }
                    }
                }
                else
                {
                    ValType *d_val = (ValType *)blk_head_index;
                    for (int col = col_begin; col < col_end;
                         col += col_stride)
                    {
                        sum += d_val[row * BLOCK_SIZE + col] * s_x_wave[col];
                    }
                }
                for (int delta = BLOCK_SIZE; delta < WAVE_SIZE; delta <<= 1)
                {
                    sum += __shfl_xor(sum, delta, WAVE_SIZE);
                }
                if (tid_in_wave < BLOCK_SIZE)
                {
                    accumulate_output(block_output, row, sum);
                }
            }
        }
    }
#if CB_ROW_OWNED
    }
    __syncthreads();
    if (wave_in_block == 0)
    {
        ValType row_sum = 0.0;
#pragma unroll
        for (int wave = 0; wave < WARP_PER_BLOCK; ++wave)
        {
            row_sum += s_y[wave * BLOCK_SIZE
                + (owned_uniform ? 0 : tid_in_wave)];
        }
        if (owned_uniform)
        {
            row_sum = __shfl(row_sum, 0, WAVE_SIZE);
        }
        if (tid_in_wave < BLOCK_SIZE)
        {
            if (direct_output)
            {
                d_y[owned_row * BLOCK_SIZE + tid_in_wave] = row_sum;
            }
            else
            {
                atomicAdd(&d_y[owned_row * BLOCK_SIZE + tid_in_wave],
                    row_sum);
            }
        }
    }
#endif
}

template <bool UNIT_VALUES, bool COMPACT_COO, int EXECUTION_MODE>
__global__ void spmv_hip_kernel_gather(int gb_nnzb, bool aggregate_coo,
    bool direct_coo_columns, int *nnz_per_blk,
    BlockType *type_per_blk, int *gb_row_idx, int *gb_col_idx,
    int *segment_rows, int *segment_begin, int *segment_end,
    uint8_t *segment_direct, uint8_t *segment_uniform,
    int *output_epochs, int output_epoch,
    char **blk_head_index_array, int *d_block_cols,
    int *d_block_cols_offset, ValType *d_x, ValType *d_y)
{
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int wave_in_block = threadIdx.x >> 6;
    int tid_in_wave = threadIdx.x & 0b111111;

    if constexpr (EXECUTION_MODE == EXEC_PATTERN_REUSE_SINGLE)
    {
        const int pattern = global_id >> 6;
        if (pattern < gb_nnzb)
        {
            ValType sum = 0.0;
            for (int index = segment_rows[pattern] + tid_in_wave;
                 index < segment_rows[pattern + 1]; index += WAVE_SIZE)
            {
                sum += __ldg(&d_x[__ldg(&segment_begin[index])]);
            }
            sum = __shfl(wave_reduce_64(sum), 0, WAVE_SIZE);
            for (int index = segment_end[pattern] + tid_in_wave;
                 index < segment_end[pattern + 1]; index += WAVE_SIZE)
            {
                d_y[output_epochs[index]] = sum;
            }
        }
        return;
    }

    if constexpr (EXECUTION_MODE == EXEC_PATTERN_REUSE_UNROLLED)
    {
        const int pattern = global_id >> 6;
        if (pattern < gb_nnzb)
        {
            ValType sum = 0.0;
            int index = segment_rows[pattern] + tid_in_wave;
            const int pattern_end = segment_rows[pattern + 1];
            for (; index + WAVE_SIZE < pattern_end;
                 index += 2 * WAVE_SIZE)
            {
                sum += __ldg(&d_x[__ldg(&segment_begin[index])]);
                sum += __ldg(&d_x[
                    __ldg(&segment_begin[index + WAVE_SIZE])]);
            }
            if (index < pattern_end)
            {
                sum += __ldg(&d_x[__ldg(&segment_begin[index])]);
            }
            sum = __shfl(wave_reduce_64(sum), 0, WAVE_SIZE);
            for (int index = segment_end[pattern] + tid_in_wave;
                 index < segment_end[pattern + 1]; index += WAVE_SIZE)
            {
                d_y[output_epochs[index]] = sum;
            }
        }
        return;
    }
#if CB_ROW_OWNED
    __shared__ ValType s_y[WARP_PER_BLOCK * BLOCK_SIZE];
    __shared__ int initialize_output;
    ValType *s_y_wave = &s_y[wave_in_block * BLOCK_SIZE];
    const int segment_id = blockIdx.x;
    const int owned_row = segment_rows[segment_id];
    const int owned_begin = segment_begin[segment_id];
    const int owned_end = segment_end[segment_id];
    constexpr bool small_direct = EXECUTION_MODE == EXEC_SMALL_DIRECT;
    const bool direct_output = small_direct ? true
        : segment_direct[segment_id];
    const bool owned_uniform = small_direct ? false
        : segment_uniform[segment_id];
    if constexpr (!small_direct)
    {
        if (!direct_output)
        {
            if (threadIdx.x == 0)
            {
                while (true)
                {
                    const int state = atomicAdd(&output_epochs[owned_row], 0);
                    if (state == output_epoch)
                    {
                        initialize_output = 0;
                        break;
                    }
                    if (state >= 0
                        && atomicCAS(&output_epochs[owned_row], state,
                            -output_epoch) == state)
                    {
                        initialize_output = 1;
                        break;
                    }
                }
            }
            __syncthreads();
            if (initialize_output && wave_in_block == 0
                && tid_in_wave < BLOCK_SIZE)
            {
                d_y[owned_row * BLOCK_SIZE + tid_in_wave] = 0.0;
            }
            __syncthreads();
            if (threadIdx.x == 0)
            {
                if (initialize_output)
                {
                    __threadfence();
                    atomicExch(&output_epochs[owned_row], output_epoch);
                }
                else
                {
                    while (atomicAdd(&output_epochs[owned_row], 0)
                        != output_epoch)
                    {
                    }
                }
            }
            __syncthreads();
        }
    }
    if (tid_in_wave < BLOCK_SIZE)
    {
        s_y_wave[tid_in_wave] = 0.0;
    }
    __builtin_amdgcn_wave_barrier();
    const int owned_iterations = (owned_end - owned_begin
        + WARP_PER_BLOCK - 1) / WARP_PER_BLOCK;
    for (int owned_iteration = 0;
         owned_iteration < owned_iterations; ++owned_iteration)
    {
        int blk_id = owned_begin + wave_in_block
            + owned_iteration * WARP_PER_BLOCK;
        bool active_block = blk_id < owned_end
            && nnz_per_blk[blk_id] > 0;
#else
    constexpr bool direct_output = false;
    constexpr bool owned_uniform = false;
    int blk_id = global_id >> 6;
    bool active_block = blk_id < gb_nnzb && nnz_per_blk[blk_id] > 0;
#endif

    if (active_block)
    {
        int blk_nnz = nnz_per_blk[blk_id];
        BlockType encoded_type = type_per_blk[blk_id];
        BlockType blk_type = encoded_type & BLOCK_TYPE_MASK;
        bool uniform_rows = encoded_type & UNIFORM_ROWS;
        char *blk_head_index = blk_head_index_array[blk_id];
        int blk_row_idx = gb_row_idx[blk_id];
        int blk_col_idx = gb_col_idx[blk_id];
        int gb_row_offset = blk_row_idx * BLOCK_SIZE;
#if CB_ROW_OWNED
        ValType *block_output = s_y_wave;
#else
        ValType *block_output = d_y + gb_row_offset;
#endif
        int gathered_row_offset = d_block_cols_offset[blk_row_idx];
        int gathered_col_count = d_block_cols_offset[blk_row_idx + 1]
            - gathered_row_offset;

        if (blk_type == COO)
        {
            constexpr int index_size = COMPACT_COO
                ? sizeof(uint16_t) : sizeof(IndexTypeComp);
            int padding = (blk_nnz * index_size) % sizeof(ValType);
            if (padding != 0) padding = sizeof(ValType) - padding;
            ValType *d_val = (ValType *)(blk_head_index
                + blk_nnz * index_size + padding);

            if (tid_in_wave < blk_nnz)
            {
                IndexTypeComp coo_idx = COMPACT_COO
                    ? reinterpret_cast<uint16_t *>(blk_head_index)[tid_in_wave]
                    : reinterpret_cast<IndexTypeComp *>(
                        blk_head_index)[tid_in_wave];
                int original_col;
                int row;
                if constexpr (COMPACT_COO)
                {
                    original_col = coo_idx & 0xfff;
                    row = (coo_idx >> 12) & BLOCK_COLUMN_MASK;
                }
                else if constexpr (BLOCK_SIZE == 2 || BLOCK_SIZE == 4
                    || BLOCK_SIZE == 8 || BLOCK_SIZE == 16
                    || BLOCK_SIZE == 32 || BLOCK_SIZE == 64)
                {
                    const int column = direct_coo_columns
                        ? coo_idx & 0xffff
                        : coo_idx & BLOCK_COLUMN_MASK;
                    row = (coo_idx >> (direct_coo_columns
                        ? 16 : BLOCK_COORD_BITS)) & BLOCK_COLUMN_MASK;
                    original_col = direct_coo_columns ? column
                        : d_block_cols[gathered_row_offset
                            + blk_col_idx * BLOCK_SIZE + column];
                }
                else
                {
                    const int column = coo_idx & BLOCK_COLUMN_MASK;
                    row = (coo_idx >> BLOCK_COORD_BITS) & BLOCK_COLUMN_MASK;
                    original_col = d_block_cols[gathered_row_offset
                        + blk_col_idx * BLOCK_SIZE + column];
                }
                const ValType contribution = owned_uniform && row != 0
                    ? ValType(0) : UNIT_VALUES
                        ? d_x[original_col]
                        : d_val[tid_in_wave] * d_x[original_col];
                if (aggregate_coo || CB_ROW_OWNED)
                {
                    if constexpr (COMPACT_COO)
                    {
                        const ValType row_sum = coo_segmented_reduce(
                            contribution, static_cast<int>(row), tid_in_wave);
                        const int next_row = __shfl_down(
                            static_cast<int>(row), 1, WAVE_SIZE);
                        if (tid_in_wave == blk_nnz - 1 || row != next_row)
                        {
                            accumulate_output(block_output, row, row_sum);
                        }
                    }
                    else if constexpr (BLOCK_SIZE == 64)
                    {
                        const ValType row_sum = coo_segmented_reduce(
                            contribution, static_cast<int>(row), tid_in_wave);
                        const int next_row = __shfl_down(
                            static_cast<int>(row), 1, WAVE_SIZE);
                        if (tid_in_wave == blk_nnz - 1 || row != next_row)
                        {
                            accumulate_output(block_output, row, row_sum);
                        }
                    }
                    else if (direct_coo_columns || BLOCK_SIZE == 32)
                    {
                        const int position = (coo_idx
                            >> (direct_coo_columns
                                ? 16 + BLOCK_COORD_BITS : 10)) & 0x1f;
                        const ValType row_sum = coo_position_reduce(
                            contribution, position, coo_idx
                                & (direct_coo_columns
                                    ? (1 << (21 + BLOCK_COORD_BITS))
                                    : 0x8000));
                        const int next_position = __shfl_down(
                            position, 1, WAVE_SIZE);
                        if (tid_in_wave == blk_nnz - 1 || next_position == 0)
                        {
                            accumulate_output(block_output, row, row_sum);
                        }
                    }
                    else
                    {
                        const ValType row_sum = coo_segmented_reduce(
                            contribution, static_cast<int>(row), tid_in_wave);
                        const int next_row = __shfl_down(
                            static_cast<int>(row), 1, WAVE_SIZE);
                        if (tid_in_wave == blk_nnz - 1 || row != next_row)
                        {
                            accumulate_output(block_output, row, row_sum);
                        }
                    }
                }
                else
                {
                    accumulate_output(block_output, row, contribution);
                }
            }
        }
        else if (blk_type == CSR)
        {
            int padding = ((BLOCK_SIZE + 1 + blk_nnz)
                * sizeof(IndexTypeOrig)) % sizeof(ValType);
            if (padding != 0) padding = sizeof(ValType) - padding;
            IndexTypeOrig *d_row_ptr = (IndexTypeOrig *)blk_head_index;
            IndexTypeOrig *d_col_idx = (IndexTypeOrig *)((char *)d_row_ptr
                + (BLOCK_SIZE + 1) * sizeof(IndexTypeOrig));
            ValType *d_val = (ValType *)((char *)d_col_idx
                + blk_nnz * sizeof(IndexTypeOrig) + padding);
            ValType sum = 0.0;
            int row_id = uniform_rows ? 0 : tid_in_wave / LANES_PER_ROW;
            int offset = uniform_rows ? tid_in_wave
                : tid_in_wave % LANES_PER_ROW;
            int stride = uniform_rows ? BLOCK_SIZE : LANES_PER_ROW;
            for (int k = d_row_ptr[row_id] + offset;
                 k < d_row_ptr[row_id + 1]; k += stride)
            {
                int gathered_col = blk_col_idx * BLOCK_SIZE + d_col_idx[k];
                int original_col = d_block_cols[
                    gathered_row_offset + gathered_col];
                if constexpr (UNIT_VALUES)
                {
                    sum += d_x[original_col];
                }
                else
                {
                    sum += d_x[original_col] * d_val[k];
                }
            }
            if (uniform_rows)
            {
                sum = reduce_uniform_row(sum);
                if (owned_uniform)
                {
                    if (tid_in_wave == 0 && sum != 0.0)
                    {
                        accumulate_output(block_output, 0, sum);
                    }
                }
                else if (tid_in_wave < BLOCK_SIZE && sum != 0.0)
                {
                    accumulate_output(block_output, tid_in_wave, sum);
                }
            }
            else
            {
                if constexpr (LANES_PER_ROW == 1)
                {
                }
                else if constexpr (LANES_PER_ROW > 8)
                {
                    sum = shuffle_reduce_large_row(sum);
                }
                else if constexpr (LANES_PER_ROW == 8)
                {
                    sum = dpp_reduce_8(sum);
                }
                else if constexpr (LANES_PER_ROW == 4)
                {
                    sum = dpp_reduce_4(sum);
                }
                else
                {
                    sum = dpp_reduce_2(sum);
                }
                if (offset == LANES_PER_ROW - 1 && sum != 0.0)
                {
                    accumulate_output(block_output, row_id, sum);
                }
            }
        }
        else if (blk_type == DENSE)
        {
            ValType sum = 0.0;
            int gathered_col_begin = blk_col_idx * BLOCK_SIZE;
            if (uniform_rows)
            {
                if (tid_in_wave < BLOCK_SIZE)
                {
                    int gathered_col = gathered_col_begin + tid_in_wave;
                    if (gathered_col < gathered_col_count)
                    {
                        int original_col = d_block_cols[
                            gathered_row_offset + gathered_col];
                        if constexpr (UNIT_VALUES)
                        {
                            const DenseMaskType row_mask =
                                reinterpret_cast<const DenseMaskType *>(
                                    blk_head_index)[0];
                            if (row_mask
                                & (DenseMaskType(1) << tid_in_wave))
                            {
                                sum = d_x[original_col];
                            }
                        }
                        else
                        {
                            ValType *d_val = (ValType *)blk_head_index;
                            sum = d_val[tid_in_wave] * d_x[original_col];
                        }
                    }
                }
                sum = reduce_uniform_row(sum);
                if (owned_uniform)
                {
                    if (tid_in_wave == 0 && sum != 0.0)
                    {
                        accumulate_output(block_output, 0, sum);
                    }
                }
                else if (tid_in_wave < BLOCK_SIZE && sum != 0.0)
                {
                    accumulate_output(block_output, tid_in_wave, sum);
                }
            }
            else
            {
                int row = tid_in_wave % BLOCK_SIZE;
                int offset = tid_in_wave / BLOCK_SIZE;
                constexpr int columns_per_lane = LANES_PER_ROW <= BLOCK_SIZE
                    ? BLOCK_SIZE / LANES_PER_ROW : 1;
                const int col_begin = LANES_PER_ROW <= BLOCK_SIZE
                    ? offset * columns_per_lane : offset;
                const int col_end = LANES_PER_ROW <= BLOCK_SIZE
                    ? (offset + 1) * columns_per_lane : BLOCK_SIZE;
                constexpr int col_stride = LANES_PER_ROW <= BLOCK_SIZE
                    ? 1 : LANES_PER_ROW;
                if constexpr (UNIT_VALUES)
                {
                    const DenseMaskType row_mask =
                        reinterpret_cast<const DenseMaskType *>(
                            blk_head_index)[row];
                    for (int col = col_begin; col < col_end;
                         col += col_stride)
                    {
                        int gathered_col = gathered_col_begin + col;
                        if (gathered_col < gathered_col_count
                            && (row_mask & (DenseMaskType(1) << col)))
                        {
                            int original_col = d_block_cols[
                                gathered_row_offset + gathered_col];
                            sum += d_x[original_col];
                        }
                    }
                }
                else
                {
                    ValType *d_val = (ValType *)blk_head_index;
                    for (int col = col_begin; col < col_end;
                         col += col_stride)
                    {
                        int gathered_col = gathered_col_begin + col;
                        if (gathered_col < gathered_col_count)
                        {
                            int original_col = d_block_cols[
                                gathered_row_offset + gathered_col];
                            sum += d_val[row * BLOCK_SIZE + col]
                                * d_x[original_col];
                        }
                    }
                }
                for (int delta = BLOCK_SIZE; delta < WAVE_SIZE; delta <<= 1)
                {
                    sum += __shfl_xor(sum, delta, WAVE_SIZE);
                }
                if (tid_in_wave < BLOCK_SIZE)
                {
                    accumulate_output(block_output, row, sum);
                }
            }
        }
    }
#if CB_ROW_OWNED
    }
    __syncthreads();
    if (wave_in_block == 0)
    {
        ValType row_sum = 0.0;
#pragma unroll
        for (int wave = 0; wave < WARP_PER_BLOCK; ++wave)
        {
            row_sum += s_y[wave * BLOCK_SIZE
                + (owned_uniform ? 0 : tid_in_wave)];
        }
        if (owned_uniform)
        {
            row_sum = __shfl(row_sum, 0, WAVE_SIZE);
        }
        if (tid_in_wave < BLOCK_SIZE)
        {
            if (direct_output)
            {
                d_y[owned_row * BLOCK_SIZE + tid_in_wave] = row_sum;
            }
            else
            {
                atomicAdd(&d_y[owned_row * BLOCK_SIZE + tid_in_wave],
                    row_sum);
            }
        }
    }
#endif
}

}

inline PROGRAM_METRICS cb_spmv(
    const GLOBAL_BLOCK& gb_block, const ValType *x, ValType *y)
{
    if (gb_block.mtx_nnz == 0)
    {
        for (int row = 0; row < gb_block.mtx_m; row++)
        {
            y[row] = 0.0;
        }
        return {};
    }

    const size_t x_elements = static_cast<size_t>(gb_block.gb_n) * BLOCK_SIZE;
    const size_t y_elements = static_cast<size_t>(gb_block.gb_m) * BLOCK_SIZE;
    std::vector<ValType> h_x(x_elements, 0.0);
    memcpy(h_x.data(), x, gb_block.mtx_n * sizeof(ValType));

    bool unit_values = true;
    for (const auto& entry : gb_block.data_per_coo_blk_comp)
    {
        for (ValType value : entry.second.val)
        {
            unit_values = unit_values && value == ValType(1);
        }
    }
    for (const auto& entry : gb_block.data_per_csr_blk)
    {
        for (ValType value : entry.second.val)
        {
            unit_values = unit_values && value == ValType(1);
        }
    }
    for (const auto& entry : gb_block.data_per_dense_blk)
    {
        for (ValType value : entry.second.val)
        {
            unit_values = unit_values
                && (value == ValType(0) || value == ValType(1));
        }
    }
    const size_t block_count = static_cast<size_t>(gb_block.gb_nnzb);

    size_t device_bytes = 0;
    auto reserve_bytes = [&device_bytes](size_t bytes, size_t alignment)
    {
        device_bytes = cb_spmv_detail::align_up(device_bytes, alignment);
        const size_t offset = device_bytes;
        device_bytes += bytes;
        return offset;
    };

    const size_t gb_row_idx_offset = reserve_bytes(block_count * sizeof(int), alignof(int));
    const size_t gb_col_idx_offset = reserve_bytes(block_count * sizeof(int), alignof(int));
    const size_t segment_count = gb_block.segment_rows.size();
    const size_t segment_rows_offset = CB_ROW_OWNED
        ? reserve_bytes(segment_count * sizeof(int), alignof(int)) : 0;
    const size_t segment_begin_offset = CB_ROW_OWNED
        ? reserve_bytes(segment_count * sizeof(int), alignof(int)) : 0;
    const size_t segment_end_offset = CB_ROW_OWNED
        ? reserve_bytes(segment_count * sizeof(int), alignof(int)) : 0;
    const size_t segment_direct_offset = CB_ROW_OWNED
        ? reserve_bytes(segment_count * sizeof(uint8_t), alignof(uint8_t)) : 0;
    const size_t segment_uniform_offset = CB_ROW_OWNED
        ? reserve_bytes(segment_count * sizeof(uint8_t), alignof(uint8_t)) : 0;
    const size_t nnz_per_blk_offset = reserve_bytes(block_count * sizeof(int), alignof(int));
    const size_t type_per_blk_offset = reserve_bytes(block_count * sizeof(BlockType), alignof(BlockType));
    const size_t pattern_row_ptr_offset = CB_PATTERN_WAVE
        ? reserve_bytes(gb_block.pattern_row_ptr.size() * sizeof(int),
            alignof(int)) : 0;
    const size_t pattern_col_idx_offset = CB_PATTERN_WAVE
        ? reserve_bytes(gb_block.pattern_col_idx.size() * sizeof(int),
            alignof(int)) : 0;
    const size_t pattern_output_ptr_offset = CB_PATTERN_WAVE
        ? reserve_bytes(gb_block.pattern_output_ptr.size() * sizeof(int),
            alignof(int)) : 0;
    const size_t pattern_output_rows_offset = CB_PATTERN_WAVE
        ? reserve_bytes(gb_block.pattern_output_rows.size() * sizeof(int),
            alignof(int)) : 0;
    size_t block_cols_offset = 0;
    size_t block_cols_offsets_offset = 0;
    if (gb_block.gather_flag)
    {
        block_cols_offset = reserve_bytes(
            gb_block.block_cols.size() * sizeof(int), alignof(int));
        block_cols_offsets_offset = reserve_bytes(gb_block.block_cols_offset.size() * sizeof(int), alignof(int));
    }
    const size_t block_addresses_offset = reserve_bytes(block_count * sizeof(char *), alignof(char *));

    std::vector<size_t> block_offsets(block_count, 0);
    std::vector<bool> block_has_payload(block_count, false);
    for (int i = 0; i < gb_block.gb_nnzb; i++)
    {
        if (gb_block.nnz_per_blk[i] > 0)
        {
            int old_idx = gb_block.block_idx_array[i].origin_idx;
            BlockType block_type = gb_block.type_per_blk[i] & BLOCK_TYPE_MASK;
            if (block_type == COO)
            {
                const COO_BLOCK_COMP& block = gb_block.data_per_coo_blk_comp.at(old_idx);
                const size_t index_bytes = static_cast<size_t>(block.blk_nnz)
                    * (gb_block.compact_coo_indices
                        ? sizeof(uint16_t) : sizeof(IndexTypeComp));
                const size_t value_offset = cb_spmv_detail::align_up(index_bytes, sizeof(ValType));
                block_offsets[i] = reserve_bytes(unit_values
                    ? index_bytes
                    : value_offset
                        + static_cast<size_t>(block.blk_nnz) * sizeof(ValType),
                    unit_values ? alignof(IndexTypeComp) : alignof(ValType));
                block_has_payload[i] = true;
            }
            else if (block_type == CSR)
            {
                const CSR_BLOCK& block = gb_block.data_per_csr_blk.at(old_idx);
                const size_t row_ptr_bytes = (BLOCK_SIZE + 1) * sizeof(IndexTypeOrig);
                const size_t col_idx_bytes = static_cast<size_t>(block.blk_nnz) * sizeof(IndexTypeOrig);
                const size_t value_offset = cb_spmv_detail::align_up(
                    row_ptr_bytes + col_idx_bytes, sizeof(ValType));
                block_offsets[i] = reserve_bytes(unit_values
                    ? row_ptr_bytes + col_idx_bytes
                    : value_offset
                        + static_cast<size_t>(block.blk_nnz) * sizeof(ValType),
                    unit_values ? alignof(IndexTypeOrig) : alignof(ValType));
                block_has_payload[i] = true;
            }
            else if (block_type == DENSE)
            {
                block_offsets[i] = reserve_bytes(unit_values
                    ? BLOCK_SIZE * sizeof(DenseMaskType)
                    : BLOCK_SIZE * BLOCK_SIZE * sizeof(ValType),
                    unit_values ? alignof(DenseMaskType) : alignof(ValType));
                block_has_payload[i] = true;
            }
        }
    }

    const size_t x_offset = reserve_bytes(x_elements * sizeof(ValType), alignof(ValType));
    const size_t y_offset = reserve_bytes(y_elements * sizeof(ValType), alignof(ValType));
    const size_t output_epochs_offset = CB_ROW_OWNED
        ? reserve_bytes(gb_block.gb_m * sizeof(int), alignof(int)) : 0;

    std::vector<char> staging_buffer(device_bytes, 0);
    auto pack_bytes = [&staging_buffer](size_t offset, const void *source, size_t bytes)
    {
        if (bytes != 0)
        {
            memcpy(staging_buffer.data() + offset, source, bytes);
        }
    };

    pack_bytes(gb_row_idx_offset, gb_block.gb_row_idx.data(), block_count * sizeof(int));
    pack_bytes(gb_col_idx_offset, gb_block.gb_col_idx.data(), block_count * sizeof(int));
    if (CB_ROW_OWNED)
    {
        pack_bytes(segment_rows_offset, gb_block.segment_rows.data(),
            segment_count * sizeof(int));
        pack_bytes(segment_begin_offset, gb_block.segment_begin.data(),
            segment_count * sizeof(int));
        pack_bytes(segment_end_offset, gb_block.segment_end.data(),
            segment_count * sizeof(int));
        pack_bytes(segment_direct_offset, gb_block.segment_direct.data(),
            segment_count * sizeof(uint8_t));
        pack_bytes(segment_uniform_offset, gb_block.segment_uniform.data(),
            segment_count * sizeof(uint8_t));
    }
    pack_bytes(nnz_per_blk_offset, gb_block.nnz_per_blk.data(), block_count * sizeof(int));
    pack_bytes(type_per_blk_offset, gb_block.type_per_blk.data(), block_count * sizeof(BlockType));
    if (CB_PATTERN_WAVE)
    {
        pack_bytes(pattern_row_ptr_offset, gb_block.pattern_row_ptr.data(),
            gb_block.pattern_row_ptr.size() * sizeof(int));
        pack_bytes(pattern_col_idx_offset, gb_block.pattern_col_idx.data(),
            gb_block.pattern_col_idx.size() * sizeof(int));
        pack_bytes(pattern_output_ptr_offset,
            gb_block.pattern_output_ptr.data(),
            gb_block.pattern_output_ptr.size() * sizeof(int));
        pack_bytes(pattern_output_rows_offset,
            gb_block.pattern_output_rows.data(),
            gb_block.pattern_output_rows.size() * sizeof(int));
    }
    if (gb_block.gather_flag)
    {
        pack_bytes(block_cols_offset, gb_block.block_cols.data(),
            gb_block.block_cols.size() * sizeof(int));
        pack_bytes(block_cols_offsets_offset, gb_block.block_cols_offset.data(), gb_block.block_cols_offset.size() * sizeof(int));
    }
    for (int i = 0; i < gb_block.gb_nnzb; i++)
    {
        if (!block_has_payload[i])
        {
            continue;
        }
        const size_t payload_offset = block_offsets[i];
        const int old_idx = gb_block.block_idx_array[i].origin_idx;
        const BlockType block_type = gb_block.type_per_blk[i] & BLOCK_TYPE_MASK;
        if (block_type == COO)
        {
            const COO_BLOCK_COMP& block = gb_block.data_per_coo_blk_comp.at(old_idx);
            const size_t index_bytes = static_cast<size_t>(block.blk_nnz)
                * (gb_block.compact_coo_indices
                    ? sizeof(uint16_t) : sizeof(IndexTypeComp));
            const size_t value_offset = cb_spmv_detail::align_up(index_bytes, sizeof(ValType));
            if (gb_block.compact_coo_indices)
            {
                std::vector<uint16_t> compact_indices(block.blk_nnz);
                for (int k = 0; k < block.blk_nnz; ++k)
                {
                    compact_indices[k] = static_cast<uint16_t>(block.idx[k]);
                }
                pack_bytes(payload_offset, compact_indices.data(), index_bytes);
            }
            else
            {
                pack_bytes(payload_offset, block.idx.data(), index_bytes);
            }
            if (!unit_values)
            {
                pack_bytes(payload_offset + value_offset, block.val.data(),
                           static_cast<size_t>(block.blk_nnz) * sizeof(ValType));
            }
        }
        else if (block_type == CSR)
        {
            const CSR_BLOCK& block = gb_block.data_per_csr_blk.at(old_idx);
            const size_t row_ptr_bytes = (BLOCK_SIZE + 1) * sizeof(IndexTypeOrig);
            const size_t col_idx_bytes = static_cast<size_t>(block.blk_nnz) * sizeof(IndexTypeOrig);
            const size_t value_offset = cb_spmv_detail::align_up(
                row_ptr_bytes + col_idx_bytes, sizeof(ValType));
            pack_bytes(payload_offset, block.row_ptr.data(), row_ptr_bytes);
            pack_bytes(payload_offset + row_ptr_bytes, block.col_idx.data(), col_idx_bytes);
            if (!unit_values)
            {
                pack_bytes(payload_offset + value_offset, block.val.data(),
                           static_cast<size_t>(block.blk_nnz) * sizeof(ValType));
            }
        }
        else if (block_type == DENSE)
        {
            const DENSE_BLOCK& block = gb_block.data_per_dense_blk.at(old_idx);
            if (unit_values)
            {
                DenseMaskType row_masks[BLOCK_SIZE]{};
                for (int row = 0; row < BLOCK_SIZE; ++row)
                {
                    for (int column = 0; column < BLOCK_SIZE; ++column)
                    {
                        if (block.val[row * BLOCK_SIZE + column] != ValType(0))
                        {
                            row_masks[row] |= DenseMaskType(1) << column;
                        }
                    }
                }
                pack_bytes(payload_offset, row_masks, sizeof(row_masks));
            }
            else
            {
                pack_bytes(payload_offset, block.val.data(),
                           BLOCK_SIZE * BLOCK_SIZE * sizeof(ValType));
            }
        }
    }
    pack_bytes(x_offset, h_x.data(), x_elements * sizeof(ValType));

    char *device_base = nullptr;
    int *d_gb_row_idx = nullptr;
    int *d_gb_col_idx = nullptr;
    int *d_segment_rows = nullptr;
    int *d_segment_begin = nullptr;
    int *d_segment_end = nullptr;
    uint8_t *d_segment_direct = nullptr;
    uint8_t *d_segment_uniform = nullptr;
    int *d_output_epochs = nullptr;
    int *d_nnz_per_blk = nullptr;
    BlockType *d_type_per_blk = nullptr;
    int *d_pattern_row_ptr = nullptr;
    int *d_pattern_col_idx = nullptr;
    int *d_pattern_output_ptr = nullptr;
    int *d_pattern_output_rows = nullptr;
    int *d_block_cols = nullptr;
    int *d_block_cols_offset = nullptr;
    char **d_block_addresses = nullptr;
    ValType *d_x = nullptr;
    ValType *d_y = nullptr;

    CHECK_HIP(hipMalloc((void **)(&device_base), device_bytes));
    d_gb_row_idx = reinterpret_cast<int *>(device_base + gb_row_idx_offset);
    d_gb_col_idx = reinterpret_cast<int *>(device_base + gb_col_idx_offset);
    if (CB_ROW_OWNED)
    {
        d_segment_rows = reinterpret_cast<int *>(device_base
            + segment_rows_offset);
        d_segment_begin = reinterpret_cast<int *>(device_base
            + segment_begin_offset);
        d_segment_end = reinterpret_cast<int *>(device_base
            + segment_end_offset);
        d_segment_direct = reinterpret_cast<uint8_t *>(device_base
            + segment_direct_offset);
        d_segment_uniform = reinterpret_cast<uint8_t *>(device_base
            + segment_uniform_offset);
        d_output_epochs = reinterpret_cast<int *>(device_base
            + output_epochs_offset);
    }
    d_nnz_per_blk = reinterpret_cast<int *>(device_base + nnz_per_blk_offset);
    d_type_per_blk = reinterpret_cast<BlockType *>(device_base + type_per_blk_offset);
    if (CB_PATTERN_WAVE)
    {
        d_pattern_row_ptr = reinterpret_cast<int *>(device_base
            + pattern_row_ptr_offset);
        d_pattern_col_idx = reinterpret_cast<int *>(device_base
            + pattern_col_idx_offset);
        d_pattern_output_ptr = reinterpret_cast<int *>(device_base
            + pattern_output_ptr_offset);
        d_pattern_output_rows = reinterpret_cast<int *>(device_base
            + pattern_output_rows_offset);
    }
    d_block_addresses = reinterpret_cast<char **>(device_base + block_addresses_offset);
    d_x = reinterpret_cast<ValType *>(device_base + x_offset);
    d_y = reinterpret_cast<ValType *>(device_base + y_offset);
    if (gb_block.gather_flag)
    {
        d_block_cols = reinterpret_cast<int *>(device_base + block_cols_offset);
        d_block_cols_offset = reinterpret_cast<int *>(device_base + block_cols_offsets_offset);
    }
    std::vector<char *> block_addresses(block_count, nullptr);
    for (size_t i = 0; i < block_count; i++)
    {
        if (block_has_payload[i])
        {
            block_addresses[i] = device_base + block_offsets[i];
        }
    }
    pack_bytes(block_addresses_offset, block_addresses.data(), block_count * sizeof(char *));

    CHECK_HIP(hipMemcpy(device_base, staging_buffer.data(), y_offset, hipMemcpyHostToDevice));
    CHECK_HIP(hipMemset(d_y, 0, y_elements * sizeof(ValType)));
    if (CB_ROW_OWNED)
    {
        CHECK_HIP(hipMemset(d_output_epochs, 0,
            gb_block.gb_m * sizeof(int)));
    }

    const int pattern_count = static_cast<int>(
        gb_block.pattern_row_ptr.empty()
            ? 0 : gb_block.pattern_row_ptr.size() - 1);
    const bool use_pattern = CB_PATTERN_WAVE && unit_values
        && pattern_count > 0
        && static_cast<int64_t>(pattern_count) * 2 < gb_block.mtx_m;
    const bool use_unrolled_pattern = use_pattern
        && gb_block.pattern_col_idx.size()
            >= static_cast<size_t>(pattern_count) * 128;
    const bool use_small_direct = CB_ROW_OWNED && !use_pattern
        && unit_values && gb_block.gather_flag
        && gb_block.compact_coo_indices && gb_block.mtx_nnz < 10000;
    const int launch_waves = use_pattern && pattern_count < 1000
        ? 9 : WARP_PER_BLOCK;
    int num_threads = launch_waves * WAVE_SIZE;
    int num_blocks = use_pattern
        ? (pattern_count + launch_waves - 1) / launch_waves
        : CB_ROW_OWNED ? static_cast<int>(segment_count)
            : (gb_block.gb_nnzb + WARP_PER_BLOCK - 1) / WARP_PER_BLOCK;
    const bool aggregate_coo = gb_block.mtx_nnz >= 10000;
    const bool direct_coo_columns = gb_block.direct_coo_columns;
    const bool compact_coo_indices = gb_block.compact_coo_indices;
    int next_output_epoch = 1;

    auto launch_kernel = [&]()
    {
        const int current_output_epoch = next_output_epoch++;
        if (use_pattern)
        {
            if (gb_block.gather_flag)
            {
                if (use_unrolled_pattern)
                {
                    hipLaunchKernelGGL(
                        HIP_KERNEL_NAME(cb_spmv_detail::spmv_hip_kernel_gather<true, false, cb_spmv_detail::EXEC_PATTERN_REUSE_UNROLLED>),
                        dim3(num_blocks), dim3(num_threads), 0, 0,
                        pattern_count, aggregate_coo, direct_coo_columns,
                        d_nnz_per_blk, d_type_per_blk,
                        d_gb_row_idx, d_gb_col_idx, d_pattern_row_ptr,
                        d_pattern_col_idx, d_pattern_output_ptr, nullptr,
                        nullptr, d_pattern_output_rows, current_output_epoch,
                        d_block_addresses, d_block_cols,
                        d_block_cols_offset, d_x, d_y);
                }
                else
                {
                    hipLaunchKernelGGL(
                        HIP_KERNEL_NAME(cb_spmv_detail::spmv_hip_kernel_gather<true, false, cb_spmv_detail::EXEC_PATTERN_REUSE_SINGLE>),
                        dim3(num_blocks), dim3(num_threads), 0, 0,
                        pattern_count, aggregate_coo, direct_coo_columns,
                        d_nnz_per_blk, d_type_per_blk,
                        d_gb_row_idx, d_gb_col_idx, d_pattern_row_ptr,
                        d_pattern_col_idx, d_pattern_output_ptr, nullptr,
                        nullptr, d_pattern_output_rows, current_output_epoch,
                        d_block_addresses, d_block_cols,
                        d_block_cols_offset, d_x, d_y);
                }
            }
            else
            {
                if (use_unrolled_pattern)
                {
                    hipLaunchKernelGGL(
                        HIP_KERNEL_NAME(cb_spmv_detail::spmv_hip_kernel_comp<true, false, cb_spmv_detail::EXEC_PATTERN_REUSE_UNROLLED>),
                        dim3(num_blocks), dim3(num_threads), 0, 0,
                        pattern_count, aggregate_coo, direct_coo_columns,
                        d_nnz_per_blk, d_type_per_blk,
                        d_gb_row_idx, d_gb_col_idx, d_pattern_row_ptr,
                        d_pattern_col_idx, d_pattern_output_ptr, nullptr,
                        nullptr, d_pattern_output_rows, current_output_epoch,
                        d_block_addresses, d_x, d_y);
                }
                else
                {
                    hipLaunchKernelGGL(
                        HIP_KERNEL_NAME(cb_spmv_detail::spmv_hip_kernel_comp<true, false, cb_spmv_detail::EXEC_PATTERN_REUSE_SINGLE>),
                        dim3(num_blocks), dim3(num_threads), 0, 0,
                        pattern_count, aggregate_coo, direct_coo_columns,
                        d_nnz_per_blk, d_type_per_blk,
                        d_gb_row_idx, d_gb_col_idx, d_pattern_row_ptr,
                        d_pattern_col_idx, d_pattern_output_ptr, nullptr,
                        nullptr, d_pattern_output_rows, current_output_epoch,
                        d_block_addresses, d_x, d_y);
                }
            }
            return;
        }
        if (use_small_direct)
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(cb_spmv_detail::spmv_hip_kernel_gather<true, true, cb_spmv_detail::EXEC_SMALL_DIRECT>),
                dim3(num_blocks), dim3(num_threads), 0, 0,
                gb_block.gb_nnzb, aggregate_coo, direct_coo_columns,
                d_nnz_per_blk, d_type_per_blk,
                d_gb_row_idx, d_gb_col_idx, d_segment_rows,
                d_segment_begin, d_segment_end, d_segment_direct,
                d_segment_uniform, d_output_epochs, current_output_epoch,
                d_block_addresses, d_block_cols, d_block_cols_offset,
                d_x, d_y);
            return;
        }
        if (gb_block.gather_flag)
        {
            if (unit_values)
            {
                if (compact_coo_indices)
                {
                    hipLaunchKernelGGL(
                        HIP_KERNEL_NAME(cb_spmv_detail::spmv_hip_kernel_gather<true, true, cb_spmv_detail::EXEC_STANDARD>),
                        dim3(num_blocks), dim3(num_threads), 0, 0,
                        gb_block.gb_nnzb, aggregate_coo, direct_coo_columns,
                        d_nnz_per_blk, d_type_per_blk,
                        d_gb_row_idx, d_gb_col_idx, d_segment_rows,
                        d_segment_begin, d_segment_end,
                        d_segment_direct, d_segment_uniform,
                        d_output_epochs, current_output_epoch,
                        d_block_addresses,
                        d_block_cols, d_block_cols_offset, d_x, d_y);
                }
                else
                {
                    hipLaunchKernelGGL(
                        HIP_KERNEL_NAME(cb_spmv_detail::spmv_hip_kernel_gather<true, false, cb_spmv_detail::EXEC_STANDARD>),
                        dim3(num_blocks), dim3(num_threads), 0, 0,
                        gb_block.gb_nnzb, aggregate_coo, direct_coo_columns,
                        d_nnz_per_blk, d_type_per_blk,
                        d_gb_row_idx, d_gb_col_idx, d_segment_rows,
                        d_segment_begin, d_segment_end,
                        d_segment_direct, d_segment_uniform,
                        d_output_epochs, current_output_epoch,
                        d_block_addresses,
                        d_block_cols, d_block_cols_offset, d_x, d_y);
                }
            }
            else
            {
                if (compact_coo_indices)
                {
                    hipLaunchKernelGGL(
                        HIP_KERNEL_NAME(cb_spmv_detail::spmv_hip_kernel_gather<false, true, cb_spmv_detail::EXEC_STANDARD>),
                        dim3(num_blocks), dim3(num_threads), 0, 0,
                        gb_block.gb_nnzb, aggregate_coo, direct_coo_columns,
                        d_nnz_per_blk, d_type_per_blk,
                        d_gb_row_idx, d_gb_col_idx, d_segment_rows,
                        d_segment_begin, d_segment_end,
                        d_segment_direct, d_segment_uniform,
                        d_output_epochs, current_output_epoch,
                        d_block_addresses,
                        d_block_cols, d_block_cols_offset, d_x, d_y);
                }
                else
                {
                    hipLaunchKernelGGL(
                        HIP_KERNEL_NAME(cb_spmv_detail::spmv_hip_kernel_gather<false, false, cb_spmv_detail::EXEC_STANDARD>),
                        dim3(num_blocks), dim3(num_threads), 0, 0,
                        gb_block.gb_nnzb, aggregate_coo, direct_coo_columns,
                        d_nnz_per_blk, d_type_per_blk,
                        d_gb_row_idx, d_gb_col_idx, d_segment_rows,
                        d_segment_begin, d_segment_end,
                        d_segment_direct, d_segment_uniform,
                        d_output_epochs, current_output_epoch,
                        d_block_addresses,
                        d_block_cols, d_block_cols_offset, d_x, d_y);
                }
            }
        }
        else
        {
            if (unit_values)
            {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(cb_spmv_detail::spmv_hip_kernel_comp<true, false, cb_spmv_detail::EXEC_STANDARD>),
                    dim3(num_blocks), dim3(num_threads), 0, 0,
                    gb_block.gb_nnzb, aggregate_coo, direct_coo_columns,
                    d_nnz_per_blk, d_type_per_blk,
                    d_gb_row_idx, d_gb_col_idx, d_segment_rows,
                    d_segment_begin, d_segment_end,
                    d_segment_direct, d_segment_uniform,
                    d_output_epochs, current_output_epoch,
                    d_block_addresses, d_x, d_y);
            }
            else
            {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(cb_spmv_detail::spmv_hip_kernel_comp<false, false, cb_spmv_detail::EXEC_STANDARD>),
                    dim3(num_blocks), dim3(num_threads), 0, 0,
                    gb_block.gb_nnzb, aggregate_coo, direct_coo_columns,
                    d_nnz_per_blk, d_type_per_blk,
                    d_gb_row_idx, d_gb_col_idx, d_segment_rows,
                    d_segment_begin, d_segment_end,
                    d_segment_direct, d_segment_uniform,
                    d_output_epochs, current_output_epoch,
                    d_block_addresses, d_x, d_y);
            }
        }
    };

    launch_kernel();
    CHECK_HIP(hipGetLastError());
    CHECK_HIP(hipDeviceSynchronize());
    CHECK_HIP(hipMemcpy(y, d_y, gb_block.mtx_m * sizeof(ValType), hipMemcpyDeviceToHost));
    CHECK_HIP(hipMemset(d_y, 0, y_elements * sizeof(ValType)));

    for (int i = 0; i < WARMUP_ITERATIONS; ++i)
    {
        launch_kernel();
    }
    CHECK_HIP(hipGetLastError());
    CHECK_HIP(hipDeviceSynchronize());
    CHECK_HIP(hipMemset(d_y, 0, y_elements * sizeof(ValType)));

    hipEvent_t start;
    hipEvent_t stop;
    CHECK_HIP(hipEventCreate(&start));
    CHECK_HIP(hipEventCreate(&stop));

    CHECK_HIP(hipEventRecord(start));
    for (int i = 0; i < ITERATIONS; ++i)
    {
        launch_kernel();
    }
    CHECK_HIP(hipGetLastError());
    CHECK_HIP(hipEventRecord(stop));
    CHECK_HIP(hipEventSynchronize(stop));

    float elapsed_time = 0.0f;
    CHECK_HIP(hipEventElapsedTime(&elapsed_time, start, stop));
    PROGRAM_METRICS metrics;
    metrics.runtime = static_cast<double>(elapsed_time) / ITERATIONS;
    if (metrics.runtime > 0.0)
    {
        metrics.gflops = 2.0 * static_cast<double>(gb_block.mtx_nnz) * 1.0e-6 / metrics.runtime;
    }

    CHECK_HIP(hipEventDestroy(start));
    CHECK_HIP(hipEventDestroy(stop));
    CHECK_HIP(hipFree(device_base));

    return metrics;
}
