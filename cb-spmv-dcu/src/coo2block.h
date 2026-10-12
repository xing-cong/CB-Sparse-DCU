#include "format.h"
#include <algorithm>
#include <cmath>

using namespace std;

void gather(const int* row_idx, const int* col_idx, std::vector<int>& gather_col_idx, GLOBAL_BLOCK &gb_block){
    vector<vector<int>> block_cols(gb_block.gb_m);

    #pragma omp parallel if(gb_block.mtx_nnz >= 10000)
    {
        unordered_map<int, vector<int>> local_block_cols;
        #pragma omp for nowait
        for (int i=0; i<gb_block.mtx_nnz; i++){
            int blk_idx = row_idx[i] / BLOCK_SIZE;
            local_block_cols[blk_idx].push_back(col_idx[i]);
        }

        #pragma omp critical
        {
            for (auto &entry : local_block_cols) {
                vector<int> &destination = block_cols[entry.first];
                if (destination.empty()) {
                    destination = move(entry.second);
                } else {
                    destination.insert(destination.end(), entry.second.begin(), entry.second.end());
                }
            }
        }
    }

    #pragma omp parallel for if(gb_block.mtx_nnz >= 10000)
    for (int i=0; i<gb_block.gb_m; i++){
        sort(block_cols[i].begin(), block_cols[i].end());
        block_cols[i].erase(unique(block_cols[i].begin(), block_cols[i].end()), block_cols[i].end());
    }

    #pragma omp parallel for if(gb_block.mtx_nnz >= 10000)
    for (int i=0; i<gb_block.mtx_nnz; i++){
        int blk_idx = row_idx[i] / BLOCK_SIZE;
        int pos = (int)(lower_bound(block_cols[blk_idx].begin(), block_cols[blk_idx].end(), col_idx[i]) - block_cols[blk_idx].begin());
        gather_col_idx[i] = pos;
    }

    vector<int> block_cols_offset(gb_block.gb_m+1, 0);

    for(int i=1; i<=gb_block.gb_m; i++){
        block_cols_offset[i] = block_cols_offset[i-1] + (int)block_cols[i-1].size();
    }

    vector<int> flat_block_cols(block_cols_offset.back());
    #pragma omp parallel for if(gb_block.mtx_nnz >= 10000)
    for (int i = 0; i < gb_block.gb_m; i++) {
        int base = block_cols_offset[i];
        for (size_t j = 0; j < block_cols[i].size(); j++) {
            flat_block_cols[base + j] = block_cols[i][j];
        }
    }
    gb_block.block_cols_offset = move(block_cols_offset);
    gb_block.block_cols = move(flat_block_cols);
}

bool gather_flag(GLOBAL_BLOCK &gb_block){
    int blk_0_8 = 0;
    for(int i=0; i<gb_block.gb_nnzb; i++){
        if(gb_block.nnz_per_blk[i] <= 8){
            blk_0_8++;
        }
    }
    int blk_sum = gb_block.gb_nnzb;
    gb_block.super_sparse_blocks = blk_0_8;
    gb_block.analyzed_blocks = blk_sum;
    gb_block.super_sparse_ratio = blk_sum > 0 ? blk_0_8 * 1.0 / blk_sum : 0.0;
    return gb_block.mtx_nnz >= 10 && gb_block.super_sparse_ratio > 0.15;
}

void coo2block_pre_metadata(const int* row_idx, const int* col_idx,
                            GLOBAL_BLOCK &gb_block,
                            vector<int> &block_ids){
    int nnz = gb_block.mtx_nnz;
    block_ids.resize(nnz);
    const size_t grid_cells = static_cast<size_t>(gb_block.gb_m) * gb_block.gb_n;
    const size_t dense_map_limit = 64 * 1024 * 1024 / sizeof(int);
    const bool use_dense_map = grid_cells <= dense_map_limit
        && grid_cells <= static_cast<size_t>(nnz) * 4;
    vector<int> dense_block_map;
    std::unordered_map<long long, int> blk_map;
    if (use_dense_map) {
        dense_block_map.resize(grid_cells, -1);
    }
    for(int i = 0; i < nnz; i++){
        int blk_idx = row_idx[i] / BLOCK_SIZE;
        int blk_idy = col_idx[i] / BLOCK_SIZE;
        long long blk_id = static_cast<long long>(blk_idx) * gb_block.gb_n + blk_idy;
        int block_id;
        bool inserted;
        if (use_dense_map) {
            int &entry = dense_block_map[static_cast<size_t>(blk_id)];
            inserted = entry == -1;
            if (inserted) {
                entry = gb_block.gb_nnzb;
            }
            block_id = entry;
        } else {
            auto insertion = blk_map.emplace(blk_id, gb_block.gb_nnzb);
            inserted = insertion.second;
            block_id = insertion.first->second;
        }
        if(inserted){
            gb_block.gb_nnzb++;
            gb_block.gb_row_idx.push_back(blk_idx);
            gb_block.gb_col_idx.push_back(blk_idy);
            gb_block.nnz_per_blk.push_back(1);
            gb_block.type_per_blk.push_back(COO);
        }
        else{
            gb_block.nnz_per_blk[block_id]++;
        }
        block_ids[i] = block_id;
    }
}

void coo2block_pre_payload(const int* row_idx, const int* col_idx,
                           const ValType* val, const vector<int> &block_ids,
                           GLOBAL_BLOCK &gb_block){
    gb_block.data_per_coo_blk_orig.resize(gb_block.gb_nnzb);
    for(int i = 0; i < gb_block.gb_nnzb; i++){
        COO_BLOCK_ORIG &block = gb_block.data_per_coo_blk_orig[i];
        block.blk_nnz = gb_block.nnz_per_blk[i];
        block.row_idx.resize(block.blk_nnz);
        block.col_idx.resize(block.blk_nnz);
        block.val.resize(block.blk_nnz);
    }

    vector<int> block_offsets(gb_block.gb_nnzb, 0);
    for(int i = 0; i < gb_block.mtx_nnz; i++){
        int block_id = block_ids[i];
        int offset = block_offsets[block_id]++;
        COO_BLOCK_ORIG &block = gb_block.data_per_coo_blk_orig[block_id];
        block.row_idx[offset] = IndexTypeOrig(
            row_idx[i] - gb_block.gb_row_idx[block_id] * BLOCK_SIZE);
        block.col_idx[offset] = IndexTypeOrig(
            col_idx[i] - gb_block.gb_col_idx[block_id] * BLOCK_SIZE);
        block.val[offset] = ValType(val[i]);
    }
}

void coo2block_pre(const int* row_idx, const int* col_idx, const ValType* val, GLOBAL_BLOCK &gb_block){
    vector<int> block_ids;
    coo2block_pre_metadata(row_idx, col_idx, gb_block, block_ids);
    coo2block_pre_payload(row_idx, col_idx, val, block_ids, gb_block);
}

void coo2block_post(const int* row_idx, const int* col_idx, const ValType* val, GLOBAL_BLOCK &gb_block){
    constexpr int dense_nnz_threshold = BLOCK_SIZE * BLOCK_SIZE
        * CB_DENSE_SIXTEENTHS / 16;
    gb_block.direct_coo_columns = gb_block.gather_flag
        && gb_block.mtx_n <= UINT16_MAX;
    gb_block.compact_coo_indices = gb_block.direct_coo_columns
        && BLOCK_SIZE == 16 && gb_block.mtx_n <= 4095
        && gb_block.mtx_nnz < 10000;
    int extended_coo_candidates = 0;
    int balanced_row_candidates = 0;
    for(const COO_BLOCK_ORIG& block : gb_block.data_per_coo_blk_orig){
        if(block.blk_nnz <= WAVE_SIZE / 2 || block.blk_nnz > WAVE_SIZE){
            continue;
        }
        int row_nnz[BLOCK_SIZE]{};
        for(IndexTypeOrig row : block.row_idx){
            row_nnz[row]++;
        }
        int minimum = INT_MAX;
        int maximum = 0;
        for(int row = 0; row < BLOCK_SIZE; row++){
            if(row_nnz[row] > 0){
                minimum = std::min(minimum, row_nnz[row]);
                maximum = std::max(maximum, row_nnz[row]);
            }
        }
        extended_coo_candidates++;
        balanced_row_candidates += maximum - minimum <= 1;
    }
    const bool extend_coo_to_wave64 = CB_FORCE_COO64
        || (extended_coo_candidates > 0
            && balanced_row_candidates * 2 < extended_coo_candidates);
    auto use_coo_format = [extend_coo_to_wave64](const COO_BLOCK_ORIG& block){
        if(block.blk_nnz <= WAVE_SIZE / 2){
            return true;
        }
        return extend_coo_to_wave64 && block.blk_nnz <= WAVE_SIZE;
    };
    size_t coo_blocks = 0;
    size_t csr_blocks = 0;
    size_t dense_blocks = 0;
    for(int i = 0; i < gb_block.gb_nnzb; i++){
        const COO_BLOCK_ORIG &block = gb_block.data_per_coo_blk_orig[i];
        if(use_coo_format(block)){
            coo_blocks++;
        }else if(gb_block.nnz_per_blk[i] >= dense_nnz_threshold){
            dense_blocks++;
        }else{
            csr_blocks++;
        }
    }
    gb_block.data_per_coo_blk_comp.reserve(coo_blocks);
    gb_block.data_per_csr_blk.reserve(csr_blocks);
    gb_block.data_per_dense_blk.reserve(dense_blocks);
    for(int i = 0; i < gb_block.gb_nnzb; i++){
        const COO_BLOCK_ORIG &coo_block_orig_tmp = gb_block.data_per_coo_blk_orig[i];
        if(use_coo_format(coo_block_orig_tmp)){
            COO_BLOCK_COMP coo_block_comp_tmp;
            coo_block_comp_tmp.blk_nnz = coo_block_orig_tmp.blk_nnz;
            coo_block_comp_tmp.idx.reserve(coo_block_orig_tmp.blk_nnz);
            coo_block_comp_tmp.val.reserve(coo_block_orig_tmp.blk_nnz);
            vector<int> order(coo_block_orig_tmp.blk_nnz);
            for(int j = 0; j < coo_block_orig_tmp.blk_nnz; j++){
                order[j] = j;
            }
            sort(order.begin(), order.end(), [&coo_block_orig_tmp](int left, int right){
                if(coo_block_orig_tmp.row_idx[left]
                    != coo_block_orig_tmp.row_idx[right]){
                    return coo_block_orig_tmp.row_idx[left]
                        < coo_block_orig_tmp.row_idx[right];
                }
                return coo_block_orig_tmp.col_idx[left]
                    < coo_block_orig_tmp.col_idx[right];
            });
            for(int j : order){
                IndexTypeComp idx_row = coo_block_orig_tmp.row_idx[j];
                IndexTypeComp idx_col = coo_block_orig_tmp.col_idx[j];
                IndexTypeComp idx;
                if(gb_block.compact_coo_indices){
                    const int gathered_col = gb_block.gb_col_idx[i]
                        * BLOCK_SIZE + static_cast<int>(idx_col);
                    const int original_col = gb_block.block_cols[
                        gb_block.block_cols_offset[gb_block.gb_row_idx[i]]
                            + gathered_col];
                    idx = (idx_row << 12) | IndexTypeComp(original_col);
                }else if(gb_block.direct_coo_columns){
                    const int gathered_col = gb_block.gb_col_idx[i]
                        * BLOCK_SIZE + static_cast<int>(idx_col);
                    const int original_col = gb_block.block_cols[
                        gb_block.block_cols_offset[gb_block.gb_row_idx[i]]
                            + gathered_col];
                    idx = (idx_row << 16) | IndexTypeComp(original_col);
                }else{
                    idx = (idx_row << BLOCK_COORD_BITS) | idx_col;
                }
                if(!coo_block_comp_tmp.idx.empty()
                    && coo_block_comp_tmp.idx.back() == idx){
                    coo_block_comp_tmp.val.back() += coo_block_orig_tmp.val[j];
                }else{
                    coo_block_comp_tmp.idx.push_back(idx);
                    coo_block_comp_tmp.val.push_back(coo_block_orig_tmp.val[j]);
                }
            }
            coo_block_comp_tmp.blk_nnz = coo_block_comp_tmp.idx.size();
            if(BLOCK_SIZE != 64 && ((gb_block.direct_coo_columns
                    && !gb_block.compact_coo_indices)
                || BLOCK_SIZE == 32)){
                const int row_shift = gb_block.direct_coo_columns
                    ? 16 : BLOCK_COORD_BITS;
                const int position_shift = gb_block.direct_coo_columns
                    ? 16 + BLOCK_COORD_BITS : 10;
                const int long_row_mask = gb_block.direct_coo_columns
                    ? (1 << (position_shift + 5)) : 0x8000;
                int row_begin = 0;
                bool long_rows = false;
                while(row_begin < coo_block_comp_tmp.blk_nnz){
                    const int row = (coo_block_comp_tmp.idx[row_begin]
                        >> row_shift) & BLOCK_COLUMN_MASK;
                    int row_end = row_begin + 1;
                    while(row_end < coo_block_comp_tmp.blk_nnz
                        && (((coo_block_comp_tmp.idx[row_end]
                            >> row_shift) & BLOCK_COLUMN_MASK) == row)){
                        row_end++;
                    }
                    for(int position = 0; row_begin + position < row_end;
                        position++){
                        coo_block_comp_tmp.idx[row_begin + position]
                            |= IndexTypeComp(position << position_shift);
                    }
                    long_rows = long_rows || row_end - row_begin > 8;
                    row_begin = row_end;
                }
                if(long_rows){
                    for(IndexTypeComp& index : coo_block_comp_tmp.idx){
                        index |= IndexTypeComp(long_row_mask);
                    }
                }
            }
            vector<ValType> dense_rows(BLOCK_SIZE * BLOCK_SIZE, 0.0);
            for(int j = 0; j < coo_block_orig_tmp.blk_nnz; j++){
                dense_rows[coo_block_orig_tmp.row_idx[j] * BLOCK_SIZE
                    + coo_block_orig_tmp.col_idx[j]]
                    += coo_block_orig_tmp.val[j];
            }
            bool uniform_rows = true;
            for(int row = 1; row < BLOCK_SIZE && uniform_rows; row++){
                for(int column = 0; column < BLOCK_SIZE; column++){
                    uniform_rows = uniform_rows
                        && dense_rows[row * BLOCK_SIZE + column]
                            == dense_rows[column];
                }
            }
            gb_block.type_per_blk[i] = COO
                | (uniform_rows ? UNIFORM_ROWS : 0);
            gb_block.nnz_per_blk[i] = coo_block_comp_tmp.blk_nnz;
            gb_block.data_per_coo_blk_comp.emplace(i, move(coo_block_comp_tmp));

        }else if (coo_block_orig_tmp.blk_nnz >= dense_nnz_threshold){
            DENSE_BLOCK dense_block_tmp;
            dense_block_tmp.blk_nnz = coo_block_orig_tmp.blk_nnz;
            dense_block_tmp.val.resize(BLOCK_SIZE*BLOCK_SIZE);
            for(int j = 0; j < coo_block_orig_tmp.blk_nnz; j++){
                IndexTypeOrig idx_row = coo_block_orig_tmp.row_idx[j];
                IndexTypeOrig idx_col = coo_block_orig_tmp.col_idx[j];
                dense_block_tmp.val[idx_row*BLOCK_SIZE + idx_col] += coo_block_orig_tmp.val[j];
            }
            bool uniform_rows = true;
            for(int row = 1; row < BLOCK_SIZE && uniform_rows; row++){
                for(int column = 0; column < BLOCK_SIZE; column++){
                    uniform_rows = uniform_rows
                        && dense_block_tmp.val[row * BLOCK_SIZE + column]
                            == dense_block_tmp.val[column];
                }
            }
            gb_block.type_per_blk[i] = DENSE
                | (uniform_rows ? UNIFORM_ROWS : 0);
            gb_block.data_per_dense_blk.emplace(i, move(dense_block_tmp));
        }else{
            CSR_BLOCK csr_block_tmp;
            csr_block_tmp.blk_nnz = coo_block_orig_tmp.blk_nnz;
            csr_block_tmp.row_ptr.resize(BLOCK_SIZE+1, 0);
            csr_block_tmp.col_idx.resize(coo_block_orig_tmp.blk_nnz);
            csr_block_tmp.val.resize(coo_block_orig_tmp.blk_nnz);

            for(int j = 0; j < coo_block_orig_tmp.blk_nnz; j++){
                csr_block_tmp.row_ptr[coo_block_orig_tmp.row_idx[j]]++;
            }

            for(int j = 0, sum = 0; j < BLOCK_SIZE; j++){
                int temp = csr_block_tmp.row_ptr[j];
                csr_block_tmp.row_ptr[j] = sum;
                sum += temp;
            }
            csr_block_tmp.row_ptr[BLOCK_SIZE] = csr_block_tmp.blk_nnz;

            std::vector<IndexTypeOrig> row_start = csr_block_tmp.row_ptr;

            for(int j = 0; j < coo_block_orig_tmp.blk_nnz; j++){
                int row = coo_block_orig_tmp.row_idx[j];
                int dest = row_start[row]++;
                csr_block_tmp.col_idx[dest] = coo_block_orig_tmp.col_idx[j];
                csr_block_tmp.val[dest] = coo_block_orig_tmp.val[j];
            }
            bool uniform_rows = true;
            const int first_length = csr_block_tmp.row_ptr[1]
                - csr_block_tmp.row_ptr[0];
            for(int row = 1; row < BLOCK_SIZE && uniform_rows; row++){
                const int begin = csr_block_tmp.row_ptr[row];
                const int length = csr_block_tmp.row_ptr[row + 1] - begin;
                if(length != first_length){
                    uniform_rows = false;
                    break;
                }
                for(int position = 0; position < length; position++){
                    uniform_rows = uniform_rows
                        && csr_block_tmp.col_idx[begin + position]
                            == csr_block_tmp.col_idx[position]
                        && csr_block_tmp.val[begin + position]
                            == csr_block_tmp.val[position];
                }
            }
            gb_block.type_per_blk[i] = CSR
                | (uniform_rows ? UNIFORM_ROWS : 0);
            gb_block.data_per_csr_blk.emplace(i, move(csr_block_tmp));
        }
    }
}

void coo2block_banlance(GLOBAL_BLOCK &gb_block){
    vector<BLOCK_IDX> blk_idx_array(gb_block.gb_nnzb,{0,0,0});
    #pragma omp parallel for
    for(int i=0; i<gb_block.gb_nnzb; i++){
        blk_idx_array[i].blk_nnz = gb_block.nnz_per_blk[i];
        blk_idx_array[i].origin_idx = i;
    }

    int num_warps_per_block = WARP_PER_BLOCK;
    int num_blocks_before = (gb_block.gb_nnzb + num_warps_per_block - 1) / num_warps_per_block;
    vector<int> block_loads_before(num_blocks_before, 0);
    for(int i=0; i<gb_block.gb_nnzb; i++){
        block_loads_before[i/num_warps_per_block] += gb_block.nnz_per_blk[i];
    }
    double load_sum_before = 0.0;
    for(int load : block_loads_before){
        load_sum_before += load;
    }
    double load_mean_before = load_sum_before / block_loads_before.size();
    double load_variance_before = 0.0;
    for(int load : block_loads_before){
        double difference = load - load_mean_before;
        load_variance_before += difference * difference;
    }
    load_variance_before /= block_loads_before.size();
    double load_stddev_before = std::sqrt(load_variance_before);

    std::sort(blk_idx_array.begin(), blk_idx_array.end(), cmp_nnz);

    int num_blocks = 0;
    if(gb_block.gb_nnzb % num_warps_per_block != 0) {
        num_blocks = gb_block.gb_nnzb / num_warps_per_block + 1;
        int pad_count = num_warps_per_block
            - gb_block.gb_nnzb % num_warps_per_block;
        for(int i=0; i<pad_count; i++){
            blk_idx_array.push_back({0,INT_MAX,-1});
        }
        gb_block.gb_nnzb += pad_count;
    } else {
        num_blocks = gb_block.gb_nnzb / num_warps_per_block;
    }

    struct PQItem {
        int total;
        int index;
        int assigned_count;
        bool operator>(const PQItem &other) const {
            if (total == other.total) return index > other.index;
            return total > other.total;
        }
    };

    priority_queue<PQItem, vector<PQItem>, greater<PQItem>> pq;
    for (int j = 0; j < num_blocks; j++) {
        pq.push({0, j, 0});
    }

    for (int i = 0; i < gb_block.gb_nnzb; i++) {
        PQItem top_item = pq.top();
        pq.pop();
        blk_idx_array[i].end_idx = top_item.index * num_warps_per_block + top_item.assigned_count;
        top_item.total += blk_idx_array[i].blk_nnz;
        top_item.assigned_count += 1;
        if (top_item.assigned_count < num_warps_per_block) {
            pq.push(top_item);
        }
    }

    std::sort(blk_idx_array.begin(), blk_idx_array.end(), cmp_end_idx);

    vector<int> gb_row_idx_tmp(gb_block.gb_nnzb);
    vector<int> gb_col_idx_tmp(gb_block.gb_nnzb);
    vector<int> nnz_per_blk_tmp(gb_block.gb_nnzb);
    vector<BlockType> type_per_blk_tmp(gb_block.gb_nnzb);

    #pragma omp parallel for
    for(int i=0; i<gb_block.gb_nnzb; i++){
        int origin_idx = blk_idx_array[i].origin_idx;
        if(origin_idx != INT_MAX){
            gb_row_idx_tmp[i] = gb_block.gb_row_idx[origin_idx];
            gb_col_idx_tmp[i] = gb_block.gb_col_idx[origin_idx];
            nnz_per_blk_tmp[i] = gb_block.nnz_per_blk[origin_idx];
            type_per_blk_tmp[i] = gb_block.type_per_blk[origin_idx];
        } else {
            gb_row_idx_tmp[i] = -1;
            gb_col_idx_tmp[i] = -1;
            nnz_per_blk_tmp[i] = 0;
            type_per_blk_tmp[i] = COO;
        }
    }

    vector<int> block_loads_after(num_blocks, 0);
    for(int i=0; i<gb_block.gb_nnzb; i++){
        block_loads_after[i/num_warps_per_block] += blk_idx_array[i].blk_nnz;
    }
    double load_sum_after = 0.0;
    for(int load : block_loads_after){
        load_sum_after += load;
    }
    double load_mean_after = load_sum_after / block_loads_after.size();
    double load_variance_after = 0.0;
    for(int load : block_loads_after){
        double difference = load - load_mean_after;
        load_variance_after += difference * difference;
    }
    load_variance_after /= block_loads_after.size();
    double load_stddev_after = std::sqrt(load_variance_after);
    gb_block.load_balanced = true;
    gb_block.load_stddev_before = load_stddev_before;
    gb_block.load_stddev_after = load_stddev_after;

    gb_block.gb_row_idx = move(gb_row_idx_tmp);
    gb_block.gb_col_idx = move(gb_col_idx_tmp);
    gb_block.nnz_per_blk = move(nnz_per_blk_tmp);
    gb_block.type_per_blk = move(type_per_blk_tmp);
    gb_block.block_idx_array = move(blk_idx_array);
}

void coo2block_build_row_ptr(GLOBAL_BLOCK &gb_block){
    gb_block.gb_row_ptr.assign(gb_block.gb_m + 1, 0);
    for(int block_row : gb_block.gb_row_idx){
        gb_block.gb_row_ptr[block_row + 1]++;
    }
    for(int row = 0; row < gb_block.gb_m; row++){
        gb_block.gb_row_ptr[row + 1] += gb_block.gb_row_ptr[row];
    }
}

void coo2block_build_segments(GLOBAL_BLOCK &gb_block){
    gb_block.segment_rows.clear();
    gb_block.segment_begin.clear();
    gb_block.segment_end.clear();
    gb_block.segment_direct.clear();
    gb_block.segment_uniform.clear();
    gb_block.block_row_uniform.assign(gb_block.gb_m, 1);
    vector<uint8_t> row_has_blocks(gb_block.gb_m, 0);
    for(int block = 0; block < gb_block.gb_nnzb; block++){
        const int row = gb_block.gb_row_idx[block];
        row_has_blocks[row] = 1;
        gb_block.block_row_uniform[row]
            = gb_block.block_row_uniform[row]
            && !!(gb_block.type_per_blk[block] & UNIFORM_ROWS);
    }
    for(int row = 0; row < gb_block.gb_m; row++){
        const int begin = gb_block.gb_row_ptr[row];
        const int end = gb_block.gb_row_ptr[row + 1];
        const bool uniform = row_has_blocks[row]
            && gb_block.block_row_uniform[row];
        const bool direct = gb_block.mtx_nnz < 10000 || uniform;
        const int count = end == begin ? 0 : direct ? 1
            : (end - begin + WARP_PER_BLOCK - 1) / WARP_PER_BLOCK;
        for(int segment = 0; segment < count; segment++){
            const int segment_begin = direct
                ? begin : begin + segment * WARP_PER_BLOCK;
            gb_block.segment_rows.push_back(row);
            gb_block.segment_begin.push_back(segment_begin);
            gb_block.segment_end.push_back(direct ? end
                : std::min(segment_begin + WARP_PER_BLOCK, end));
            gb_block.segment_direct.push_back(direct);
            gb_block.segment_uniform.push_back(uniform);
        }
    }
}

void coo2block_build_flat_csr(const int* row_idx, const int* col_idx,
    GLOBAL_BLOCK &gb_block){
    gb_block.flat_row_ptr.assign(gb_block.mtx_m + 1, 0);
    gb_block.flat_col_idx.resize(gb_block.mtx_nnz);
    for(int i = 0; i < gb_block.mtx_nnz; i++){
        gb_block.flat_row_ptr[row_idx[i] + 1]++;
    }
    for(int row = 0; row < gb_block.mtx_m; row++){
        gb_block.flat_row_ptr[row + 1] += gb_block.flat_row_ptr[row];
    }
    vector<int> cursor = gb_block.flat_row_ptr;
    for(int i = 0; i < gb_block.mtx_nnz; i++){
        const int position = cursor[row_idx[i]]++;
        gb_block.flat_col_idx[position] = col_idx[i];
    }
}

// Rows with the same column pattern produce the same result for unit-valued
// matrices.  Keep this metadata beside the normal CB representation so the
// existing comp/gather kernels can select a wave64 pattern-reuse schedule.
void coo2block_build_patterns(GLOBAL_BLOCK &gb_block){
    std::map<std::vector<int>, int> pattern_ids;
    std::vector<std::vector<int>> pattern_columns;
    std::vector<std::vector<int>> pattern_rows;

    for(int row = 0; row < gb_block.mtx_m; row++){
        std::vector<int> columns(
            gb_block.flat_col_idx.begin() + gb_block.flat_row_ptr[row],
            gb_block.flat_col_idx.begin() + gb_block.flat_row_ptr[row + 1]);
        std::sort(columns.begin(), columns.end());
        auto inserted = pattern_ids.emplace(columns,
            static_cast<int>(pattern_columns.size()));
        int pattern = inserted.first->second;
        if(inserted.second){
            pattern_columns.push_back(std::move(columns));
            pattern_rows.emplace_back();
        }
        pattern_rows[pattern].push_back(row);
    }

    gb_block.pattern_row_ptr.assign(pattern_columns.size() + 1, 0);
    gb_block.pattern_output_ptr.assign(pattern_columns.size() + 1, 0);
    gb_block.pattern_col_idx.clear();
    gb_block.pattern_output_rows.clear();
    gb_block.pattern_col_idx.reserve(gb_block.mtx_nnz);
    gb_block.pattern_output_rows.reserve(gb_block.mtx_m);
    for(size_t pattern = 0; pattern < pattern_columns.size(); pattern++){
        gb_block.pattern_col_idx.insert(gb_block.pattern_col_idx.end(),
            pattern_columns[pattern].begin(), pattern_columns[pattern].end());
        gb_block.pattern_output_rows.insert(gb_block.pattern_output_rows.end(),
            pattern_rows[pattern].begin(), pattern_rows[pattern].end());
        gb_block.pattern_row_ptr[pattern + 1]
            = static_cast<int>(gb_block.pattern_col_idx.size());
        gb_block.pattern_output_ptr[pattern + 1]
            = static_cast<int>(gb_block.pattern_output_rows.size());
    }
}


void coo2block_comp(const int* row_idx, const int* col_idx, const ValType* val, GLOBAL_BLOCK &gb_block){
    coo2block_pre(row_idx, col_idx, val, gb_block);
    coo2block_post(row_idx, col_idx, val, gb_block);
    coo2block_build_row_ptr(gb_block);
    if(CB_ROW_OWNED){
        coo2block_build_segments(gb_block);
    }
    if(!CB_ROW_OWNED && gb_block.mtx_nnz >= 10000){
        coo2block_banlance(gb_block);
    }else{
        vector<BLOCK_IDX> blk_idx_array(gb_block.gb_nnzb,{0,0,0});
        #pragma omp parallel for
        for(int i=0; i<gb_block.gb_nnzb; i++){
            blk_idx_array[i].blk_nnz = gb_block.nnz_per_blk[i];
            blk_idx_array[i].origin_idx = i;
            blk_idx_array[i].end_idx = i;
        }
        gb_block.block_idx_array = move(blk_idx_array);
    }
    coo2block_build_flat_csr(row_idx, col_idx, gb_block);
    coo2block_build_patterns(gb_block);
}

void coo2block_gather(const int* row_idx, const int* col_idx, const ValType* val, GLOBAL_BLOCK &gb_block){
    vector<int> block_ids;
    coo2block_pre_metadata(row_idx, col_idx, gb_block, block_ids);
    if(gather_flag(gb_block)){
        int super_sparse_blocks = gb_block.super_sparse_blocks;
        int analyzed_blocks = gb_block.analyzed_blocks;
        double super_sparse_ratio = gb_block.super_sparse_ratio;
        vector<int> gather_col_idx(gb_block.mtx_nnz);
        gather(row_idx, col_idx, gather_col_idx, gb_block);
        vector<int> block_cols_offset = move(gb_block.block_cols_offset);
        vector<int> block_cols = move(gb_block.block_cols);
        gb_block.reset(gb_block.mtx_m, gb_block.mtx_n, gb_block.mtx_nnz);
        gb_block.gather_flag = true;
        gb_block.super_sparse_blocks = super_sparse_blocks;
        gb_block.analyzed_blocks = analyzed_blocks;
        gb_block.super_sparse_ratio = super_sparse_ratio;
        gb_block.block_cols_offset = move(block_cols_offset);
        gb_block.block_cols = move(block_cols);
        coo2block_pre(row_idx, gather_col_idx.data(), val, gb_block);
        coo2block_post(row_idx, gather_col_idx.data(), val, gb_block);
    }else{
        coo2block_pre_payload(row_idx, col_idx, val, block_ids, gb_block);
        coo2block_post(row_idx, col_idx, val, gb_block);
    }
    coo2block_build_row_ptr(gb_block);
    if(CB_ROW_OWNED){
        coo2block_build_segments(gb_block);
    }
    if(!CB_ROW_OWNED && gb_block.mtx_nnz >= 10000){
        coo2block_banlance(gb_block);
    }else{
        vector<BLOCK_IDX> blk_idx_array(gb_block.gb_nnzb,{0,0,0});
        #pragma omp parallel for
        for(int i=0; i<gb_block.gb_nnzb; i++){
            blk_idx_array[i].blk_nnz = gb_block.nnz_per_blk[i];
            blk_idx_array[i].origin_idx = i;
            blk_idx_array[i].end_idx = i;
        }
        gb_block.block_idx_array = move(blk_idx_array);
    }
    coo2block_build_flat_csr(row_idx, col_idx, gb_block);
    coo2block_build_patterns(gb_block);
}
