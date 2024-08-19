#pragma once
#include <costa/grid2grid/profiler.hpp>
#include <costa/grid2grid/grid_layout.hpp>
#include <costa/grid2grid/grid_cover.hpp>
#include <costa/grid2grid/communication_data.hpp>
#include <unordered_map>
#include <algorithm>
#include <vector>

namespace costa {

std::vector<std::vector<int>> topology_cost(MPI_Comm comm);

namespace utils {

bool if_should_transpose(const char src_ordering,
const char dest_ordering,
const char trans);

std::unordered_map<int, int> rank_to_comm_vol_for_block(
const assigned_grid2D& g_init,
const block_coordinates &b_coord,
grid_cover &g_cover,
const assigned_grid2D& g_final);

template <typename T>
std::vector<message<T>> decompose_block(const block<T> &b,
grid_cover &g_cover,
const assigned_grid2D &g,
const char final_ordering,
const T alpha, const T beta,
bool transpose, bool conjugate
) {
block_cover b_cover = g_cover.decompose_block(b);

int row_first = b_cover.rows_cover.start_index;
int row_last = b_cover.rows_cover.end_index;

int col_first = b_cover.cols_cover.start_index;
int col_last = b_cover.cols_cover.end_index;

int n_blocks = (col_last - col_first) * (row_last - row_first);

std::vector<message<T>> decomposed_blocks;
decomposed_blocks.reserve(n_blocks);

int col_start = b.cols_interval.start;
for (int j = col_first; j < col_last; ++j) {
int row_start = b.rows_interval.start;
int col_end =
std::min(g.grid().cols_split[j + 1], b.cols_interval.end);
for (int i = row_first; i < row_last; ++i) {
int row_end = std::min(g.grid().rows_split[i + 1], b.rows_interval.end);
int rank = g.owner(i, j);

block<T> subblock =
b.subblock({row_start, row_end}, {col_start, col_end});

assert(subblock.non_empty());
if (subblock.non_empty()) {
decomposed_blocks.push_back({subblock, rank,
final_ordering,
alpha, beta,
transpose, conjugate});
}
row_start = row_end;
}
col_start = col_end;
}
return decomposed_blocks;
}

template <typename T>
std::vector<message<T>> decompose_blocks(grid_layout<T> &init_layout,
grid_layout<T> &final_layout,
const T alpha, const T beta,
bool transpose,
bool conjugate,
int tag = 0) {
PE(transform_decompose);
grid_cover g_overlap(init_layout.grid.grid(), final_layout.grid.grid());

std::vector<message<T>> messages;

for (int i = 0; i < init_layout.blocks.num_blocks(); ++i) {
auto blk = init_layout.blocks.get_block(i);
blk.tag = tag;
assert(blk.non_empty());
std::vector<message<T>> decomposed =
decompose_block(blk, g_overlap, 
final_layout.grid,
final_layout.ordering,
alpha, beta, transpose, conjugate);
messages.insert(messages.end(), decomposed.begin(), decomposed.end());
}

PL();
return messages;
}


template <typename T>
void merge_messages(std::vector<message<T>> &messages) {
std::sort(messages.begin(), messages.end());
}

template <typename T>
communication_data<T> prepare_to_send(grid_layout<T> &init_layout,
grid_layout<T> &final_layout,
int rank,
const T alpha, const T beta,
bool transpose, bool conjugate) {
std::vector<message<T>> messages =
decompose_blocks(init_layout, final_layout, 
alpha, beta, transpose, conjugate);
merge_messages(messages);

return communication_data<T>(messages, rank, std::max(final_layout.num_ranks(), init_layout.num_ranks()), costa::CommType::send);
}

template <typename T>
communication_data<T> prepare_to_send(
std::vector<layout_ref<T>>& from,
std::vector<layout_ref<T>>& to,
int rank,
const T* alpha, const T* beta,
bool* transpose,
bool* conjugate) {
std::vector<message<T>> messages;
int n_ranks = 0;

for (unsigned i = 0u; i < from.size(); ++i) {
auto& init_layout = from[i].get();
auto& final_layout = to[i].get();

auto decomposed_blocks = decompose_blocks(init_layout, final_layout, 
alpha[i], beta[i], 
transpose[i], conjugate[i], 
i);
messages.insert(messages.end(), decomposed_blocks.begin(), decomposed_blocks.end());
n_ranks = std::max(n_ranks, std::max(final_layout.num_ranks(), init_layout.num_ranks()));
}
merge_messages(messages);
return communication_data<T>(messages, rank, n_ranks, costa::CommType::send);
}
template <typename T> 
communication_data<T> prepare_to_recv(grid_layout<T> &final_layout,
grid_layout<T> &init_layout,
int rank,
const T alpha, const T beta,
const bool transpose, const bool conjugate) {
std::vector<message<T>> messages =
decompose_blocks(final_layout, init_layout, 
alpha, beta, transpose, conjugate);
merge_messages(messages);

return communication_data<T>(messages, rank, std::max(init_layout.num_ranks(), final_layout.num_ranks()), costa::CommType::recv);
}

template <typename T>
communication_data<T> prepare_to_recv(
std::vector<layout_ref<T>>& to,
std::vector<layout_ref<T>>& from,
int rank,
const T* alpha, const T* beta,
bool* transpose,
bool* conjugate) {
std::vector<message<T>> messages;
int n_ranks = 0;

for (unsigned i = 0u; i < from.size(); ++i) {
auto& init_layout = from[i].get();
auto& final_layout = to[i].get();

auto decomposed_blocks = decompose_blocks(final_layout, init_layout, 
alpha[i], beta[i], 
transpose[i], conjugate[i], 
i);
messages.insert(messages.end(), decomposed_blocks.begin(), decomposed_blocks.end());
n_ranks = std::max(n_ranks, std::max(init_layout.num_ranks(), final_layout.num_ranks()));
}
merge_messages(messages);
return communication_data<T>(messages, rank, n_ranks, costa::CommType::recv);
}
} 
} 

