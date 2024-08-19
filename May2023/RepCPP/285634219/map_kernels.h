























#pragma once

#include <cstdint>

namespace GpuBTree {
namespace kernels {
template <typename KeyT, typename ValueT, typename SizeT>
void insert_keys(uint32_t *__restrict d_root, KeyT *__restrict d_keys,
ValueT *__restrict d_values, SizeT num_keys,
uint32_t *__restrict d_pool,
uint32_t *__restrict d_count,
sycl::nd_item<1> &item) {
uint32_t tid = item.get_global_id(0);
uint32_t laneId = lane_id(item);

KeyT myKey;
ValueT myValue;
bool to_insert = false;

if ((tid - laneId) >= num_keys)
return;

if (tid < num_keys) {
myKey = d_keys[tid] + 2;
myValue = d_values[tid] + 2;
to_insert = true;
}

warps::insertion_unit(to_insert, myKey, myValue, d_root, d_pool, d_count, item);
}

void init_btree(uint32_t *__restrict d_root,
uint32_t *__restrict d_pool,
uint32_t *__restrict d_count,
sycl::nd_item<1> &item) {
uint32_t laneId = lane_id(item);

uint32_t root_id;
if (laneId == 0)
root_id = allocate();

root_id = sycl::select_from_group(item.get_sub_group(), root_id, 0);

*d_root = root_id;
uint32_t* tree_root = getAddressPtr(root_id);

if (laneId < 2)
tree_root[laneId] = 1 - laneId;
}

template <typename KeyT, typename ValueT, typename SizeT>
void search_b_tree(uint32_t *__restrict d_root, KeyT *__restrict d_queries,
ValueT *__restrict d_results, SizeT num_queries,
uint32_t *__restrict d_pool,
sycl::nd_item<1> &item) {
uint32_t tid = item.get_global_id(0);
uint32_t laneId = lane_id(item);
if ((tid - laneId) >= num_queries)
return;

uint32_t myQuery = 0;
uint32_t myResult = SEARCH_NOT_FOUND;
bool to_search = false;

if (tid < num_queries) {
myQuery = d_queries[tid] + 2;
to_search = true;
}

warps::search_unit(to_search, laneId, myQuery, myResult, d_root, d_pool, item);

if (tid < num_queries)
myResult = myResult ? myResult - 2 : myResult;
d_results[tid] = myResult;
}

template <typename KeyT, typename SizeT>
void delete_b_tree(uint32_t *__restrict d_root,
KeyT *__restrict d_queries,
SizeT num_queries,
uint32_t *__restrict d_pool,
sycl::nd_item<1> &item) {
uint32_t tid = item.get_global_id(0);
uint32_t laneId = lane_id(item);
if ((tid - laneId) >= num_queries)
return;

KeyT myQuery = 0xFFFFFFFF;

if (tid < uint32_t(num_queries)) {
myQuery = d_queries[tid] + 2;
}

warps::delete_unit_bulk(laneId, myQuery, d_root, d_pool, item);
}

template <typename KeyT, typename ValueT, typename SizeT>
void range_b_tree(
uint32_t *__restrict d_root, KeyT *__restrict d_queries_lower,
KeyT *__restrict d_queries_upper,
ValueT *__restrict d_range_results, SizeT num_queries,
SizeT range_length, uint32_t *__restrict d_pool,
sycl::nd_item<1> &item) {

uint32_t tid = item.get_global_id(0);
uint32_t laneId = lane_id(item);
if ((tid - laneId) >= num_queries)
return;

uint32_t lower_bound = 0;
uint32_t upper_bound = 0;
bool to_search = false;

if (tid < num_queries) {
lower_bound = d_queries_lower[tid] + 2;
upper_bound = d_queries_upper[tid] + 2;
to_search = true;
}

warps::range_unit(laneId, to_search, lower_bound, upper_bound,
d_range_results, d_root, range_length, d_pool,
item);
}

};  
};  
