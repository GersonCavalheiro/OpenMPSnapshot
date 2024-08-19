

#ifndef GKO_DPCPP_COMPONENTS_SEARCHING_DP_HPP_
#define GKO_DPCPP_COMPONENTS_SEARCHING_DP_HPP_


#include <CL/sycl.hpp>


#include "dpcpp/base/config.hpp"
#include "dpcpp/base/dpct.hpp"
#include "dpcpp/components/intrinsics.dp.hpp"


namespace gko {
namespace kernels {
namespace dpcpp {



template <typename IndexType, typename Predicate>
__dpct_inline__ IndexType binary_search(IndexType offset, IndexType length,
Predicate p)
{
while (length > 0) {
auto half_length = length / 2;
auto mid = offset + half_length;
auto pred = p(mid);
length = pred ? half_length : length - (half_length + 1);
offset = pred ? offset : mid + 1;
}
return offset;
}



template <int size, typename Predicate>
__dpct_inline__ int synchronous_fixed_binary_search(Predicate p)
{
if (size == 0) {
return 0;
}
int begin{};
static_assert(size > 0, "size must be positive");
static_assert(!(size & (size - 1)), "size must be a power of two");
#pragma unroll
for (auto cur_size = size; cur_size > 1; cur_size /= 2) {
auto half_size = cur_size / 2;
auto mid = begin + half_size;
begin = p(mid) ? begin : mid;
}
return p(begin) ? begin : begin + 1;
}



template <typename Predicate>
__dpct_inline__ int synchronous_binary_search(int size, Predicate p)
{
if (size == 0) {
return 0;
}
int begin{};
for (auto cur_size = size; cur_size > 1; cur_size /= 2) {
auto half_size = cur_size / 2;
auto mid = begin + half_size;
begin = p(mid) ? begin : mid;
}
return p(begin) ? begin : begin + 1;
}



template <typename IndexType, typename Group, typename Predicate>
__dpct_inline__ IndexType group_wide_search(IndexType offset, IndexType length,
Group group, Predicate p)
{
IndexType num_blocks = (length + group.size() - 1) / group.size();
auto group_pos = binary_search(IndexType{}, num_blocks, [&](IndexType i) {
auto idx = i * group.size();
return p(offset + idx);
});
if (group_pos == 0) {
return offset;
}

auto base_idx = (group_pos - 1) * group.size() + 1;
auto idx = base_idx + group.thread_rank();
auto pos = ffs(group.ballot(idx >= length || p(offset + idx))) - 1;
return offset + base_idx + pos;
}



template <typename IndexType, typename Group, typename Predicate>
__dpct_inline__ IndexType group_ary_search(IndexType offset, IndexType length,
Group group, Predicate p)
{
IndexType end = offset + length;
while (length > group.size()) {
auto stride = length / group.size();
auto idx = offset + group.thread_rank() * stride;
auto mask = group.ballot(p(idx));
auto pos = mask == 0 ? group.size() - 1 : ffs(mask >> 1) - 1;
auto last_length = length - stride * (group.size() - 1);
length = pos == group.size() - 1 ? last_length : stride;
offset += stride * pos;
}
auto idx = offset + group.thread_rank();
auto mask = group.ballot(idx >= end || p(idx));
auto pos = mask == 0 ? group.size() : ffs(mask) - 1;
return offset + pos;
}


}  
}  
}  


#endif  
