

#ifndef GKO_DPCPP_COMPONENTS_REDUCTION_DP_HPP_
#define GKO_DPCPP_COMPONENTS_REDUCTION_DP_HPP_


#include <type_traits>


#include <CL/sycl.hpp>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/synthesizer/containers.hpp>


#include "core/base/types.hpp"
#include "core/synthesizer/implementation_selection.hpp"
#include "dpcpp/base/config.hpp"
#include "dpcpp/base/dim3.dp.hpp"
#include "dpcpp/base/dpct.hpp"
#include "dpcpp/base/helper.hpp"
#include "dpcpp/components/cooperative_groups.dp.hpp"
#include "dpcpp/components/thread_ids.dp.hpp"
#include "dpcpp/components/uninitialized_array.hpp"
#include "dpcpp/synthesizer/implementation_selection.hpp"


namespace gko {
namespace kernels {
namespace dpcpp {


static constexpr int default_block_size = 256;
static constexpr auto dcfg_1d_list = dcfg_1d_list_t();
static constexpr auto dcfg_1d_array = as_array(dcfg_1d_list);


template <
typename Group, typename ValueType, typename Operator,
typename = std::enable_if_t<group::is_communicator_group<Group>::value>>
__dpct_inline__ ValueType reduce(const Group& group, ValueType local_data,
Operator reduce_op = Operator{})
{
#pragma unroll
for (int32 bitmask = 1; bitmask < group.size(); bitmask <<= 1) {
const auto remote_data = group.shfl_xor(local_data, bitmask);
local_data = reduce_op(local_data, remote_data);
}
return local_data;
}



template <
typename Group, typename ValueType,
typename = std::enable_if_t<group::is_communicator_group<Group>::value>>
__dpct_inline__ int choose_pivot(const Group& group, ValueType local_data,
bool is_pivoted)
{
using real = remove_complex<ValueType>;
real lmag = is_pivoted ? -one<real>() : abs(local_data);
const auto pivot = ::gko::kernels::dpcpp::reduce(
group, group.thread_rank(), [&](int lidx, int ridx) {
const auto rmag = group.shfl(lmag, ridx);
if (rmag > lmag) {
lmag = rmag;
lidx = ridx;
}
return lidx;
});
return group.shfl(pivot, 0);
}



template <
unsigned int sg_size = config::warp_size, typename Group,
typename ValueType, typename Operator,
typename = std::enable_if_t<group::is_synchronizable_group<Group>::value>>
void reduce(const Group& __restrict__ group, ValueType* __restrict__ data,
Operator reduce_op = Operator{})
{
const auto local_id = group.thread_rank();

for (int k = group.size() / 2; k >= sg_size; k /= 2) {
group.sync();
if (local_id < k) {
data[local_id] = reduce_op(data[local_id], data[local_id + k]);
}
}

const auto warp = group::tiled_partition<sg_size>(group);
const auto warp_id = group.thread_rank() / warp.size();
if (warp_id > 0) {
return;
}
auto result = ::gko::kernels::dpcpp::reduce(warp, data[warp.thread_rank()],
reduce_op);
if (warp.thread_rank() == 0) {
data[0] = result;
}
}



template <unsigned int sg_size = config::warp_size, typename Operator,
typename ValueType>
void reduce_array(size_type size, const ValueType* __restrict__ source,
ValueType* __restrict__ result, sycl::nd_item<3> item_ct1,
Operator reduce_op = Operator{})
{
const auto tidx = thread::get_thread_id_flat(item_ct1);
auto thread_result = zero<ValueType>();
for (auto i = tidx; i < size;
i += item_ct1.get_local_range().get(2) * item_ct1.get_group_range(2)) {
thread_result = reduce_op(thread_result, source[i]);
}
result[item_ct1.get_local_id(2)] = thread_result;

group::this_thread_block(item_ct1).sync();

reduce<sg_size>(group::this_thread_block(item_ct1), result, reduce_op);
}



template <typename DeviceConfig, typename ValueType>
void reduce_add_array(
size_type size, const ValueType* __restrict__ source,
ValueType* __restrict__ result, sycl::nd_item<3> item_ct1,
uninitialized_array<ValueType, DeviceConfig::block_size>& block_sum)
{
reduce_array<DeviceConfig::subgroup_size>(
size, source, static_cast<ValueType*>(block_sum), item_ct1,
[](const ValueType& x, const ValueType& y) { return x + y; });

if (item_ct1.get_local_id(2) == 0) {
result[item_ct1.get_group(2)] = block_sum[0];
}
}

template <typename DeviceConfig = device_config<256, 16>, typename ValueType>
void reduce_add_array(dim3 grid, dim3 block, size_type dynamic_shared_memory,
sycl::queue* queue, size_type size,
const ValueType* source, ValueType* result)
{
queue->submit([&](sycl::handler& cgh) {
sycl::accessor<uninitialized_array<ValueType, DeviceConfig::block_size>,
0, sycl::access::mode::read_write,
sycl::access::target::local>
block_sum_acc_ct1(cgh);

cgh.parallel_for(
sycl_nd_range(grid, block),
[=](sycl::nd_item<3> item_ct1)
[[sycl::reqd_sub_group_size(DeviceConfig::subgroup_size)]] {
reduce_add_array<DeviceConfig>(
size, source, result, item_ct1,
*block_sum_acc_ct1.get_pointer());
});
});
}

GKO_ENABLE_IMPLEMENTATION_CONFIG_SELECTION_TOTYPE(reduce_add_array_config,
reduce_add_array, DCFG_1D);

GKO_ENABLE_DEFAULT_CONFIG_CALL(reduce_add_array_call, reduce_add_array_config,
dcfg_1d_list);



template <typename ValueType>
ValueType reduce_add_array(std::shared_ptr<const DpcppExecutor> exec,
size_type size, const ValueType* source)
{
auto block_results_val = source;
size_type grid_dim = size;
auto block_results = array<ValueType>(exec);
ValueType answer = zero<ValueType>();
auto queue = exec->get_queue();
constexpr auto dcfg_1d_array = as_array(dcfg_1d_list);
const std::uint32_t cfg =
get_first_cfg(dcfg_1d_array, [&queue](std::uint32_t cfg) {
return validate(queue, DCFG_1D::decode<0>(cfg),
DCFG_1D::decode<1>(cfg));
});
const auto wg_size = DCFG_1D::decode<0>(cfg);
const auto sg_size = DCFG_1D::decode<1>(cfg);

if (size > wg_size) {
const auto n = ceildiv(size, wg_size);
grid_dim = (n <= wg_size) ? n : wg_size;

block_results.resize_and_reset(grid_dim);

reduce_add_array_call(cfg, grid_dim, wg_size, 0, exec->get_queue(),
size, source, block_results.get_data());

block_results_val = block_results.get_const_data();
}

auto d_result = array<ValueType>(exec, 1);

reduce_add_array_call(cfg, 1, wg_size, 0, exec->get_queue(), grid_dim,
block_results_val, d_result.get_data());
answer = exec->copy_val_to_host(d_result.get_const_data());
return answer;
}


}  
}  
}  


#endif  