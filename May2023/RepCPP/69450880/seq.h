

#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/detail/allocator_aware_execution_policy.h>
#include <hydra/detail/external/hydra_thrust/system/detail/sequential/execution_policy.h>

namespace hydra_thrust
{
namespace detail
{


struct seq_t : hydra_thrust::system::detail::sequential::execution_policy<seq_t>,
hydra_thrust::detail::allocator_aware_execution_policy<
hydra_thrust::system::detail::sequential::execution_policy>
{
__host__ __device__
seq_t() : hydra_thrust::system::detail::sequential::execution_policy<seq_t>() {}

template<typename DerivedPolicy>
__host__ __device__
seq_t(const hydra_thrust::execution_policy<DerivedPolicy> &)
: hydra_thrust::system::detail::sequential::execution_policy<seq_t>()
{}
};


} 


#ifdef __CUDA_ARCH__
static const __device__ detail::seq_t seq;
#else
static const detail::seq_t seq;
#endif


} 


