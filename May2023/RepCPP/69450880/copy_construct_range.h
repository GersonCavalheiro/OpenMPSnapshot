

#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/detail/execution_policy.h>

namespace hydra_thrust
{
namespace detail
{

template<typename System, typename Allocator, typename InputIterator, typename Pointer>
__host__ __device__
Pointer copy_construct_range(hydra_thrust::execution_policy<System> &from_system,
Allocator &a,
InputIterator first,
InputIterator last,
Pointer result);

template<typename System, typename Allocator, typename InputIterator, typename Size, typename Pointer>
__host__ __device__
Pointer copy_construct_range_n(hydra_thrust::execution_policy<System> &from_system,
Allocator &a,
InputIterator first,
Size n,
Pointer result);

} 
} 

#include <hydra/detail/external/hydra_thrust/detail/allocator/copy_construct_range.inl>

