

#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>

namespace hydra_thrust
{
namespace detail
{

template<typename Allocator, typename Pointer, typename Size>
__host__ __device__
inline void destroy_range(Allocator &a, Pointer p, Size n);

} 
} 

#include <hydra/detail/external/hydra_thrust/detail/allocator/destroy_range.inl>

