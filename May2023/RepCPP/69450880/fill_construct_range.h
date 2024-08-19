

#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>

namespace hydra_thrust
{
namespace detail
{


template<typename Allocator, typename Pointer, typename Size, typename T>
__host__ __device__
inline void fill_construct_range(Allocator &a, Pointer p, Size n, const T &value);


} 
} 

#include <hydra/detail/external/hydra_thrust/detail/allocator/fill_construct_range.inl>

