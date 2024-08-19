



#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <cstring>
#include <hydra/detail/external/hydra_thrust/system/detail/sequential/general_copy.h>

namespace hydra_thrust
{
namespace system
{
namespace detail
{
namespace sequential
{


template<typename T>
__host__ __device__
T *trivial_copy_n(const T *first,
std::ptrdiff_t n,
T *result)
{
#ifndef __CUDA_ARCH__
std::memmove(result, first, n * sizeof(T));
return result + n;
#else
return hydra_thrust::system::detail::sequential::general_copy_n(first, n, result);
#endif
} 


} 
} 
} 
} 

