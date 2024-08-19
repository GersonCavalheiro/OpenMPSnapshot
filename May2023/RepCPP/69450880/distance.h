


#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/iterator/iterator_traits.h>

namespace hydra_thrust
{
namespace system
{
namespace detail
{
namespace generic
{

template<typename InputIterator>
inline __host__ __device__
typename hydra_thrust::iterator_traits<InputIterator>::difference_type
distance(InputIterator first, InputIterator last);

} 
} 
} 
} 

#include <hydra/detail/external/hydra_thrust/system/detail/generic/distance.inl>

