


#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>

namespace hydra_thrust
{
namespace system
{
namespace detail
{
namespace generic
{

template<typename InputIterator, typename Distance>
__host__ __device__
void advance(InputIterator& i, Distance n);

} 
} 
} 
} 

#include <hydra/detail/external/hydra_thrust/system/detail/generic/advance.inl>

