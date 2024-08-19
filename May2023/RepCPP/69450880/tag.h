




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

struct tag
{
template<typename T>
__host__ __device__ inline
tag(const T &) {}
};

} 
} 
} 
} 

