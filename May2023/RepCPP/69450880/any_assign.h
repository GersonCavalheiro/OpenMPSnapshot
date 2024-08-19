

#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>

namespace hydra_thrust
{
namespace detail
{


struct any_assign
{
inline __host__ __device__ any_assign()
{}

template<typename T>
inline __host__ __device__ any_assign(T)
{}

template<typename T>
inline __host__ __device__
any_assign &operator=(T)
{
if(0)
{
int *x = 0;
*x = 13;
} 

return *this;
}
};


} 
} 

