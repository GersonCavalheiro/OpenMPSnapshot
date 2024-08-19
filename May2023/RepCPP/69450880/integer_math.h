

#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <limits>

#if HYDRA_THRUST_CPP_DIALECT >= 2011
#include <hydra/detail/external/hydra_thrust/detail/type_deduction.h>
#endif

namespace hydra_thrust
{
namespace detail
{

template <typename Integer>
__host__ __device__ __hydra_thrust_forceinline__
Integer clz(Integer x)
{
#if __CUDA_ARCH__
return ::__clz(x);
#else
int num_bits = 8 * sizeof(Integer);
int num_bits_minus_one = num_bits - 1;

for (int i = num_bits_minus_one; i >= 0; --i)
{
if ((Integer(1) << i) & x)
{
return num_bits_minus_one - i;
}
}

return num_bits;
#endif
}

template <typename Integer>
__host__ __device__ __hydra_thrust_forceinline__
bool is_power_of_2(Integer x)
{
return 0 == (x & (x - 1));
}

template <typename Integer>
__host__ __device__ __hydra_thrust_forceinline__
bool is_odd(Integer x)
{
return 1 & x;
}

template <typename Integer>
__host__ __device__ __hydra_thrust_forceinline__
Integer log2(Integer x)
{
Integer num_bits = 8 * sizeof(Integer);
Integer num_bits_minus_one = num_bits - 1;

return num_bits_minus_one - clz(x);
}


template <typename Integer>
__host__ __device__ __hydra_thrust_forceinline__
Integer log2_ri(Integer x)
{
Integer result = log2(x);

if (!is_power_of_2(x))
++result;

return result;
}

template <typename Integer0, typename Integer1>
__host__ __device__ __hydra_thrust_forceinline__
#if HYDRA_THRUST_CPP_DIALECT >= 2011
auto divide_ri(Integer0 const x, Integer1 const y)
HYDRA_THRUST_DECLTYPE_RETURNS((x + (y - 1)) / y)
#else
Integer0 divide_ri(Integer0 const x, Integer1 const y)
{
return (x + (y - 1)) / y;
}
#endif

template <typename Integer0, typename Integer1>
__host__ __device__ __hydra_thrust_forceinline__
#if HYDRA_THRUST_CPP_DIALECT >= 2011
auto divide_rz(Integer0 const x, Integer1 const y)
HYDRA_THRUST_DECLTYPE_RETURNS(x / y)
#else
Integer0 divide_rz(Integer0 const x, Integer1 const y)
{
return x / y;
}
#endif

template <typename Integer0, typename Integer1>
__host__ __device__ __hydra_thrust_forceinline__
#if HYDRA_THRUST_CPP_DIALECT >= 2011
auto round_i(Integer0 const x, Integer1 const y)
HYDRA_THRUST_DECLTYPE_RETURNS(y * divide_ri(x, y))
#else
Integer0 round_i(Integer0 const x, Integer1 const y)
{
return y * divide_ri(x, y);
}
#endif

template <typename Integer0, typename Integer1>
__host__ __device__ __hydra_thrust_forceinline__
#if HYDRA_THRUST_CPP_DIALECT >= 2011
auto round_z(Integer0 const x, Integer1 const y)
HYDRA_THRUST_DECLTYPE_RETURNS(y * divide_rz(x, y))
#else
Integer0 round_z(Integer0 const x, Integer1 const y)
{
return y * divide_rz(x, y);
}
#endif

} 
} 

