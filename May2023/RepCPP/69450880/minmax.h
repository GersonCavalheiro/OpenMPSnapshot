

#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>

namespace hydra_thrust
{


template<typename T, typename BinaryPredicate>
__host__ __device__
T min HYDRA_THRUST_PREVENT_MACRO_SUBSTITUTION (const T &lhs, const T &rhs, BinaryPredicate comp)
{
return comp(rhs, lhs) ? rhs : lhs;
} 

template<typename T>
__host__ __device__
T min HYDRA_THRUST_PREVENT_MACRO_SUBSTITUTION (const T &lhs, const T &rhs)
{
return rhs < lhs ? rhs : lhs;
} 

template<typename T, typename BinaryPredicate>
__host__ __device__
T max HYDRA_THRUST_PREVENT_MACRO_SUBSTITUTION (const T &lhs, const T &rhs, BinaryPredicate comp)
{
return comp(lhs,rhs) ? rhs : lhs;
} 

template<typename T>
__host__ __device__
T max HYDRA_THRUST_PREVENT_MACRO_SUBSTITUTION (const T &lhs, const T &rhs)
{
return lhs < rhs ? rhs : lhs;
} 


} 

