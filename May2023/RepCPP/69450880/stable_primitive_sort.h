

#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/system/detail/sequential/execution_policy.h>

namespace hydra_thrust
{
namespace system
{
namespace detail
{
namespace sequential
{


template<typename DerivedPolicy,
typename RandomAccessIterator>
__host__ __device__
void stable_primitive_sort(sequential::execution_policy<DerivedPolicy> &exec,
RandomAccessIterator first,
RandomAccessIterator last);


template<typename DerivedPolicy,
typename RandomAccessIterator1,
typename RandomAccessIterator2>
__host__ __device__
void stable_primitive_sort_by_key(sequential::execution_policy<DerivedPolicy> &exec,
RandomAccessIterator1 keys_first,
RandomAccessIterator1 keys_last,
RandomAccessIterator2 values_first);


} 
} 
} 
} 

#include <hydra/detail/external/hydra_thrust/system/detail/sequential/stable_primitive_sort.inl>

