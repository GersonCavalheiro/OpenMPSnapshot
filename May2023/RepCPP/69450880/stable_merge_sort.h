

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
typename RandomAccessIterator,
typename StrictWeakOrdering>
__host__ __device__
void stable_merge_sort(sequential::execution_policy<DerivedPolicy> &exec,
RandomAccessIterator begin,
RandomAccessIterator end,
StrictWeakOrdering comp);


template<typename DerivedPolicy,
typename RandomAccessIterator1,
typename RandomAccessIterator2,
typename StrictWeakOrdering>
__host__ __device__
void stable_merge_sort_by_key(sequential::execution_policy<DerivedPolicy> &exec,
RandomAccessIterator1 keys_begin,
RandomAccessIterator1 keys_end,
RandomAccessIterator2 values_begin,
StrictWeakOrdering comp);


} 
} 
} 
} 

#include <hydra/detail/external/hydra_thrust/system/detail/sequential/stable_merge_sort.inl>

