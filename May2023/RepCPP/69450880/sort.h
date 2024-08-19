

#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/system/tbb/detail/execution_policy.h>

namespace hydra_thrust
{
namespace system
{
namespace tbb
{
namespace detail
{

template<typename DerivedPolicy,
typename RandomAccessIterator,
typename StrictWeakOrdering>
void stable_sort(execution_policy<DerivedPolicy> &exec,
RandomAccessIterator first,
RandomAccessIterator last,
StrictWeakOrdering comp);

template<typename DerivedPolicy,
typename RandomAccessIterator1,
typename RandomAccessIterator2,
typename StrictWeakOrdering>
void stable_sort_by_key(execution_policy<DerivedPolicy> &exec,
RandomAccessIterator1 keys_first,
RandomAccessIterator1 keys_last,
RandomAccessIterator2 values_first,
StrictWeakOrdering comp);

} 
} 
} 
} 

#include <hydra/detail/external/hydra_thrust/system/tbb/detail/sort.inl>

