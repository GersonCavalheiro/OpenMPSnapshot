

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
typename UnaryFunction>
RandomAccessIterator for_each(execution_policy<DerivedPolicy> &exec,
RandomAccessIterator first,
RandomAccessIterator last,
UnaryFunction f);

template<typename DerivedPolicy,
typename RandomAccessIterator,
typename Size,
typename UnaryFunction>
RandomAccessIterator for_each_n(execution_policy<DerivedPolicy> &exec,
RandomAccessIterator first,
Size n,
UnaryFunction f);

} 
} 
} 
} 

#include <hydra/detail/external/hydra_thrust/system/tbb/detail/for_each.inl>

