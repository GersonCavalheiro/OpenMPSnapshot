

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
typename InputIterator,
typename OutputIterator>
OutputIterator copy(execution_policy<DerivedPolicy> &exec,
InputIterator first,
InputIterator last,
OutputIterator result);


template<typename DerivedPolicy,
typename InputIterator,
typename Size,
typename OutputIterator>
OutputIterator copy_n(execution_policy<DerivedPolicy> &exec,
InputIterator first,
Size n,
OutputIterator result);


} 
} 
} 
} 

#include <hydra/detail/external/hydra_thrust/system/tbb/detail/copy.inl>

