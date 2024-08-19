

#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/system/tbb/detail/execution_policy.h>
#include <hydra/detail/external/hydra_thrust/system/detail/generic/extrema.h>

namespace hydra_thrust
{
namespace system
{
namespace tbb
{
namespace detail
{

template <typename DerivedPolicy, typename ForwardIterator, typename BinaryPredicate>
ForwardIterator max_element(execution_policy<DerivedPolicy> &exec,
ForwardIterator first, 
ForwardIterator last,
BinaryPredicate comp)
{
return hydra_thrust::system::detail::generic::max_element(exec, first, last, comp);
} 

template <typename DerivedPolicy, typename ForwardIterator, typename BinaryPredicate>
ForwardIterator min_element(execution_policy<DerivedPolicy> &exec,
ForwardIterator first, 
ForwardIterator last,
BinaryPredicate comp)
{
return hydra_thrust::system::detail::generic::min_element(exec, first, last, comp);
} 

template <typename DerivedPolicy, typename ForwardIterator, typename BinaryPredicate>
hydra_thrust::pair<ForwardIterator,ForwardIterator> minmax_element(execution_policy<DerivedPolicy> &exec,
ForwardIterator first, 
ForwardIterator last,
BinaryPredicate comp)
{
return hydra_thrust::system::detail::generic::minmax_element(exec, first, last, comp);
} 

} 
} 
} 
} 


