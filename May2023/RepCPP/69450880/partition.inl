

#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/system/tbb/detail/partition.h>
#include <hydra/detail/external/hydra_thrust/system/detail/generic/partition.h>

namespace hydra_thrust
{
namespace system
{
namespace tbb
{
namespace detail
{


template<typename DerivedPolicy,
typename ForwardIterator,
typename Predicate>
ForwardIterator stable_partition(execution_policy<DerivedPolicy> &exec,
ForwardIterator first,
ForwardIterator last,
Predicate pred)
{
return hydra_thrust::system::detail::generic::stable_partition(exec, first, last, pred);
} 


template<typename DerivedPolicy,
typename ForwardIterator,
typename InputIterator,
typename Predicate>
ForwardIterator stable_partition(execution_policy<DerivedPolicy> &exec,
ForwardIterator first,
ForwardIterator last,
InputIterator stencil,
Predicate pred)
{
return hydra_thrust::system::detail::generic::stable_partition(exec, first, last, stencil, pred);
} 

template<typename DerivedPolicy,
typename InputIterator,
typename OutputIterator1,
typename OutputIterator2,
typename Predicate>
hydra_thrust::pair<OutputIterator1,OutputIterator2>
stable_partition_copy(execution_policy<DerivedPolicy> &exec,
InputIterator first,
InputIterator last,
OutputIterator1 out_true,
OutputIterator2 out_false,
Predicate pred)
{
return hydra_thrust::system::detail::generic::stable_partition_copy(exec, first, last, out_true, out_false, pred);
} 


template<typename DerivedPolicy,
typename InputIterator1,
typename InputIterator2,
typename OutputIterator1,
typename OutputIterator2,
typename Predicate>
hydra_thrust::pair<OutputIterator1,OutputIterator2>
stable_partition_copy(execution_policy<DerivedPolicy> &exec,
InputIterator1 first,
InputIterator1 last,
InputIterator2 stencil,
OutputIterator1 out_true,
OutputIterator2 out_false,
Predicate pred)
{
return hydra_thrust::system::detail::generic::stable_partition_copy(exec, first, last, stencil, out_true, out_false, pred);
} 


} 
} 
} 
} 

