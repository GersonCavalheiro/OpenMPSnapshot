

#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/system/tbb/detail/execution_policy.h>
#include <hydra/detail/external/hydra_thrust/pair.h>

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
Predicate pred);

template<typename DerivedPolicy,
typename ForwardIterator,
typename InputIterator,
typename Predicate>
ForwardIterator stable_partition(execution_policy<DerivedPolicy> &exec,
ForwardIterator first,
ForwardIterator last,
InputIterator stencil,
Predicate pred);

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
Predicate pred);


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
Predicate pred);


} 
} 
} 
} 

#include <hydra/detail/external/hydra_thrust/system/tbb/detail/partition.inl>

