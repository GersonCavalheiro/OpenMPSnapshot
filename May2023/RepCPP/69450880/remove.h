

#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/system/omp/detail/execution_policy.h>

namespace hydra_thrust
{
namespace system
{
namespace omp
{
namespace detail
{

template<typename ExecutionPolicy,
typename ForwardIterator,
typename Predicate>
ForwardIterator remove_if(execution_policy<ExecutionPolicy> &exec,
ForwardIterator first,
ForwardIterator last,
Predicate pred);


template<typename ExecutionPolicy,
typename ForwardIterator,
typename InputIterator,
typename Predicate>
ForwardIterator remove_if(execution_policy<ExecutionPolicy> &exec,
ForwardIterator first,
ForwardIterator last,
InputIterator stencil,
Predicate pred);


template<typename ExecutionPolicy,
typename InputIterator,
typename OutputIterator,
typename Predicate>
OutputIterator remove_copy_if(execution_policy<ExecutionPolicy> &exec,
InputIterator first,
InputIterator last,
OutputIterator result,
Predicate pred);


template<typename ExecutionPolicy,
typename InputIterator1,
typename InputIterator2,
typename OutputIterator,
typename Predicate>
OutputIterator remove_copy_if(execution_policy<ExecutionPolicy> &exec,
InputIterator1 first,
InputIterator1 last,
InputIterator2 stencil,
OutputIterator result,
Predicate pred);


} 
} 
} 
} 

#include <hydra/detail/external/hydra_thrust/system/tbb/detail/remove.inl>

