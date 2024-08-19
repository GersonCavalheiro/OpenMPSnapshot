

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


template<typename ExecutionPolicy,
typename ForwardIterator,
typename BinaryPredicate>
ForwardIterator unique(execution_policy<ExecutionPolicy> &exec,
ForwardIterator first,
ForwardIterator last,
BinaryPredicate binary_pred);


template<typename ExecutionPolicy,
typename InputIterator,
typename OutputIterator,
typename BinaryPredicate>
OutputIterator unique_copy(execution_policy<ExecutionPolicy> &exec,
InputIterator first,
InputIterator last,
OutputIterator output,
BinaryPredicate binary_pred);


} 
} 
} 
} 

#include <hydra/detail/external/hydra_thrust/system/tbb/detail/unique.inl>

