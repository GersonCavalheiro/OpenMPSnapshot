

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
typename InputIterator1,
typename InputIterator2,
typename OutputIterator1,
typename OutputIterator2,
typename BinaryPredicate,
typename BinaryFunction>
hydra_thrust::pair<OutputIterator1,OutputIterator2>
reduce_by_key(execution_policy<DerivedPolicy> &exec,
InputIterator1 keys_first, 
InputIterator1 keys_last,
InputIterator2 values_first,
OutputIterator1 keys_output,
OutputIterator2 values_output,
BinaryPredicate binary_pred,
BinaryFunction binary_op);


} 
} 
} 
} 

#include <hydra/detail/external/hydra_thrust/system/tbb/detail/reduce_by_key.inl>

