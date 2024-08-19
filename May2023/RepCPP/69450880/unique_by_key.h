

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
typename ForwardIterator1,
typename ForwardIterator2,
typename BinaryPredicate>
hydra_thrust::pair<ForwardIterator1,ForwardIterator2>
unique_by_key(execution_policy<DerivedPolicy> &exec,
ForwardIterator1 keys_first, 
ForwardIterator1 keys_last,
ForwardIterator2 values_first,
BinaryPredicate binary_pred);


template<typename DerivedPolicy,
typename InputIterator1,
typename InputIterator2,
typename OutputIterator1,
typename OutputIterator2,
typename BinaryPredicate>
hydra_thrust::pair<OutputIterator1,OutputIterator2>
unique_by_key_copy(execution_policy<DerivedPolicy> &exec,
InputIterator1 keys_first, 
InputIterator1 keys_last,
InputIterator2 values_first,
OutputIterator1 keys_output,
OutputIterator2 values_output,
BinaryPredicate binary_pred);


} 
} 
} 
} 

#include <hydra/detail/external/hydra_thrust/system/tbb/detail/unique_by_key.inl>

