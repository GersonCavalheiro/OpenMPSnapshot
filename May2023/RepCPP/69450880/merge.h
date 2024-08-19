

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

template<typename ExecutionPolicy,
typename InputIterator1,
typename InputIterator2,
typename OutputIterator,
typename StrictWeakOrdering>
OutputIterator merge(execution_policy<ExecutionPolicy> &exec,
InputIterator1 first1,
InputIterator1 last1,
InputIterator2 first2,
InputIterator2 last2,
OutputIterator result,
StrictWeakOrdering comp);

template <typename ExecutionPolicy,
typename InputIterator1,
typename InputIterator2,
typename InputIterator3,
typename InputIterator4,
typename OutputIterator1,
typename OutputIterator2,
typename StrictWeakOrdering>
hydra_thrust::pair<OutputIterator1,OutputIterator2>
merge_by_key(execution_policy<ExecutionPolicy> &exec,
InputIterator1 keys_first1,
InputIterator1 keys_last1,
InputIterator2 keys_first2,
InputIterator2 keys_last2,
InputIterator3 values_first3,
InputIterator4 values_first4,
OutputIterator1 keys_result,
OutputIterator2 values_result,
StrictWeakOrdering comp);

} 
} 
} 
} 

#include <hydra/detail/external/hydra_thrust/system/tbb/detail/merge.inl>

