

#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/system/tbb/detail/execution_policy.h>
#include <hydra/detail/external/hydra_thrust/system/detail/generic/adjacent_difference.h>

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
typename OutputIterator,
typename BinaryFunction>
OutputIterator adjacent_difference(execution_policy<DerivedPolicy> &exec,
InputIterator first,
InputIterator last,
OutputIterator result,
BinaryFunction binary_op)
{
return hydra_thrust::system::detail::generic::adjacent_difference(exec, first, last, result, binary_op);
} 

} 
} 
} 
} 

