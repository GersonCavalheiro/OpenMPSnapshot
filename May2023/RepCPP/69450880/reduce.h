




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
typename OutputType,
typename BinaryFunction>
OutputType reduce(execution_policy<DerivedPolicy> &exec,
InputIterator begin,
InputIterator end,
OutputType init,
BinaryFunction binary_op);


} 
} 
} 
} 

#include <hydra/detail/external/hydra_thrust/system/tbb/detail/reduce.inl>

