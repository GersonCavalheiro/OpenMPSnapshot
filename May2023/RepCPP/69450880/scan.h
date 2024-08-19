




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

template<typename InputIterator,
typename OutputIterator,
typename BinaryFunction>
OutputIterator inclusive_scan(tag,
InputIterator first,
InputIterator last,
OutputIterator result,
BinaryFunction binary_op);


template<typename InputIterator,
typename OutputIterator,
typename T,
typename BinaryFunction>
OutputIterator exclusive_scan(tag,
InputIterator first,
InputIterator last,
OutputIterator result,
T init,
BinaryFunction binary_op);


} 
} 
} 
} 

#include <hydra/detail/external/hydra_thrust/system/tbb/detail/scan.inl>

