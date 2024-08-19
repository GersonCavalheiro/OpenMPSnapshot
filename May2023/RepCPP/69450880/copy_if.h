

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


template<typename InputIterator1,
typename InputIterator2,
typename OutputIterator,
typename Predicate>
OutputIterator copy_if(tag,
InputIterator1 first,
InputIterator1 last,
InputIterator2 stencil,
OutputIterator result,
Predicate pred);


} 
} 
} 
} 

#include <hydra/detail/external/hydra_thrust/system/tbb/detail/copy_if.inl>

