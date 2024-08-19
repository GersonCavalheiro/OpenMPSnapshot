

#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/system/detail/generic/find.h>
#include <hydra/detail/external/hydra_thrust/system/tbb/detail/execution_policy.h>

namespace hydra_thrust
{
namespace system
{
namespace tbb
{
namespace detail
{

template <typename DerivedPolicy, typename InputIterator, typename Predicate>
InputIterator find_if(execution_policy<DerivedPolicy> &exec,
InputIterator first,
InputIterator last,
Predicate pred)
{
return hydra_thrust::system::detail::generic::find_if(exec, first, last, pred);
}

} 
} 
} 
} 

