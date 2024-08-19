

#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/system/tbb/detail/remove.h>
#include <hydra/detail/external/hydra_thrust/system/detail/generic/remove.h>

namespace hydra_thrust
{
namespace system
{
namespace tbb
{
namespace detail
{

template<typename DerivedPolicy,
typename ForwardIterator,
typename Predicate>
ForwardIterator remove_if(execution_policy<DerivedPolicy> &exec,
ForwardIterator first,
ForwardIterator last,
Predicate pred)
{
return hydra_thrust::system::detail::generic::remove_if(exec, first, last, pred);
}


template<typename DerivedPolicy,
typename ForwardIterator,
typename InputIterator,
typename Predicate>
ForwardIterator remove_if(execution_policy<DerivedPolicy> &exec,
ForwardIterator first,
ForwardIterator last,
InputIterator stencil,
Predicate pred)
{
return hydra_thrust::system::detail::generic::remove_if(exec, first, last, stencil, pred);
}


template<typename DerivedPolicy,
typename InputIterator,
typename OutputIterator,
typename Predicate>
OutputIterator remove_copy_if(execution_policy<DerivedPolicy> &exec,
InputIterator first,
InputIterator last,
OutputIterator result,
Predicate pred)
{
return hydra_thrust::system::detail::generic::remove_copy_if(exec, first, last, result, pred);
}

template<typename DerivedPolicy,
typename InputIterator1,
typename InputIterator2,
typename OutputIterator,
typename Predicate>
OutputIterator remove_copy_if(execution_policy<DerivedPolicy> &exec,
InputIterator1 first,
InputIterator1 last,
InputIterator2 stencil,
OutputIterator result,
Predicate pred)
{
return hydra_thrust::system::detail::generic::remove_copy_if(exec, first, last, stencil, result, pred);
}

} 
} 
} 
} 

