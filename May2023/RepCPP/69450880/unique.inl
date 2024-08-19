

#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/system/tbb/detail/unique.h>
#include <hydra/detail/external/hydra_thrust/system/detail/generic/unique.h>
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
typename ForwardIterator,
typename BinaryPredicate>
ForwardIterator unique(execution_policy<DerivedPolicy> &exec,
ForwardIterator first,
ForwardIterator last,
BinaryPredicate binary_pred)
{
return hydra_thrust::system::detail::generic::unique(exec,first,last,binary_pred);
} 


template<typename DerivedPolicy,
typename InputIterator,
typename OutputIterator,
typename BinaryPredicate>
OutputIterator unique_copy(execution_policy<DerivedPolicy> &exec,
InputIterator first,
InputIterator last,
OutputIterator output,
BinaryPredicate binary_pred)
{
return hydra_thrust::system::detail::generic::unique_copy(exec,first,last,output,binary_pred);
} 


} 
} 
} 
} 

