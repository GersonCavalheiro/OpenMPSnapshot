

#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/system/tbb/detail/copy.h>
#include <hydra/detail/external/hydra_thrust/system/detail/generic/copy.h>
#include <hydra/detail/external/hydra_thrust/system/detail/sequential/copy.h>
#include <hydra/detail/external/hydra_thrust/detail/type_traits/minimum_type.h>
#include <hydra/detail/external/hydra_thrust/detail/copy.h>

namespace hydra_thrust
{
namespace system
{
namespace tbb
{
namespace detail
{
namespace dispatch
{


template<typename DerivedPolicy,
typename InputIterator,
typename OutputIterator>
OutputIterator copy(execution_policy<DerivedPolicy> &exec,
InputIterator first,
InputIterator last,
OutputIterator result,
hydra_thrust::incrementable_traversal_tag)
{
return hydra_thrust::system::detail::sequential::copy(exec, first, last, result);
} 


template<typename DerivedPolicy,
typename InputIterator,
typename OutputIterator>
OutputIterator copy(execution_policy<DerivedPolicy> &exec,
InputIterator first,
InputIterator last,
OutputIterator result,
hydra_thrust::random_access_traversal_tag)
{
return hydra_thrust::system::detail::generic::copy(exec, first, last, result);
} 


template<typename DerivedPolicy,
typename InputIterator,
typename Size,
typename OutputIterator>
OutputIterator copy_n(execution_policy<DerivedPolicy> &exec,
InputIterator first,
Size n,
OutputIterator result,
hydra_thrust::incrementable_traversal_tag)
{
return hydra_thrust::system::detail::sequential::copy_n(exec, first, n, result);
} 


template<typename DerivedPolicy,
typename InputIterator,
typename Size,
typename OutputIterator>
OutputIterator copy_n(execution_policy<DerivedPolicy> &exec,
InputIterator first,
Size n,
OutputIterator result,
hydra_thrust::random_access_traversal_tag)
{
return hydra_thrust::system::detail::generic::copy_n(exec, first, n, result);
} 


} 


template<typename DerivedPolicy,
typename InputIterator,
typename OutputIterator>
OutputIterator copy(execution_policy<DerivedPolicy> &exec,
InputIterator first,
InputIterator last,
OutputIterator result)
{
typedef typename hydra_thrust::iterator_traversal<InputIterator>::type  traversal1;
typedef typename hydra_thrust::iterator_traversal<OutputIterator>::type traversal2;

typedef typename hydra_thrust::detail::minimum_type<traversal1,traversal2>::type traversal;

return hydra_thrust::system::tbb::detail::dispatch::copy(exec,first,last,result,traversal());
} 



template<typename DerivedPolicy,
typename InputIterator,
typename Size,
typename OutputIterator>
OutputIterator copy_n(execution_policy<DerivedPolicy> &exec,
InputIterator first,
Size n,
OutputIterator result)
{
typedef typename hydra_thrust::iterator_traversal<InputIterator>::type  traversal1;
typedef typename hydra_thrust::iterator_traversal<OutputIterator>::type traversal2;

typedef typename hydra_thrust::detail::minimum_type<traversal1,traversal2>::type traversal;

return hydra_thrust::system::tbb::detail::dispatch::copy_n(exec,first,n,result,traversal());
} 


} 
} 
} 
} 

