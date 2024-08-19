

#pragma once

#include <hydra/detail/external/hydra_thrust/iterator/iterator_adaptor.h>
#include <hydra/detail/external/hydra_thrust/iterator/iterator_traits.h>
#include <hydra/detail/external/hydra_thrust/detail/type_traits.h>
#include <hydra/detail/external/hydra_thrust/iterator/detail/minimum_system.h>

namespace hydra_thrust
{

template<typename,typename> class permutation_iterator;


namespace detail
{

template<typename ElementIterator,
typename IndexIterator>
struct permutation_iterator_base
{
typedef typename hydra_thrust::iterator_system<ElementIterator>::type System1;
typedef typename hydra_thrust::iterator_system<IndexIterator>::type System2;

typedef hydra_thrust::iterator_adaptor<
permutation_iterator<ElementIterator,IndexIterator>,
IndexIterator,
typename hydra_thrust::iterator_value<ElementIterator>::type,
typename detail::minimum_system<System1,System2>::type,
hydra_thrust::use_default,
typename hydra_thrust::iterator_reference<ElementIterator>::type
> type;
}; 

} 

} 

