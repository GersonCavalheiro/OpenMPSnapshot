

#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/iterator/counting_iterator.h>
#include <hydra/detail/external/hydra_thrust/iterator/iterator_adaptor.h>
#include <hydra/detail/external/hydra_thrust/iterator/detail/any_assign.h>
#include <cstddef> 

namespace hydra_thrust
{

template<typename> class discard_iterator;

namespace detail
{


template<typename System>
struct discard_iterator_base
{
typedef any_assign         value_type;
typedef any_assign&        reference;
typedef std::ptrdiff_t     incrementable;

typedef typename hydra_thrust::counting_iterator<
incrementable,
System,
hydra_thrust::random_access_traversal_tag
> base_iterator;

typedef typename hydra_thrust::iterator_adaptor<
discard_iterator<System>,
base_iterator,
value_type,
typename hydra_thrust::iterator_system<base_iterator>::type,
typename hydra_thrust::iterator_traversal<base_iterator>::type,
reference
> type;
}; 


} 

} 


