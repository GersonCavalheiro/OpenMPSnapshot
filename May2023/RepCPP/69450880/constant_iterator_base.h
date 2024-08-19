

#pragma once

#include <hydra/detail/external/hydra_thrust/iterator/counting_iterator.h>
#include <hydra/detail/external/hydra_thrust/iterator/iterator_adaptor.h>

namespace hydra_thrust
{

template<typename,typename,typename> class constant_iterator;

namespace detail
{

template<typename Value,
typename Incrementable,
typename System>
struct constant_iterator_base
{
typedef Value              value_type;

typedef value_type         reference;

typedef typename hydra_thrust::detail::ia_dflt_help<
Incrementable,
hydra_thrust::detail::identity_<hydra_thrust::detail::intmax_t>
>::type incrementable;

typedef typename hydra_thrust::counting_iterator<
incrementable,
System,
hydra_thrust::random_access_traversal_tag
> base_iterator;

typedef typename hydra_thrust::iterator_adaptor<
constant_iterator<Value, Incrementable, System>,
base_iterator,
value_type, 
typename hydra_thrust::iterator_system<base_iterator>::type,
typename hydra_thrust::iterator_traversal<base_iterator>::type,
reference
> type;
}; 

} 

} 

