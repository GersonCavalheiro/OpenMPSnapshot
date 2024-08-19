

#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/detail/type_traits.h>
#include <hydra/detail/external/hydra_thrust/detail/type_traits/is_metafunction_defined.h>
#include <hydra/detail/external/hydra_thrust/iterator/iterator_traits.h>
#include <hydra/detail/external/hydra_thrust/iterator/detail/any_assign.h>

namespace hydra_thrust
{

namespace detail
{


template<typename T>
struct is_void_like
: hydra_thrust::detail::or_<
hydra_thrust::detail::is_void<T>,
hydra_thrust::detail::is_same<T,hydra_thrust::detail::any_assign>
>
{}; 


template<typename T>
struct lazy_is_void_like
: is_void_like<typename T::type>
{}; 


template<typename T>
struct is_output_iterator
: eval_if<
is_metafunction_defined<hydra_thrust::iterator_value<T> >::value,
lazy_is_void_like<hydra_thrust::iterator_value<T> >,
hydra_thrust::detail::true_type
>::type
{
}; 

} 

} 

