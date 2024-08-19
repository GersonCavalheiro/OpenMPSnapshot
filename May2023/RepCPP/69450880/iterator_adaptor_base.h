

#pragma once

#include <hydra/detail/external/hydra_thrust/detail/type_traits.h>
#include <hydra/detail/external/hydra_thrust/iterator/iterator_traits.h>
#include <hydra/detail/external/hydra_thrust/detail/use_default.h>
#include <hydra/detail/external/hydra_thrust/iterator/iterator_facade.h>

namespace hydra_thrust
{


template<typename Derived,
typename Base,
typename Value,
typename System,
typename Traversal,
typename Reference,
typename Difference
>
class iterator_adaptor;


namespace detail
{

template <class T, class DefaultNullaryFn>
struct ia_dflt_help
: hydra_thrust::detail::eval_if<
hydra_thrust::detail::is_same<T, hydra_thrust::use_default>::value
, DefaultNullaryFn
, hydra_thrust::detail::identity_<T>
>
{
}; 


template<typename Derived,
typename Base,
typename Value,
typename System,
typename Traversal,
typename Reference,
typename Difference
>
struct iterator_adaptor_base
{
typedef typename ia_dflt_help<
Value,
iterator_value<Base>
>::type value;

typedef typename ia_dflt_help<
System,
hydra_thrust::iterator_system<Base>
>::type system;

typedef typename ia_dflt_help<
Traversal,
hydra_thrust::iterator_traversal<Base>
>::type traversal;

typedef typename ia_dflt_help<
Reference,
hydra_thrust::detail::eval_if<
hydra_thrust::detail::is_same<Value,use_default>::value,
hydra_thrust::iterator_reference<Base>,
hydra_thrust::detail::add_reference<Value>
>
>::type reference;

typedef typename ia_dflt_help<
Difference,
iterator_difference<Base>
>::type difference;

typedef hydra_thrust::iterator_facade<
Derived,
value,
system,
traversal,
reference,
difference
> type;
}; 


} 
} 

