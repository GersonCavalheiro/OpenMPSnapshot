

#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/detail/type_traits.h>
#include <hydra/detail/external/hydra_thrust/detail/type_traits/is_metafunction_defined.h>
#include <hydra/detail/external/hydra_thrust/detail/type_traits/minimum_type.h>

namespace hydra_thrust
{
namespace detail
{ 


template<typename T1,
typename T2  = void,
typename T3  = void,
typename T4  = void,
typename T5  = void,
typename T6  = void,
typename T7  = void,
typename T8  = void,
typename T9  = void,
typename T10 = void,
typename T11 = void,
typename T12 = void,
typename T13 = void,
typename T14 = void,
typename T15 = void,
typename T16 = void>
struct unrelated_systems {};


template<typename T1,
typename T2  = minimum_type_detail::any_conversion,
typename T3  = minimum_type_detail::any_conversion,
typename T4  = minimum_type_detail::any_conversion,
typename T5  = minimum_type_detail::any_conversion,
typename T6  = minimum_type_detail::any_conversion,
typename T7  = minimum_type_detail::any_conversion,
typename T8  = minimum_type_detail::any_conversion,
typename T9  = minimum_type_detail::any_conversion,
typename T10 = minimum_type_detail::any_conversion,
typename T11 = minimum_type_detail::any_conversion,
typename T12 = minimum_type_detail::any_conversion,
typename T13 = minimum_type_detail::any_conversion,
typename T14 = minimum_type_detail::any_conversion,
typename T15 = minimum_type_detail::any_conversion,
typename T16 = minimum_type_detail::any_conversion>
struct minimum_system
: hydra_thrust::detail::eval_if<
is_metafunction_defined<
minimum_type<T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15,T16>
>::value,
minimum_type<T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15,T16>,
hydra_thrust::detail::identity_<
unrelated_systems<T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15,T16>
>
>
{}; 


} 
} 

