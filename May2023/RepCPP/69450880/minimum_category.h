

#pragma once

#include <hydra/detail/external/hydra_thrust/detail/type_traits/minimum_type.h>

namespace hydra_thrust
{

namespace detail
{ 

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
struct minimum_category
: minimum_type<T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15,T16>
{
}; 

} 

} 


