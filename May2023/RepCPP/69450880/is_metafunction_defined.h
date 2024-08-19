

#pragma once

#include <hydra/detail/external/hydra_thrust/detail/type_traits/has_nested_type.h>
#include <hydra/detail/external/hydra_thrust/detail/type_traits.h>

namespace hydra_thrust
{

namespace detail
{

__HYDRA_THRUST_DEFINE_HAS_NESTED_TYPE(is_metafunction_defined, type)

template<typename Metafunction>
struct enable_if_defined
: hydra_thrust::detail::lazy_enable_if<
is_metafunction_defined<Metafunction>::value,
Metafunction
>
{};

} 

} 

