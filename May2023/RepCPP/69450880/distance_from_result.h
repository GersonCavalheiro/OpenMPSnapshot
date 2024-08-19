

#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/detail/type_traits.h>

namespace hydra_thrust
{

namespace detail
{

template<typename IteratorFacade1, typename IteratorFacade2>
struct distance_from_result
: eval_if<
is_convertible<IteratorFacade2,IteratorFacade1>::value,
identity_<typename IteratorFacade1::difference_type>,
identity_<typename IteratorFacade2::difference_type>
>
{};

} 

} 

