

#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>

#include <hydra/detail/external/hydra_thrust/detail/type_traits.h>
#include <hydra/detail/external/hydra_thrust/detail/execution_policy.h>

HYDRA_THRUST_BEGIN_NS

template <typename T>
#if HYDRA_THRUST_CPP_DIALECT >= 2011
using is_execution_policy =
#else
struct is_execution_policy :
#endif
detail::is_base_of<detail::execution_policy_marker, T>
#if HYDRA_THRUST_CPP_DIALECT < 2011
{}
#endif
;

#if HYDRA_THRUST_CPP_DIALECT >= 2014
template <typename T>
constexpr bool is_execution_policy_v = is_execution_policy<T>::value;
#endif

HYDRA_THRUST_END_NS


