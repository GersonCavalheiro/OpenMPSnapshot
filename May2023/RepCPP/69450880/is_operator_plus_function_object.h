



#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/functional.h>
#include <hydra/detail/external/hydra_thrust/detail/type_traits.h>
#include <hydra/detail/external/hydra_thrust/detail/type_traits/pointer_traits.h>

HYDRA_THRUST_BEGIN_NS

namespace detail
{

template <typename FunctionObject>
struct is_operator_plus_function_object_impl;

} 

template <typename FunctionObject>
#if HYDRA_THRUST_CPP_DIALECT >= 2011
using is_operator_plus_function_object =
#else
struct is_operator_plus_function_object :
#endif
detail::is_operator_plus_function_object_impl<FunctionObject>
#if HYDRA_THRUST_CPP_DIALECT < 2011
{}
#endif
;

#if HYDRA_THRUST_CPP_DIALECT >= 2014
template <typename FunctionObject>
constexpr bool is_operator_plus_function_object_v
= is_operator_plus_function_object<FunctionObject>::value;
#endif


namespace detail
{

template <typename FunctionObject>
struct is_operator_plus_function_object_impl                   : false_type {};
template <typename T>
struct is_operator_plus_function_object_impl<hydra_thrust::plus<T> > : true_type {};
template <typename T>
struct is_operator_plus_function_object_impl<std::plus<T>    > : true_type {};

} 

HYDRA_THRUST_END_NS

