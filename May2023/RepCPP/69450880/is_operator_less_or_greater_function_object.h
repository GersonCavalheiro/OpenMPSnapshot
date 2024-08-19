




#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/functional.h>
#include <hydra/detail/external/hydra_thrust/detail/type_traits.h>
#include <hydra/detail/external/hydra_thrust/detail/type_traits/pointer_traits.h>

HYDRA_THRUST_BEGIN_NS

namespace detail
{

template <typename FunctionObject>
struct is_operator_less_function_object_impl;

template <typename FunctionObject>
struct is_operator_greater_function_object_impl;

} 

template <typename FunctionObject>
#if HYDRA_THRUST_CPP_DIALECT >= 2011
using is_operator_less_function_object =
#else
struct is_operator_less_function_object :
#endif
detail::is_operator_less_function_object_impl<FunctionObject>
#if HYDRA_THRUST_CPP_DIALECT < 2011
{}
#endif
;

#if HYDRA_THRUST_CPP_DIALECT >= 2014
template <typename FunctionObject>
constexpr bool is_operator_less_function_object_v
= is_operator_less_function_object<FunctionObject>::value;
#endif

template <typename FunctionObject>
#if HYDRA_THRUST_CPP_DIALECT >= 2011
using is_operator_greater_function_object =
#else
struct is_operator_greater_function_object :
#endif
detail::is_operator_greater_function_object_impl<FunctionObject>
#if HYDRA_THRUST_CPP_DIALECT < 2011
{}
#endif
;

#if HYDRA_THRUST_CPP_DIALECT >= 2014
template <typename FunctionObject>
constexpr bool is_operator_greater_function_object_v
= is_operator_greater_function_object<FunctionObject>::value;
#endif

template <typename FunctionObject>
#if HYDRA_THRUST_CPP_DIALECT >= 2011
using is_operator_less_or_greater_function_object =
#else
struct is_operator_less_or_greater_function_object :
#endif
integral_constant<
bool 
,    detail::is_operator_less_function_object_impl<FunctionObject>::value
|| detail::is_operator_greater_function_object_impl<FunctionObject>::value
>
#if HYDRA_THRUST_CPP_DIALECT < 2011
{}
#endif
;

#if HYDRA_THRUST_CPP_DIALECT >= 2014
template <typename FunctionObject>
constexpr bool is_operator_less_or_greater_function_object_v
= is_operator_less_or_greater_function_object<FunctionObject>::value;
#endif


namespace detail
{

template <typename FunctionObject>
struct is_operator_less_function_object_impl                   : false_type {};
template <typename T>
struct is_operator_less_function_object_impl<hydra_thrust::less<T> > : true_type {};
template <typename T>
struct is_operator_less_function_object_impl<std::less<T>    > : true_type {};

template <typename FunctionObject>
struct is_operator_greater_function_object_impl                      : false_type {};
template <typename T>
struct is_operator_greater_function_object_impl<hydra_thrust::greater<T> > : true_type {};
template <typename T>
struct is_operator_greater_function_object_impl<std::greater<T>    > : true_type {};

} 

HYDRA_THRUST_END_NS

