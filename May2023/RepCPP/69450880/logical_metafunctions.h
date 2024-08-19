


#pragma once

#include <hydra/detail/external/hydra_thrust/detail/config.h>
#include <hydra/detail/external/hydra_thrust/detail/cpp11_required.h>

#if HYDRA_THRUST_CPP_DIALECT >= 2011

#include <type_traits>

HYDRA_THRUST_BEGIN_NS

#if HYDRA_THRUST_CPP_DIALECT >= 2017

template <typename... Ts>
using conjunction = std::conjunction<Ts...>;

template <typename... Ts>
constexpr bool conjunction_v = conjunction<Ts...>::value;

template <typename... Ts>
using disjunction = std::disjunction<Ts...>;

template <typename... Ts>
constexpr bool disjunction_v = disjunction<Ts...>::value;

template <typename T>
using negation = std::negation<T>;

template <typename T>
constexpr bool negation_v = negation<T>::value;


#else 

template <typename... Ts>
struct conjunction;

#if HYDRA_THRUST_CPP_DIALECT >= 2014
template <typename... Ts>
constexpr bool conjunction_v = conjunction<Ts...>::value;
#endif

template <>
struct conjunction<> : std::true_type {};

template <typename T>
struct conjunction<T> : T {};

template <typename T0, typename T1>
struct conjunction<T0, T1> : std::conditional<T0::value, T1, T0>::type {};

template<typename T0, typename T1, typename T2, typename... TN>
struct conjunction<T0, T1, T2, TN...>
: std::conditional<T0::value, conjunction<T1, T2, TN...>, T0>::type {};


template <typename... Ts>
struct disjunction;

#if HYDRA_THRUST_CPP_DIALECT >= 2014
template <typename... Ts>
constexpr bool disjunction_v = disjunction<Ts...>::value;
#endif

template <>
struct disjunction<> : std::false_type {};

template <typename T>
struct disjunction<T> : T {};

template <typename T0, typename... TN>
struct disjunction<T0, TN...>
: std::conditional<T0::value != false, T0, disjunction<TN...> >::type {};


template <typename T>
struct negation;

#if HYDRA_THRUST_CPP_DIALECT >= 2014
template <typename T>
constexpr bool negation_v = negation<T>::value;
#endif

template <typename T>
struct negation : std::integral_constant<bool, !T::value> {};

#endif 


template <bool... Bs>
struct conjunction_value;

#if HYDRA_THRUST_CPP_DIALECT >= 2014
template <bool... Bs>
constexpr bool conjunction_value_v = conjunction_value<Bs...>::value;
#endif

template <>
struct conjunction_value<> : std::true_type {};

template <bool B>
struct conjunction_value<B> : std::integral_constant<bool, B> {};

template <bool B0, bool... BN>
struct conjunction_value<B0, BN...>
: std::integral_constant<bool, B0 && conjunction_value<BN...>::value> {};


template <bool... Bs>
struct disjunction_value;

#if HYDRA_THRUST_CPP_DIALECT >= 2014
template <bool... Bs>
constexpr bool disjunction_value_v = disjunction_value<Bs...>::value;
#endif

template <>
struct disjunction_value<> : std::false_type {};

template <bool B>
struct disjunction_value<B> : std::integral_constant<bool, B> {};

template <bool B0, bool... BN>
struct disjunction_value<B0, BN...>
: std::integral_constant<bool, B0 || disjunction_value<BN...>::value> {};


template <bool B>
struct negation_value;

#if HYDRA_THRUST_CPP_DIALECT >= 2014
template <bool B>
constexpr bool negation_value_v = negation_value<B>::value;
#endif

template <bool B>
struct negation_value : std::integral_constant<bool, !B> {};

HYDRA_THRUST_END_NS

#endif 

