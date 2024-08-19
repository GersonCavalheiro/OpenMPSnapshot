#pragma once

#include <type_traits>
#include <tuple>
#include "execution_policy.h"
#include "scalar_categories.h"
#include "tensor_traits.h"

namespace dg{
namespace detail{
template<template <typename> class Predicate, unsigned n, class Default, class... Ts>
struct find_if_impl;

template<template <typename> class Predicate, unsigned n, class Default, class T>
struct find_if_impl<Predicate, n, Default, T>
{
using type = std::conditional_t< Predicate<T>::value, T, Default>;
static constexpr unsigned value = Predicate<T>::value ? n : n+1;
};

template<template <typename> class Predicate, unsigned n, class Default, class T, class... Ts>
struct find_if_impl<Predicate, n, Default, T, Ts...>
{
using type = std::conditional_t< Predicate<T>::value, T, typename find_if_impl<Predicate, n+1, Default, Ts...>::type>;
static constexpr unsigned value = Predicate<T>::value ? n : find_if_impl<Predicate, n+1, Default, Ts...>::value;
};
}

template<size_t index, typename T, typename... Ts>
inline std::enable_if_t<index==0, T>
get_idx(T&& t, Ts&&... ts) {
return std::forward<T>(t);
}

template<size_t index, typename T, typename... Ts>
inline std::enable_if_t<(index > 0) && index <= sizeof...(Ts),
std::tuple_element_t<index, std::tuple<T, Ts...>>>
get_idx(T&& t, Ts&&... ts) {
return get_idx<index-1>(std::forward<Ts>(ts)...);
}

template<template <typename> class Predicate, class Default, class T, class... Ts>
using find_if_t = typename detail::find_if_impl<Predicate,0, Default, T, Ts...>::type;
template<template <typename> class Predicate, class Default, class T, class... Ts>
using find_if_v = std::integral_constant<unsigned, detail::find_if_impl<Predicate,0, Default, T, Ts...>::value>;

template< class T>
using is_scalar = std::conditional_t< std::is_base_of<AnyScalarTag, get_tensor_category<T>>::value, std::true_type, std::false_type>;
template< class T>
using is_not_scalar = std::conditional_t< !std::is_base_of<AnyScalarTag, get_tensor_category<T>>::value, std::true_type, std::false_type>;
template< class T>
using is_vector = std::conditional_t< std::is_base_of<AnyVectorTag, get_tensor_category<T>>::value, std::true_type, std::false_type>;
template< class T>
using is_matrix = std::conditional_t< std::is_base_of<AnyMatrixTag, get_tensor_category<T>>::value, std::true_type, std::false_type>;

template< class T>
using is_tensor = std::conditional_t< std::is_same<NotATensorTag, get_tensor_category<T>>::value, std::false_type, std::true_type>;

namespace detail
{
template<class Category>
using find_base_category = std::conditional_t< std::is_base_of<SharedVectorTag, Category>::value, SharedVectorTag,
std::conditional_t< std::is_base_of<RecursiveVectorTag, Category>::value, RecursiveVectorTag, MPIVectorTag>>;
}
template<class T, class Category>
using is_scalar_or_same_base_category = std::conditional_t< std::is_base_of<detail::find_base_category<Category>, get_tensor_category<T>>::value || is_scalar<T>::value , std::true_type, std::false_type>;


template< class T>
using has_any_policy = std::conditional_t< std::is_same<AnyPolicyTag, get_execution_policy<T>>::value, std::true_type, std::false_type>;
template< class T>
using has_not_any_policy = std::conditional_t< !std::is_same<AnyPolicyTag, get_execution_policy<T>>::value, std::true_type, std::false_type>;
template<class U, class Policy>
using has_any_or_same_policy = std::conditional_t< std::is_same<get_execution_policy<U>, Policy>::value || has_any_policy<U>::value, std::true_type, std::false_type>;
template< class T>
using is_not_scalar_has_not_any_policy = std::conditional_t< !is_scalar<T>::value && !has_any_policy<T>::value, std::true_type, std::false_type>;

template < bool...> struct bool_pack;

template<bool... v>
using all_true = std::is_same<bool_pack<true,v...>, bool_pack<v..., true>>;

template< typename ...> struct WhichType; 


}
