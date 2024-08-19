

#ifndef LBT_TUPLE_UTILITIES
#define LBT_TUPLE_UTILITIES
#pragma once

#include <tuple>
#include <type_traits>


namespace lbt {
namespace detail {
constexpr std::false_type is_tuple_impl(...) noexcept;
template <typename... Ts>
constexpr std::true_type is_tuple_impl(std::tuple<Ts...> const volatile&) noexcept;
}

template <typename T>
using is_tuple = decltype(detail::is_tuple_impl(std::declval<T&>()));
template <typename T>
static constexpr bool is_tuple_v = is_tuple<T>::value;


template <typename T1, typename T2,
typename std::enable_if_t<is_tuple_v<T1>>* = nullptr,
typename std::enable_if_t<is_tuple_v<T2>>* = nullptr>
class CartesianProduct {
protected:
CartesianProduct() = delete;
CartesianProduct(CartesianProduct const&) = delete;
CartesianProduct(CartesianProduct&&) = delete;
CartesianProduct& operator=(CartesianProduct const&) = delete;
CartesianProduct& operator=(CartesianProduct&&) = delete;


template <typename T, typename... Ts>
static constexpr auto innerHelper(T&&, std::tuple<Ts...>&&) noexcept
-> decltype(std::make_tuple(std::make_tuple(std::declval<T>(), std::declval<Ts>()) ...));


template <typename... Ts, typename T,
typename std::enable_if_t<is_tuple_v<T>>* = nullptr>
static constexpr auto outerHelper(std::tuple<Ts...>&&, T&&) noexcept 
-> decltype(std::tuple_cat(innerHelper(std::declval<Ts>(), std::declval<T>()) ...));

public:
using type = decltype(outerHelper(std::declval<T1>(), std::declval<T2>()));
};
template <typename T1, typename T2>
using CartesianProduct_t = typename CartesianProduct<T1, T2>::type;


template <template <typename> class TC, typename TT,
typename std::enable_if_t<is_tuple_v<TT>>* = nullptr>
class CartesianProductApply {
protected:
CartesianProductApply() = delete;
CartesianProductApply(CartesianProductApply const&) = delete;
CartesianProductApply(CartesianProductApply&&) = delete;
CartesianProductApply& operator=(CartesianProductApply const&) = delete;
CartesianProductApply& operator=(CartesianProductApply&&) = delete;


template <typename... Ts>
static constexpr auto helper(std::tuple<Ts...>&&) noexcept
-> decltype(std::tuple_cat(std::declval<TC<Ts>>() ...));

public:
using type = decltype(helper(std::declval<TT>()));
};
template<template <typename> class TC, typename... Ts>
using CartesianProductApply_t = typename CartesianProductApply<TC,Ts...>::type;
}

#endif 
