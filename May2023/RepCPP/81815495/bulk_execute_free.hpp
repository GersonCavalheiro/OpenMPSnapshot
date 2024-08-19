
#ifndef ASIO_TRAITS_BULK_EXECUTE_FREE_HPP
#define ASIO_TRAITS_BULK_EXECUTE_FREE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"
#include "asio/detail/type_traits.hpp"

#if defined(ASIO_HAS_DECLTYPE) \
&& defined(ASIO_HAS_NOEXCEPT) \
&& defined(ASIO_HAS_WORKING_EXPRESSION_SFINAE)
# define ASIO_HAS_DEDUCED_BULK_EXECUTE_FREE_TRAIT 1
#endif 

#include "asio/detail/push_options.hpp"

namespace asio {
namespace traits {

template <typename T, typename F, typename N, typename = void>
struct bulk_execute_free_default;

template <typename T, typename F, typename N, typename = void>
struct bulk_execute_free;

} 
namespace detail {

struct no_bulk_execute_free
{
ASIO_STATIC_CONSTEXPR(bool, is_valid = false);
ASIO_STATIC_CONSTEXPR(bool, is_noexcept = false);
};

#if defined(ASIO_HAS_DEDUCED_BULK_EXECUTE_FREE_TRAIT)

template <typename T, typename F, typename N, typename = void>
struct bulk_execute_free_trait : no_bulk_execute_free
{
};

template <typename T, typename F, typename N>
struct bulk_execute_free_trait<T, F, N,
typename void_type<
decltype(bulk_execute(declval<T>(), declval<F>(), declval<N>()))
>::type>
{
ASIO_STATIC_CONSTEXPR(bool, is_valid = true);

using result_type = decltype(
bulk_execute(declval<T>(), declval<F>(), declval<N>()));

ASIO_STATIC_CONSTEXPR(bool, is_noexcept = noexcept(
bulk_execute(declval<T>(), declval<F>(), declval<N>())));
};

#else 

template <typename T, typename F, typename N, typename = void>
struct bulk_execute_free_trait :
conditional<
is_same<T, typename remove_reference<T>::type>::value
&& is_same<F, typename decay<F>::type>::value
&& is_same<N, typename decay<N>::type>::value,
typename conditional<
is_same<T, typename add_const<T>::type>::value,
no_bulk_execute_free,
traits::bulk_execute_free<typename add_const<T>::type, F, N>
>::type,
traits::bulk_execute_free<
typename remove_reference<T>::type,
typename decay<F>::type,
typename decay<N>::type>
>::type
{
};

#endif 

} 
namespace traits {

template <typename T, typename F, typename N, typename>
struct bulk_execute_free_default :
detail::bulk_execute_free_trait<T, F, N>
{
};

template <typename T, typename F, typename N, typename>
struct bulk_execute_free :
bulk_execute_free_default<T, F, N>
{
};

} 
} 

#include "asio/detail/pop_options.hpp"

#endif 
