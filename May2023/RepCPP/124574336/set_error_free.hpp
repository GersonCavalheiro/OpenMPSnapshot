
#ifndef BOOST_ASIO_TRAITS_SET_ERROR_FREE_HPP
#define BOOST_ASIO_TRAITS_SET_ERROR_FREE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>
#include <boost/asio/detail/type_traits.hpp>

#if defined(BOOST_ASIO_HAS_DECLTYPE) \
&& defined(BOOST_ASIO_HAS_NOEXCEPT) \
&& defined(BOOST_ASIO_HAS_WORKING_EXPRESSION_SFINAE)
# define BOOST_ASIO_HAS_DEDUCED_SET_ERROR_FREE_TRAIT 1
#endif 

#include <boost/asio/detail/push_options.hpp>

namespace boost {
namespace asio {
namespace traits {

template <typename T, typename E, typename = void>
struct set_error_free_default;

template <typename T, typename E, typename = void>
struct set_error_free;

} 
namespace detail {

struct no_set_error_free
{
BOOST_ASIO_STATIC_CONSTEXPR(bool, is_valid = false);
BOOST_ASIO_STATIC_CONSTEXPR(bool, is_noexcept = false);
};

#if defined(BOOST_ASIO_HAS_DEDUCED_SET_ERROR_FREE_TRAIT)

template <typename T, typename E, typename = void>
struct set_error_free_trait : no_set_error_free
{
};

template <typename T, typename E>
struct set_error_free_trait<T, E,
typename void_type<
decltype(set_error(declval<T>(), declval<E>()))
>::type>
{
BOOST_ASIO_STATIC_CONSTEXPR(bool, is_valid = true);

using result_type = decltype(
set_error(declval<T>(), declval<E>()));

BOOST_ASIO_STATIC_CONSTEXPR(bool, is_noexcept = noexcept(
set_error(declval<T>(), declval<E>())));
};

#else 

template <typename T, typename E, typename = void>
struct set_error_free_trait :
conditional<
is_same<T, typename remove_reference<T>::type>::value
&& is_same<E, typename decay<E>::type>::value,
typename conditional<
is_same<T, typename add_const<T>::type>::value,
no_set_error_free,
traits::set_error_free<typename add_const<T>::type, E>
>::type,
traits::set_error_free<
typename remove_reference<T>::type,
typename decay<E>::type>
>::type
{
};

#endif 

} 
namespace traits {

template <typename T, typename E, typename>
struct set_error_free_default :
detail::set_error_free_trait<T, E>
{
};

template <typename T, typename E, typename>
struct set_error_free :
set_error_free_default<T, E>
{
};

} 
} 
} 

#include <boost/asio/detail/pop_options.hpp>

#endif 
