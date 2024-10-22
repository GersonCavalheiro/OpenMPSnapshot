
#ifndef BOOST_ASIO_TRAITS_CONNECT_FREE_HPP
#define BOOST_ASIO_TRAITS_CONNECT_FREE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>
#include <boost/asio/detail/type_traits.hpp>

#if defined(BOOST_ASIO_HAS_DECLTYPE) \
&& defined(BOOST_ASIO_HAS_NOEXCEPT) \
&& defined(BOOST_ASIO_HAS_WORKING_EXPRESSION_SFINAE)
# define BOOST_ASIO_HAS_DEDUCED_CONNECT_FREE_TRAIT 1
#endif 

#include <boost/asio/detail/push_options.hpp>

namespace boost {
namespace asio {
namespace traits {

template <typename S, typename R, typename = void>
struct connect_free_default;

template <typename S, typename R, typename = void>
struct connect_free;

} 
namespace detail {

struct no_connect_free
{
BOOST_ASIO_STATIC_CONSTEXPR(bool, is_valid = false);
BOOST_ASIO_STATIC_CONSTEXPR(bool, is_noexcept = false);
};

#if defined(BOOST_ASIO_HAS_DEDUCED_CONNECT_FREE_TRAIT)

template <typename S, typename R, typename = void>
struct connect_free_trait : no_connect_free
{
};

template <typename S, typename R>
struct connect_free_trait<S, R,
typename void_type<
decltype(connect(declval<S>(), declval<R>()))
>::type>
{
BOOST_ASIO_STATIC_CONSTEXPR(bool, is_valid = true);

using result_type = decltype(
connect(declval<S>(), declval<R>()));

BOOST_ASIO_STATIC_CONSTEXPR(bool, is_noexcept = noexcept(
connect(declval<S>(), declval<R>())));
};

#else 

template <typename S, typename R, typename = void>
struct connect_free_trait :
conditional<
is_same<S, typename remove_reference<S>::type>::value
&& is_same<R, typename decay<R>::type>::value,
typename conditional<
is_same<S, typename add_const<S>::type>::value,
no_connect_free,
traits::connect_free<typename add_const<S>::type, R>
>::type,
traits::connect_free<
typename remove_reference<S>::type,
typename decay<R>::type>
>::type
{
};

#endif 

} 
namespace traits {

template <typename S, typename R, typename>
struct connect_free_default :
detail::connect_free_trait<S, R>
{
};

template <typename S, typename R, typename>
struct connect_free :
connect_free_default<S, R>
{
};

} 
} 
} 

#include <boost/asio/detail/pop_options.hpp>

#endif 
