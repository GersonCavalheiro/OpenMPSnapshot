
#ifndef BOOST_ASIO_TRAITS_START_FREE_HPP
#define BOOST_ASIO_TRAITS_START_FREE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>
#include <boost/asio/detail/type_traits.hpp>

#if defined(BOOST_ASIO_HAS_DECLTYPE) \
&& defined(BOOST_ASIO_HAS_NOEXCEPT) \
&& defined(BOOST_ASIO_HAS_WORKING_EXPRESSION_SFINAE)
# define BOOST_ASIO_HAS_DEDUCED_START_FREE_TRAIT 1
#endif 

#include <boost/asio/detail/push_options.hpp>

namespace boost {
namespace asio {
namespace traits {

template <typename T, typename = void>
struct start_free_default;

template <typename T, typename = void>
struct start_free;

} 
namespace detail {

struct no_start_free
{
BOOST_ASIO_STATIC_CONSTEXPR(bool, is_valid = false);
BOOST_ASIO_STATIC_CONSTEXPR(bool, is_noexcept = false);
};

#if defined(BOOST_ASIO_HAS_DEDUCED_START_FREE_TRAIT)

template <typename T, typename = void>
struct start_free_trait : no_start_free
{
};

template <typename T>
struct start_free_trait<T,
typename void_type<
decltype(start(declval<T>()))
>::type>
{
BOOST_ASIO_STATIC_CONSTEXPR(bool, is_valid = true);

using result_type = decltype(start(declval<T>()));

BOOST_ASIO_STATIC_CONSTEXPR(bool,
is_noexcept = noexcept(start(declval<T>())));
};

#else 

template <typename T, typename = void>
struct start_free_trait :
conditional<
is_same<T, typename remove_reference<T>::type>::value,
typename conditional<
is_same<T, typename add_const<T>::type>::value,
no_start_free,
traits::start_free<typename add_const<T>::type>
>::type,
traits::start_free<typename remove_reference<T>::type>
>::type
{
};

#endif 

} 
namespace traits {

template <typename T, typename>
struct start_free_default :
detail::start_free_trait<T>
{
};

template <typename T, typename>
struct start_free :
start_free_default<T>
{
};

} 
} 
} 

#include <boost/asio/detail/pop_options.hpp>

#endif 
