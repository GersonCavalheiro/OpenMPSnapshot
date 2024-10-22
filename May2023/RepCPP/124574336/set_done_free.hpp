
#ifndef BOOST_ASIO_TRAITS_SET_DONE_FREE_HPP
#define BOOST_ASIO_TRAITS_SET_DONE_FREE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>
#include <boost/asio/detail/type_traits.hpp>

#if defined(BOOST_ASIO_HAS_DECLTYPE) \
&& defined(BOOST_ASIO_HAS_NOEXCEPT) \
&& defined(BOOST_ASIO_HAS_WORKING_EXPRESSION_SFINAE)
# define BOOST_ASIO_HAS_DEDUCED_SET_DONE_FREE_TRAIT 1
#endif 

#include <boost/asio/detail/push_options.hpp>

namespace boost {
namespace asio {
namespace traits {

template <typename T, typename = void>
struct set_done_free_default;

template <typename T, typename = void>
struct set_done_free;

} 
namespace detail {

struct no_set_done_free
{
BOOST_ASIO_STATIC_CONSTEXPR(bool, is_valid = false);
BOOST_ASIO_STATIC_CONSTEXPR(bool, is_noexcept = false);
};

#if defined(BOOST_ASIO_HAS_DEDUCED_SET_DONE_FREE_TRAIT)

template <typename T, typename = void>
struct set_done_free_trait : no_set_done_free
{
};

template <typename T>
struct set_done_free_trait<T,
typename void_type<
decltype(set_done(declval<T>()))
>::type>
{
BOOST_ASIO_STATIC_CONSTEXPR(bool, is_valid = true);

using result_type = decltype(set_done(declval<T>()));

BOOST_ASIO_STATIC_CONSTEXPR(bool,
is_noexcept = noexcept(set_done(declval<T>())));
};

#else 

template <typename T, typename = void>
struct set_done_free_trait :
conditional<
is_same<T, typename remove_reference<T>::type>::value,
typename conditional<
is_same<T, typename add_const<T>::type>::value,
no_set_done_free,
traits::set_done_free<typename add_const<T>::type>
>::type,
traits::set_done_free<typename remove_reference<T>::type>
>::type
{
};

#endif 

} 
namespace traits {

template <typename T, typename>
struct set_done_free_default :
detail::set_done_free_trait<T>
{
};

template <typename T, typename>
struct set_done_free :
set_done_free_default<T>
{
};

} 
} 
} 

#include <boost/asio/detail/pop_options.hpp>

#endif 
