
#ifndef BOOST_ASIO_TRAITS_EXECUTE_FREE_HPP
#define BOOST_ASIO_TRAITS_EXECUTE_FREE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>
#include <boost/asio/detail/type_traits.hpp>

#if defined(BOOST_ASIO_HAS_DECLTYPE) \
&& defined(BOOST_ASIO_HAS_NOEXCEPT) \
&& defined(BOOST_ASIO_HAS_WORKING_EXPRESSION_SFINAE)
# define BOOST_ASIO_HAS_DEDUCED_EXECUTE_FREE_TRAIT 1
#endif 

#include <boost/asio/detail/push_options.hpp>

namespace boost {
namespace asio {
namespace traits {

template <typename T, typename F, typename = void>
struct execute_free_default;

template <typename T, typename F, typename = void>
struct execute_free;

} 
namespace detail {

struct no_execute_free
{
BOOST_ASIO_STATIC_CONSTEXPR(bool, is_valid = false);
BOOST_ASIO_STATIC_CONSTEXPR(bool, is_noexcept = false);
};

#if defined(BOOST_ASIO_HAS_DEDUCED_EXECUTE_FREE_TRAIT)

template <typename T, typename F, typename = void>
struct execute_free_trait : no_execute_free
{
};

template <typename T, typename F>
struct execute_free_trait<T, F,
typename void_type<
decltype(execute(declval<T>(), declval<F>()))
>::type>
{
BOOST_ASIO_STATIC_CONSTEXPR(bool, is_valid = true);

using result_type = decltype(
execute(declval<T>(), declval<F>()));

BOOST_ASIO_STATIC_CONSTEXPR(bool, is_noexcept = noexcept(
execute(declval<T>(), declval<F>())));
};

#else 

template <typename T, typename F, typename = void>
struct execute_free_trait :
conditional<
is_same<T, typename decay<T>::type>::value
&& is_same<F, typename decay<F>::type>::value,
no_execute_free,
traits::execute_free<
typename decay<T>::type,
typename decay<F>::type>
>::type
{
};

#endif 

} 
namespace traits {

template <typename T, typename F, typename>
struct execute_free_default :
detail::execute_free_trait<T, F>
{
};

template <typename T, typename F, typename>
struct execute_free :
execute_free_default<T, F>
{
};

} 
} 
} 

#include <boost/asio/detail/pop_options.hpp>

#endif 
