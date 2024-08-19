
#ifndef BOOST_ASIO_TRAITS_REQUIRE_CONCEPT_FREE_HPP
#define BOOST_ASIO_TRAITS_REQUIRE_CONCEPT_FREE_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include <boost/asio/detail/config.hpp>
#include <boost/asio/detail/type_traits.hpp>

#if defined(BOOST_ASIO_HAS_DECLTYPE) \
&& defined(BOOST_ASIO_HAS_NOEXCEPT) \
&& defined(BOOST_ASIO_HAS_WORKING_EXPRESSION_SFINAE)
# define BOOST_ASIO_HAS_DEDUCED_REQUIRE_CONCEPT_FREE_TRAIT 1
#endif 

#include <boost/asio/detail/push_options.hpp>

namespace boost {
namespace asio {
namespace traits {

template <typename T, typename Property, typename = void>
struct require_concept_free_default;

template <typename T, typename Property, typename = void>
struct require_concept_free;

} 
namespace detail {

struct no_require_concept_free
{
BOOST_ASIO_STATIC_CONSTEXPR(bool, is_valid = false);
BOOST_ASIO_STATIC_CONSTEXPR(bool, is_noexcept = false);
};

#if defined(BOOST_ASIO_HAS_DEDUCED_REQUIRE_CONCEPT_FREE_TRAIT)

template <typename T, typename Property, typename = void>
struct require_concept_free_trait : no_require_concept_free
{
};

template <typename T, typename Property>
struct require_concept_free_trait<T, Property,
typename void_type<
decltype(require_concept(declval<T>(), declval<Property>()))
>::type>
{
BOOST_ASIO_STATIC_CONSTEXPR(bool, is_valid = true);

using result_type = decltype(
require_concept(declval<T>(), declval<Property>()));

BOOST_ASIO_STATIC_CONSTEXPR(bool, is_noexcept = noexcept(
require_concept(declval<T>(), declval<Property>())));
};

#else 

template <typename T, typename Property, typename = void>
struct require_concept_free_trait :
conditional<
is_same<T, typename decay<T>::type>::value
&& is_same<Property, typename decay<Property>::type>::value,
no_require_concept_free,
traits::require_concept_free<
typename decay<T>::type,
typename decay<Property>::type>
>::type
{
};

#endif 

} 
namespace traits {

template <typename T, typename Property, typename>
struct require_concept_free_default :
detail::require_concept_free_trait<T, Property>
{
};

template <typename T, typename Property, typename>
struct require_concept_free :
require_concept_free_default<T, Property>
{
};

} 
} 
} 

#include <boost/asio/detail/pop_options.hpp>

#endif 
