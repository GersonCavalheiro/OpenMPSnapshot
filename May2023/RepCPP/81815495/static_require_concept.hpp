
#ifndef ASIO_TRAITS_STATIC_REQUIRE_CONCEPT_HPP
#define ASIO_TRAITS_STATIC_REQUIRE_CONCEPT_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"
#include "asio/detail/type_traits.hpp"
#include "asio/traits/static_query.hpp"

#if defined(ASIO_HAS_DECLTYPE) \
&& defined(ASIO_HAS_NOEXCEPT)
# define ASIO_HAS_DEDUCED_STATIC_REQUIRE_CONCEPT_TRAIT 1
#endif 

#include "asio/detail/push_options.hpp"

namespace asio {
namespace traits {

template <typename T, typename Property, typename = void>
struct static_require_concept_default;

template <typename T, typename Property, typename = void>
struct static_require_concept;

} 
namespace detail {

struct no_static_require_concept
{
ASIO_STATIC_CONSTEXPR(bool, is_valid = false);
};

template <typename T, typename Property, typename = void>
struct static_require_concept_trait :
conditional<
is_same<T, typename decay<T>::type>::value
&& is_same<Property, typename decay<Property>::type>::value,
no_static_require_concept,
traits::static_require_concept<
typename decay<T>::type,
typename decay<Property>::type>
>::type
{
};

#if defined(ASIO_HAS_DEDUCED_STATIC_REQUIRE_CONCEPT_TRAIT)

#if defined(ASIO_HAS_WORKING_EXPRESSION_SFINAE)

template <typename T, typename Property>
struct static_require_concept_trait<T, Property,
typename enable_if<
decay<Property>::type::value() == traits::static_query<T, Property>::value()
>::type>
{
ASIO_STATIC_CONSTEXPR(bool, is_valid = true);
};

#else 

false_type static_require_concept_test(...);

template <typename T, typename Property>
true_type static_require_concept_test(T*, Property*,
typename enable_if<
Property::value() == traits::static_query<T, Property>::value()
>::type* = 0);

template <typename T, typename Property>
struct has_static_require_concept
{
ASIO_STATIC_CONSTEXPR(bool, value =
decltype((static_require_concept_test)(
static_cast<T*>(0), static_cast<Property*>(0)))::value);
};

template <typename T, typename Property>
struct static_require_concept_trait<T, Property,
typename enable_if<
has_static_require_concept<typename decay<T>::type,
typename decay<Property>::type>::value
>::type>
{
ASIO_STATIC_CONSTEXPR(bool, is_valid = true);
};

#endif 

#endif 

} 
namespace traits {

template <typename T, typename Property, typename>
struct static_require_concept_default :
detail::static_require_concept_trait<T, Property>
{
};

template <typename T, typename Property, typename>
struct static_require_concept : static_require_concept_default<T, Property>
{
};

} 
} 

#include "asio/detail/pop_options.hpp"

#endif 