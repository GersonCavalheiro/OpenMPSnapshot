
#ifndef ASIO_IS_APPLICABLE_PROPERTY_HPP
#define ASIO_IS_APPLICABLE_PROPERTY_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif 

#include "asio/detail/config.hpp"
#include "asio/detail/type_traits.hpp"

namespace asio {
namespace detail {

template <typename T, typename Property, typename = void>
struct is_applicable_property_trait : false_type
{
};

#if defined(ASIO_HAS_VARIABLE_TEMPLATES)

template <typename T, typename Property>
struct is_applicable_property_trait<T, Property,
typename void_type<
typename enable_if<
!!Property::template is_applicable_property_v<T>
>::type
>::type> : true_type
{
};

#endif 

} 

template <typename T, typename Property, typename = void>
struct is_applicable_property :
detail::is_applicable_property_trait<T, Property>
{
};

#if defined(ASIO_HAS_VARIABLE_TEMPLATES)

template <typename T, typename Property>
ASIO_CONSTEXPR const bool is_applicable_property_v
= is_applicable_property<T, Property>::value;

#endif 

} 

#endif 
