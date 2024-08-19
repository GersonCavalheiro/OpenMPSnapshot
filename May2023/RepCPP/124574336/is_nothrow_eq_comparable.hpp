

#ifndef BOOST_POLY_COLLECTION_DETAIL_IS_NOTHROW_EQ_COMPARABLE_HPP
#define BOOST_POLY_COLLECTION_DETAIL_IS_NOTHROW_EQ_COMPARABLE_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/poly_collection/detail/is_equality_comparable.hpp>
#include <type_traits>
#include <utility>

namespace boost{

namespace poly_collection{

namespace detail{

template<typename T,typename=void>
struct is_nothrow_equality_comparable:std::false_type{};

template<typename T>
struct is_nothrow_equality_comparable<
T,
typename std::enable_if<
is_equality_comparable<T>::value
>::type
>:std::integral_constant<
bool,
noexcept(std::declval<T>()==std::declval<T>())
>{};

} 

} 

} 

#endif
