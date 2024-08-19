

#ifndef BOOST_POLY_COLLECTION_DETAIL_IS_EQUALITY_COMPARABLE_HPP
#define BOOST_POLY_COLLECTION_DETAIL_IS_EQUALITY_COMPARABLE_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/type_traits/has_equal_to.hpp>
#include <type_traits>

namespace boost{

namespace poly_collection{

namespace detail{

template<typename T>
using is_equality_comparable=std::integral_constant<
bool,
has_equal_to<T,T,bool>::value
>;

} 

} 

} 

#endif
