

#ifndef BOOST_POLY_COLLECTION_DETAIL_IS_ACCEPTABLE_HPP
#define BOOST_POLY_COLLECTION_DETAIL_IS_ACCEPTABLE_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <type_traits>

namespace boost{

namespace poly_collection{

namespace detail{



template<typename T,typename Model,typename=void>
struct is_acceptable:std::integral_constant<
bool,
Model::template is_implementation<T>::value&&
std::is_move_constructible<typename std::decay<T>::type>::value&&
(std::is_move_assignable<typename std::decay<T>::type>::value||
std::is_nothrow_move_constructible<typename std::decay<T>::type>::value)
>{};

} 

} 

} 

#endif
