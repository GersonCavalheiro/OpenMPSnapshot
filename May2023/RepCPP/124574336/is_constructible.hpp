

#ifndef BOOST_POLY_COLLECTION_DETAIL_IS_CONSTRUCIBLE_HPP
#define BOOST_POLY_COLLECTION_DETAIL_IS_CONSTRUCIBLE_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/config.hpp>
#include <boost/detail/workaround.hpp>

#if BOOST_WORKAROUND(BOOST_MSVC_FULL_VER,<190023918)


#include <boost/type_traits/is_constructible.hpp>

namespace boost{

namespace poly_collection{

namespace detail{

template<typename T,typename... Args>
struct is_constructible:std::integral_constant<
bool,
boost::is_constructible<T,Args...>::value
>{};

} 

} 

} 

#else
#include <type_traits>

namespace boost{

namespace poly_collection{

namespace detail{

template<typename T,typename... Args>
using is_constructible=std::is_constructible<T,Args...>;

} 

} 

} 

#endif
#endif
