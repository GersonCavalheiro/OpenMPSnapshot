#ifndef  BOOST_SERIALIZATION_DETAIL_IS_DEFAULT_CONSTRUCTIBLE_HPP
#define BOOST_SERIALIZATION_DETAIL_IS_DEFAULT_CONSTRUCTIBLE_HPP

#if defined(_MSC_VER)
# pragma once
#endif



#include <boost/config.hpp>

#if ! defined(BOOST_NO_CXX11_HDR_TYPE_TRAITS)
#include <type_traits>
namespace boost{
namespace serialization {
namespace detail {

template<typename T>
struct is_default_constructible : public std::is_default_constructible<T> {};

} 
} 
} 
#else
#include <boost/type_traits/has_trivial_constructor.hpp>
namespace boost{
namespace serialization {
namespace detail {

template<typename T>
struct is_default_constructible : public boost::has_trivial_constructor<T> {};

} 
} 
} 

#endif


#endif 
