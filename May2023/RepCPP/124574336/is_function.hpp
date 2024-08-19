

#ifndef BOOST_MULTI_INDEX_DETAIL_IS_FUNCTION_HPP
#define BOOST_MULTI_INDEX_DETAIL_IS_FUNCTION_HPP

#if defined(_MSC_VER)
#pragma once
#endif

#include <boost/config.hpp> 
#include <boost/detail/workaround.hpp>

#if !defined(BOOST_NO_CXX11_HDR_TYPE_TRAITS)||\
BOOST_WORKAROUND(_LIBCPP_VERSION,<30700)||\
BOOST_WORKAROUND(BOOST_LIBSTDCXX_VERSION,<40802)


#include <boost/type_traits/is_function.hpp>

namespace boost{namespace multi_index{namespace detail{

template<typename T>
struct is_function:boost::is_function<T>{};

}}} 

#else

#include <type_traits>

namespace boost{namespace multi_index{namespace detail{

template<typename T>
struct is_function:std::is_function<T>{};

}}} 

#endif
#endif
