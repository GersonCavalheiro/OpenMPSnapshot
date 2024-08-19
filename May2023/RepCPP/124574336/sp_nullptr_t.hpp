#ifndef BOOST_SMART_PTR_DETAIL_SP_NULLPTR_T_HPP_INCLUDED
#define BOOST_SMART_PTR_DETAIL_SP_NULLPTR_T_HPP_INCLUDED


#if defined(_MSC_VER) && (_MSC_VER >= 1020)
# pragma once
#endif


#include <boost/config.hpp>
#include <cstddef>

#if !defined( BOOST_NO_CXX11_NULLPTR )

namespace boost
{

namespace detail
{

#if !defined( BOOST_NO_CXX11_DECLTYPE ) && ( ( defined( __clang__ ) && !defined( _LIBCPP_VERSION ) ) || defined( __INTEL_COMPILER ) )

typedef decltype(nullptr) sp_nullptr_t;

#else

typedef std::nullptr_t sp_nullptr_t;

#endif

} 

} 

#endif 

#endif  
