#ifndef BOOST_SMART_PTR_DETAIL_SP_FORWARD_HPP_INCLUDED
#define BOOST_SMART_PTR_DETAIL_SP_FORWARD_HPP_INCLUDED


#if defined(_MSC_VER) && (_MSC_VER >= 1020)
# pragma once
#endif


#include <boost/config.hpp>

namespace boost
{

namespace detail
{

#if !defined( BOOST_NO_CXX11_RVALUE_REFERENCES )

#if defined( BOOST_GCC ) && __GNUC__ * 100 + __GNUC_MINOR__ <= 404

template< class T > T&& sp_forward( T && t ) BOOST_NOEXCEPT
{
return t;
}

#else

template< class T > T&& sp_forward( T & t ) BOOST_NOEXCEPT
{
return static_cast< T&& >( t );
}

#endif

#endif

} 

} 

#endif  
