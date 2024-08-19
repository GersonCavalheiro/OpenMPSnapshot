#ifndef BOOST_CORE_IS_SAME_HPP_INCLUDED
#define BOOST_CORE_IS_SAME_HPP_INCLUDED


#if defined(_MSC_VER) && (_MSC_VER >= 1020)
# pragma once
#endif


#include <boost/config.hpp>

namespace boost
{

namespace core
{

template< class T1, class T2 > struct is_same
{
BOOST_STATIC_CONSTANT( bool, value = false );
};

template< class T > struct is_same< T, T >
{
BOOST_STATIC_CONSTANT( bool, value = true );
};

} 

} 

#endif 
