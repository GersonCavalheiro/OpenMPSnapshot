
#ifndef BOOST_RANGE_EMPTY_HPP
#define BOOST_RANGE_EMPTY_HPP

#if defined(_MSC_VER)
# pragma once
#endif

#include <boost/range/config.hpp>
#include <boost/range/begin.hpp>
#include <boost/range/end.hpp>

namespace boost 
{ 

template< class T >
inline bool empty( const T& r )
{
return boost::begin( r ) == boost::end( r );
}

} 


#endif
