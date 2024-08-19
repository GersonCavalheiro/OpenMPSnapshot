
#ifndef BOOST_RANGE_DETAIL_SIZER_HPP
#define BOOST_RANGE_DETAIL_SIZER_HPP

#if defined(_MSC_VER)
# pragma once
#endif

#include <boost/range/config.hpp>
#include <cstddef>

namespace boost 
{

template< typename T, std::size_t sz >
char (& sizer( const T BOOST_RANGE_ARRAY_REF()[sz] ) )[sz];

template< typename T, std::size_t sz >
char (& sizer( T BOOST_RANGE_ARRAY_REF()[sz] ) )[sz];

} 

#endif
