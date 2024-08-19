
#ifndef BOOST_RANGE_REND_HPP
#define BOOST_RANGE_REND_HPP

#if defined(_MSC_VER)
# pragma once
#endif

#include <boost/range/begin.hpp>
#include <boost/range/reverse_iterator.hpp>

namespace boost
{

template< class C >
inline BOOST_DEDUCED_TYPENAME range_reverse_iterator<C>::type
rend( C& c )
{
typedef BOOST_DEDUCED_TYPENAME range_reverse_iterator<C>::type
iter_type;
return iter_type( boost::begin( c ) );
}

template< class C >
inline BOOST_DEDUCED_TYPENAME range_reverse_iterator<const C>::type
rend( const C& c )
{
typedef BOOST_DEDUCED_TYPENAME range_reverse_iterator<const C>::type
iter_type;
return iter_type( boost::begin( c ) );
}

template< class T >
inline BOOST_DEDUCED_TYPENAME range_reverse_iterator<const T>::type
const_rend( const T& r )
{
return boost::rend( r );
}

} 

#endif

