
#ifndef BOOST_RANGE_RBEGIN_HPP
#define BOOST_RANGE_RBEGIN_HPP

#if defined(_MSC_VER)
# pragma once
#endif

#include <boost/range/end.hpp>
#include <boost/range/reverse_iterator.hpp>

namespace boost
{

template< class C >
inline BOOST_DEDUCED_TYPENAME range_reverse_iterator<C>::type
rbegin( C& c )
{
typedef BOOST_DEDUCED_TYPENAME range_reverse_iterator<C>::type
iter_type;
return iter_type( boost::end( c ) );
}

template< class C >
inline BOOST_DEDUCED_TYPENAME range_reverse_iterator<const C>::type
rbegin( const C& c )
{
typedef BOOST_DEDUCED_TYPENAME range_reverse_iterator<const C>::type
iter_type;
return iter_type( boost::end( c ) );
}

template< class T >
inline BOOST_DEDUCED_TYPENAME range_reverse_iterator<const T>::type
const_rbegin( const T& r )
{
return boost::rbegin( r );
}

} 

#endif

