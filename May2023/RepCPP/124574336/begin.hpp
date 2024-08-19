
#ifndef BOOST_RANGE_BEGIN_HPP
#define BOOST_RANGE_BEGIN_HPP

#if defined(_MSC_VER)
# pragma once
#endif

#include <boost/range/config.hpp>

#include <boost/range/iterator.hpp>
#include <boost/config.hpp>
#include <boost/config/workaround.hpp>

namespace boost
{

#if !BOOST_WORKAROUND(BOOST_BORLANDC, BOOST_TESTED_AT(0x564))
namespace range_detail
{
#endif


template< typename C >
BOOST_CONSTEXPR inline BOOST_DEDUCED_TYPENAME range_iterator<C>::type
range_begin( C& c )
{
return c.begin();
}


template< typename Iterator >
BOOST_CONSTEXPR inline Iterator range_begin( const std::pair<Iterator,Iterator>& p )
{
return p.first;
}

template< typename Iterator >
BOOST_CONSTEXPR inline Iterator range_begin( std::pair<Iterator,Iterator>& p )
{
return p.first;
}


template< typename T, std::size_t sz >
BOOST_CONSTEXPR inline const T* range_begin( const T (&a)[sz] ) BOOST_NOEXCEPT
{
return a;
}

template< typename T, std::size_t sz >
BOOST_CONSTEXPR inline T* range_begin( T (&a)[sz] ) BOOST_NOEXCEPT
{
return a;
}


#if !BOOST_WORKAROUND(BOOST_BORLANDC, BOOST_TESTED_AT(0x564))
} 
#endif

namespace range_adl_barrier
{

template< class T >
#if !BOOST_WORKAROUND(BOOST_GCC, < 40700)
BOOST_CONSTEXPR
#endif
inline BOOST_DEDUCED_TYPENAME range_iterator<T>::type begin( T& r )
{
#if !BOOST_WORKAROUND(BOOST_BORLANDC, BOOST_TESTED_AT(0x564))
using namespace range_detail;
#endif
return range_begin( r );
}

template< class T >
#if !BOOST_WORKAROUND(BOOST_GCC, < 40700)
BOOST_CONSTEXPR
#endif
inline BOOST_DEDUCED_TYPENAME range_iterator<const T>::type begin( const T& r )
{
#if !BOOST_WORKAROUND(BOOST_BORLANDC, BOOST_TESTED_AT(0x564))
using namespace range_detail;
#endif
return range_begin( r );
}

} 
} 

namespace boost
{
namespace range_adl_barrier
{
template< class T >
inline BOOST_DEDUCED_TYPENAME range_iterator<const T>::type
const_begin( const T& r )
{
return boost::range_adl_barrier::begin( r );
}
} 

using namespace range_adl_barrier;
} 

#endif

