#ifndef BOOST_RANGE_ITERATOR_RANGE_IO_HPP_INCLUDED
#define BOOST_RANGE_ITERATOR_RANGE_IO_HPP_INCLUDED

#include <boost/config.hpp>
#include <boost/detail/workaround.hpp>

#if BOOST_WORKAROUND(BOOST_MSVC, BOOST_TESTED_AT(1500))
#pragma warning( push )
#pragma warning( disable : 4996 )
#endif

#ifndef BOOST_OLD_IOSTREAMS 
# if defined(__STL_CONFIG_H) && \
!defined (__STL_USE_NEW_IOSTREAMS) && !defined(__crayx1) \

#  define BOOST_OLD_IOSTREAMS
# endif
#endif 

#ifndef _STLP_NO_IOSTREAMS
# ifndef BOOST_OLD_IOSTREAMS
#  include <ostream>
# else
#  include <ostream.h>
# endif
#endif 

#include <boost/range/iterator_range_core.hpp>
#include <iterator>
#include <algorithm>
#include <cstddef>

namespace boost
{

#ifndef _STLP_NO_IOSTREAMS
# ifndef BOOST_OLD_IOSTREAMS   


template< typename IteratorT, typename Elem, typename Traits >
inline std::basic_ostream<Elem,Traits>& operator<<( 
std::basic_ostream<Elem, Traits>& Os,
const iterator_range<IteratorT>& r )
{
std::copy( r.begin(), r.end(), 
std::ostream_iterator< BOOST_DEDUCED_TYPENAME 
iterator_value<IteratorT>::type, 
Elem, Traits>(Os) );
return Os;
}

# else


template< typename IteratorT >
inline std::ostream& operator<<( 
std::ostream& Os,
const iterator_range<IteratorT>& r )
{
std::copy( r.begin(), r.end(), std::ostream_iterator<char>(Os));
return Os;
}

# endif
#endif 

} 

#undef BOOST_OLD_IOSTREAMS

#if BOOST_WORKAROUND(BOOST_MSVC, BOOST_TESTED_AT(1500))
#pragma warning(pop)
#endif

#endif 
