
#ifndef BOOST_VARIANT_DETAIL_FORCED_RETURN_HPP
#define BOOST_VARIANT_DETAIL_FORCED_RETURN_HPP

#include <boost/config.hpp>
#include <boost/assert.hpp>


#ifdef BOOST_MSVC
# pragma warning( push )
# pragma warning( disable : 4702 ) 
#endif

namespace boost { namespace detail { namespace variant {

template <typename T>
BOOST_NORETURN inline T
forced_return()
{
BOOST_ASSERT(false);

T (*dummy)() = 0;
(void)dummy;
BOOST_UNREACHABLE_RETURN(dummy());
}

}}} 


#ifdef BOOST_MSVC
# pragma warning( pop )
#endif

#endif 
