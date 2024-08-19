#ifndef BOOST_CONFIG_HEADER_DEPRECATED_HPP_INCLUDED
#define BOOST_CONFIG_HEADER_DEPRECATED_HPP_INCLUDED


#include <boost/config/pragma_message.hpp>

#if defined(BOOST_ALLOW_DEPRECATED_HEADERS)
# define BOOST_HEADER_DEPRECATED(a)
#else
# define BOOST_HEADER_DEPRECATED(a) BOOST_PRAGMA_MESSAGE("This header is deprecated. Use " a " instead.")
#endif

#endif 
