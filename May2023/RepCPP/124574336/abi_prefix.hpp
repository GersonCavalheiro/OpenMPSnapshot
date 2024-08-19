


#ifndef BOOST_CONFIG_ABI_PREFIX_HPP
# define BOOST_CONFIG_ABI_PREFIX_HPP
#else
# error double inclusion of header boost/config/abi_prefix.hpp is an error
#endif

#include <boost/config.hpp>

#ifdef BOOST_HAS_ABI_HEADERS
#  include BOOST_ABI_PREFIX
#endif

#if defined( BOOST_BORLANDC )
#pragma nopushoptwarn
#endif

