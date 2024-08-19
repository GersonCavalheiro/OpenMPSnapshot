


#ifndef BOOST_IOSTREAMS_DETAIL_CONFIG_BROKEN_OVERLOAD_RESOLUTION_HPP_INCLUDED
#define BOOST_IOSTREAMS_DETAIL_CONFIG_BROKEN_OVERLOAD_RESOLUTION_HPP_INCLUDED

#if defined(_MSC_VER)
# pragma once
#endif             

#include <boost/config.hpp> 
#include <boost/detail/workaround.hpp>
#include <boost/iostreams/detail/config/gcc.hpp>

#if !defined(BOOST_IOSTREAMS_BROKEN_OVERLOAD_RESOLUTION)
# if BOOST_WORKAROUND(__MWERKS__, <= 0x3003) || \
BOOST_WORKAROUND(BOOST_BORLANDC, < 0x600) \

#  define BOOST_IOSTREAMS_BROKEN_OVERLOAD_RESOLUTION
# endif
#endif

#endif 
