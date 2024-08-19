

#ifndef BOOST_IOSTREAMS_DETAIL_CONFIG_FPOS_HPP_INCLUDED
#define BOOST_IOSTREAMS_DETAIL_CONFIG_FPOS_HPP_INCLUDED

#if defined(_MSC_VER)
# pragma once
#endif

#include <boost/config.hpp>

# if (defined(_YVALS) || defined(_CPPLIB_VER)) && !defined(__SGI_STL_PORT) && \
!defined(_STLPORT_VERSION) && !defined(__QNX__) && !defined(_VX_CPU) && !defined(__VXWORKS__) \
&& !((defined(BOOST_MSVC) || defined(BOOST_CLANG)) && _MSVC_STL_VERSION >= 141) \
&& !defined(_LIBCPP_VERSION)


#include <boost/iostreams/detail/ios.hpp>

#  define BOOST_IOSTREAMS_HAS_DINKUMWARE_FPOS

#if !defined(_FPOSOFF)
#define BOOST_IOSTREAMS_FPOSOFF(fp) ((long long)(fp))
#else
#define BOOST_IOSTREAMS_FPOSOFF(fp) _FPOSOFF(fp)
#endif

# endif

#endif 
