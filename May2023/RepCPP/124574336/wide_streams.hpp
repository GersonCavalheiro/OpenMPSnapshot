


#ifndef BOOST_IOSTREAMS_DETAIL_CONFIG_WIDE_STREAMS_HPP_INCLUDED
#define BOOST_IOSTREAMS_DETAIL_CONFIG_WIDE_STREAMS_HPP_INCLUDED

#include <boost/config.hpp>
#include <boost/detail/workaround.hpp>
#include <cstddef>

#if defined(_MSC_VER)
# pragma once
#endif       


#ifndef BOOST_IOSTREAMS_NO_STREAM_TEMPLATES
# if defined(__STL_CONFIG_H) && \
!defined (__STL_USE_NEW_IOSTREAMS) && !defined(__crayx1) \

#  define BOOST_IOSTREAMS_NO_STREAM_TEMPLATES
# endif
#endif 


#ifndef BOOST_IOSTREAMS_NO_WIDE_STREAMS
# if defined(BOOST_IOSTREAMS_NO_STREAM_TEMPLATES) || \
defined (BOOST_NO_STD_WSTREAMBUF) && \
( !defined(__MSL_CPP__) || defined(_MSL_NO_WCHART_CPP_SUPPORT) ) \

#  define BOOST_IOSTREAMS_NO_WIDE_STREAMS
# endif
#endif 


#ifndef BOOST_IOSTREAMS_NO_LOCALE
# if defined(BOOST_NO_STD_LOCALE) && \
( !defined(__MSL_CPP__) || defined(_MSL_NO_WCHART_CPP_SUPPORT) ) \

#  define BOOST_IOSTREAMS_NO_LOCALE
# endif
#endif 

#endif 
