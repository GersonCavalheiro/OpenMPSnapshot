


#ifndef BOOST_IOSTREAMS_DETAIL_AUTO_LINK_HPP_INCLUDED
#define BOOST_IOSTREAMS_DETAIL_AUTO_LINK_HPP_INCLUDED

#if defined(_MSC_VER)
# pragma once
#endif              

#if defined(BOOST_EXTERNAL_LIB_NAME)
# if defined(BOOST_MSVC) \
|| defined(__BORLANDC__) && !defined(__clang__) \
|| (defined(__MWERKS__) && defined(_WIN32) && (__MWERKS__ >= 0x3000)) \
|| (defined(__ICL) && defined(_MSC_EXTENSIONS)) \

#  pragma comment(lib, BOOST_EXTERNAL_LIB_NAME)
# endif
# undef BOOST_EXTERNAL_LIB_NAME
#endif


#if !defined(BOOST_IOSTREAMS_SOURCE) && \
!defined(BOOST_ALL_NO_LIB) && \
!defined(BOOST_IOSTREAMS_NO_LIB) \


# define BOOST_LIB_NAME boost_iostreams

# if defined(BOOST_ALL_DYN_LINK) || defined(BOOST_IOSTREAMS_DYN_LINK)
#  define BOOST_DYN_LINK
# endif

# include <boost/config/auto_link.hpp>
#endif  

#endif 
