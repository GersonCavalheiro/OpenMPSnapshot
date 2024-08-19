#ifndef BOOST_ARCHIVE_DETAIL_AUTO_LINK_WARCHIVE_HPP
#define BOOST_ARCHIVE_DETAIL_AUTO_LINK_WARCHIVE_HPP

#if defined(_MSC_VER)
# pragma once
#endif






#include <boost/archive/detail/decl.hpp>

#if !defined(BOOST_WARCHIVE_SOURCE) \
&& !defined(BOOST_ALL_NO_LIB) && !defined(BOOST_SERIALIZATION_NO_LIB)

#define BOOST_LIB_NAME boost_wserialization
#if defined(BOOST_ALL_DYN_LINK) || defined(BOOST_SERIALIZATION_DYN_LINK)
#  define BOOST_DYN_LINK
#endif
#include <boost/config/auto_link.hpp>
#endif  

#endif 
