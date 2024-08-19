#ifndef BOOST_ARCHIVE_BASIC_XML_TEXT_ARCHIVE_HPP
#define BOOST_ARCHIVE_BASIC_XML_TEXT_ARCHIVE_HPP

#if defined(_MSC_VER)
# pragma once
#endif




#include <boost/archive/archive_exception.hpp>

#include <boost/archive/detail/auto_link_archive.hpp>
#include <boost/archive/detail/abi_prefix.hpp> 

namespace boost {
namespace archive {


extern
BOOST_ARCHIVE_DECL const char *
BOOST_ARCHIVE_XML_OBJECT_ID();

extern
BOOST_ARCHIVE_DECL const char *
BOOST_ARCHIVE_XML_OBJECT_REFERENCE();

extern
BOOST_ARCHIVE_DECL const char *
BOOST_ARCHIVE_XML_CLASS_ID();

extern
BOOST_ARCHIVE_DECL const char *
BOOST_ARCHIVE_XML_CLASS_ID_REFERENCE();

extern
BOOST_ARCHIVE_DECL const char *
BOOST_ARCHIVE_XML_CLASS_NAME();

extern
BOOST_ARCHIVE_DECL const char *
BOOST_ARCHIVE_XML_TRACKING();

extern
BOOST_ARCHIVE_DECL const char *
BOOST_ARCHIVE_XML_VERSION();

extern
BOOST_ARCHIVE_DECL const char *
BOOST_ARCHIVE_XML_SIGNATURE();

}
}

#include <boost/archive/detail/abi_suffix.hpp> 

#endif 

