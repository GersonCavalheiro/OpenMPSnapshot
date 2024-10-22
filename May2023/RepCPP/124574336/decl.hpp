#ifndef BOOST_ARCHIVE_DETAIL_DECL_HPP
#define BOOST_ARCHIVE_DETAIL_DECL_HPP

#if defined(_MSC_VER)
# pragma once
#endif





#include <boost/config.hpp>

#if (defined(BOOST_ALL_DYN_LINK) || defined(BOOST_SERIALIZATION_DYN_LINK))
#if defined(BOOST_ARCHIVE_SOURCE)
#define BOOST_ARCHIVE_DECL BOOST_SYMBOL_EXPORT
#else
#define BOOST_ARCHIVE_DECL BOOST_SYMBOL_IMPORT
#endif

#if defined(BOOST_WARCHIVE_SOURCE)
#define BOOST_WARCHIVE_DECL BOOST_SYMBOL_EXPORT
#else
#define BOOST_WARCHIVE_DECL BOOST_SYMBOL_IMPORT
#endif

#if defined(BOOST_WARCHIVE_SOURCE) || defined(BOOST_ARCHIVE_SOURCE)
#define BOOST_ARCHIVE_OR_WARCHIVE_DECL BOOST_SYMBOL_EXPORT
#else
#define BOOST_ARCHIVE_OR_WARCHIVE_DECL BOOST_SYMBOL_IMPORT
#endif

#endif

#if ! defined(BOOST_ARCHIVE_DECL)
#define BOOST_ARCHIVE_DECL
#endif
#if ! defined(BOOST_WARCHIVE_DECL)
#define BOOST_WARCHIVE_DECL
#endif
#if ! defined(BOOST_ARCHIVE_OR_WARCHIVE_DECL)
#define BOOST_ARCHIVE_OR_WARCHIVE_DECL
#endif

#endif 
