#ifndef BOOST_ARCHIVE_XML_ARCHIVE_EXCEPTION_HPP
#define BOOST_ARCHIVE_XML_ARCHIVE_EXCEPTION_HPP

#if defined(_MSC_VER)
# pragma once
#endif




#include <exception>
#include <boost/assert.hpp>

#include <boost/config.hpp>
#include <boost/archive/detail/decl.hpp>
#include <boost/archive/archive_exception.hpp>

#include <boost/archive/detail/abi_prefix.hpp> 

namespace boost {
namespace archive {

class BOOST_SYMBOL_VISIBLE xml_archive_exception :
public virtual boost::archive::archive_exception
{
public:
typedef enum {
xml_archive_parsing_error,    
xml_archive_tag_mismatch,
xml_archive_tag_name_error
} exception_code;
BOOST_ARCHIVE_DECL xml_archive_exception(
exception_code c,
const char * e1 = NULL,
const char * e2 = NULL
);
BOOST_ARCHIVE_DECL xml_archive_exception(xml_archive_exception const &);
BOOST_ARCHIVE_DECL ~xml_archive_exception() BOOST_NOEXCEPT_OR_NOTHROW BOOST_OVERRIDE;
};

}
}

#include <boost/archive/detail/abi_suffix.hpp> 

#endif 
