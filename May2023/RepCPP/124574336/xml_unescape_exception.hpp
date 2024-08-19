#ifndef BOOST_ARCHIVE_ITERATORS_XML_UNESCAPE_EXCEPTION_HPP
#define BOOST_ARCHIVE_ITERATORS_XML_UNESCAPE_EXCEPTION_HPP

#if defined(_MSC_VER)
# pragma once
#endif




#include <boost/config.hpp>
#ifndef BOOST_NO_EXCEPTIONS
#include <exception>

#include <boost/assert.hpp>

namespace boost {
namespace archive {
namespace iterators {

class xml_unescape_exception : public std::exception
{
public:
xml_unescape_exception()
{}

virtual const char *what( ) const throw( )
{
return "xml contained un-recognized escape code";
}
};

} 
} 
} 

#endif 
#endif 
