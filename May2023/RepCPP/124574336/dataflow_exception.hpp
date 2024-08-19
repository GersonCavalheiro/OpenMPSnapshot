#ifndef BOOST_ARCHIVE_ITERATORS_DATAFLOW_EXCEPTION_HPP
#define BOOST_ARCHIVE_ITERATORS_DATAFLOW_EXCEPTION_HPP

#if defined(_MSC_VER)
# pragma once
#endif




#include <boost/config.hpp>
#ifndef BOOST_NO_EXCEPTIONS
#include <exception>
#endif 

#include <boost/assert.hpp>

namespace boost {
namespace archive {
namespace iterators {

class dataflow_exception : public std::exception
{
public:
typedef enum {
invalid_6_bitcode,
invalid_base64_character,
invalid_xml_escape_sequence,
comparison_not_permitted,
invalid_conversion,
other_exception
} exception_code;
exception_code code;

dataflow_exception(exception_code c = other_exception) : code(c)
{}

const char *what( ) const throw( ) BOOST_OVERRIDE
{
const char *msg = "unknown exception code";
switch(code){
case invalid_6_bitcode:
msg = "attempt to encode a value > 6 bits";
break;
case invalid_base64_character:
msg = "attempt to decode a value not in base64 char set";
break;
case invalid_xml_escape_sequence:
msg = "invalid xml escape_sequence";
break;
case comparison_not_permitted:
msg = "cannot invoke iterator comparison now";
break;
case invalid_conversion:
msg = "invalid multbyte/wide char conversion";
break;
default:
BOOST_ASSERT(false);
break;
}
return msg;
}
};

} 
} 
} 

#endif 
