#ifndef BOOST_ARCHIVE_ITERATORS_BASE64_EXCEPTION_HPP
#define BOOST_ARCHIVE_ITERATORS_BASE64_EXCEPTION_HPP

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

class base64_exception : public std::exception
{
public:
typedef enum {
invalid_code,       
invalid_character,  
other_exception
} exception_code;
exception_code code;

base64_exception(exception_code c = other_exception) : code(c)
{}

virtual const char *what( ) const throw( )
{
const char *msg = "unknown exception code";
switch(code){
case invalid_code:
msg = "attempt to encode a value > 6 bits";
break;
case invalid_character:
msg = "attempt to decode a value not in base64 char set";
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
#endif 
