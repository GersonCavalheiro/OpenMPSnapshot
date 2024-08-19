#ifndef BOOST_LOCALE_ENCODING_ERRORS_HPP_INCLUDED
#define BOOST_LOCALE_ENCODING_ERRORS_HPP_INCLUDED

#include <boost/locale/definitions.hpp>
#ifdef BOOST_MSVC
#  pragma warning(push)
#  pragma warning(disable : 4275 4251 4231 4660)
#endif
#include <stdexcept>



namespace boost {
namespace locale {
namespace conv {

class BOOST_SYMBOL_VISIBLE conversion_error : public std::runtime_error {
public:
conversion_error() : std::runtime_error("Conversion failed") {}
};

class BOOST_SYMBOL_VISIBLE invalid_charset_error : public std::runtime_error {
public:

invalid_charset_error(std::string charset) : 
std::runtime_error("Invalid or unsupported charset:" + charset)
{
}
};


typedef enum {
skip            = 0,    
stop            = 1,    
default_method  = skip  
} method_type;



} 

} 
} 

#ifdef BOOST_MSVC
#pragma warning(pop)
#endif

#endif


