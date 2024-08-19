#ifndef BOOST_ARCHIVE_ARCHIVE_EXCEPTION_HPP
#define BOOST_ARCHIVE_ARCHIVE_EXCEPTION_HPP

#if defined(_MSC_VER)
# pragma once
#endif




#include <exception>
#include <boost/assert.hpp>
#include <string>

#include <boost/config.hpp>
#include <boost/archive/detail/decl.hpp>

#if defined(BOOST_WINDOWS)
#include <excpt.h>
#endif

#include <boost/archive/detail/abi_prefix.hpp> 

namespace boost {
namespace archive {

class BOOST_SYMBOL_VISIBLE archive_exception :
public virtual std::exception
{
private:
char m_buffer[128];
protected:
BOOST_ARCHIVE_DECL unsigned int
append(unsigned int l, const char * a);
BOOST_ARCHIVE_DECL
archive_exception() BOOST_NOEXCEPT;
public:
typedef enum {
no_exception,       
other_exception,    
unregistered_class, 
invalid_signature,  
unsupported_version,
pointer_conflict,   
incompatible_native_format, 
array_size_too_short,
input_stream_error, 
invalid_class_name, 
unregistered_cast,   
unsupported_class_version, 
multiple_code_instantiation, 
output_stream_error 
} exception_code;
exception_code code;

BOOST_ARCHIVE_DECL archive_exception(
exception_code c,
const char * e1 = NULL,
const char * e2 = NULL
) BOOST_NOEXCEPT;
BOOST_ARCHIVE_DECL archive_exception(archive_exception const &) BOOST_NOEXCEPT;
BOOST_ARCHIVE_DECL ~archive_exception() BOOST_NOEXCEPT_OR_NOTHROW BOOST_OVERRIDE;
BOOST_ARCHIVE_DECL const char * what() const BOOST_NOEXCEPT_OR_NOTHROW BOOST_OVERRIDE;
};

}
}

#include <boost/archive/detail/abi_suffix.hpp> 

#endif 
