

#ifndef BOOST_ASSIGN_ASSIGNMENT_EXCEPTION_HPP
#define BOOST_ASSIGN_ASSIGNMENT_EXCEPTION_HPP

#include <boost/config.hpp>
#include <exception>

#if defined(BOOST_HAS_PRAGMA_ONCE)
# pragma once
#endif

namespace boost
{    
namespace assign
{
class assignment_exception : public std::exception
{
public:
assignment_exception( const char* _what ) 
: what_( _what )
{ }

virtual const char* what() const BOOST_NOEXCEPT_OR_NOTHROW
{
return what_;
}

private:
const char* what_;
};
}
}

#endif
