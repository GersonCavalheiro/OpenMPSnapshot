
#ifndef BOOST_PTR_CONTAINER_EXCEPTION_HPP
#define BOOST_PTR_CONTAINER_EXCEPTION_HPP

#if defined(_MSC_VER) && (_MSC_VER >= 1200)
# pragma once
#endif

#include <exception>

namespace boost
{
class bad_ptr_container_operation : public std::exception
{
const char* what_;
public:
bad_ptr_container_operation( const char* what ) : what_( what )
{ }

virtual const char* what() const throw()
{
return what_;
}
};



class bad_index : public bad_ptr_container_operation
{
public:
bad_index( const char* what ) : bad_ptr_container_operation( what )
{ }
};



class bad_pointer : public bad_ptr_container_operation
{
public:
bad_pointer() : bad_ptr_container_operation( "Null pointer not allowed in a pointer container!" )
{ }

bad_pointer( const char* text ) : bad_ptr_container_operation( text )
{ }
};
}

#endif
