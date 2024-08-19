#ifndef BOOST_SMART_PTR_BAD_WEAK_PTR_HPP_INCLUDED
#define BOOST_SMART_PTR_BAD_WEAK_PTR_HPP_INCLUDED


#if defined(_MSC_VER) && (_MSC_VER >= 1020)
# pragma once
#endif


#include <boost/config.hpp>
#include <exception>

#ifdef BOOST_BORLANDC
# pragma warn -8026     
#endif

namespace boost
{


#if defined(BOOST_BORLANDC) && BOOST_BORLANDC <= 0x564
# pragma option push -pc
#endif

#if defined(BOOST_CLANG)
# pragma clang diagnostic push
# pragma clang diagnostic ignored "-Wweak-vtables"
#endif

class bad_weak_ptr: public std::exception
{
public:

char const * what() const BOOST_NOEXCEPT_OR_NOTHROW BOOST_OVERRIDE
{
return "tr1::bad_weak_ptr";
}
};

#if defined(BOOST_CLANG)
# pragma clang diagnostic pop
#endif

#if defined(BOOST_BORLANDC) && BOOST_BORLANDC <= 0x564
# pragma option pop
#endif

} 

#ifdef BOOST_BORLANDC
# pragma warn .8026     
#endif

#endif  
