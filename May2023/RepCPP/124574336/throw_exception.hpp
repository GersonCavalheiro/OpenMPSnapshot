#ifndef BOOST_SERIALIZATION_THROW_EXCEPTION_HPP_INCLUDED
#define BOOST_SERIALIZATION_THROW_EXCEPTION_HPP_INCLUDED


#if defined(_MSC_VER)
# pragma once
#endif


#include <boost/config.hpp>

#ifndef BOOST_NO_EXCEPTIONS
#include <exception>
#endif

namespace boost {
namespace serialization {

#ifdef BOOST_NO_EXCEPTIONS

BOOST_NORETURN inline void throw_exception(std::exception const & e) {
::boost::throw_exception(e);
}

#else

template<class E> BOOST_NORETURN inline void throw_exception(E const & e){
throw e;
}

#endif

} 
} 

#endif 
