#ifndef BOOST_SERIALIZATION_FORCE_INCLUDE_HPP
#define BOOST_SERIALIZATION_FORCE_INCLUDE_HPP

#if defined(_MSC_VER)
# pragma once
#endif




#include <boost/config.hpp>


#if defined(BOOST_HAS_DECLSPEC) && !defined(__COMO__)
#   define BOOST_DLLEXPORT __declspec(dllexport)
#elif ! defined(_WIN32) && ! defined(_WIN64)
#   if defined(__MWERKS__)
#       define BOOST_DLLEXPORT __declspec(dllexport)
#   elif defined(__GNUC__) && (__GNUC__ >= 3)
#       define BOOST_USED __attribute__ ((__used__))
#   elif defined(__IBMCPP__) && (__IBMCPP__ >= 1110)
#       define BOOST_USED __attribute__ ((__used__))
#   elif defined(__INTEL_COMPILER) && (BOOST_INTEL_CXX_VERSION >= 800)
#       define BOOST_USED __attribute__ ((__used__))
#   endif
#endif

#ifndef BOOST_USED
#    define BOOST_USED
#endif

#ifndef BOOST_DLLEXPORT
#    define BOOST_DLLEXPORT
#endif

#endif 
