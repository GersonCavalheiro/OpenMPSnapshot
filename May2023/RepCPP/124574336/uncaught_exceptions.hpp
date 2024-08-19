


#ifndef BOOST_CORE_UNCAUGHT_EXCEPTIONS_HPP_INCLUDED_
#define BOOST_CORE_UNCAUGHT_EXCEPTIONS_HPP_INCLUDED_

#include <exception>
#include <boost/config.hpp>

#if defined(BOOST_HAS_PRAGMA_ONCE)
#pragma once
#endif

#if defined(__APPLE__)
#include <Availability.h>
#if (defined(__cpp_lib_uncaught_exceptions) && __cpp_lib_uncaught_exceptions >= 201411) && \
( \
(defined(__MAC_OS_X_VERSION_MIN_REQUIRED) && __MAC_OS_X_VERSION_MIN_REQUIRED >= 101200) || \
(defined(__IPHONE_OS_VERSION_MIN_REQUIRED) && __IPHONE_OS_VERSION_MIN_REQUIRED >= 100000) \
)
#define BOOST_CORE_HAS_UNCAUGHT_EXCEPTIONS
#endif
#elif (defined(__cpp_lib_uncaught_exceptions) && __cpp_lib_uncaught_exceptions >= 201411) || \
(defined(_MSC_VER) && _MSC_VER >= 1900)
#define BOOST_CORE_HAS_UNCAUGHT_EXCEPTIONS
#endif

#if !defined(BOOST_CORE_HAS_UNCAUGHT_EXCEPTIONS)

#if defined(__has_include) && (!defined(BOOST_GCC) || (__GNUC__ >= 5))
#   if __has_include(<cxxabi.h>)
#       define BOOST_CORE_HAS_CXXABI_H
#   endif
#elif defined(__GLIBCXX__) || defined(__GLIBCPP__)
#   define BOOST_CORE_HAS_CXXABI_H
#endif

#if defined(BOOST_CORE_HAS_CXXABI_H)
#if !( \
(defined(__MINGW32__) && (defined(__GNUC__) && (__GNUC__ * 100 + __GNUC_MINOR__) < 405)) || \
defined(__ibmxl__) \
)
#include <cxxabi.h>
#include <cstring>
#define BOOST_CORE_HAS_CXA_GET_GLOBALS
#if !defined(__FreeBSD__) && \
( \
(defined(__GNUC__) && (__GNUC__ * 100 + __GNUC_MINOR__) < 407) || \
defined(__OpenBSD__) || \
(defined(__QNXNTO__) && !defined(__GLIBCXX__) && !defined(__GLIBCPP__)) || \
defined(_LIBCPPABI_VERSION) \
)
namespace __cxxabiv1 {
struct __cxa_eh_globals;
#if defined(__OpenBSD__)
extern "C" __cxa_eh_globals* __cxa_get_globals();
#else
extern "C" __cxa_eh_globals* __cxa_get_globals() BOOST_NOEXCEPT_OR_NOTHROW __attribute__((__const__));
#endif
} 
#endif
#endif
#endif 

#if defined(_MSC_VER) && _MSC_VER >= 1400
#include <cstring>
#define BOOST_CORE_HAS_GETPTD
namespace boost {
namespace core {
namespace detail {
extern "C" void* _getptd();
} 
} 
} 
#endif 

#endif 

#if !defined(BOOST_CORE_HAS_UNCAUGHT_EXCEPTIONS) && !defined(BOOST_CORE_HAS_CXA_GET_GLOBALS) && !defined(BOOST_CORE_HAS_GETPTD)
#define BOOST_CORE_UNCAUGHT_EXCEPTIONS_EMULATED
#endif

namespace boost {

namespace core {

inline unsigned int uncaught_exceptions() BOOST_NOEXCEPT
{
#if defined(BOOST_CORE_HAS_UNCAUGHT_EXCEPTIONS)
return static_cast< unsigned int >(std::uncaught_exceptions());
#elif defined(BOOST_CORE_HAS_CXA_GET_GLOBALS)
unsigned int count;
std::memcpy(&count, reinterpret_cast< const unsigned char* >(::abi::__cxa_get_globals()) + sizeof(void*), sizeof(count)); 
return count;
#elif defined(BOOST_CORE_HAS_GETPTD)
unsigned int count;
std::memcpy(&count, static_cast< const unsigned char* >(boost::core::detail::_getptd()) + (sizeof(void*) == 8u ? 0x100 : 0x90), sizeof(count)); 
return count;
#else
return static_cast< unsigned int >(std::uncaught_exception());
#endif
}

} 

} 

#undef BOOST_CORE_HAS_CXXABI_H
#undef BOOST_CORE_HAS_CXA_GET_GLOBALS
#undef BOOST_CORE_HAS_UNCAUGHT_EXCEPTIONS
#undef BOOST_CORE_HAS_GETPTD

#endif 
