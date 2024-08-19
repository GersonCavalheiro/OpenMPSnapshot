
#ifndef JSON_CONFIG_H_INCLUDED
#define JSON_CONFIG_H_INCLUDED
#include <stddef.h>
#include <string> 
#include <stdint.h> 



#ifndef JSON_USE_EXCEPTION
#define JSON_USE_EXCEPTION 1
#endif


#ifdef JSON_IN_CPPTL
#include <cpptl/config.h>
#ifndef JSON_USE_CPPTL
#define JSON_USE_CPPTL 1
#endif
#endif

#ifdef JSON_IN_CPPTL
#define JSON_API CPPTL_API
#elif defined(JSON_DLL_BUILD)
#if defined(_MSC_VER) || defined(__MINGW32__)
#define JSON_API __declspec(dllexport)
#define JSONCPP_DISABLE_DLL_INTERFACE_WARNING
#endif 
#elif defined(JSON_DLL)
#if defined(_MSC_VER) || defined(__MINGW32__)
#define JSON_API __declspec(dllimport)
#define JSONCPP_DISABLE_DLL_INTERFACE_WARNING
#endif 
#endif 
#if !defined(JSON_API)
#define JSON_API
#endif


#if defined(_MSC_VER) 
#  if _MSC_VER <= 1200 
#    define JSON_USE_INT64_DOUBLE_CONVERSION 1
#    pragma warning(disable : 4786)
#  endif 

#  if _MSC_VER >= 1500 
#    define JSONCPP_DEPRECATED(message) __declspec(deprecated(message))
#  endif

#endif 

#if __cplusplus >= 201103L
# define JSONCPP_OVERRIDE override
# define JSONCPP_NOEXCEPT noexcept
#elif defined(_MSC_VER) && _MSC_VER > 1600 && _MSC_VER < 1900
# define JSONCPP_OVERRIDE override
# define JSONCPP_NOEXCEPT throw()
#elif defined(_MSC_VER) && _MSC_VER >= 1900
# define JSONCPP_OVERRIDE override
# define JSONCPP_NOEXCEPT noexcept
#else
# define JSONCPP_OVERRIDE
# define JSONCPP_NOEXCEPT throw()
#endif

#ifndef JSON_HAS_RVALUE_REFERENCES

#if defined(_MSC_VER) && _MSC_VER >= 1600 
#define JSON_HAS_RVALUE_REFERENCES 1
#endif 

#ifdef __clang__
#if __has_feature(cxx_rvalue_references)
#define JSON_HAS_RVALUE_REFERENCES 1
#endif  

#elif defined __GNUC__ 
#if defined(__GXX_EXPERIMENTAL_CXX0X__) || (__cplusplus >= 201103L)
#define JSON_HAS_RVALUE_REFERENCES 1
#endif  

#endif 

#endif 

#ifndef JSON_HAS_RVALUE_REFERENCES
#define JSON_HAS_RVALUE_REFERENCES 0
#endif

#ifdef __clang__
#  if __has_extension(attribute_deprecated_with_message)
#    define JSONCPP_DEPRECATED(message)  __attribute__ ((deprecated(message)))
#  endif
#elif defined __GNUC__ 
#  if (__GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 5))
#    define JSONCPP_DEPRECATED(message)  __attribute__ ((deprecated(message)))
#  elif (__GNUC__ > 3 || (__GNUC__ == 3 && __GNUC_MINOR__ >= 1))
#    define JSONCPP_DEPRECATED(message)  __attribute__((__deprecated__))
#  endif  
#endif 

#if !defined(JSONCPP_DEPRECATED)
#define JSONCPP_DEPRECATED(message)
#endif 

#if __GNUC__ >= 6
#  define JSON_USE_INT64_DOUBLE_CONVERSION 1
#endif

#if !defined(JSON_IS_AMALGAMATION)

# include "version.h"

# if JSONCPP_USING_SECURE_MEMORY
#  include "allocator.h" 
# endif

#endif 

namespace Json {
typedef int Int;
typedef unsigned int UInt;
#if defined(JSON_NO_INT64)
typedef int LargestInt;
typedef unsigned int LargestUInt;
#undef JSON_HAS_INT64
#else                 
#if defined(_MSC_VER) 
typedef __int64 Int64;
typedef unsigned __int64 UInt64;
#else                 
typedef int64_t Int64;
typedef uint64_t UInt64;
#endif 
typedef Int64 LargestInt;
typedef UInt64 LargestUInt;
#define JSON_HAS_INT64
#endif 
#if JSONCPP_USING_SECURE_MEMORY
#define JSONCPP_STRING        std::basic_string<char, std::char_traits<char>, Json::SecureAllocator<char> >
#define JSONCPP_OSTRINGSTREAM std::basic_ostringstream<char, std::char_traits<char>, Json::SecureAllocator<char> >
#define JSONCPP_OSTREAM       std::basic_ostream<char, std::char_traits<char>>
#define JSONCPP_ISTRINGSTREAM std::basic_istringstream<char, std::char_traits<char>, Json::SecureAllocator<char> >
#define JSONCPP_ISTREAM       std::istream
#else
#define JSONCPP_STRING        std::string
#define JSONCPP_OSTRINGSTREAM std::ostringstream
#define JSONCPP_OSTREAM       std::ostream
#define JSONCPP_ISTRINGSTREAM std::istringstream
#define JSONCPP_ISTREAM       std::istream
#endif 
} 

#endif 
