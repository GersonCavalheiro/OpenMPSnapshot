


#if __BORLANDC__ < 0x540
#  error "Compiler not supported or configured - please reconfigure"
#endif

#if (__BORLANDC__ > 0x613)
#     error "Unknown compiler version - please run the configure tests and report the results"
#elif (__BORLANDC__ == 0x600)
#  error "CBuilderX preview compiler is no longer supported"
#endif

#if (__BORLANDC__ < 0x560) || defined(_USE_OLD_RW_STL)
#  define BOOST_BCB_WITH_ROGUE_WAVE
#elif __BORLANDC__ < 0x570
#  define BOOST_BCB_WITH_STLPORT
#else
#  define BOOST_BCB_WITH_DINKUMWARE
#endif

#   if __BORLANDC__ <= 0x0550
#     define BOOST_NO_MEMBER_TEMPLATE_FRIENDS
#     if __BORLANDC__ == 0x0550
#       define BOOST_NO_OPERATORS_IN_NAMESPACE
#     endif
#   endif

#if (__BORLANDC__ <= 0x551)
#  define BOOST_NO_CV_SPECIALIZATIONS
#  define BOOST_NO_CV_VOID_SPECIALIZATIONS
#  define BOOST_NO_DEDUCED_TYPENAME
#include <climits>
#include <cwchar>
#ifndef WCHAR_MAX
#  define WCHAR_MAX 0xffff
#endif
#ifndef WCHAR_MIN
#  define WCHAR_MIN 0
#endif
#endif

#if (__BORLANDC__ <= 0x564)

#  ifdef NDEBUG
#     include <cstring>
#     undef strcmp
#  endif
#  include <errno.h>
#  ifndef errno
#     define errno errno
#  endif

#endif

#if (__BORLANDC__ >= 0x561) && (__BORLANDC__ <= 0x580)
#  define BOOST_NO_MEMBER_FUNCTION_SPECIALIZATIONS
#endif

#if (__BORLANDC__ <= 0x582)
#  define BOOST_NO_SFINAE
#  define BOOST_BCB_PARTIAL_SPECIALIZATION_BUG
#  define BOOST_NO_TEMPLATE_TEMPLATES

#  define BOOST_NO_PRIVATE_IN_AGGREGATE

#  ifdef _WIN32
#     define BOOST_NO_SWPRINTF
#  elif defined(linux) || defined(__linux__) || defined(__linux)
#     define BOOST_NO_STDC_NAMESPACE
#     pragma defineonoption BOOST_CPPUNWIND -x
#  endif
#endif

#if (__BORLANDC__ <= 0x613)  
#  define BOOST_NO_LIMITS_COMPILE_TIME_CONSTANTS
#  define BOOST_NO_IS_ABSTRACT
#  define BOOST_NO_FUNCTION_TYPE_SPECIALIZATIONS
#  define BOOST_NO_USING_TEMPLATE
#  define BOOST_SP_NO_SP_CONVERTIBLE

#define BOOST_MPL_CFG_NO_PREPROCESSED_HEADERS
#endif

#  define BOOST_NO_INTEGRAL_INT64_T
#  define BOOST_FUNCTION_SCOPE_USING_DECLARATION_BREAKS_ADL
#  define BOOST_NO_DEPENDENT_NESTED_DERIVATIONS
#  define BOOST_NO_MEMBER_TEMPLATE_FRIENDS
#  define BOOST_NO_TWO_PHASE_NAME_LOOKUP
#  define BOOST_NO_USING_DECLARATION_OVERLOADS_FROM_TYPENAME_BASE
#  define BOOST_NO_NESTED_FRIENDSHIP
#  define BOOST_NO_TYPENAME_WITH_CTOR
#if (__BORLANDC__ < 0x600)
#  define BOOST_ILLEGAL_CV_REFERENCES
#endif

#if (__BORLANDC__ >= 0x599)
#  pragma defineonoption BOOST_CODEGEAR_0X_SUPPORT -Ax
#endif
#if !defined( BOOST_CODEGEAR_0X_SUPPORT ) || (__BORLANDC__ < 0x610)
#  define BOOST_NO_CHAR16_T
#  define BOOST_NO_CHAR32_T
#  define BOOST_NO_DECLTYPE
#  define BOOST_NO_EXPLICIT_CONVERSION_OPERATORS
#  define BOOST_NO_EXTERN_TEMPLATE
#  define BOOST_NO_RVALUE_REFERENCES 
#  define BOOST_NO_SCOPED_ENUMS
#  define BOOST_NO_STATIC_ASSERT
#else
#  define BOOST_HAS_ALIGNOF
#  define BOOST_HAS_CHAR16_T
#  define BOOST_HAS_CHAR32_T
#  define BOOST_HAS_DECLTYPE
#  define BOOST_HAS_EXPLICIT_CONVERSION_OPS
#  define BOOST_HAS_REF_QUALIFIER
#  define BOOST_HAS_RVALUE_REFS
#  define BOOST_HAS_STATIC_ASSERT
#endif

#define BOOST_NO_AUTO_DECLARATIONS
#define BOOST_NO_AUTO_MULTIDECLARATIONS
#define BOOST_NO_CONCEPTS
#define BOOST_NO_CONSTEXPR
#define BOOST_NO_DEFAULTED_FUNCTIONS
#define BOOST_NO_DELETED_FUNCTIONS
#define BOOST_NO_FUNCTION_TEMPLATE_DEFAULT_ARGS
#define BOOST_NO_INITIALIZER_LISTS
#define BOOST_NO_LAMBDAS
#define BOOST_NO_NULLPTR
#define BOOST_NO_RAW_LITERALS
#define BOOST_NO_RVALUE_REFERENCES
#define BOOST_NO_SCOPED_ENUMS
#define BOOST_NO_SFINAE_EXPR
#define BOOST_NO_TEMPLATE_ALIASES
#define BOOST_NO_UNICODE_LITERALS    
#define BOOST_NO_VARIADIC_TEMPLATES

#if __BORLANDC__ >= 0x590
#  define BOOST_HAS_TR1_HASH

#  define BOOST_HAS_MACRO_USE_FACET
#endif

#if __BORLANDC__ >= 0x561
#  ifndef __NO_LONG_LONG
#     define BOOST_HAS_LONG_LONG
#  else
#     define BOOST_NO_LONG_LONG
#  endif
#  ifdef _WIN32
#      define BOOST_HAS_STDINT_H
#  endif
#endif

#if defined( BOOST_BCB_WITH_ROGUE_WAVE )
#include <float.h>
#endif
#if (__BORLANDC__ >= 0x530) && !defined(__STRICT_ANSI__)
#  define BOOST_HAS_MS_INT64
#endif
#if !defined(_CPPUNWIND) && !defined(BOOST_CPPUNWIND) && !defined(__EXCEPTIONS)
#  define BOOST_NO_EXCEPTIONS
#endif
#ifndef __STRICT_ANSI__
#  define BOOST_HAS_DIRENT_H
#endif
#ifndef __STRICT_ANSI__
#  define BOOST_HAS_DECLSPEC
#endif
#if __BORLANDC__ != 0x600 
#ifndef BOOST_ABI_PREFIX
#  define BOOST_ABI_PREFIX "boost/config/abi/borland_prefix.hpp"
#endif
#ifndef BOOST_ABI_SUFFIX
#  define BOOST_ABI_SUFFIX "boost/config/abi/borland_suffix.hpp"
#endif
#endif
#if __BORLANDC__ < 0x600
#  pragma defineonoption BOOST_DISABLE_WIN32 -A
#elif defined(__STRICT_ANSI__)
#  define BOOST_DISABLE_WIN32
#endif
#if defined(_MSC_VER) && (_MSC_VER <= 1200)
#  define BOOST_NO_ARGUMENT_DEPENDENT_LOOKUP
#  define BOOST_NO_VOID_RETURNS
#endif

#define BOOST_COMPILER "Borland C++ version " BOOST_STRINGIZE(__BORLANDC__)



