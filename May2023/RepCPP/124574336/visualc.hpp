

#define BOOST_MSVC _MSC_VER

#if _MSC_FULL_VER > 100000000
#  define BOOST_MSVC_FULL_VER _MSC_FULL_VER
#else
#  define BOOST_MSVC_FULL_VER (_MSC_FULL_VER * 10)
#endif

#pragma warning( disable : 4503 ) 

#define BOOST_HAS_PRAGMA_ONCE

#if _MSC_VER < 1310
#  error "Compiler not supported or configured - please reconfigure"
#endif

#define BOOST_UNREACHABLE_RETURN(x) __assume(0);

#if _MSC_FULL_VER < 180020827
#  define BOOST_NO_FENV_H
#endif

#if _MSC_VER < 1400
#  define BOOST_NO_SWPRINTF
#  define BOOST_NO_CXX11_EXTERN_TEMPLATE
#  define BOOST_NO_CXX11_VARIADIC_MACROS
#  define BOOST_NO_CXX11_LOCAL_CLASS_TEMPLATE_PARAMETERS
#endif

#if _MSC_VER < 1500  
#  define BOOST_NO_MEMBER_TEMPLATE_FRIENDS
#endif

#if _MSC_VER < 1600  
#  define BOOST_NO_ADL_BARRIER
#endif


#ifndef _NATIVE_WCHAR_T_DEFINED
#  define BOOST_NO_INTRINSIC_WCHAR_T
#endif

#if !defined(_CPPUNWIND) && !defined(BOOST_NO_EXCEPTIONS)
#  define BOOST_NO_EXCEPTIONS
#endif

#define BOOST_HAS_MS_INT64
#if defined(_MSC_EXTENSIONS) || (_MSC_VER >= 1400)
#   define BOOST_HAS_LONG_LONG
#else
#   define BOOST_NO_LONG_LONG
#endif
#if (_MSC_VER >= 1400) && !defined(_DEBUG)
#   define BOOST_HAS_NRVO
#endif
#if _MSC_VER >= 1600  
#  define BOOST_HAS_PRAGMA_DETECT_MISMATCH
#endif
#if !defined(_MSC_EXTENSIONS) && !defined(BOOST_DISABLE_WIN32)
#  define BOOST_DISABLE_WIN32
#endif
#if !defined(_CPPRTTI) && !defined(BOOST_NO_RTTI)
#  define BOOST_NO_RTTI
#endif

#if (_MSC_VER >= 1700) && defined(_HAS_CXX17) && (_HAS_CXX17 > 0)
# define BOOST_HAS_TR1_UNORDERED_MAP
# define BOOST_HAS_TR1_UNORDERED_SET
#endif


#if _MSC_VER < 1600
#  define BOOST_NO_CXX11_AUTO_DECLARATIONS
#  define BOOST_NO_CXX11_AUTO_MULTIDECLARATIONS
#  define BOOST_NO_CXX11_LAMBDAS
#  define BOOST_NO_CXX11_RVALUE_REFERENCES
#  define BOOST_NO_CXX11_STATIC_ASSERT
#  define BOOST_NO_CXX11_NULLPTR
#  define BOOST_NO_CXX11_DECLTYPE
#endif 

#if _MSC_VER >= 1600
#  define BOOST_HAS_STDINT_H
#endif

#if _MSC_VER < 1700
#  define BOOST_NO_CXX11_FINAL
#  define BOOST_NO_CXX11_RANGE_BASED_FOR
#  define BOOST_NO_CXX11_SCOPED_ENUMS
#  define BOOST_NO_CXX11_OVERRIDE
#endif 

#if _MSC_FULL_VER < 180020827
#  define BOOST_NO_CXX11_DEFAULTED_FUNCTIONS
#  define BOOST_NO_CXX11_DELETED_FUNCTIONS
#  define BOOST_NO_CXX11_EXPLICIT_CONVERSION_OPERATORS
#  define BOOST_NO_CXX11_FUNCTION_TEMPLATE_DEFAULT_ARGS
#  define BOOST_NO_CXX11_RAW_LITERALS
#  define BOOST_NO_CXX11_TEMPLATE_ALIASES
#  define BOOST_NO_CXX11_TRAILING_RESULT_TYPES
#  define BOOST_NO_CXX11_VARIADIC_TEMPLATES
#  define BOOST_NO_CXX11_UNIFIED_INITIALIZATION_SYNTAX
#  define BOOST_NO_CXX11_DECLTYPE_N3276
#endif

#if _MSC_FULL_VER >= 180020827
#define BOOST_HAS_EXPM1
#define BOOST_HAS_LOG1P
#endif

#if (_MSC_FULL_VER < 190023026)
#  define BOOST_NO_CXX11_NOEXCEPT
#  define BOOST_NO_CXX11_DEFAULTED_MOVES
#  define BOOST_NO_CXX11_REF_QUALIFIERS
#  define BOOST_NO_CXX11_USER_DEFINED_LITERALS
#  define BOOST_NO_CXX11_ALIGNAS
#  define BOOST_NO_CXX11_INLINE_NAMESPACES
#  define BOOST_NO_CXX11_CHAR16_T
#  define BOOST_NO_CXX11_CHAR32_T
#  define BOOST_NO_CXX11_UNICODE_LITERALS
#  define BOOST_NO_CXX14_DECLTYPE_AUTO
#  define BOOST_NO_CXX14_INITIALIZED_LAMBDA_CAPTURES
#  define BOOST_NO_CXX14_RETURN_TYPE_DEDUCTION
#  define BOOST_NO_CXX14_BINARY_LITERALS
#  define BOOST_NO_CXX14_GENERIC_LAMBDAS
#  define BOOST_NO_CXX14_DIGIT_SEPARATORS
#  define BOOST_NO_CXX11_THREAD_LOCAL
#  define BOOST_NO_CXX11_UNRESTRICTED_UNION
#endif
#if (_MSC_FULL_VER < 190024210)
#  define BOOST_NO_CXX14_VARIABLE_TEMPLATES
#  define BOOST_NO_SFINAE_EXPR
#  define BOOST_NO_CXX11_CONSTEXPR
#endif

#if (_MSC_VER < 1910)
#  define BOOST_NO_CXX14_AGGREGATE_NSDMI
#endif

#if (_MSC_VER < 1911) || (_MSVC_LANG < 201703)
#  define BOOST_NO_CXX17_STRUCTURED_BINDINGS
#  define BOOST_NO_CXX17_IF_CONSTEXPR
#endif

#define BOOST_NO_COMPLETE_VALUE_INITIALIZATION
#  define BOOST_NO_TWO_PHASE_NAME_LOOKUP

#if (_MSC_VER < 1912) || (_MSVC_LANG < 201402)
#define BOOST_NO_CXX11_SFINAE_EXPR
#endif
#if (_MSC_VER < 1915) || (_MSVC_LANG < 201402)
#  define BOOST_NO_CXX14_CONSTEXPR
#endif
#if (_MSC_VER < 1912) || (_MSVC_LANG < 201703)
#define BOOST_NO_CXX17_INLINE_VARIABLES
#define BOOST_NO_CXX17_FOLD_EXPRESSIONS
#endif

#ifdef _M_CEE
#ifndef BOOST_NO_CXX11_THREAD_LOCAL
#  define BOOST_NO_CXX11_THREAD_LOCAL
#endif
#ifndef BOOST_NO_SFINAE_EXPR
#  define BOOST_NO_SFINAE_EXPR
#endif
#ifndef BOOST_NO_CXX11_REF_QUALIFIERS
#  define BOOST_NO_CXX11_REF_QUALIFIERS
#endif
#endif
#ifdef _M_CEE_PURE
#ifndef BOOST_NO_CXX11_CONSTEXPR
#  define BOOST_NO_CXX11_CONSTEXPR
#endif
#endif

#ifndef BOOST_ABI_PREFIX
#  define BOOST_ABI_PREFIX "boost/config/abi/msvc_prefix.hpp"
#endif
#ifndef BOOST_ABI_SUFFIX
#  define BOOST_ABI_SUFFIX "boost/config/abi/msvc_suffix.hpp"
#endif

#ifndef BOOST_COMPILER
# if defined(UNDER_CE)
#   if _MSC_VER < 1400
#      if defined(BOOST_ASSERT_CONFIG)
#         error "boost: Unknown EVC++ compiler version - please run the configure tests and report the results"
#      else
#         pragma message("boost: Unknown EVC++ compiler version - please run the configure tests and report the results")
#      endif
#   elif _MSC_VER < 1500
#     define BOOST_COMPILER_VERSION evc8
#   elif _MSC_VER < 1600
#     define BOOST_COMPILER_VERSION evc9
#   elif _MSC_VER < 1700
#     define BOOST_COMPILER_VERSION evc10
#   elif _MSC_VER < 1800 
#     define BOOST_COMPILER_VERSION evc11 
#   elif _MSC_VER < 1900 
#     define BOOST_COMPILER_VERSION evc12
#   elif _MSC_VER < 2000  
#     define BOOST_COMPILER_VERSION evc14
#   else
#      if defined(BOOST_ASSERT_CONFIG)
#         error "boost: Unknown EVC++ compiler version - please run the configure tests and report the results"
#      else
#         pragma message("boost: Unknown EVC++ compiler version - please run the configure tests and report the results")
#      endif
#   endif
# else
#   if _MSC_VER < 1200
#     define BOOST_COMPILER_VERSION 5.0
#   elif _MSC_VER < 1300
#     define BOOST_COMPILER_VERSION 6.0
#   elif _MSC_VER < 1310
#     define BOOST_COMPILER_VERSION 7.0
#   elif _MSC_VER < 1400
#     define BOOST_COMPILER_VERSION 7.1
#   elif _MSC_VER < 1500
#     define BOOST_COMPILER_VERSION 8.0
#   elif _MSC_VER < 1600
#     define BOOST_COMPILER_VERSION 9.0
#   elif _MSC_VER < 1700
#     define BOOST_COMPILER_VERSION 10.0
#   elif _MSC_VER < 1800 
#     define BOOST_COMPILER_VERSION 11.0
#   elif _MSC_VER < 1900
#     define BOOST_COMPILER_VERSION 12.0
#   elif _MSC_VER < 1910
#     define BOOST_COMPILER_VERSION 14.0
#   elif _MSC_VER < 1920
#     define BOOST_COMPILER_VERSION 14.1
#   elif _MSC_VER < 1930
#     define BOOST_COMPILER_VERSION 14.2
#   else
#     define BOOST_COMPILER_VERSION _MSC_VER
#   endif
# endif

#  define BOOST_COMPILER "Microsoft Visual C++ version " BOOST_STRINGIZE(BOOST_COMPILER_VERSION)
#endif

#include <boost/config/pragma_message.hpp>

#if (_MSC_VER > 1920)
#  if defined(BOOST_ASSERT_CONFIG)
#     error "Boost.Config is older than your current compiler version."
#  elif !defined(BOOST_CONFIG_SUPPRESS_OUTDATED_MESSAGE)
#  endif
#endif
