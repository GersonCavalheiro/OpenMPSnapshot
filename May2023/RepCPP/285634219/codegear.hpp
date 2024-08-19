


#if !defined( BOOST_WITH_CODEGEAR_WARNINGS )
# pragma warn -8004 
# pragma warn -8008 
# pragma warn -8066 
# pragma warn -8104 
# pragma warn -8105 
#endif
#if (__CODEGEARC__ > 0x620)
#  if defined(BOOST_ASSERT_CONFIG)
#     error "Unknown compiler version - please run the configure tests and report the results"
#  else
#     pragma message( "Unknown compiler version - please run the configure tests and report the results")
#  endif
#endif

#if (__CODEGEARC__ <= 0x613)
#  define BOOST_NO_INTEGRAL_INT64_T
#  define BOOST_NO_DEPENDENT_NESTED_DERIVATIONS
#  define BOOST_NO_PRIVATE_IN_AGGREGATE
#  define BOOST_NO_USING_DECLARATION_OVERLOADS_FROM_TYPENAME_BASE
#  define BOOST_NO_LIMITS_COMPILE_TIME_CONSTANTS
#  define BOOST_SP_NO_SP_CONVERTIBLE
#endif

#if (__CODEGEARC__ <= 0x620)
#  define BOOST_NO_TYPENAME_WITH_CTOR    
#  define BOOST_FUNCTION_SCOPE_USING_DECLARATION_BREAKS_ADL
#  define BOOST_NO_MEMBER_TEMPLATE_FRIENDS
#  define BOOST_NO_NESTED_FRIENDSHIP     
#  define BOOST_NO_USING_TEMPLATE
#  define BOOST_NO_TWO_PHASE_NAME_LOOKUP
#  define BOOST_MPL_CFG_NO_PREPROCESSED_HEADERS

#  ifdef NDEBUG
#     include <cstring>
#     undef strcmp
#  endif
#  include <errno.h>
#  ifndef errno
#     define errno errno
#  endif

#endif
#define BOOST_HAS_CHAR16_T
#define BOOST_HAS_CHAR32_T
#define BOOST_HAS_LONG_LONG
#define BOOST_HAS_DECLTYPE
#define BOOST_HAS_EXPLICIT_CONVERSION_OPS
#define BOOST_HAS_SCOPED_ENUM
#define BOOST_HAS_STD_TYPE_TRAITS

#define BOOST_NO_AUTO_DECLARATIONS
#define BOOST_NO_AUTO_MULTIDECLARATIONS
#define BOOST_NO_CONCEPTS
#define BOOST_NO_CONSTEXPR
#define BOOST_NO_DEFAULTED_FUNCTIONS
#define BOOST_NO_DELETED_FUNCTIONS
#define BOOST_NO_EXTERN_TEMPLATE
#define BOOST_NO_FUNCTION_TEMPLATE_DEFAULT_ARGS
#define BOOST_NO_INITIALIZER_LISTS
#define BOOST_NO_LAMBDAS
#define BOOST_NO_NULLPTR
#define BOOST_NO_RAW_LITERALS
#define BOOST_NO_RVALUE_REFERENCES
#define BOOST_NO_SFINAE_EXPR
#define BOOST_NO_STATIC_ASSERT
#define BOOST_NO_TEMPLATE_ALIASES
#define BOOST_NO_UNICODE_LITERALS
#define BOOST_NO_VARIADIC_TEMPLATES

#define BOOST_HAS_TR1_HASH
#define BOOST_HAS_TR1_TYPE_TRAITS
#define BOOST_HAS_TR1_UNORDERED_MAP
#define BOOST_HAS_TR1_UNORDERED_SET

#define BOOST_HAS_MACRO_USE_FACET

#define BOOST_NO_INITIALIZER_LISTS

#ifdef _WIN32
#  define BOOST_HAS_STDINT_H
#endif

#if !defined(__STRICT_ANSI__)
#  define BOOST_HAS_MS_INT64
#endif
#if !defined(_CPPUNWIND) && !defined(BOOST_CPPUNWIND) && !defined(__EXCEPTIONS)
#  define BOOST_NO_EXCEPTIONS
#endif
#if !defined(__STRICT_ANSI__)
#  define BOOST_HAS_DIRENT_H
#endif
#if !defined(__STRICT_ANSI__)
#  define BOOST_HAS_DECLSPEC
#endif
#ifndef BOOST_ABI_PREFIX
#  define BOOST_ABI_PREFIX "boost/config/abi/borland_prefix.hpp"
#endif
#ifndef BOOST_ABI_SUFFIX
#  define BOOST_ABI_SUFFIX "boost/config/abi/borland_suffix.hpp"
#endif
#  pragma defineonoption BOOST_DISABLE_WIN32 -A
#if defined(_MSC_VER) && (_MSC_VER <= 1200)
#  define BOOST_NO_ARGUMENT_DEPENDENT_LOOKUP
#  define BOOST_NO_VOID_RETURNS
#endif

#define BOOST_COMPILER "CodeGear C++ version " BOOST_STRINGIZE(__CODEGEARC__)

